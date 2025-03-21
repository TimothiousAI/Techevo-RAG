"""
Tool implementations for the Techevo-RAG agent.
These tools are used by the agent to interact with Gmail, Drive, and perform RAG operations.
"""

import io
import json
import os
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import base64
from datetime import datetime
import aiohttp
from tqdm import tqdm

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

from pydantic_ai.agent.tool import tool
from pydantic_ai.agent.context import AgentContext
from pydantic_ai.agent.retry import ModelRetry
from pydantic_ai.agent.parallel import parallel

# Cache file for storing email search results
CACHE_FILE = "cache.json"

# Initialize cache
def load_cache() -> Dict[str, Any]:
    """Load the cache from the cache file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"email_searches": {}, "drive_searches": {}}
    return {"email_searches": {}, "drive_searches": {}}

def save_cache(cache: Dict[str, Any]) -> None:
    """Save the cache to the cache file."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

@tool("search_emails")
async def search_emails(ctx: AgentContext, query: str) -> List[Dict[str, Any]]:
    """
    Search for emails matching the query in Gmail.
    
    Args:
        ctx: The agent context
        query: The search query for emails
        
    Returns:
        A list of emails with id, snippet, and attachment metadata
    """
    # Check if we have cached results
    cache = load_cache()
    cache_key = f"email_{query}"
    
    if cache_key in cache["email_searches"]:
        ctx.log.info(f"Using cached results for email search query: {query}")
        return cache["email_searches"][cache_key]
    
    try:
        ctx.log.info(f"Searching emails with query: {query}")
        gmail_service = ctx.deps.gmail_service
        
        response = gmail_service.users().messages().list(userId='me', q=query).execute()
        messages = response.get('messages', [])
        
        results = []
        for message in messages:
            msg_id = message['id']
            msg = gmail_service.users().messages().get(userId='me', id=msg_id).execute()
            
            # Extract basic info
            headers = msg['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
            
            # Look for attachments
            attachments = []
            
            if 'parts' in msg['payload']:
                for part in msg['payload']['parts']:
                    if part.get('filename') and part.get('body', {}).get('attachmentId'):
                        attachments.append({
                            'id': part['body']['attachmentId'],
                            'filename': part['filename'],
                            'mimeType': part.get('mimeType', 'application/octet-stream')
                        })
            
            results.append({
                'id': msg_id,
                'snippet': msg.get('snippet', ''),
                'subject': subject,
                'sender': sender,
                'date': date,
                'attachments': attachments
            })
        
        # Cache the results
        cache["email_searches"][cache_key] = results
        save_cache(cache)
        
        # Store in state for UI
        if 'processed_emails' not in ctx.deps.state:
            ctx.deps.state['processed_emails'] = []
        ctx.deps.state['processed_emails'].extend(results)
        
        return results
    
    except Exception as e:
        ctx.log.error(f"Error searching emails: {str(e)}")
        raise ModelRetry(f"Error searching emails: {str(e)}")

@tool("download_attachment")
async def download_attachment(
    ctx: AgentContext, 
    email_id: str, 
    attachment_id: str, 
    folder_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download an attachment from an email and upload it to Google Drive.
    
    Args:
        ctx: The agent context
        email_id: The ID of the email
        attachment_id: The ID of the attachment
        folder_id: The ID of the Drive folder to upload to (optional)
        
    Returns:
        Information about the downloaded attachment and Drive upload
    """
    try:
        ctx.log.info(f"Downloading attachment {attachment_id} from email {email_id}")
        gmail_service = ctx.deps.gmail_service
        drive_service = ctx.deps.drive_service
        
        # Use default folder if none provided
        if not folder_id and os.getenv('DEFAULT_DRIVE_FOLDER_ID'):
            folder_id = os.getenv('DEFAULT_DRIVE_FOLDER_ID')
        
        # Get the attachment
        attachment = gmail_service.users().messages().attachments().get(
            userId='me', messageId=email_id, id=attachment_id
        ).execute()
        
        # Get the email to extract filename
        msg = gmail_service.users().messages().get(userId='me', id=email_id).execute()
        
        # Find the filename for this attachment
        filename = "unknown_attachment"
        if 'parts' in msg['payload']:
            for part in msg['payload']['parts']:
                if part.get('body', {}).get('attachmentId') == attachment_id:
                    filename = part.get('filename', filename)
        
        # Decode the attachment data
        data = base64.urlsafe_b64decode(attachment['data'])
        
        # Compute hash for deduplication
        file_hash = hashlib.sha256(data).hexdigest()
        
        # Check if already processed via Supabase
        supabase = ctx.deps.supabase
        
        try:
            # First check if the item has already been processed
            check_result = await supabase.table('processed_items').select('*').eq('file_hash', file_hash).execute()
            if check_result.data and len(check_result.data) > 0:
                ctx.log.info(f"Attachment {filename} already processed (hash: {file_hash})")
                return {
                    'status': 'skipped',
                    'reason': 'already_processed',
                    'file_hash': file_hash,
                    'filename': filename,
                    'email_id': email_id
                }
        except Exception as db_err:
            ctx.log.error(f"Error checking Supabase: {str(db_err)}")
            # Continue processing even if the check fails
        
        # Upload to Drive
        media = MediaIoBaseUpload(
            io.BytesIO(data),
            mimetype='application/octet-stream',
            resumable=True
        )
        
        drive_params = {
            'name': filename,
            'media_body': media
        }
        
        if folder_id:
            drive_params['parents'] = [folder_id]
        
        drive_file = drive_service.files().create(**drive_params).execute()
        
        # Track progress in Supabase
        try:
            result = await track_progress(
                ctx, 
                email_id=email_id, 
                file_hash=file_hash, 
                status='downloaded_and_uploaded',
                file_id=drive_file.get('id'),
                filename=filename
            )
        except Exception as track_err:
            ctx.log.error(f"Error tracking progress: {str(track_err)}")
        
        return {
            'status': 'success',
            'file_hash': file_hash,
            'filename': filename,
            'email_id': email_id,
            'drive_file_id': drive_file.get('id')
        }
    
    except Exception as e:
        ctx.log.error(f"Error downloading attachment: {str(e)}")
        
        # Track failure
        try:
            await track_progress(
                ctx, 
                email_id=email_id, 
                file_hash="unknown",  # We might not have been able to compute the hash
                status='error',
                error_message=str(e)
            )
        except Exception as track_err:
            ctx.log.error(f"Error tracking error progress: {str(track_err)}")
        
        raise ModelRetry(f"Error downloading attachment: {str(e)}")

@tool("search_drive")
async def search_drive(ctx: AgentContext, folder_id: Optional[str] = None, query: str = "") -> List[Dict[str, Any]]:
    """
    Search for files in Google Drive.
    
    Args:
        ctx: The agent context
        folder_id: The ID of the folder to search in (optional)
        query: Additional search query (optional)
        
    Returns:
        A list of files with id, name, and other metadata
    """
    # Check if we have cached results
    cache = load_cache()
    cache_key = f"drive_{folder_id}_{query}"
    
    if cache_key in cache["drive_searches"]:
        ctx.log.info(f"Using cached results for drive search in folder {folder_id}")
        return cache["drive_searches"][cache_key]
    
    try:
        ctx.log.info(f"Searching Drive folder {folder_id}")
        drive_service = ctx.deps.drive_service
        
        # Build the query
        drive_query = ""
        
        if folder_id:
            drive_query = f"'{folder_id}' in parents"
        
        if query:
            if drive_query:
                drive_query += f" and {query}"
            else:
                drive_query = query
        
        # If no specific query is provided, just return all files
        response = drive_service.files().list(
            q=drive_query if drive_query else None,
            fields="files(id, name, mimeType, size, createdTime, modifiedTime)"
        ).execute()
        
        files = response.get('files', [])
        
        # Cache the results
        cache["drive_searches"][cache_key] = files
        save_cache(cache)
        
        return files
    
    except Exception as e:
        ctx.log.error(f"Error searching Drive: {str(e)}")
        raise ModelRetry(f"Error searching Drive: {str(e)}")

@parallel
async def chunk_and_embed_document(text: str, model: SentenceTransformer) -> List[Tuple[str, np.ndarray]]:
    """
    Chunk a document and embed the chunks in parallel.
    
    Args:
        text: The document text
        model: The embedding model
        
    Returns:
        A list of (chunk, embedding) tuples
    """
    # Split into chunks of approximately 500 tokens
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # ~500 tokens
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Embed chunks
    embeddings = model.encode(chunks)
    
    return list(zip(chunks, embeddings))

@tool("perform_rag")
async def perform_rag(ctx: AgentContext, query: str, documents: List[str]) -> Dict[str, Any]:
    """
    Perform RAG operations on documents.
    
    Args:
        ctx: The agent context
        query: The query to answer
        documents: List of document contents
        
    Returns:
        The RAG result with query, response, and used chunks
    """
    try:
        ctx.log.info(f"Performing RAG on {len(documents)} documents for query: {query}")
        
        # Get the embedding model and FAISS index
        # Use all-MiniLM-L6-v2 which outputs 768-dimensional embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        faiss_index = ctx.deps.faiss_index
        
        # Process documents in parallel
        all_chunks_with_embeddings = []
        for document in documents:
            # Use parallelized chunking and embedding
            chunk_embeddings = await chunk_and_embed_document(document, model)
            all_chunks_with_embeddings.extend(chunk_embeddings)
        
        chunks = [c for c, _ in all_chunks_with_embeddings]
        embeddings = np.array([e for _, e in all_chunks_with_embeddings])
        
        # Add to FAISS index
        if len(embeddings) > 0:
            faiss_index.add(embeddings)
        
        # Embed the query
        query_embedding = model.encode([query])[0]
        
        # Search for relevant chunks
        k = min(5, len(chunks))  # Get top-5 chunks or all if fewer
        if k > 0:
            D, I = faiss_index.search(np.array([query_embedding]), k)
            
            # Get the top chunks
            top_chunks = [chunks[i] for i in I[0]]
            
            # Generate response with Archon MCP
            from agent_prompts import RAG_PROMPT
            
            # Keep prompt under 5k tokens to avoid Archon MCP limits
            chunks_text = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])
            if len(chunks_text) > 4000:  # Leave room for the rest of the prompt
                chunks_text = chunks_text[:4000] + "...[truncated]"
                
            prompt = RAG_PROMPT.format(
                query=query,
                chunks=chunks_text
            )
            
            # Use Archon MCP to generate the response
            try:
                # In an actual implementation using Cursor's MCP protocol
                import cursor.mcp.archon as archon_mcp
                
                # Create a thread if needed
                if not hasattr(ctx.deps, 'archon_thread_id'):
                    ctx.deps.archon_thread_id = await archon_mcp.create_thread(random_string="init")
                
                # Run the agent
                response_text = await archon_mcp.run_agent(
                    thread_id=ctx.deps.archon_thread_id,
                    user_input=prompt
                )
            except Exception as archon_err:
                ctx.log.error(f"Error using Archon MCP: {str(archon_err)}")
                # Use the ArchonClient directly as fallback
                response = await ctx.deps.archon_client.generate(
                    model="openai:gpt-4o",
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=1000
                )
                response_text = response.get('response', 'No response generated')
            
            # Store in Supabase
            try:
                result = {
                    'query': query,
                    'response': response_text,
                    'chunks_used': len(top_chunks),
                    'timestamp': datetime.now().isoformat()
                }
                
                supabase = ctx.deps.supabase
                await supabase.table('rag_results').insert(result).execute()
            except Exception as db_err:
                ctx.log.error(f"Error storing RAG result in Supabase: {str(db_err)}")
            
            return {
                'query': query,
                'response': response_text,
                'chunks_used': top_chunks,
                'status': 'success'
            }
        else:
            return {
                'query': query,
                'response': 'No relevant information found for this query.',
                'chunks_used': [],
                'status': 'no_chunks'
            }
    
    except Exception as e:
        ctx.log.error(f"Error performing RAG: {str(e)}")
        raise ModelRetry(f"Error performing RAG: {str(e)}")

@tool("track_progress")
async def track_progress(
    ctx: AgentContext, 
    email_id: str, 
    file_hash: str, 
    status: str,
    file_id: Optional[str] = None,
    filename: Optional[str] = None,
    error_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Track progress of processing in Supabase.
    
    Args:
        ctx: The agent context
        email_id: The ID of the email
        file_hash: The hash of the file
        status: The status (e.g., 'downloaded', 'error')
        file_id: The ID of the file in Drive (optional)
        filename: The name of the file (optional)
        error_message: Error message if status is 'error' (optional)
        
    Returns:
        The tracking record
    """
    try:
        ctx.log.info(f"Tracking progress for {filename} ({file_hash}): {status}")
        
        record = {
            'email_id': email_id,
            'file_hash': file_hash,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        if file_id:
            record['file_id'] = file_id
            
        if filename:
            record['filename'] = filename
            
        if error_message:
            record['error_message'] = error_message
        
        # Insert into Supabase
        try:
            supabase = ctx.deps.supabase
            
            # Using MCP directly when available
            try:
                import cursor.mcp.supabase as supabase_mcp
                
                # Enable unsafe mode for DB operations
                await supabase_mcp.live_dangerously(service="database", enable=True)
                
                # Insert with a direct SQL query using MCP
                insert_query = f"""
                INSERT INTO processed_items (email_id, file_hash, status, timestamp, 
                                           {', '.join([k for k in record if k not in ['email_id', 'file_hash', 'status', 'timestamp']])})
                VALUES ('{email_id}', '{file_hash}', '{status}', '{record['timestamp']}',
                       {', '.join(["'" + str(record[k]) + "'" for k in record if k not in ['email_id', 'file_hash', 'status', 'timestamp']])})
                """
                result = await supabase_mcp.execute_sql_query(query=insert_query)
                
            except ImportError:
                # Fall back to the Supabase SDK
                result = await supabase.table('processed_items').insert(record).execute()
        
        except Exception as db_err:
            ctx.log.error(f"Error inserting into Supabase: {str(db_err)}")
            # Create a JSON file as backup if database fails
            backup_file = f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, 'w') as f:
                json.dump(record, f)
            ctx.log.info(f"Progress saved to backup file: {backup_file}")
        
        return record
    
    except Exception as e:
        ctx.log.error(f"Error tracking progress: {str(e)}")
        # Don't retry this as it's a tracking function
        return {
            'status': 'error',
            'error': str(e),
            'email_id': email_id,
            'file_hash': file_hash
        } 