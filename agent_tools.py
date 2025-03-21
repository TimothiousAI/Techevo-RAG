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

from agent_prompts import (
    RAG_PROMPT,
    EMAIL_SEARCH_PROMPT, 
    DOCUMENT_PROCESSING_PROMPT,
    SEARCH_CRITERIA_PROMPT
)

# Cache file for storing email search results
CACHE_FILE = "cache/search_results.json"
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

# Initialize cache
def load_cache() -> Dict[str, Any]:
    """
    Load cached search results.
    
    Returns:
        Dict containing cached results
    """
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse cache file {CACHE_FILE}")
            logfire.warning(f"Could not parse cache file {CACHE_FILE}")
            return {}
    return {}

def save_cache(cache: Dict[str, Any]) -> None:
    """
    Save search results to cache.
    
    Args:
        cache: Dict containing results to cache
    """
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
    logfire.info(f"Searching emails with query: {query}")
    
    # Check if results are cached
    cache = load_cache()
    cache_key = f"email_{hashlib.md5(query.encode()).hexdigest()}"
    
    if cache_key in cache:
        logger.info(f"Using cached email results for query: {query}")
        logfire.info(f"Using cached email results for query: {query}")
        return cache[cache_key]
    
    try:
        ctx.log.info(f"Searching emails with query: {query}")
        gmail_service = ctx.deps.gmail_service
        
        # Use Archon to refine the search criteria
        archon_client = ctx.deps.archon_client
        prompt = EMAIL_SEARCH_PROMPT.format(query=query)
        
        response = await archon_client.generate(
            model="openai:gpt-4o", 
            prompt=prompt,
            temperature=0.2
        )
        
        search_criteria = response.get('response', '').strip()
        
        # Default to original query if no criteria generated
        if not search_criteria or len(search_criteria) < 5:
            search_criteria = query
            
        logger.info(f"Refined search criteria: {search_criteria}")
        logfire.info(f"Refined search criteria: {search_criteria}")
        
        # Perform the search
        response = gmail_service.users().messages().list(
            userId='me',
            q=search_criteria,
            maxResults=10
        ).execute()
        
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
        cache[cache_key] = results
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
    logfire.info(f"Searching Drive folder {folder_id}")
    
    drive_service = ctx.deps.drive_service
    
    # Generate search criteria if query provided
    search_query = f"'{folder_id}' in parents"
    
    if query:
        try:
            # Use Archon to refine search criteria
            archon_client = ctx.deps.archon_client
            prompt = SEARCH_CRITERIA_PROMPT.format(query=query)
            
            response = await archon_client.generate(
                model="openai:gpt-4o",
                prompt=prompt,
                temperature=0.2
            )
            
            criteria = response.get('response', '').strip()
            
            if criteria and len(criteria) > 3:
                search_query += f" and (name contains '{criteria}' or fullText contains '{criteria}')"
                logger.info(f"Using refined Drive search criteria: {criteria}")
                logfire.info(f"Using refined Drive search criteria: {criteria}")
        except Exception as e:
            logger.error(f"Error generating Drive search criteria: {str(e)}")
            logfire.error(f"Error generating Drive search criteria: {str(e)}")
            
            # Basic fallback
            if query:
                search_query += f" and (name contains '{query}' or fullText contains '{query}')"
    
    # Perform the search
    results = drive_service.files().list(
        q=search_query,
        spaces='drive',
        fields='files(id, name, mimeType, modifiedTime, size)',
        orderBy='modifiedTime desc',
        pageSize=20
    ).execute()
    
    files = results.get('files', [])
    
    # Fetch file contents for text files that could be useful for RAG
    for file in files:
        try:
            # Only fetch content for text files or PDFs
            mime_type = file.get('mimeType', '')
            if mime_type.startswith('text/') or mime_type in [
                'application/pdf',
                'application/json',
                'application/vnd.google-apps.document'
            ]:
                file_id = file['id']
                
                # Handle Google Docs differently
                if mime_type == 'application/vnd.google-apps.document':
                    # Export as plain text
                    content = drive_service.files().export(
                        fileId=file_id,
                        mimeType='text/plain'
                    ).execute()
                    
                    # For exported files, content is already bytes
                    text_content = content.decode('utf-8', errors='replace') if isinstance(content, bytes) else content
                    
                else:
                    # Download binary files
                    content = drive_service.files().get_media(
                        fileId=file_id
                    ).execute()
                    
                    # Try to decode text files
                    if mime_type.startswith('text/') or mime_type == 'application/json':
                        text_content = content.decode('utf-8', errors='replace')
                    else:
                        # For non-text files (like PDFs), we would need special handling
                        # This is a placeholder - real implementation would use libraries like PyPDF2
                        text_content = f"[Binary content of type {mime_type}]"
                
                # Store the content in state for RAG
                if 'drive_contents' not in ctx.deps.state:
                    ctx.deps.state['drive_contents'] = {}
                
                ctx.deps.state['drive_contents'][file_id] = text_content
                
                # Add a snippet to the file metadata
                if text_content:
                    file['snippet'] = text_content[:200] + '...' if len(text_content) > 200 else text_content
                
                logger.info(f"Fetched content for Drive file: {file['name']}")
                logfire.info(f"Fetched content for Drive file: {file['name']}")
                
        except Exception as e:
            logger.error(f"Error fetching content for file {file.get('name', 'unknown')}: {str(e)}")
            logfire.error(f"Error fetching content for file {file.get('name', 'unknown')}: {str(e)}")
    
    logfire.info(f"Found {len(files)} files in Drive")
    return files

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
    logfire.info(f"Performing RAG for query: {query}")
    
    # Gather all text content from state
    all_documents = list(documents)  # Start with the provided documents
    
    # Add email content from state if available
    if 'email_contents' in ctx.deps.state:
        all_documents.extend(ctx.deps.state['email_contents'].values())
    
    # Add document content from state if available
    if 'document_contents' in ctx.deps.state:
        all_documents.extend(ctx.deps.state['document_contents'].values())
    
    # Add Drive content from state if available
    if 'drive_contents' in ctx.deps.state:
        all_documents.extend(ctx.deps.state['drive_contents'].values())
    
    # Remove empty documents and truncate very long ones
    filtered_documents = []
    for doc in all_documents:
        if doc and isinstance(doc, str):
            # Truncate very long documents to a reasonable size
            if len(doc) > 5000:
                doc = doc[:5000] + "... [truncated]"
            filtered_documents.append(doc)
    
    if not filtered_documents:
        logger.warning("No valid documents available for RAG")
        logfire.warning("No valid documents available for RAG")
        return {
            'query': query,
            'response': "I don't have enough context to answer this query.",
            'chunks_used': []
        }
    
    # Load the embedding model - using the all-MiniLM-L6-v2 model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings for the documents
        document_embeddings = embedding_model.encode(filtered_documents)
        
        # Generate embedding for the query
        query_embedding = embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = np.dot(document_embeddings, query_embedding) / (
            np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top 5 most relevant document chunks
        top_indices = similarities.argsort()[-5:][::-1]
        top_chunks = [filtered_documents[i] for i in top_indices]
        
        # Use Archon to generate a response using the retrieved chunks
        context = "\n\n---\n\n".join(top_chunks)
        
        # Ensure the prompt stays under 5k tokens
        if len(context) > 4000:
            context = context[:4000] + "... [truncated]"
        
        prompt = RAG_PROMPT.format(query=query, context=context)
        
        try:
            # Direct MCP call using the Cursor MCP protocol
            try:
                import cursor.mcp.archon as archon_mcp
                
                # Create a thread if we don't have one
                if not hasattr(ctx.deps.archon_client, 'thread_id') or not ctx.deps.archon_client.thread_id:
                    thread_id = await archon_mcp.create_thread(random_string="init")
                    ctx.deps.archon_client.thread_id = thread_id
                
                # Run the agent with the generated prompt
                response_text = await archon_mcp.run_agent(
                    thread_id=ctx.deps.archon_client.thread_id,
                    user_input=prompt
                )
                
            except ImportError:
                # Fall back to the client's generate method
                response = await ctx.deps.archon_client.generate(
                    model="openai:gpt-4o", 
                    prompt=prompt,
                    temperature=0.5,
                    max_tokens=500
                )
                response_text = response.get('response', '')
            
            # Store the result in the database
            try:
                rag_result = {
                    'query': query,
                    'response': response_text,
                    'chunks_used': top_chunks,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Use direct MCP for Supabase insert
                try:
                    import cursor.mcp.supabase as supabase_mcp
                    
                    await supabase_mcp.insert('rag_results', {
                        'query': query,
                        'result': response_text,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    logger.info("Stored RAG result in Supabase")
                    logfire.info("Stored RAG result in Supabase")
                    
                except ImportError:
                    # Fall back to Supabase client
                    if ctx.deps.supabase:
                        await ctx.deps.supabase.table('rag_results').insert({
                            'query': query,
                            'result': response_text,
                            'timestamp': datetime.now().isoformat()
                        }).execute()
                        
                        logger.info("Stored RAG result in Supabase")
                        logfire.info("Stored RAG result in Supabase")
                
            except Exception as e:
                logger.error(f"Error storing RAG result in database: {str(e)}")
                logfire.error(f"Error storing RAG result in database: {str(e)}")
                
                # Create a backup file
                os.makedirs('cache/rag_results', exist_ok=True)
                backup_file = f"cache/rag_results/rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(backup_file, 'w') as f:
                    json.dump(rag_result, f, indent=2)
                
                logger.info(f"Saved RAG result to backup file: {backup_file}")
                logfire.info(f"Saved RAG result to backup file: {backup_file}")
            
            return rag_result
            
        except Exception as e:
            error_msg = f"Error generating RAG response: {str(e)}"
            logger.error(error_msg)
            logfire.error(error_msg)
            
            return {
                'query': query,
                'response': "Error generating response. Please try again.",
                'chunks_used': top_chunks,
                'error': str(e)
            }
            
    except Exception as e:
        error_msg = f"Error in RAG processing: {str(e)}"
        logger.error(error_msg)
        logfire.error(error_msg)
        
        return {
            'query': query,
            'response': "Error processing documents. Please try again.",
            'chunks_used': [],
            'error': str(e)
        }

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