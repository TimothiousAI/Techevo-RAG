"""
Agent tools for the Techevo RAG system.

Implements tools for email search, attachment download, drive search, and RAG operations.
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import asyncio
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
import logfire
import io
import base64
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import httpx

from agent_prompts import (
    RAG_PROMPT, 
    EMAIL_SEARCH_PROMPT, 
    DOCUMENT_PROCESSING_PROMPT,
    SEARCH_CRITERIA_PROMPT
)

# Configure logging
logger = logfire.configure()

# Cache files
CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / 'search_results.json'

class AgentContext:
    """Context for agent tool functions."""
    
    def __init__(self, agent, deps, log=None):
        """Initialize agent context.
        
        Args:
            agent: The agent instance
            deps: Dependencies including services and state
            log: Logger instance (defaults to module logger)
        """
        self.agent = agent
        self.deps = deps
        self.log = log or logger

def load_cache():
    """Load cached email and drive search results."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    return {
        'email_searches': {},
        'drive_searches': {},
        'processed_emails': [],
        'attachments_content': {},
        'drive_contents': {}
    }

def save_cache(data):
    """Save search results to cache file."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

async def search_emails(ctx: AgentContext, query: str) -> List[Dict]:
    """
    Search for emails matching the query in Gmail.
    
    Args:
        ctx: Agent context containing deps and logger
        query: The user query
        
    Returns:
        List of emails with metadata
    """
    ctx.log.info(f"Searching emails with raw query: {query}")
    gmail_service = ctx.deps.gmail_service
    
    if not gmail_service:
        ctx.log.error("Gmail service not initialized")
        raise ValueError("Gmail service not initialized")
    
    try:
        # Use raw query directly without refinement
        ctx.log.info(f"Performing Gmail search with raw query: {query}")
        
        # Format 'full' retrieves the full email content
        results = gmail_service.users().messages().list(
            userId='me',
            q=query,
            maxResults=30  # Increased from 20 to ensure we get more results
        ).execute()
        
        ctx.log.info(f"Raw Gmail API response: {results}")
        
        messages = results.get('messages', [])
        ctx.log.info(f"Found {len(messages)} potential messages")
        emails = []
        
        for message in messages:
            email_data = gmail_service.users().messages().get(
                userId='me',
                id=message['id'],
                format='full'
            ).execute()
            
            # Extract headers
            headers = {}
            for header in email_data['payload']['headers']:
                headers[header['name'].lower()] = header['value']
            
            # Extract body content
            body = ''
            if 'parts' in email_data['payload']:
                for part in email_data['payload']['parts']:
                    if part['mimeType'] == 'text/plain' and 'data' in part.get('body', {}):
                        body += base64.urlsafe_b64decode(
                            part['body']['data'].encode('ASCII')
                        ).decode('utf-8')
            elif 'body' in email_data['payload'] and 'data' in email_data['payload']['body']:
                body += base64.urlsafe_b64decode(
                    email_data['payload']['body']['data'].encode('ASCII')
                ).decode('utf-8')
            
            # Extract attachments metadata
            attachments = []
            if 'parts' in email_data['payload']:
                for part in email_data['payload']['parts']:
                    if 'filename' in part and part['filename']:
                        attachments.append({
                            'id': part['body'].get('attachmentId', ''),
                            'filename': part['filename'],
                            'mimeType': part['mimeType']
                        })
            
            email = {
                'id': email_data['id'],
                'threadId': email_data['threadId'],
                'subject': headers.get('subject', 'No Subject'),
                'from': headers.get('from', 'Unknown'),
                'to': headers.get('to', 'Unknown'),
                'date': headers.get('date', 'Unknown'),
                'snippet': email_data.get('snippet', ''),
                'body': body,
                'attachments': attachments
            }
            
            # Log detailed info about each email
            ctx.log.info(f"Processed email: id={email['id']}, from={email['from']}, subject={email['subject']}, attachments={len(attachments)}")
                
            emails.append(email)
        
        # Store in state
        if hasattr(ctx.deps, 'state'):
            ctx.deps.state['processed_emails'] = emails
        
        ctx.log.info(f"Retrieved {len(emails)} matching emails (raw query search)")
        return emails
    
    except Exception as e:
        ctx.log.error(f"Error searching emails: {e}")
        ctx.log.error(f"Traceback: {traceback.format_exc()}")
        return []  # Return empty list instead of raising to continue with workflow

async def download_attachment(ctx: AgentContext, query: str) -> List[Dict]:
    """
    Download attachments from emails matching the query.
    
    Args:
        ctx: Agent context containing deps and logger
        query: The user query
        
    Returns:
        List of downloaded attachments
    """
    gmail_service = ctx.deps.gmail_service
    drive_service = ctx.deps.drive_service
    
    if not gmail_service or not drive_service:
        ctx.log.error("Gmail or Drive service not initialized")
        raise ValueError("Gmail or Drive service not initialized")
    
    # First, search for emails if not already in state
    if not hasattr(ctx.deps, 'state') or 'processed_emails' not in ctx.deps.state or not ctx.deps.state['processed_emails']:
        ctx.log.info("No emails in state, searching first")
        await search_emails(ctx, query)
    
    if not hasattr(ctx.deps, 'state') or 'processed_emails' not in ctx.deps.state or not ctx.deps.state['processed_emails']:
        ctx.log.error("No emails found to download attachments from")
        return []
    
    # Create attachments directory if it doesn't exist
    os.makedirs('attachments', exist_ok=True)
    
    # Find the default Drive folder ID, create one if not exists
    folder_id = os.getenv('DEFAULT_DRIVE_FOLDER_ID')
    if not folder_id:
        folder_metadata = {
            'name': 'TechevoRAG_Attachments',
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = drive_service.files().create(
            body=folder_metadata,
            fields='id'
        ).execute()
        folder_id = folder.get('id')
        ctx.log.info(f"Created new Drive folder with ID: {folder_id}")
    
    # Process attachments
    downloaded = []
    cache = load_cache()
    
    try:
        for email in ctx.deps.state['processed_emails']:
            if not email.get('attachments'):
                continue
                
            for attachment in email['attachments']:
                try:
                    ctx.log.info(f"Downloading attachment: {attachment['filename']} from email {email['id']}")
                    
                    # Download the attachment
                    attachment_data = gmail_service.users().messages().attachments().get(
                        userId='me',
                        messageId=email['id'],
                        id=attachment['id']
                    ).execute()
                    
                    file_data = base64.urlsafe_b64decode(attachment_data['data'])
                    
                    # Save locally
                    local_path = os.path.join('attachments', attachment['filename'])
                    with open(local_path, 'wb') as f:
                        f.write(file_data)
                    
                    # Upload to Drive
                    file_metadata = {
                        'name': attachment['filename'],
                        'parents': [folder_id]
                    }
                    
                    media = io.BytesIO(file_data)
                    uploaded_file = drive_service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields='id,name,webViewLink'
                    ).execute()
                    
                    result = {
                        'email_id': email['id'],
                        'attachment_id': attachment['id'],
                        'filename': attachment['filename'],
                        'local_path': local_path,
                        'drive_id': uploaded_file['id'],
                        'drive_link': uploaded_file['webViewLink']
                    }
                    
                    downloaded.append(result)
                    
                    # Store content in state for RAG
                    if 'attachments_content' not in cache:
                        cache['attachments_content'] = {}
                    
                    # Store content as base64 string in the cache
                    cache['attachments_content'][attachment['filename']] = base64.b64encode(file_data).decode('utf-8')
                    
                    if hasattr(ctx.deps, 'state'):
                        if 'attachments_content' not in ctx.deps.state:
                            ctx.deps.state['attachments_content'] = {}
                        ctx.deps.state['attachments_content'][attachment['filename']] = file_data
                    
                except Exception as e:
                    ctx.log.error(f"Error downloading attachment {attachment['filename']}: {e}")
                    ctx.log.error(f"Traceback: {traceback.format_exc()}")
                    downloaded.append({
                        'error': str(e),
                        'filename': attachment['filename'],
                        'email_id': email['id']
                    })
        
        # Save cache
        save_cache(cache)
        
        ctx.log.info(f"Downloaded {len(downloaded)} attachments")
        return downloaded
    
    except Exception as e:
        ctx.log.error(f"Error in download_attachment: {e}")
        ctx.log.error(f"Traceback: {traceback.format_exc()}")
        raise

async def search_drive(ctx: AgentContext, query: str) -> List[Dict]:
    """
    Search for files in Google Drive matching the query.
    
    Args:
        ctx: Agent context containing deps and logger
        query: The user query
        
    Returns:
        List of files with metadata
    """
    drive_service = ctx.deps.drive_service
    
    if not drive_service:
        ctx.log.error("Drive service not initialized")
        raise ValueError("Drive service not initialized")
    
    try:
        ctx.log.info(f"Searching Drive with query: {query}")
        
        # Execute search request (no caching - always live)
        search_results = drive_service.files().list(
            q=f"fullText contains '{query}' and trashed = false",
            spaces='drive',
            fields='files(id, name, mimeType, webViewLink, description)',
            pageSize=20
        ).execute()
        
        files = search_results.get('files', [])
        ctx.log.info(f"Found {len(files)} files in Drive")
        
        results = []
        for file in files:
            # Get more detailed metadata for each file
            try:
                file_metadata = drive_service.files().get(
                    fileId=file['id'], 
                    fields='id, name, mimeType, webViewLink, description, createdTime, modifiedTime, size, owners'
                ).execute()
                
                results.append({
                    'id': file_metadata.get('id', ''),
                    'name': file_metadata.get('name', ''),
                    'mimeType': file_metadata.get('mimeType', ''),
                    'link': file_metadata.get('webViewLink', ''),
                    'description': file_metadata.get('description', ''),
                    'created': file_metadata.get('createdTime', ''),
                    'modified': file_metadata.get('modifiedTime', ''),
                    'size': file_metadata.get('size', ''),
                    'owner': file_metadata.get('owners', [{}])[0].get('displayName', '') if file_metadata.get('owners') else ''
                })
            except Exception as e:
                ctx.log.error(f"Error getting metadata for file {file['id']}: {str(e)}")
                # Add basic info we already have
                results.append({
                    'id': file.get('id', ''),
                    'name': file.get('name', ''),
                    'mimeType': file.get('mimeType', ''),
                    'link': file.get('webViewLink', ''),
                    'error': str(e)
                })
                
        # Store in state
        if hasattr(ctx.deps, 'state'):
            ctx.deps.state['drive_files'] = results
        
        return results
    
    except Exception as e:
        ctx.log.error(f"Error searching Drive: {str(e)}")
        ctx.log.error(f"Traceback: {traceback.format_exc()}")
        return []

async def chunk_and_embed_document(document: str, model: SentenceTransformer, chunk_size: int = 500, overlap: int = 50) -> List[Tuple[str, np.ndarray]]:
    """
    Chunk a document into smaller pieces with overlap and generate embeddings.
    
    Args:
        document: The text document to chunk
        model: The SentenceTransformer model to use for embeddings
        chunk_size: The size of each chunk in words
        overlap: The number of words to overlap between chunks
        
    Returns:
        List of (chunk, embedding) tuples
    """
    try:
        # Split into words for more precise chunking with overlap
        words = document.split()
        chunks = []
        
        # Create chunks with overlap
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if not chunk_words:
                break
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
        # Generate embeddings for all chunks
        if chunks:
            try:
                embeddings = model.encode(chunks)
                return list(zip(chunks, embeddings))
            except Exception as e:
                # If batch is too large, process in smaller batches
                result = []
                batch_size = 10
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    batch_embeddings = model.encode(batch)
                    result.extend([(batch[j], batch_embeddings[j]) for j in range(len(batch))])
                return result
        return []
    except Exception as e:
        logger.error(f"Error in chunk_and_embed_document: {str(e)}")
        logger.error(traceback.format_exc())
        return []

async def perform_rag(ctx, query: str, documents: List[str]) -> Dict[str, Any]:
    """
    Perform RAG operation on documents using Gemini API.
    
    Args:
        ctx: Agent context containing deps and logger
        query: The user query
        documents: List of documents to process
        
    Returns:
        RAG operation result
    """
    try:
        ctx.log.info(f"Performing RAG on {len(documents)} documents for query: {query}")
        
        # Ensure we have the model loaded
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            ctx.log.info("Loaded SentenceTransformer model")
        except Exception as e:
            ctx.log.error(f"Error loading SentenceTransformer: {str(e)}")
            raise ValueError(f"Failed to load embedding model: {str(e)}")
        
        # Get or create FAISS index
        faiss_index = ctx.deps.faiss_index
        if faiss_index is None:
            ctx.log.warning("FAISS index not found in deps, creating new one")
            try:
                import faiss
                faiss_index = faiss.IndexFlatL2(768)  # dimension for all-MiniLM-L6-v2
                ctx.deps.faiss_index = faiss_index
            except Exception as e:
                ctx.log.error(f"Error creating FAISS index: {str(e)}")
                raise ValueError(f"Failed to create FAISS index: {str(e)}")
        
        all_chunks_with_embeddings = []
        
        # Chunk and embed documents with improved chunking
        for doc_idx, document in enumerate(documents):
            try:
                # Skip empty or invalid documents
                if not document or not isinstance(document, str):
                    ctx.log.warning(f"Skipping invalid document at index {doc_idx}")
                    continue
                
                # Use the improved chunking function with overlap
                chunk_embeddings = await chunk_and_embed_document(document, model)
                all_chunks_with_embeddings.extend(chunk_embeddings)
                ctx.log.info(f"Processed document {doc_idx+1}/{len(documents)} - created {len(chunk_embeddings)} chunks")
                
            except Exception as e:
                ctx.log.error(f"Error processing document {doc_idx}: {str(e)}")
                ctx.log.error(traceback.format_exc())
                # Continue with other documents
        
        # Check if we have any chunks
        if not all_chunks_with_embeddings:
            ctx.log.warning("No valid chunks extracted from documents")
            return {
                'query': query,
                'response': 'I could not extract any useful information from the provided documents.',
                'chunks_used': [],
                'status': 'no_chunks'
            }
        
        chunks = [c for c, _ in all_chunks_with_embeddings]
        embeddings = np.array([e for _, e in all_chunks_with_embeddings])
        
        # Add embeddings to FAISS index
        if len(embeddings) > 0:
            try:
                faiss_index.add(embeddings)
                ctx.log.info(f"Added {len(embeddings)} embeddings to FAISS index")
            except Exception as e:
                ctx.log.error(f"Error adding embeddings to FAISS index: {str(e)}")
                ctx.log.error(traceback.format_exc())
                # Continue with search using just the current embeddings
        
        # Query the index
        try:
            query_embedding = model.encode([query])[0]
            k = min(10, len(chunks))  # Increased to 10 chunks
            
            if k > 0:
                # Perform search
                D, I = faiss_index.search(np.array([query_embedding]), k)
                top_chunks = [chunks[i] for i in I[0]]
                ctx.log.info(f"Retrieved {len(top_chunks)} chunks from FAISS search")
                
                # Create prompt with retrieved chunks
                from agent_prompts import RAG_PROMPT
                chunks_text = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])
                
                # Increased truncation limit for Gemini
                if len(chunks_text) > 500000:
                    ctx.log.warning("Chunks text too long, truncating")
                    chunks_text = chunks_text[:500000] + "...[truncated]"
                
                prompt = RAG_PROMPT.format(
                    query=query,
                    chunks=chunks_text
                )
                
                # Use Gemini API instead of OpenAI
                ctx.log.info("Sending prompt to Gemini API for RAG generation")
                try:
                    gemini_api_key = os.getenv('GEMINI_API_KEY')
                    if not gemini_api_key:
                        raise ValueError("GEMINI_API_KEY environment variable not set")
                    
                    # Use httpx for async HTTP requests
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}",
                            headers={'Content-Type': 'application/json'},
                            json={
                                "contents": [{
                                    "parts": [{"text": prompt}]
                                }]
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        response_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No response generated')
                    
                    ctx.log.info("Successfully generated RAG response with Gemini")
                    
                except Exception as e:
                    ctx.log.error(f"Gemini API error: {str(e)}")
                    ctx.log.error(traceback.format_exc())
                    
                    # Try fallback to Archon
                    ctx.log.info("Attempting fallback to Archon after Gemini API failure")
                    try:
                        archon = ctx.deps.archon_client
                        if archon:
                            archon_response = await archon.generate(
                                model="gpt-4o",
                                prompt=prompt
                            )
                            response_text = archon_response.get('response', 'Failed to generate response')
                            ctx.log.info("Successfully generated RAG response via Archon fallback")
                        else:
                            response_text = "Error generating response. Gemini API error and Archon not available."
                    except Exception as e2:
                        ctx.log.error(f"Archon fallback also failed: {str(e2)}")
                        response_text = f"Error generating response: {str(e)}. Fallback also failed."
                
                # Store result in Supabase
                try:
                    supabase = ctx.deps.supabase
                    if supabase:
                        result = {
                            'query': query,
                            'response': response_text,
                            'chunks_used': len(top_chunks),
                            'model': 'gemini-2.0-flash',
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        await supabase.table('rag_results').insert(result).execute()
                        ctx.log.info("RAG result stored in Supabase")
                    else:
                        ctx.log.warning("Supabase client not available, RAG result not stored")
                except Exception as e:
                    ctx.log.error(f"Failed to store RAG result in Supabase: {str(e)}")
                    # Fallback to local storage
                    try:
                        os.makedirs('cache', exist_ok=True)
                        with open(f"cache/rag_result_{int(time.time())}.json", 'w') as f:
                            json.dump({
                                'query': query,
                                'response': response_text,
                                'chunks_used': len(top_chunks),
                                'model': 'gemini-2.0-flash',
                                'timestamp': datetime.now().isoformat()
                            }, f)
                        ctx.log.info("RAG result stored locally")
                    except Exception as e2:
                        ctx.log.error(f"Local storage also failed: {str(e2)}")
                
                return {
                    'query': query,
                    'response': response_text,
                    'chunks_used': top_chunks,
                    'model': 'gemini-2.0-flash',
                    'status': 'success'
                }
            else:
                ctx.log.warning("No chunks available for search")
                return {
                    'query': query,
                    'response': 'No relevant information found for this query.',
                    'chunks_used': [],
                    'status': 'no_chunks'
                }
        except Exception as e:
            ctx.log.error(f"Error during FAISS search: {str(e)}")
            ctx.log.error(traceback.format_exc())
            return {
                'query': query,
                'response': f'Error during document search: {str(e)}',
                'chunks_used': [],
                'status': 'search_error'
            }
            
    except Exception as e:
        ctx.log.error(f"Error performing RAG: {str(e)}")
        ctx.log.error(traceback.format_exc())
        return {
            'query': query,
            'response': f'Error performing RAG: {str(e)}',
            'model': 'gemini-2.0-flash',
            'status': 'error'
        }

async def track_progress(ctx, email_id: str, file_hash: str, status: str, file_id: Optional[str] = None, filename: Optional[str] = None, error_message: Optional[str] = None) -> Dict[str, Any]:
    """
    Track progress in Supabase.
    
    Args:
        ctx: Agent context containing deps and logger
        email_id: ID of the processed email
        file_hash: Hash of the processed file
        status: Status of the operation
        file_id: Optional Drive file ID
        filename: Optional filename
        error_message: Optional error message
        
    Returns:
        Result of the storage operation
    """
    try:
        ctx.log.info(f"Tracking progress for {filename} ({file_hash}): {status}")
        supabase = ctx.deps.supabase
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
            
        # Handle missing Supabase client
        if not supabase:
            ctx.log.warning("Supabase client not initialized, storing record locally")
            local_path = Path(f'cache/track_{int(time.time())}_{file_hash[:8]}.json')
            with open(local_path, 'w') as f:
                json.dump(record, f)
            return {**record, 'stored': 'local', 'path': str(local_path)}
            
        # Try with retry mechanism
        for attempt in range(3):
            try:
                result = await supabase.table('processed_items').insert(record).execute()
                ctx.log.info(f"Progress tracked successfully after {attempt+1} attempt(s)")
                return {**record, 'stored': 'supabase', 'data': result.data}
            except Exception as e:
                ctx.log.warning(f"Supabase insert attempt {attempt+1} failed: {str(e)}")
                if attempt < 2:
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                else:
                    # Final fallback to local storage
                    ctx.log.warning("All Supabase attempts failed, storing record locally")
                    local_path = Path(f'cache/track_{int(time.time())}_{file_hash[:8]}.json')
                    with open(local_path, 'w') as f:
                        json.dump(record, f)
                    return {**record, 'stored': 'local', 'path': str(local_path), 'error': str(e)}
        
    except Exception as e:
        ctx.log.error(f"Failed to track progress: {str(e)}")
        # Emergency local backup
        try:
            backup_path = Path(f'cache/track_error_{int(time.time())}_{file_hash[:8]}.json')
            with open(backup_path, 'w') as f:
                json.dump({
                    'email_id': email_id,
                    'file_hash': file_hash,
                    'status': status,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }, f)
            return {
                'status': 'error',
                'error': str(e),
                'email_id': email_id,
                'file_hash': file_hash,
                'backup_path': str(backup_path)
            }
        except:
            # Last resort
            return {
                'status': 'error',
                'error': str(e),
                'email_id': email_id,
                'file_hash': file_hash
            } 