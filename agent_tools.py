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

async def search_emails(query: str, services: Dict, deps) -> List[Dict]:
    """
    Search for emails matching the query in Gmail.
    
    Args:
        query: The user query
        services: Dict containing initialized services
        deps: Dependencies object with state
        
    Returns:
        List of emails with metadata
    """
    gmail_service = services.get('gmail')
    archon = services.get('archon')
    
    if not gmail_service:
        logger.error("Gmail service not initialized")
        raise ValueError("Gmail service not initialized")
    
    # First check if we have cached results
    cache = load_cache()
    cache_key = f"email_{query.lower().strip()}"
    
    if cache_key in cache.get('email_searches', {}):
        logger.info(f"Using cached email results for query: {query}")
        emails = cache['email_searches'][cache_key]
        
        # Store in state
        if hasattr(deps, 'state'):
            deps.state['processed_emails'] = emails
        
        return emails
    
    # Use Archon to parse search parameters
    try:
        logger.info(f"Refining email search criteria for: {query}")
        search_prompt = EMAIL_SEARCH_PROMPT.format(query=query)
        
        search_response = await archon.generate(
            model="gpt-4o",
            prompt=search_prompt
        )
        
        if 'error' in search_response:
            logger.error(f"Error getting search criteria: {search_response['error']}")
            search_criteria = query
        else:
            search_criteria = search_response['response']
            logger.info(f"Refined search criteria: {search_criteria}")
    except Exception as e:
        logger.error(f"Error generating search criteria: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        search_criteria = query  # Fallback to original query
    
    try:
        logger.info(f"Searching emails with criteria: {search_criteria}")
        
        # Format 'full' retrieves the full email content
        results = gmail_service.users().messages().list(
            userId='me',
            q=search_criteria,
            maxResults=10
        ).execute()
        
        messages = results.get('messages', [])
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
            
            emails.append(email)
        
        # Cache the results
        cache['email_searches'][cache_key] = emails
        save_cache(cache)
        
        # Store in state
        if hasattr(deps, 'state'):
            deps.state['processed_emails'] = emails
        
        logger.info(f"Retrieved {len(emails)} matching emails")
        return emails
    
    except Exception as e:
        logger.error(f"Error searching emails: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

async def download_attachment(query: str, services: Dict, deps) -> List[Dict]:
    """
    Download attachments from emails matching the query.
    
    Args:
        query: The user query
        services: Dict containing initialized services
        deps: Dependencies object with state
        
    Returns:
        List of downloaded attachments
    """
    gmail_service = services.get('gmail')
    drive_service = services.get('drive')
    
    if not gmail_service or not drive_service:
        logger.error("Gmail or Drive service not initialized")
        raise ValueError("Gmail or Drive service not initialized")
    
    # First, search for emails if not already in state
    if not hasattr(deps, 'state') or 'processed_emails' not in deps.state or not deps.state['processed_emails']:
        logger.info("No emails in state, searching first")
        await search_emails(query, services, deps)
    
    if not hasattr(deps, 'state') or 'processed_emails' not in deps.state:
        logger.error("No emails found to download attachments from")
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
        logger.info(f"Created new Drive folder with ID: {folder_id}")
    
    # Process attachments
    downloaded = []
    cache = load_cache()
    
    try:
        for email in deps.state['processed_emails']:
            if not email.get('attachments'):
                continue
                
            for attachment in email['attachments']:
                try:
                    logger.info(f"Downloading attachment: {attachment['filename']} from email {email['id']}")
                    
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
                    
                    if hasattr(deps, 'state'):
                        if 'attachments_content' not in deps.state:
                            deps.state['attachments_content'] = {}
                        deps.state['attachments_content'][attachment['filename']] = file_data
                    
                except Exception as e:
                    logger.error(f"Error downloading attachment {attachment['filename']}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    downloaded.append({
                        'error': str(e),
                        'filename': attachment['filename'],
                        'email_id': email['id']
                    })
        
        # Save cache
        save_cache(cache)
        
        logger.info(f"Downloaded {len(downloaded)} attachments")
        return downloaded
    
    except Exception as e:
        logger.error(f"Error in download_attachment: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

async def search_drive(query: str, services: Dict, deps) -> List[Dict]:
    """
    Search for files in Google Drive matching the query.
    
    Args:
        query: The user query
        services: Dict containing initialized services
        deps: Dependencies object with state
        
    Returns:
        List of found files
    """
    drive_service = services.get('drive')
    archon = services.get('archon')
    
    if not drive_service:
        logger.error("Drive service not initialized")
        raise ValueError("Drive service not initialized")
    
    # First check if we have cached results
    cache = load_cache()
    cache_key = f"drive_{query.lower().strip()}"
    
    if cache_key in cache.get('drive_searches', {}):
        logger.info(f"Using cached Drive results for query: {query}")
        files = cache['drive_searches'][cache_key]
        
        # Store in state
        if hasattr(deps, 'state'):
            deps.state['drive_files'] = files
        
        return files
    
    # Use Archon to parse search parameters
    try:
        logger.info(f"Refining Drive search criteria for: {query}")
        search_prompt = SEARCH_CRITERIA_PROMPT.format(query=query)
        
        search_response = await archon.generate(
            model="gpt-4o",
            prompt=search_prompt
        )
        
        if 'error' in search_response:
            logger.error(f"Error getting search criteria: {search_response['error']}")
            search_criteria = query
        else:
            search_criteria = search_response['response']
            logger.info(f"Refined search criteria: {search_criteria}")
    except Exception as e:
        logger.error(f"Error generating search criteria: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        search_criteria = query  # Fallback to original query
    
    try:
        logger.info(f"Searching Drive with criteria: {search_criteria}")
        
        # Search Drive for files
        results = drive_service.files().list(
            q=f"name contains '{search_criteria}' and trashed=false",
            spaces='drive',
            fields='files(id, name, mimeType, webViewLink, createdTime, modifiedTime)',
            orderBy='modifiedTime desc',
            pageSize=10
        ).execute()
        
        files = results.get('files', [])
        
        # For text files, fetch the content for RAG
        if 'drive_contents' not in cache:
            cache['drive_contents'] = {}
        
        if hasattr(deps, 'state') and 'drive_contents' not in deps.state:
            deps.state['drive_contents'] = {}
        
        for file in files:
            # Only process text files, PDFs, Google Docs etc.
            if file['mimeType'] in [
                'text/plain', 
                'application/pdf',
                'application/vnd.google-apps.document',
                'application/vnd.google-apps.spreadsheet',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ]:
                try:
                    if file['mimeType'] == 'application/vnd.google-apps.document':
                        # Export Google Docs as plain text
                        content = drive_service.files().export(
                            fileId=file['id'],
                            mimeType='text/plain'
                        ).execute()
                        
                        if isinstance(content, bytes):
                            text_content = content.decode('utf-8')
                        else:
                            text_content = str(content)
                    else:
                        # Download other supported files
                        request = drive_service.files().get_media(fileId=file['id'])
                        file_content = io.BytesIO()
                        downloader = request.execute()
                        
                        if isinstance(downloader, bytes):
                            text_content = downloader.decode('utf-8', errors='ignore')
                        else:
                            text_content = str(downloader)
                    
                    # Cache the content
                    cache['drive_contents'][file['id']] = text_content
                    
                    # Store in state
                    if hasattr(deps, 'state'):
                        deps.state['drive_contents'][file['id']] = text_content
                        
                    # Add content to the file metadata
                    file['content'] = text_content[:1000] + "..." if len(text_content) > 1000 else text_content
                    
                except Exception as e:
                    logger.error(f"Error fetching content for file {file['name']}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    file['content_error'] = str(e)
        
        # Cache the results
        cache['drive_searches'][cache_key] = files
        save_cache(cache)
        
        # Store in state
        if hasattr(deps, 'state'):
            deps.state['drive_files'] = files
        
        logger.info(f"Retrieved {len(files)} matching files from Drive")
        return files
    
    except Exception as e:
        logger.error(f"Error searching Drive: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

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