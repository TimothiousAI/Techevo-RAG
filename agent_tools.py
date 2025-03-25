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
from googleapiclient.http import MediaIoBaseUpload
import re

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
    ctx.log.info(f"Searching emails with query: {query}")
    gmail_service = ctx.deps.gmail_service
    
    if not gmail_service:
        ctx.log.error("Gmail service not initialized")
        raise ValueError("Gmail service not initialized")
    
    try:
        # Use query directly
        ctx.log.info(f"Performing Gmail search with query: {query}")
        
        # Format 'full' retrieves the full email content
        results = gmail_service.users().messages().list(
            userId='me',
            q=query,
            maxResults=30
        ).execute()
        
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
            
            # Extract only true attachments (excluding inline content)
            attachments = []
            if 'parts' in email_data['payload']:
                for part in email_data['payload']['parts']:
                    # Check if it's a true attachment (has filename and proper content disposition)
                    is_attachment = False
                    if part.get('filename') and part['filename'].strip():
                        # Check for Content-Disposition header
                        for header in part.get('headers', []):
                            if header.get('name', '').lower() == 'content-disposition':
                                if 'attachment' in header.get('value', '').lower():
                                    is_attachment = True
                                    break
                        
                        # If no content-disposition header, check if it's a common attachment type
                        if not is_attachment:
                            mime_type = part.get('mimeType', '')
                            common_attachment_types = [
                                'application/pdf', 
                                'application/msword',
                                'application/vnd.openxmlformats-officedocument',
                                'application/vnd.ms-excel',
                                'application/zip',
                                'application/x-zip-compressed',
                                'image/',
                                'audio/',
                                'video/'
                            ]
                            if any(mime_type.startswith(t) for t in common_attachment_types):
                                is_attachment = True
                        
                        # Add to attachments if it's a true attachment
                        if is_attachment and 'body' in part and 'attachmentId' in part['body']:
                            attachments.append({
                                'id': part['body'].get('attachmentId', ''),
                                'filename': part['filename'],
                                'mimeType': part['mimeType'],
                                'size': part['body'].get('size', 0)
                            })
            
            # Only include emails that have true attachments
            if attachments:
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
                
                # Log detailed info about each email with true attachments
                ctx.log.info(f"Found email with true attachments: id={email['id']}, from={email['from']}, subject={email['subject']}, attachments={len(attachments)}")
                emails.append(email)
        
        # Store in state
        if hasattr(ctx.deps, 'state'):
            ctx.deps.state['processed_emails'] = emails
        
        ctx.log.info(f"Retrieved {len(emails)} emails with true attachments")
        return emails
    
    except Exception as e:
        ctx.log.error(f"Error searching emails: {e}")
        ctx.log.error(f"Traceback: {traceback.format_exc()}")
        return []  # Return empty list instead of raising to continue with workflow

async def download_attachment(
    ctx: AgentContext,
    email_id: str,
    attachment_id: str,
    folder_id: str = None,
    save_local: bool = False
) -> Dict[str, Any]:
    """Download an email attachment and upload it to Google Drive.
    
    Args:
        ctx: Agent context with dependencies
        email_id: ID of the email containing the attachment
        attachment_id: ID of the attachment to download
        folder_id: ID of the Google Drive folder to upload to
        save_local: Whether to save a local copy of the attachment
        
    Returns:
        dict: Result of the download and upload operation
    """
    gmail_service = ctx.deps.gmail_service
    drive_service = ctx.deps.drive_service
    log = ctx.log
    
    try:
        # Get the email to extract attachment metadata
        log.info(f"Fetching message {email_id} to get attachment {attachment_id}")
        message = gmail_service.users().messages().get(userId='me', id=email_id).execute()
        
        # Find the attachment part
        attachment_part = None
        filename = None
        mime_type = None
        
        # Find the attachment part from message payload parts
        if 'payload' in message and 'parts' in message['payload']:
            for part in message['payload']['parts']:
                if part.get('body', {}).get('attachmentId') == attachment_id:
                    attachment_part = part
                    break
        
        # If attachment part found, get filename and mime_type
        if attachment_part:
            # Get filename from headers
            if 'headers' in attachment_part:
                for header in attachment_part['headers']:
                    if header['name'].lower() == 'content-disposition':
                        # Look for filename in content-disposition
                        match = re.search(r'filename="?([^";]+)"?', header['value'])
                        if match:
                            filename = match.group(1).strip()
                    elif header['name'].lower() == 'content-type':
                        # Extract MIME type from content-type header
                        mime_type = header['value'].split(';')[0].strip()
            
            # If filename not found in content-disposition, try filename header
            if not filename and 'filename' in attachment_part:
                filename = attachment_part['filename']
        
        # If we still don't have a filename or no attachment part, this is not a true attachment
        if not attachment_part or not filename:
            log.warning(f"No valid attachment part or filename found for attachment ID {attachment_id}")
            return {
                'status': 'skipped',
                'error': 'Not a true attachment - no valid attachment part or filename found',
                'attachment_id': attachment_id
            }
        
        # Check if this is a true attachment (not inline content)
        # Look for content-disposition: attachment or common attachment types
        is_true_attachment = False
        
        # Check headers for content-disposition: attachment
        if 'headers' in attachment_part:
            for header in attachment_part['headers']:
                if header['name'].lower() == 'content-disposition':
                    if 'attachment' in header['value'].lower():
                        is_true_attachment = True
                        break
        
        # Check MIME type for common attachment types if not already confirmed
        if not is_true_attachment and mime_type:
            common_attachment_types = [
                'application/pdf', 
                'application/msword',
                'application/vnd.openxmlformats-officedocument',
                'application/vnd.ms-excel',
                'application/vnd.ms-powerpoint',
                'application/zip',
                'application/x-zip-compressed',
                'image/jpeg',
                'image/png',
                'image/gif',
                'application/octet-stream'
            ]
            
            if any(mime_type.startswith(t) for t in common_attachment_types):
                is_true_attachment = True
        
        # Check filename extension for common types if still not confirmed
        if not is_true_attachment and filename:
            common_extensions = [
                '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                '.zip', '.jpg', '.jpeg', '.png', '.gif', '.txt', '.csv'
            ]
            
            if any(filename.lower().endswith(ext) for ext in common_extensions):
                is_true_attachment = True
        
        # Skip if not a true attachment
        if not is_true_attachment:
            log.warning(f"Skipping non-true attachment: {filename} (mime: {mime_type})")
            return {
                'status': 'skipped',
                'error': 'Not a true attachment based on content-disposition and mime-type',
                'filename': filename,
                'mime_type': mime_type,
                'attachment_id': attachment_id
            }
        
        # Download the attachment
        log.info(f"Downloading true attachment: {filename}")
        attachment = gmail_service.users().messages().attachments().get(
            userId='me', messageId=email_id, id=attachment_id
        ).execute()
        
        file_data = base64.urlsafe_b64decode(attachment['data'])
        file_size = len(file_data)
        
        # Clean filename to remove invalid characters
        clean_filename = re.sub(r'[^a-zA-Z0-9_\-\. ]', '', filename)
        if not clean_filename:
            clean_filename = f"attachment_{attachment_id}"
        
        # Save locally if requested
        local_path = None
        if save_local:
            # Create a directory for attachments if it doesn't exist
            os.makedirs('attachments', exist_ok=True)
            local_path = os.path.join('attachments', clean_filename)
            
            with open(local_path, 'wb') as f:
                f.write(file_data)
            log.info(f"Saved attachment locally to {local_path}")
        
        # Prepare for upload to Google Drive
        if not mime_type:
            mime_type = 'application/octet-stream'  # Default MIME type if unknown
        
        # Create file metadata for Drive upload
        file_metadata = {
            'name': clean_filename,
            'mimeType': mime_type
        }
        
        # Add to folder if specified
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        # Create MediaIoBaseUpload from file data for faster uploads
        media = MediaIoBaseUpload(
            io.BytesIO(file_data),
            mimetype=mime_type,
            resumable=True
        )
        
        # Upload to Drive
        log.info(f"Uploading {clean_filename} ({file_size} bytes) to Google Drive...")
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,name,mimeType,size,webViewLink'
        ).execute()
        
        drive_id = file.get('id')
        web_link = file.get('webViewLink', '')
        
        log.info(f"Successfully uploaded to Drive with ID: {drive_id}")
        
        # Return success with file details
        return {
            'status': 'success',
            'filename': clean_filename,
            'original_filename': filename,
            'mime_type': mime_type,
            'size': file_size,
            'drive_id': drive_id,
            'web_link': web_link,
            'local_path': local_path
        }
    except Exception as e:
        log.error(f"Error downloading attachment: {str(e)}")
        log.error(traceback.format_exc())
        return {
            'status': 'error',
            'error': str(e),
            'attachment_id': attachment_id
        }

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