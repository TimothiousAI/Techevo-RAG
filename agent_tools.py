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
from typing import Dict, List, Any, Optional, Union
import logfire
import io
import base64
from sentence_transformers import SentenceTransformer
import faiss

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

async def perform_rag(query: str, services: Dict, deps) -> Dict:
    """
    Perform RAG operation on documents from emails, attachments, and Drive files.
    
    Args:
        query: The user query
        services: Dict containing initialized services
        deps: Dependencies object with state
        
    Returns:
        RAG operation result
    """
    archon = services.get('archon')
    supabase = services.get('supabase')
    
    if not archon:
        logger.error("Archon client not initialized")
        raise ValueError("Archon client not initialized")
    
    try:
        # Gather context from various sources
        context = []
        logger.info("Gathering context for RAG")
        
        # 1. Emails
        if hasattr(deps, 'state') and 'processed_emails' in deps.state:
            for email in deps.state['processed_emails']:
                # Use full body for better context
                if email.get('body'):
                    context.append(f"EMAIL: {email['subject']}\nFROM: {email['from']}\nDATE: {email['date']}\n\n{email['body']}")
                else:
                    context.append(f"EMAIL: {email['subject']}\nFROM: {email['from']}\nDATE: {email['date']}\n\n{email['snippet']}")
        
        # 2. Attachments content
        if hasattr(deps, 'state') and 'attachments_content' in deps.state:
            for filename, content in deps.state['attachments_content'].items():
                if isinstance(content, bytes):
                    # Try to decode bytes to string
                    try:
                        text_content = content.decode('utf-8', errors='ignore')
                        context.append(f"ATTACHMENT: {filename}\n\n{text_content[:2000]}...")
                    except:
                        logger.warning(f"Could not decode attachment {filename} as text")
        
        # 3. Drive files
        if hasattr(deps, 'state') and 'drive_contents' in deps.state:
            for file_id, content in deps.state['drive_contents'].items():
                if content:
                    context.append(f"DRIVE FILE: {file_id}\n\n{content[:2000]}...")
        
        # If no context gathered, use a placeholder
        if not context:
            logger.warning("No context found for RAG, using placeholder")
            context = ["No relevant context found."]
        
        # Log content size for debugging
        total_content_size = sum(len(c) for c in context)
        logger.info(f"Total context size: {total_content_size} characters")
        
        # Perform embedding-based retrieval
        try:
            # Create embeddings
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create FAISS index
            query_embedding = model.encode([query])[0]
            context_embeddings = model.encode(context)
            
            dimension = len(query_embedding)
            index = faiss.IndexFlatL2(dimension)
            index.add(context_embeddings)
            
            # Search for most similar documents
            k = min(5, len(context))  # Get up to 5 results
            distances, indices = index.search(query_embedding.reshape(1, -1), k)
            
            # Get the top results
            top_contexts = [context[idx] for idx in indices[0]]
            
            # Combine into a single context string (with truncation if needed)
            combined_context = "\n\n---\n\n".join(top_contexts)
            
            # Truncate if too long (to stay within context limits)
            if len(combined_context) > 3000:
                logger.warning(f"Context too large ({len(combined_context)} chars), truncating")
                combined_context = combined_context[:3000] + "...[truncated]"
        except Exception as e:
            logger.error(f"Error in embedding-based retrieval: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            combined_context = "\n\n".join(context[:3])  # Fallback to first 3 contexts
        
        # Generate RAG response
        logger.info("Generating RAG response with context")
        rag_prompt = RAG_PROMPT.format(query=query, context=combined_context)
        
        # Make sure prompt isn't too large for Archon MCP (5k token limit)
        if len(rag_prompt) > 5000:
            logger.warning(f"RAG prompt too large ({len(rag_prompt)} chars), truncating")
            context_limit = 5000 - len(RAG_PROMPT.format(query=query, context=""))
            combined_context = combined_context[:context_limit] + "...[truncated]"
            rag_prompt = RAG_PROMPT.format(query=query, context=combined_context)
        
        # Call Archon MCP
        try:
            if not archon.thread_id:
                archon.thread_id = await archon.create_thread()
            
            rag_response = await archon.run_agent(
                thread_id=archon.thread_id,
                user_input=rag_prompt
            )
            
            if isinstance(rag_response, dict) and 'error' in rag_response:
                logger.error(f"Error from Archon MCP: {rag_response['error']}")
                
                # Fallback to local API
                fallback_response = await archon.generate(
                    model="gpt-4o",
                    prompt=rag_prompt
                )
                
                if 'error' in fallback_response:
                    raise ValueError(f"Both MCP and fallback failed: {fallback_response['error']}")
                
                rag_result = fallback_response['response']
            elif isinstance(rag_response, str):
                rag_result = rag_response
            else:
                rag_result = rag_response.get('text', str(rag_response))
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            rag_result = f"Error generating response: {str(e)}"
        
        # Store result in Supabase if available
        try:
            if supabase:
                logger.info("Storing RAG result in Supabase")
                
                try:
                    # Try using cursor.mcp.supabase
                    import cursor.mcp.supabase as supabase_mcp
                    record = {
                        'query': query,
                        'context_size': len(combined_context),
                        'result': rag_result,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    result = await supabase_mcp.insert('rag_results', record)
                    logger.info(f"RAG result stored in Supabase: {result}")
                except Exception as mcp_error:
                    logger.error(f"Error storing result via MCP: {mcp_error}")
                    # Fallback to direct API
                    record = {
                        'query': query,
                        'context_size': len(combined_context),
                        'result': rag_result,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    result = supabase.table('rag_results').insert(record).execute()
                    logger.info(f"RAG result stored in Supabase via direct API: {result}")
        except Exception as db_error:
            logger.error(f"Error storing RAG result in database: {db_error}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Backup to local file
            try:
                os.makedirs('logs/rag_results', exist_ok=True)
                backup_file = f"logs/rag_results/{int(time.time())}.json"
                with open(backup_file, 'w') as f:
                    json.dump({
                        'query': query,
                        'result': rag_result,
                        'timestamp': datetime.now().isoformat()
                    }, f)
                logger.info(f"RAG result backed up to {backup_file}")
            except Exception as backup_error:
                logger.error(f"Error backing up RAG result: {backup_error}")
        
        result = {
            'answer': rag_result,
            'context_used': len(context),
            'context_size': len(combined_context)
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error in perform_rag: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

async def track_progress(record: Dict, services: Dict) -> Dict:
    """
    Track progress in Supabase.
    
    Args:
        record: Progress record to store
        services: Dict containing initialized services
        
    Returns:
        Result of the storage operation
    """
    supabase = services.get('supabase')
    
    if not supabase:
        logger.error("Supabase client not initialized")
        raise ValueError("Supabase client not initialized")
    
    try:
        logger.info(f"Tracking progress: {record}")
        
        try:
            # Try using cursor.mcp.supabase
            import cursor.mcp.supabase as supabase_mcp
            
            # Add timestamp if not present
            if 'timestamp' not in record:
                record['timestamp'] = datetime.now().isoformat()
                
            result = await supabase_mcp.insert('processed_items', record)
            logger.info(f"Progress record stored in Supabase: {result}")
            return result
        except Exception as mcp_error:
            logger.error(f"Error storing progress via MCP: {mcp_error}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to direct API
            if 'timestamp' not in record:
                record['timestamp'] = datetime.now().isoformat()
                
            result = supabase.table('processed_items').insert(record).execute()
            logger.info(f"Progress record stored in Supabase via direct API: {result}")
            return result
    except Exception as e:
        logger.error(f"Error tracking progress: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Backup to local file
        try:
            os.makedirs('logs/progress', exist_ok=True)
            backup_file = f"logs/progress/{int(time.time())}.json"
            with open(backup_file, 'w') as f:
                json.dump(record, f)
            logger.info(f"Progress record backed up to {backup_file}")
            return {"status": "backed up locally", "file": backup_file}
        except Exception as backup_error:
            logger.error(f"Error backing up progress record: {backup_error}")
            raise 