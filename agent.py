"""
Main agent implementation for Techevo-RAG.

This module contains the primary agent implementation, workflow orchestration,
and setup functions for Gmail, Drive, and FAISS.
"""

import os
import json
import asyncio
import aiohttp
import time
import httpx
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
import hashlib
from datetime import datetime
import copy
from pathlib import Path
import traceback
import re

# Google API imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# FAISS and embeddings
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Supabase
from supabase import create_client, Client as SupabaseClient

# Pydantic AI
from pydantic_ai.agent import Agent

# Define base Deps class locally
class Deps:
    """Base dependency class for agent."""
    pass

# Logging with logfire
import logfire

# Import our own modules
from agent_prompts import SYSTEM_PROMPT, INTENT_PROMPT, RAG_PROMPT
import agent_tools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure logfire
logfire.configure(
    token=os.getenv('LOGFIRE_TOKEN'),
    service_name='techevo-rag',
    send_to_logfire=True
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Validate environment variables at module level
def validate_env_vars() -> None:
    """
    Validate that all required environment variables are set.
    
    Raises:
        ValueError: If any required env var is missing
    """
    required_vars = [
        'CREDENTIALS_JSON_PATH',
        'SUPABASE_URL',
        'SUPABASE_KEY',
        'OPENAI_API_KEY',
        'DEFAULT_DRIVE_FOLDER_ID'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}. Please add them to your .env file."
        logfire.error(error_msg)
        raise ValueError(error_msg)

# Google API scopes
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.readonly'
]

# For MCP-based communication with Archon
class ArchonClient:
    """Client for interacting with Archon MCP and OpenAI."""
    
    def __init__(self, api_key=None, api_base=None):
        """Initialize ArchonClient with API key and MCP URL.
        
        Args:
            api_key: OpenAI API key (fallback if MCP fails)
            api_base: OpenAI API base URL (fallback if MCP fails)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or "https://api.openai.com/v1"
        self.mcp_url = os.getenv('ARCHON_MCP_URL', 'http://host.docker.internal:8100')
        self.thread_id = None
        self.session = None
        self.state = {}
        
        # Create state backup directory
        os.makedirs('backup', exist_ok=True)
    
    async def ensure_session(self):
        """Ensure an aiohttp session is available."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            logger.info("Created new aiohttp session")
    
    async def create_thread(self):
        """Create a new thread for Archon MCP."""
        try:
            # If we already have a thread ID, just return it
            if self.thread_id is not None:
                return self.thread_id
                
            # Try to connect to Archon MCP
            try:
                await self.ensure_session()
                async with self.session.post(f"{self.mcp_url}/create-thread", json={"random_string": "init"}) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.thread_id = result.get('thread_id')
                        logger.info(f"Created Archon thread: {self.thread_id}")
                        return self.thread_id
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to create Archon thread via MCP: {error_text}")
                        raise ValueError(f"MCP error: {error_text}")
            except Exception as mcp_error:
                logger.error(f"Could not create thread via MCP: {str(mcp_error)}")
                logger.warning("Falling back to local thread ID")
                    
            # Fallback to local thread ID
            import time
            local_thread_id = f"local-{int(time.time())}"
            logger.info(f"Using local thread ID: {local_thread_id}")
            self.thread_id = local_thread_id
            return local_thread_id
        except Exception as e:
            logger.error(f"Failed to create thread: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Still need a thread ID even if everything fails
            import time
            local_thread_id = f"local-{int(time.time())}"
            logger.info(f"Using local thread ID after error: {local_thread_id}")
            self.thread_id = local_thread_id
            return local_thread_id
    
    async def run_agent(self, thread_id, user_input):
        """Run Archon agent with the given input.
        
        Args:
            thread_id: Thread ID for the conversation
            user_input: User input to process
            
        Returns:
            Agent response
        """
        try:
            # Try to use Archon MCP
            try:
                await self.ensure_session()
                async with self.session.post(
                    f"{self.mcp_url}/run-agent", 
                    json={"thread_id": thread_id, "user_input": user_input}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully ran agent via Archon MCP")
                        return result.get('response')
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to run agent via MCP: {error_text}")
                        raise ValueError(f"MCP error: {error_text}")
            except Exception as mcp_error:
                logger.error(f"Could not run agent via MCP: {str(mcp_error)}")
                logger.warning("Falling back to OpenAI")
                
            # Fallback to OpenAI
            return await self.generate(
                model="gpt-4o",
                prompt=user_input
            )
        except Exception as e:
            logger.error(f"Failed to run agent: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Agent error: {str(e)}"}
            
    async def generate(self, model="gpt-4o", prompt="", temperature=0.7, max_tokens=2000):
        """Generate text using OpenAI API.
        
        Args:
            model: Model to use
            prompt: Prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not self.api_key:
            return {"error": "OpenAI API key not configured"}
        
        if not prompt:
            return {"error": "Empty prompt"}
        
        # Truncate prompt if too long to avoid token limits
        if len(prompt) > 5000:
            logger.warning(f"Prompt size exceeds recommended limit: {len(prompt)} chars")
            prompt = prompt[:4800] + "...[truncated]"
            
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)
            
            completion = await client.chat.completions.create(
                model=model.replace("openai:", "") if model.startswith("openai:") else model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_text = completion.choices[0].message.content
            return {"response": response_text}
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Generation error: {str(e)}"}

    async def create_agent(self, skills, level="sub-agent"):
        """Create a new agent with the specified skills using Archon MCP.
        
        Args:
            skills: List of skills for the agent
            level: Agent level/type
            
        Returns:
            Agent ID
        """
        try:
            await self.ensure_session()
            
            # Attempt to start agent services
            try:
                async with self.session.post(
                    f"{self.mcp_url}/start-services", 
                    json={"action": "start"}
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to start agent services: {await response.text()}")
                    else:
                        logger.info("Agent services started successfully")
            except Exception as e:
                logger.warning(f"Could not start agent services programmatically: {str(e)}. Ensure services are started manually.")
            
            # Create the new agent
            payload = {
                "skills": skills,
                "level": level,
                "agent_type": "pydantic-ai",
                "base_model": "openai:gpt-4o"
            }
            
            async with self.session.post(
                f"{self.mcp_url}/create-agent", 
                json=payload
            ) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    raise Exception(f"Failed to create agent: {error_msg}")
                
                result = await response.json()
                return result.get('agent_id')
        except Exception as e:
            logger.error(f"Error creating agent via Archon MCP: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def save_state(self, key, value):
        """Save a value to the client state.
        
        Args:
            key: State key
            value: Value to store
        """
        self.state[key] = value
        logger.info(f"Saved state key: {key}")
        return self.state

@dataclass
class EnhancedDeps(Deps):
    """Enhanced dependencies for the Techevo-RAG agent."""
    
    gmail_service: Any = None
    drive_service: Any = None
    faiss_index: Any = None
    supabase: SupabaseClient = None
    archon_client: ArchonClient = None
    state: Dict[str, Any] = field(default_factory=dict)

async def setup_google_services() -> Tuple[Any, Any]:
    """
    Set up Gmail and Drive services using credentials from credentials.json.
    
    Returns:
        Tuple of (gmail_service, drive_service)
    """
    creds = None
    credentials_path = os.getenv('CREDENTIALS_JSON_PATH')
    token_path = 'token.json'
    
    # Validate credentials path
    if not credentials_path or not os.path.exists(credentials_path):
        error_msg = f"Credentials file not found at {credentials_path}. Please set CREDENTIALS_JSON_PATH in .env to a valid credentials.json file."
        logger.error(error_msg)
        logfire.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if token already exists
    if os.path.exists(token_path):
        try:
            with open(token_path, 'r') as token_file:
                token_data = json.load(token_file)
            creds = Credentials.from_authorized_user_info(token_data, SCOPES)
            logger.info("Loaded credentials from token.json")
        except Exception as e:
            logger.error(f"Error loading token.json: {str(e)}")
            logger.error(traceback.format_exc())
            creds = None
    
    # If no valid credentials available, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                logger.info("Refreshing expired credentials")
                creds.refresh(Request())
                logger.info("Credentials refreshed successfully")
            except Exception as e:
                logger.error(f"Error refreshing credentials: {str(e)}")
                logger.error(traceback.format_exc())
                creds = None
        
        if not creds:
            try:
                logger.info(f"Starting OAuth flow with credentials from {credentials_path}")
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
                logger.info("OAuth flow completed successfully")
                
                # Save the credentials
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
                logger.info(f"Credentials saved to {token_path}")
            except Exception as e:
                logger.error(f"Error during OAuth flow: {str(e)}")
                logger.error(traceback.format_exc())
                raise ValueError(f"Failed to authenticate with Google: {str(e)}")
    
    # Verify we have valid credentials
    if not creds:
        error_msg = "Failed to authenticate with Google services."
        logger.error(error_msg)
        logfire.error(error_msg)
        raise ValueError(error_msg)
    
    # Build services
    try:
        gmail_service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail service created successfully")
        
        drive_service = build('drive', 'v3', credentials=creds)
        logger.info("Drive service created successfully")
        
        # Verify Gmail connection with a simple test
        try:
            profile = gmail_service.users().getProfile(userId='me').execute()
            logger.info(f"Connected to Gmail as: {profile.get('emailAddress', 'unknown')}")
        except Exception as e:
            logger.warning(f"Gmail connection test failed: {str(e)}")
        
        return gmail_service, drive_service
    except Exception as e:
        logger.error(f"Error building Google services: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to build Google services: {str(e)}")

def create_faiss_index(dimension=768):
    """Create an empty FAISS index with the specified dimension.
    
    Args:
        dimension: Vector dimension (768 for all-MiniLM-L6-v2)
        
    Returns:
        FAISS index object
    """
    logger.info(f"Creating FAISS index with dimension {dimension}")
    
    try:
        # Convert dimension to int explicitly
        dimension = int(dimension)
        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)
        logger.info(f"Successfully created FAISS index with dimension {dimension}")
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Fallback to a simpler index
        try:
            fallback_dim = 384  # Default dimension that usually works
            logger.info(f"Trying fallback with dimension {fallback_dim}")
            index = faiss.IndexFlatL2(fallback_dim)
            logger.info(f"Successfully created fallback FAISS index")
            return index
        except Exception as fallback_e:
            logger.error(f"Fallback also failed: {fallback_e}")
            raise ValueError(f"Failed to create FAISS index: {e}, fallback error: {fallback_e}")

async def predict_intent(query: str, archon_client: ArchonClient) -> Dict[str, Any]:
    """
    Predict the intent of a user query using Archon MCP.
    
    Args:
        query: The user query
        archon_client: The Archon MCP client
        
    Returns:
        Dictionary with tools to execute and search parameters
    """
    try:
        # Save previous state before prediction
        archon_client.save_state('predict_intent_query', query)
        
        # Use a concise prompt to stay below token limits
        prompt = INTENT_PROMPT.format(query=query)
        
        response = await archon_client.generate(
            model="openai:gpt-4o",
            prompt=prompt,
            temperature=0.2,
            max_tokens=200
        )
        
        # Parse the response
        intent_text = response.get('response', '').strip()
        logger.info(f"Intent prediction: {intent_text}")
        
        # Try to extract JSON data
        try:
            intent_data = json.loads(intent_text)
            
            # Ensure we have required fields with defaults
            if not isinstance(intent_data, dict):
                logger.warning("Response is not a dictionary, using fallback")
                return fallback_intent(query)
                
            if 'tools' not in intent_data or not isinstance(intent_data['tools'], list):
                intent_data['tools'] = ['search_emails', 'perform_rag']
                
            if 'sender' not in intent_data:
                intent_data['sender'] = None
                
            if 'keywords' not in intent_data:
                intent_data['keywords'] = query
                
            if 'has_attachment' not in intent_data:
                intent_data['has_attachment'] = False
            
            # Make sure perform_rag is included for process/summarize queries
            if ('process' in query.lower() or 'summarize' in query.lower() or 'analyze' in query.lower()) and 'perform_rag' not in intent_data['tools']:
                intent_data['tools'].append('perform_rag')
                logger.info("Added perform_rag to tools list for processing/summarization query")
            
            return intent_data
            
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON response: {intent_text}")
            # Fall back to keyword extraction
            return fallback_intent(query)
    
    except Exception as e:
        error_msg = f"Error predicting intent: {str(e)}"
        logger.error(error_msg)
        logfire.error(error_msg)
        
        # Fall back to a default set of tools and parameters
        return fallback_intent(query)

def fallback_intent(query: str) -> Dict[str, Any]:
    """
    Fallback intent prediction using keyword matching.
    
    Args:
        query: The user query
        
    Returns:
        Dictionary with tools to execute and search parameters
    """
    # Extract tools using keywords
    tools = []
    if 'email' in query.lower() or 'mail' in query.lower():
        tools.append('search_emails')
    if 'attachment' in query.lower() or 'download' in query.lower() or 'file' in query.lower():
        tools.append('download_attachment')
    if 'drive' in query.lower() or 'document' in query.lower():
        tools.append('search_drive')
    if 'process' in query.lower() or 'analyze' in query.lower() or 'summarize' in query.lower():
        tools.append('perform_rag')
    
    # Always include search_emails and perform_rag as fallback
    if not tools or len(tools) == 0:
        tools = ['search_emails', 'perform_rag']
        
    # Try to extract sender from email pattern
    sender_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    sender_match = re.search(sender_pattern, query)
    sender = sender_match.group(0) if sender_match else None
    
    # Check for attachment mention
    has_attachment = 'attachment' in query.lower()
    
    # Return structured intent data
    return {
        'tools': tools,
        'sender': sender,
        'keywords': query,
        'has_attachment': has_attachment
    }

class TechevoRagAgent:
    """Main agent class for Techevo RAG."""
    
    def __init__(self, services=None):
        """Initialize the agent with required services.
        
        Args:
            services: Dict containing initialized services (gmail, drive, archon, etc.)
        """
        self.services = services or {}
        
        # Ensure required services are available
        required_services = ['gmail', 'drive', 'archon']
        for service in required_services:
            if service not in self.services:
                logger.warning(f"Missing required service: {service}")
        
        # Define available tools
        self.available_tools = {
            'search_emails': agent_tools.search_emails,
            'download_attachment': agent_tools.download_attachment,
            'search_drive': agent_tools.search_drive,
            'perform_rag': agent_tools.perform_rag
        }
        
        # Dictionary to track created sub-agents and their skills
        self.sub_agents = {}
        
        # Services initialization timestamp
        self.initialized_at = datetime.now().isoformat()
        
        logger.info(f"Agent initialized with services: {', '.join(self.services.keys())}")
    
    async def run_workflow(self, query: str, deps: EnhancedDeps) -> Dict[str, Any]:
        """Run the main agent workflow based on the query.
        
        Args:
            query: User query text
            deps: Dependencies object with state management
            
        Returns:
            dict: Result of the workflow execution
        """
        started_at = time.time()
        try:
            # Check for required services first
            if not deps.gmail_service:
                logger.error("Gmail service not initialized")
                return {
                    'status': 'error',
                    'error': 'Gmail service not available',
                    'query': query,
                    'runtime': time.time() - started_at
                }
            
            # Ensure state persistence
            deps.state = deps.state or {
                'query': '',
                'processed_emails': [],
                'downloaded_attachments': [],
                'rag_results': [],
                'start_time': started_at
            }
            
            # Track downloaded attachment IDs for deduplication
            downloaded_ids = set()
            for a in deps.state.get('downloaded_attachments', []):
                if a.get('drive_id'):
                    downloaded_ids.add(a.get('drive_id'))
                if a.get('attachment_id'):
                    downloaded_ids.add(a.get('attachment_id'))
            
            # Store current query in state
            deps.state['query'] = query
            deps.state['start_time'] = started_at
            
            # Get intent with enhanced search parameters
            if 'archon' in self.services:
                intent_data = await predict_intent(query, self.services['archon'])
                intent_tools = intent_data.get('tools', ['search_emails', 'perform_rag'])
                sender = intent_data.get('sender')
                has_attachment = intent_data.get('has_attachment', False)
                keywords = intent_data.get('keywords', '')
                logger.info(f"Predicted intent: tools={intent_tools}, sender={sender}, has_attachment={has_attachment}")
            else:
                # Fallback without Archon
                logger.warning("Archon service not available, using default intent prediction")
                intent_data = fallback_intent(query)
                intent_tools = intent_data.get('tools', ['search_emails', 'perform_rag'])
                sender = intent_data.get('sender')
                has_attachment = intent_data.get('has_attachment', False)
                keywords = intent_data.get('keywords', '')
                logger.info(f"Fallback intent: tools={intent_tools}, sender={sender}, has_attachment={has_attachment}")
            
            # Construct Gmail search query with better handling of year/date filters
            gmail_query_parts = []
            if sender:
                gmail_query_parts.append(f"from:{sender}")
            if has_attachment:
                gmail_query_parts.append("has:attachment")
            if keywords and keywords != query:
                gmail_query_parts.append(keywords)
            
            # Add date range if query contains year references
            if '2025' in query:
                gmail_query_parts.append("after:2025/01/01 before:2026/01/01")
            elif '2024' in query:
                gmail_query_parts.append("after:2024/01/01 before:2025/01/01")
            
            # Use constructed query for Gmail search, fallback to raw query if no specific criteria
            gmail_query = " ".join(gmail_query_parts) if gmail_query_parts else query
            logger.info(f"Constructed Gmail search query: {gmail_query}")
            
            # Store the constructed query in the state
            deps.state['gmail_query'] = gmail_query
            
            result = {
                'status': 'success',
                'data': {},
                'query': query,
                'gmail_query': gmail_query,
                'timestamp': datetime.now().isoformat()
            }
            
            # Create agent context for tool execution
            from agent_tools import AgentContext
            ctx = AgentContext(agent=self, deps=deps, log=logger)
            
            # Run the workflow with available tools
            max_retries = 3  # Default retries for operations
            
            # Process each tool based on intent
            for tool in intent_tools:
                if tool in self.available_tools:
                    logger.info(f"Running tool: {tool}")
                    tool_func = self.available_tools[tool]
                    
                    if tool == 'search_emails':
                        try:
                            # Use the constructed Gmail query for search
                            logger.info(f"Searching emails with Gmail query: {gmail_query}")
                            emails = await tool_func(ctx, gmail_query)
                            result['data']['emails'] = emails
                            deps.state['processed_emails'] = emails
                            
                            # Log the result details
                            email_count = len(emails) if emails else 0
                            attachment_count = sum(len(email.get('attachments', [])) for email in emails) if emails else 0
                            logfire.info("Email search results",
                                query=query,
                                gmail_query=gmail_query,
                                email_count=email_count,
                                attachment_count=attachment_count,
                                has_emails=email_count > 0
                            )
                        except Exception as e:
                            logger.error(f"Error searching emails: {str(e)}")
                            logfire.error("Email search error",
                                query=query, 
                                gmail_query=gmail_query,
                                error=str(e),
                                traceback=traceback.format_exc()
                            )
                            result['data']['emails'] = {"status": "error", "error": str(e)}
                    
                    elif tool == 'download_attachment':
                        # Initialize attachments list from state for persistence
                        attachment_results = deps.state.get('downloaded_attachments', [])
                        
                        # Try to find emails if not already in state
                        if not deps.state.get('processed_emails'):
                            logger.info("No emails in state, running search_emails first")
                            if 'search_emails' in self.available_tools:
                                try:
                                    # Use the constructed Gmail query
                                    emails = await self.available_tools['search_emails'](ctx, gmail_query)
                                    deps.state['processed_emails'] = emails
                                except Exception as e:
                                    logger.error(f"Error searching emails before download: {str(e)}")
                                    logger.error(traceback.format_exc())
                            else:
                                logger.warning("Cannot search emails: tool not available")
                        
                        if deps.state.get('processed_emails'):
                            # Get default Drive folder ID
                            folder_id = os.getenv('DEFAULT_DRIVE_FOLDER_ID')
                            if not folder_id:
                                try:
                                    # Create default folder if it doesn't exist
                                    folder_metadata = {
                                        'name': 'TechevoRAG_Attachments',
                                        'mimeType': 'application/vnd.google-apps.folder'
                                    }
                                    folder = deps.drive_service.files().create(
                                        body=folder_metadata,
                                        fields='id'
                                    ).execute()
                                    folder_id = folder.get('id')
                                    logger.info(f"Created new Drive folder with ID: {folder_id}")
                                except Exception as e:
                                    logger.error(f"Error creating Drive folder: {str(e)}")
                                    logger.error(traceback.format_exc())
                                    continue
                            
                            # Process each email's attachments with retry mechanism
                            for email in deps.state['processed_emails']:
                                if email.get('attachments'):
                                    for attachment in email.get('attachments', []):
                                        attachment_id = attachment.get('id')
                                        
                                        # Skip if already downloaded
                                        if attachment_id and attachment_id in downloaded_ids:
                                            logger.info(f"Skipping already downloaded attachment: {attachment.get('filename', 'unknown')}")
                                            continue
                                            
                                        # Try up to max_retries times
                                        for attempt in range(max_retries):
                                            try:
                                                # Download and upload attachment
                                                download_result = await tool_func(
                                                    ctx,
                                                    email_id=email['id'],
                                                    attachment_id=attachment_id,
                                                    folder_id=folder_id
                                                )
                                                
                                                # Add email_id and attachment_id for reference
                                                download_result['email_id'] = email['id']
                                                download_result['attachment_id'] = attachment_id
                                                download_result['email_subject'] = email.get('subject', 'Unknown')
                                                download_result['email_from'] = email.get('from', 'Unknown')
                                                
                                                if download_result['status'] == 'success':
                                                    downloaded_ids.add(attachment_id)
                                                    if download_result.get('drive_id'):
                                                        downloaded_ids.add(download_result['drive_id'])
                                                    attachment_results.append(download_result)
                                                    logger.info(f"Successfully processed attachment: {download_result['filename']} (attempt {attempt+1})")
                                                    break
                                                elif download_result['status'] == 'skipped':
                                                    logger.info(f"Skipped non-true attachment: {download_result.get('filename', 'unknown')}")
                                                    attachment_results.append(download_result)
                                                    break
                                                else:
                                                    logger.warning(f"Attempt {attempt+1} failed for attachment {attachment_id}: {download_result.get('error', 'Unknown error')}")
                                                    if attempt == max_retries - 1:
                                                        attachment_results.append(download_result)
                                                        logger.error(f"Max retries reached for attachment {attachment_id}")
                                            except Exception as e:
                                                logger.error(f"Error on attempt {attempt+1} for attachment {attachment_id}: {str(e)}")
                                                if attempt == max_retries - 1:
                                                    attachment_results.append({
                                                        'status': 'error',
                                                        'error': str(e),
                                                        'email_id': email['id'],
                                                        'attachment_id': attachment_id,
                                                        'filename': attachment.get('filename', 'unknown'),
                                                        'email_subject': email.get('subject', 'Unknown'),
                                                        'email_from': email.get('from', 'Unknown')
                                                    })
                            
                            result['data']['attachments'] = attachment_results
                            deps.state['downloaded_attachments'] = attachment_results
                            
                            # Log attachment download results
                            logfire.info("Attachment download results",
                                query=query,
                                attachment_count=len(attachment_results),
                                success_count=sum(1 for a in attachment_results if a.get('status') == 'success'),
                                error_count=sum(1 for a in attachment_results if a.get('status') == 'error'),
                                skipped_count=sum(1 for a in attachment_results if a.get('status') == 'skipped')
                            )
                        else:
                            logger.warning("No emails with attachments found")
                            result['data']['attachments'] = []
                    
                    elif tool == 'search_drive':
                        try:
                            drive_files = await tool_func(ctx, query)
                            result['data']['drive_files'] = drive_files
                            deps.state['drive_files'] = drive_files
                            
                            # Log drive search results
                            logfire.info("Drive search results",
                                query=query,
                                file_count=len(drive_files) if drive_files else 0,
                                has_files=len(drive_files) > 0 if drive_files else False
                            )
                        except Exception as e:
                            logger.error(f"Error searching Drive: {str(e)}")
                            result['data']['drive_files'] = {"status": "error", "error": str(e)}
                    
                    elif tool == 'perform_rag' or 'summarize' in query.lower() or 'analyze' in query.lower():
                        # Collect documents from various sources
                        documents = []
                        document_sources = []
                        
                        # Collect documents from emails
                        if deps.state.get('processed_emails'):
                            for email in deps.state.get('processed_emails'):
                                if email.get('body'):
                                    documents.append(email.get('body'))
                                    document_sources.append({
                                        'type': 'email',
                                        'id': email.get('id'),
                                        'subject': email.get('subject', 'Unknown'),
                                        'from': email.get('from', 'Unknown')
                                    })
                        
                        # Collect documents from downloaded attachments
                        if deps.state.get('downloaded_attachments'):
                            for attachment in deps.state.get('downloaded_attachments'):
                                if attachment.get('status') == 'success':
                                    # TODO: Add actual content extraction from Drive files
                                    # For now, just record the metadata
                                    document_sources.append({
                                        'type': 'attachment',
                                        'id': attachment.get('drive_id'),
                                        'filename': attachment.get('filename', 'Unknown'),
                                        'mime_type': attachment.get('mime_type', 'Unknown'),
                                        'email_id': attachment.get('email_id')
                                    })
                        
                        # Placeholder for document text extraction
                        # This would be replaced with actual extraction logic
                        if deps.state.get('attachments_content'):
                            for filename, content in deps.state.get('attachments_content').items():
                                if isinstance(content, bytes):
                                    try:
                                        text = content.decode('utf-8', errors='ignore')
                                        documents.append(text)
                                        document_sources.append({
                                            'type': 'extracted_content',
                                            'filename': filename
                                        })
                                    except Exception as decode_err:
                                        logger.warning(f"Could not decode content for {filename}: {str(decode_err)}")
                                elif isinstance(content, str):
                                    documents.append(content)
                                    document_sources.append({
                                        'type': 'extracted_content',
                                        'filename': filename
                                    })
                        
                        if documents or 'summarize' in query.lower() or 'analyze' in query.lower():
                            try:
                                # Create a sub-agent through Archon for RAG or summarization
                                rag_sub_agent_id = None
                                
                                if 'archon' in self.services:
                                    try:
                                        # More specific skills based on query
                                        skills = ['text_extraction', 'vector_indexing', 'rag_processing']
                                        
                                        # Add specific skills based on query intent
                                        if 'summarize' in query.lower():
                                            skills.append('text_summarization')
                                        if 'analyze' in query.lower():
                                            skills.append('data_analysis') 
                                        
                                        task_description = 'Process documents and perform RAG'
                                        if 'summarize' in query.lower():
                                            task_description = 'Summarize content from emails and attachments'
                                        elif 'analyze' in query.lower():
                                            task_description = 'Analyze content from emails and attachments'
                                        
                                        rag_sub_agent_id = await self.services['archon'].create_agent(
                                            skills=skills,
                                            level='sub-agent',
                                            task=task_description,
                                            name='rag_processor'
                                        )
                                        logger.info(f"Created RAG sub-agent with ID: {rag_sub_agent_id}")
                                    except Exception as e:
                                        logger.error(f"Failed to create RAG sub-agent: {str(e)}")
                                        logger.error(f"Traceback: {traceback.format_exc()}")
                                
                                # Perform RAG with the main tool or use the sub-agent
                                if rag_sub_agent_id:
                                    # Use the sub-agent to process documents
                                    logger.info(f"Using RAG sub-agent {rag_sub_agent_id} to process content")
                                    
                                    # Prepare document summary for sub-agent
                                    document_summary = []
                                    for i, source in enumerate(document_sources):
                                        document_summary.append({
                                            'index': i,
                                            'type': source.get('type'),
                                            'id': source.get('id', 'unknown'),
                                            'metadata': {k: v for k, v in source.items() if k not in ['type', 'id']}
                                        })
                                    
                                    # Prepare a simplified request for the sub-agent
                                    rag_request = {
                                        'query': query,
                                        'document_count': len(documents),
                                        'document_sources': document_summary,
                                        'document_sample': documents[0][:500] + "..." if documents else "",
                                    }
                                    
                                    # Additional details for summarization/analysis
                                    if 'summarize' in query.lower():
                                        rag_request['operation'] = 'summarize'
                                    elif 'analyze' in query.lower():
                                        rag_request['operation'] = 'analyze'
                                    else:
                                        rag_request['operation'] = 'rag'
                                    
                                    # Run the sub-agent via Archon
                                    sub_agent_response = await self.services['archon'].run_agent(
                                        thread_id=self.services['archon'].thread_id,
                                        user_input=f"Process content with this request: {json.dumps(rag_request)}"
                                    )
                                    
                                    # Store the sub-agent result
                                    rag_result = {
                                        'query': query,
                                        'response': sub_agent_response,
                                        'document_count': len(documents),
                                        'document_sources': document_summary,
                                        'status': 'success',
                                        'processed_by': f'sub-agent:{rag_sub_agent_id}',
                                        'operation': rag_request.get('operation', 'rag')
                                    }
                                else:
                                    # Fall back to the built-in RAG tool
                                    logger.info(f"Performing RAG on {len(documents)} documents with built-in tool")
                                    
                                    # For summarization or analysis without sub-agent, use custom prompt
                                    if 'summarize' in query.lower() or 'analyze' in query.lower():
                                        # Custom prompt for summarization/analysis
                                        custom_query = query
                                        if 'summarize' in query.lower() and not query.startswith('summarize'):
                                            custom_query = f"Summarize the following content: {query}"
                                        elif 'analyze' in query.lower() and not query.startswith('analyze'):
                                            custom_query = f"Analyze the following content: {query}"
                                        
                                        rag_result = await tool_func(ctx, custom_query, documents)
                                        rag_result['operation'] = 'summarize' if 'summarize' in query.lower() else 'analyze'
                                    else:
                                        rag_result = await tool_func(ctx, query, documents)
                                        rag_result['operation'] = 'rag'
                                    
                                    # Add document sources for tracking
                                    rag_result['document_sources'] = document_sources
                                
                                result['data']['rag_result'] = rag_result
                                deps.state['rag_results'] = rag_result
                                
                                # Log RAG results
                                logfire.info("RAG/Analysis results",
                                    query=query,
                                    document_count=len(documents),
                                    operation=rag_result.get('operation', 'rag'),
                                    chunks_used=len(rag_result.get('chunks_used', [])) if 'chunks_used' in rag_result else 0,
                                    status=rag_result.get('status'),
                                    processor=rag_result.get('processed_by', 'built-in'),
                                    response_length=len(rag_result.get('response', ''))
                                )
                            except Exception as e:
                                logger.error(f"Error performing RAG/Analysis: {str(e)}")
                                logger.error(traceback.format_exc())
                                result['data']['rag_result'] = {
                                    'query': query,
                                    'response': f'Error processing content: {str(e)}',
                                    'chunks_used': [],
                                    'status': 'error',
                                    'error': str(e),
                                    'traceback': traceback.format_exc()
                                }
                        else:
                            logger.warning("No content available for RAG/Analysis processing.")
                            result['data']['rag_result'] = {
                                'query': query,
                                'response': 'No content available to process.',
                                'chunks_used': [],
                                'status': 'no_data'
                            }
                    else:
                        # For any other tools
                        try:
                            # Generic tool call with properly ordered arguments
                            tool_result = await tool_func(ctx, query)
                            result['data'][tool] = tool_result
                            deps.state[tool] = tool_result
                        except Exception as e:
                            logger.error(f"Error executing tool {tool}: {str(e)}")
                            result['data'][tool] = {"status": "error", "error": str(e)}
                
                else:
                    logger.warning(f"Tool {tool} not available")
                    result['data'][tool] = {"status": "unavailable", "message": f"Tool {tool} not available"}
            
            # Track progress with proper Supabase insert
            if deps.supabase:
                try:
                    logger.info("Storing workflow result in Supabase")
                    
                    # Store individual attachment records for better tracking
                    if 'attachments' in result['data'] and result['data']['attachments']:
                        for attachment in result['data']['attachments']:
                            if attachment['status'] == 'success':
                                # Find the corresponding email
                                email = next((e for e in deps.state.get('processed_emails', []) 
                                         if e['id'] == attachment.get('email_id')), {})
                                
                                attachment_record = {
                                    'query': query,
                                    'email_id': attachment.get('email_id', 'unknown'),
                                    'email_subject': email.get('subject', 'Unknown'),
                                    'email_from': email.get('from', 'Unknown'),
                                    'attachment_filename': attachment.get('filename', 'unknown'),
                                    'drive_id': attachment.get('drive_id', ''),
                                    'mime_type': attachment.get('mime_type', 'unknown'),
                                    'size': attachment.get('size', 0),
                                    'status': 'completed',
                                    'timestamp': datetime.now().isoformat(),
                                    'web_link': attachment.get('web_link', '')
                                }
                                
                                # Execute synchronous insert for each attachment
                                try:
                                    attachment_response = deps.supabase.table('attachment_items').insert(attachment_record).execute()
                                    logger.info(f"Supabase attachment insert response: {attachment_response}")
                                except Exception as attachment_error:
                                    logger.error(f"Failed to store attachment in Supabase: {str(attachment_error)}")
                    
                    # Create a simplified record for storage with email_count
                    record = {
                        'query': query,
                        'gmail_query': gmail_query,
                        'timestamp': datetime.now().isoformat(),
                        'email_count': len(deps.state.get('processed_emails', [])),
                        'attachment_count': len(result['data'].get('attachments', [])),
                        'success_attachments': sum(1 for a in result['data'].get('attachments', []) if a.get('status') == 'success'),
                        'has_rag_result': 'rag_result' in result['data'],
                        'status': 'completed',
                        'runtime': time.time() - started_at
                    }
                    
                    # Only add summary of result data to avoid JSON size issues
                    result_summary = {}
                    if 'emails' in result['data']:
                        result_summary['email_count'] = len(result['data']['emails'])
                    if 'attachments' in result['data']:
                        result_summary['attachment_count'] = len(result['data']['attachments'])
                    if 'rag_result' in result['data']:
                        result_summary['rag_status'] = result['data']['rag_result'].get('status')
                        # Only include the first 1000 characters of the response
                        rag_response = result['data']['rag_result'].get('response', '')
                        if isinstance(rag_response, str) and len(rag_response) > 1000:
                            result_summary['rag_response'] = rag_response[:1000] + "..."
                        else:
                            result_summary['rag_response'] = rag_response
                    
                    record['result_summary'] = json.dumps(result_summary, default=str)
                    
                    # Execute synchronous insert
                    try:
                        insert_response = deps.supabase.table('processed_items').insert(record).execute()
                        logger.info(f"Supabase insert response: {insert_response}")
                    except Exception as insert_error:
                        logger.error(f"Failed to store main record in Supabase: {str(insert_error)}")
                        try:
                            # Try again with a simpler record
                            simplified_record = {
                                'query': query,
                                'timestamp': datetime.now().isoformat(),
                                'status': 'completed',
                                'runtime': time.time() - started_at
                            }
                            retry_insert = deps.supabase.table('processed_items').insert(simplified_record).execute()
                            logger.info(f"Simplified Supabase insert response: {retry_insert}")
                        except Exception as retry_error:
                            logger.error(f"Failed to store simplified record: {str(retry_error)}")
                    
                except Exception as e:
                    logger.error(f"Failed to store result in Supabase: {str(e)}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning("Supabase client not available, skipping result storage")
            
            # Calculate runtime
            runtime = time.time() - started_at
            result['runtime'] = runtime
            logger.info(f"Workflow completed in {runtime:.2f} seconds")
            
            return result
        except Exception as e:
            logger.error(f"Error running workflow: {str(e)}")
            logger.error(traceback.format_exc())
            logfire.error("Workflow error", 
                query=query,
                error=str(e),
                traceback=traceback.format_exc()
            )
            return {
                'status': 'error',
                'error': str(e),
                'query': query,
                'runtime': time.time() - started_at
            }

# Create the agent instance
primary_agent = TechevoRagAgent()

async def main() -> None:
    """Run a test of the agent."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Validate required environment variables
        validate_env_vars()
        
        # Initialize services
        gmail_service, drive_service = await setup_google_services()
        
        # Initialize FAISS index
        index = create_faiss_index(dimension=768)  # Correct dimension for all-MiniLM-L6-v2
        
        # Create embedding model
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Supabase
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            logger.error("SUPABASE_URL or SUPABASE_KEY is missing in .env")
            raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in .env")
        
        try:
            supabase = create_client(supabase_url, supabase_key)
            # Validate connection by making a simple query
            result = supabase.table('processed_items').select('id').limit(1).execute()
            logger.info("Supabase connection validated successfully")
        except Exception as e:
            logger.error(f"Supabase connection failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to connect to Supabase: {str(e)}")
        
        # Create Archon client
        archon_client = ArchonClient()
        thread_id = await archon_client.create_thread()
        
        # Create dependencies object
        deps = EnhancedDeps(
            gmail_service=gmail_service,
            drive_service=drive_service,
            faiss_index=index,
            embedding_model=embedding_model,
            supabase=supabase,
            archon_client=archon_client,
            state={}
        )
        
        # Create a dictionary of services
        services = {
            'gmail': gmail_service,
            'drive': drive_service,
            'faiss_index': index,
            'embedding_model': embedding_model,
            'supabase': supabase,
            'archon': archon_client
        }
        
        # Create agent
        agent = TechevoRagAgent(services=services)
        
        # Define test query
        test_query = "process campaign emails"
        logger.info(f"Testing agent with query: {test_query}")
        
        # Run workflow
        result = await agent.run_workflow(test_query, deps)
        
        # Log result
        logger.info(f"Test result: {json.dumps(result, indent=2, default=str)}")
        
        with open('test_result.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Update README with test date
        readme_path = 'README.md'
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        
        today = datetime.now().strftime('%B %d, %Y')
        test_summary = f"Tested on {today}: "
        
        if 'error' in result:
            test_summary += f"Error encountered: {result['error']}"
        else:
            if 'results' in result and 'emails' in result['results']:
                emails_count = len(result['results']['emails'])
                test_summary += f"Found {emails_count} emails, "
            else:
                test_summary += "No emails found, "
                
            if 'results' in result and 'rag' in result['results']:
                test_summary += "RAG successful"
            else:
                test_summary += "RAG not performed"
        
        # Update the README file with test results
        if "## Testing" in readme_content:
            # Replace the existing test line if it exists
            if "- Tested on" in readme_content:
                import re
                updated_content = re.sub(
                    r"- Tested on .*$", 
                    f"- {test_summary}", 
                    readme_content, 
                    flags=re.MULTILINE
                )
            else:
                # Add as a new line under Testing section
                updated_content = readme_content.replace(
                    "## Testing", 
                    f"## Testing\n\n- {test_summary}"
                )
        else:
            # Add a new Testing section
            updated_content = readme_content + f"\n\n## Testing\n\n- {test_summary}"
        
        with open(readme_path, 'w') as f:
            f.write(updated_content)
        
        logger.info("Test completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}

if __name__ == "__main__":
    asyncio.run(main()) 