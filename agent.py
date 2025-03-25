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
            
            # Initialize state
            deps.state = {
                'query': query,
                'processed_emails': [],
                'downloaded_attachments': [],
                'rag_results': [],
                'start_time': started_at
            }
            
            # Create new thread if needed
            if not self.services.get('archon') or not self.services['archon'].thread_id:
                try:
                    if 'archon' in self.services:
                        self.services['archon'].thread_id = await self.services['archon'].create_thread()
                    else:
                        logger.error("Archon service not available")
                except Exception as e:
                    logger.error(f"Failed to create Archon thread: {e}")
                    logger.error(traceback.format_exc())
            
            logger.info(f"Predicting intent for query: {query}")
            
            # Get intent with enhanced search parameters
            if 'archon' in self.services:
                intent_data = await predict_intent(query, self.services['archon'])
                intent_tools = intent_data.get('tools', ['search_emails', 'perform_rag'])
                sender = intent_data.get('sender')
                keywords = intent_data.get('keywords', query)
                has_attachment = intent_data.get('has_attachment', False)
                logger.info(f"Predicted intent: tools={intent_tools}, sender={sender}, keywords={keywords}, has_attachment={has_attachment}")
            else:
                # Fallback without Archon
                logger.warning("Archon service not available, using default intent prediction")
                intent_data = fallback_intent(query)
                intent_tools = intent_data.get('tools', ['search_emails', 'perform_rag'])
                sender = intent_data.get('sender')
                keywords = intent_data.get('keywords', query)
                has_attachment = intent_data.get('has_attachment', False)
                logger.info(f"Fallback intent: tools={intent_tools}, sender={sender}, keywords={keywords}, has_attachment={has_attachment}")
            
            # Construct Gmail search query
            search_query = []
            if sender:
                search_query.append(f"from:{sender}")
            if keywords and keywords != query:  # Only add if keywords are specific and not the full query
                # Extract meaningful keywords if possible
                search_query.append(keywords)
            if has_attachment:
                search_query.append("has:attachment")
                
            # Add default time range if no specific criteria
            if not search_query:
                search_query.append(query)  # Use original query if no specific params extracted
                
            gmail_query = " ".join(search_query)
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
            
            # Check if any tools are missing and create new sub-agents if needed
            missing_tools = [tool for tool in intent_tools if tool not in self.available_tools]
            if missing_tools:
                logger.info(f"Missing tools: {missing_tools}. Attempting to create new sub-agent...")
                
                # Determine skills based on missing tools
                skills = []
                for tool in missing_tools:
                    if 'search' in tool.lower():
                        skills.append("searching")
                    elif 'download' in tool.lower():
                        skills.append("file_download")
                    elif 'rag' in tool.lower() or 'process' in tool.lower():
                        skills.append("rag_processing")
                    else:
                        skills.append("general_processing")
                
                # Create a new sub-agent with Archon MCP
                try:
                    if 'archon' in self.services:
                        agent_id = await self.services['archon'].create_agent(
                            skills=list(set(skills)),  # Remove duplicates 
                            level="sub-agent"
                        )
                        logger.info(f"Created new sub-agent with ID: {agent_id}")
                        
                        # Add the new agent to our collection
                        self.sub_agents[agent_id] = skills
                        
                        # Create simulated functions for missing tools
                        for tool_name in missing_tools:
                            async def simulated_tool_fn(ctx, **kwargs):
                                logger.info(f"Executing simulated {tool_name} with sub-agent {agent_id}")
                                try:
                                    # Run sub-agent through Archon
                                    tool_query = f"Execute {tool_name} with parameters: {json.dumps(kwargs)}"
                                    agent_response = await self.services['archon'].run_agent(
                                        thread_id=self.services['archon'].thread_id,
                                        user_input=tool_query
                                    )
                                    
                                    return {
                                        "status": "success", 
                                        "data": agent_response,
                                        "tool": tool_name,
                                        "agent_id": agent_id
                                    }
                                except Exception as e:
                                    logger.error(f"Error executing simulated {tool_name}: {str(e)}")
                                    return {
                                        "status": "error",
                                        "error": str(e),
                                        "tool": tool_name
                                    }
                            
                            # Add the function to available tools
                            self.available_tools[tool_name] = simulated_tool_fn
                            logger.info(f"Added simulated tool: {tool_name}")
                    else:
                        logger.error("Cannot create sub-agent: Archon service not available")
                except Exception as e:
                    logger.error(f"Failed to create sub-agent: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Create agent context for tool execution
            from agent_tools import AgentContext
            ctx = AgentContext(agent=self, deps=deps, log=logger)
            
            # Run the workflow with available tools
            for tool in intent_tools:
                if tool in self.available_tools:
                    logger.info(f"Running tool: {tool}")
                    tool_func = self.available_tools[tool]
                    
                    if tool == 'search_emails':
                        try:
                            # Use the raw query instead of the constructed Gmail query
                            logger.info(f"Searching emails with raw query: {query}")
                            # Pass context and raw query directly to the tool
                            emails = await tool_func(ctx, query)
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
                        attachment_results = []
                        
                        # Try to find emails if not already in state
                        if not deps.state.get('processed_emails'):
                            logger.info("No emails in state, running search_emails first")
                            if 'search_emails' in self.available_tools:
                                try:
                                    # Use raw query directly
                                    emails = await self.available_tools['search_emails'](ctx, query)
                                    deps.state['processed_emails'] = emails
                                except Exception as e:
                                    logger.error(f"Error searching emails before download: {str(e)}")
                            else:
                                logger.warning("Cannot search emails: tool not available")
                        
                        if deps.state.get('processed_emails'):
                            for email in deps.state['processed_emails']:
                                if email.get('attachments'):
                                    for attachment in email.get('attachments', []):
                                        logger.info(f"Downloading attachment {attachment.get('filename', 'unknown')} from email {email.get('id', 'unknown')}")
                                        try:
                                            # Update to use the context based method signature with raw query
                                            download_result = await tool_func(ctx, query)
                                            attachment_results.append(download_result)
                                        except Exception as e:
                                            logger.error(f"Error downloading attachment: {str(e)}")
                                            attachment_results.append({
                                                "status": "error", 
                                                "error": str(e), 
                                                "email_id": email.get('id', 'unknown'),
                                                "attachment": attachment.get('filename', 'unknown')
                                            })
                            
                            result['data']['attachments'] = attachment_results
                            deps.state['downloaded_attachments'] = attachment_results
                            
                            # Log attachment download results
                            logfire.info("Attachment download results",
                                query=query,
                                attachment_count=len(attachment_results),
                                success_count=sum(1 for a in attachment_results if a.get('status') != 'error'),
                                error_count=sum(1 for a in attachment_results if a.get('status') == 'error')
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
                    
                    elif tool == 'perform_rag':
                        documents = []
                        
                        # Collect documents from emails
                        if 'processed_emails' in deps.state and deps.state['processed_emails']:
                            for email in deps.state['processed_emails']:
                                if email.get('body'):
                                    documents.append(email.get('body'))
                        
                        # Collect documents from attachments
                        if 'attachments_content' in deps.state and deps.state['attachments_content']:
                            for filename, content in deps.state['attachments_content'].items():
                                if isinstance(content, bytes):
                                    try:
                                        text = content.decode('utf-8', errors='ignore')
                                        documents.append(text)
                                    except:
                                        logger.warning(f"Could not decode attachment content for {filename}")
                                elif isinstance(content, str):
                                    documents.append(content)
                        
                        # Collect documents from Drive
                        if 'drive_contents' in deps.state and deps.state['drive_contents']:
                            for file_id, content in deps.state['drive_contents'].items():
                                if content:
                                    documents.append(content)
                        
                        if documents:
                            try:
                                logger.info(f"Performing RAG on {len(documents)} documents")
                                rag_result = await tool_func(ctx, query, documents)
                                result['data']['rag_result'] = rag_result
                                deps.state['rag_results'] = rag_result
                                
                                # Log RAG results
                                logfire.info("RAG results",
                                    query=query,
                                    document_count=len(documents),
                                    chunks_used=len(rag_result.get('chunks_used', [])),
                                    status=rag_result.get('status'),
                                    model=rag_result.get('model', 'gemini-2.0-flash'),
                                    response_length=len(rag_result.get('response', ''))
                                )
                            except Exception as e:
                                logger.error(f"Error performing RAG: {str(e)}")
                                result['data']['rag_result'] = {
                                    'query': query,
                                    'response': f'Error performing RAG: {str(e)}',
                                    'chunks_used': [],
                                    'status': 'error'
                                }
                        else:
                            logger.warning("No documents available for RAG processing.")
                            result['data']['rag_result'] = {
                                'query': query,
                                'response': 'No documents available to process.',
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
                    logger.warning(f"Tool {tool} not available and could not be created")
                    result['data'][tool] = {"status": "unavailable", "message": f"Tool {tool} not available"}
            
            # Track progress with proper Supabase insert
            if deps.supabase:
                try:
                    logger.info("Storing workflow result in Supabase")
                    # Create a simplified record for storage
                    record = {
                        'query': query,
                        'gmail_query': gmail_query,
                        'timestamp': datetime.now().isoformat(),
                        'email_count': len(deps.state.get('processed_emails', [])),
                        'has_attachments': any(email.get('attachments') for email in deps.state.get('processed_emails', [])),
                        'status': 'completed',
                        'runtime': time.time() - started_at
                    }
                    
                    # Execute synchronous insert
                    insert_response = deps.supabase.table('processed_items').insert(record).execute()
                    logger.info(f"Supabase insert response: {insert_response}")
                    
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