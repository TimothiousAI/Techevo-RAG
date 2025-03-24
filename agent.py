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
logfire.configure()

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
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly',
          'https://www.googleapis.com/auth/drive']

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
    
    # Check if token already exists
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_info(
            json.load(open(token_path)), SCOPES)
    
    # If no valid credentials available, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path or not os.path.exists(credentials_path):
                error_msg = f"Credentials file not found at {credentials_path}. Please set CREDENTIALS_JSON_PATH in .env to a valid credentials.json file."
                logger.error(error_msg)
                logfire.error(error_msg)
                raise ValueError(error_msg)
                
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    # Build services
    gmail_service = build('gmail', 'v1', credentials=creds)
    drive_service = build('drive', 'v3', credentials=creds)
    
    return gmail_service, drive_service

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

async def predict_intent(query: str, archon_client: ArchonClient) -> List[str]:
    """
    Predict the intent of a user query using Archon MCP.
    
    Args:
        query: The user query
        archon_client: The Archon MCP client
        
    Returns:
        List of tool names to execute
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
            max_tokens=100
        )
        
        # Parse the response
        intent_text = response.get('response', '').strip()
        logger.info(f"Intent prediction: {intent_text}")
        
        # Try to extract a list of tools
        try:
            # Check if it looks like a Python list representation
            if intent_text.startswith('[') and intent_text.endswith(']'):
                tools = json.loads(intent_text)
                
                # Verify response is valid
                if not tools or not isinstance(tools, list):
                    error_msg = "Invalid tools list from Archon, retrying with context"
                    logger.error(error_msg)
                    logfire.error(error_msg)
                    
                    # Retry with more context
                    return await retry_predict_intent(query, archon_client)
                
                # Make sure perform_rag is included for process/summarize queries
                if ('process' in query.lower() or 'summarize' in query.lower() or 'analyze' in query.lower()) and 'perform_rag' not in tools:
                    tools.append('perform_rag')
                    logger.info("Added perform_rag to tools list for processing/summarization query")
                
                return tools
            else:
                # Handle plaintext responses by looking for tool names
                valid_tools = ['search_emails', 'download_attachment', 'perform_rag', 'search_drive']
                found_tools = []
                
                for tool in valid_tools:
                    if tool.lower() in intent_text.lower():
                        found_tools.append(tool)
                
                # Add perform_rag for processing/summarization queries
                if ('process' in query.lower() or 'summarize' in query.lower() or 'analyze' in query.lower()) and 'perform_rag' not in found_tools:
                    found_tools.append('perform_rag')
                    logger.info("Added perform_rag to tools list for processing/summarization query")
                
                if not found_tools:
                    logger.warning("No tools found in plaintext response, retrying")
                    logfire.warning("No tools found in plaintext response, retrying")
                    return await retry_predict_intent(query, archon_client)
                    
                return found_tools
        except json.JSONDecodeError:
            # Handle plain text responses
            found_tools = []
            
            if 'search_emails' in intent_text.lower():
                found_tools.append('search_emails')
            if 'download' in intent_text.lower() or 'attachment' in intent_text.lower():
                found_tools.append('download_attachment')
            if 'rag' in intent_text.lower():
                found_tools.append('perform_rag')
            if 'drive' in intent_text.lower():
                found_tools.append('search_drive')
            
            # Add perform_rag for processing/summarization queries
            if ('process' in query.lower() or 'summarize' in query.lower() or 'analyze' in query.lower()) and 'perform_rag' not in found_tools:
                found_tools.append('perform_rag')
                logger.info("Added perform_rag to tools list for processing/summarization query")
            
            return found_tools if found_tools else ['search_emails', 'perform_rag']
    
    except Exception as e:
        error_msg = f"Error predicting intent: {str(e)}"
        logger.error(error_msg)
        logfire.error(error_msg)
        
        # Fall back to a default set of tools
        return ['search_emails', 'perform_rag']

async def retry_predict_intent(query: str, archon_client: ArchonClient) -> List[str]:
    """
    Retry intent prediction with more context if the first attempt failed.
    
    Args:
        query: The user query
        archon_client: The Archon MCP client
        
    Returns:
        List of tool names to execute
    """
    try:
        # More detailed prompt with examples
        detailed_prompt = f"""
        Analyze this query: "{query}"
        
        Determine which tools to use from:
        - search_emails: Search for emails matching criteria
        - download_attachment: Download attachments from emails
        - perform_rag: Perform retrieval augmented generation
        - search_drive: Search files in Google Drive
        
        Examples:
        - "find emails about marketing" → ["search_emails"]
        - "download attachments from project emails" → ["search_emails", "download_attachment"]
        - "process campaign data" → ["search_emails", "download_attachment", "perform_rag"]
        
        Return only a JSON array of tool names, like: ["tool1", "tool2"]
        """
        
        response = await archon_client.generate(
            model="openai:gpt-4o",
            prompt=detailed_prompt,
            temperature=0.1,
            max_tokens=100
        )
        
        intent_text = response.get('response', '').strip()
        
        # Try to parse as JSON
        try:
            if '[' in intent_text and ']' in intent_text:
                # Extract the array part
                start = intent_text.find('[')
                end = intent_text.rfind(']') + 1
                array_text = intent_text[start:end]
                
                tools = json.loads(array_text)
                if tools and isinstance(tools, list):
                    return tools
            
            # If parsing fails, use default tools
            return ['search_emails', 'perform_rag']
        except:
            # Default fallback
            return ['search_emails', 'perform_rag']
    except Exception as e:
        logger.error(f"Error in retry_predict_intent: {str(e)}")
        logfire.error(f"Error in retry_predict_intent: {str(e)}")
        return ['search_emails', 'perform_rag']

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
            intent_tools = []
            
            if 'archon' in self.services:
                intent_tools = await predict_intent(query, self.services['archon'])
                logger.info(f"Predicted tools via Archon: {intent_tools}")
            else:
                # Fallback without Archon
                logger.warning("Archon service not available, using default intent prediction")
                
                # Simple keyword-based tool selection
                intent_tools = []
                if 'email' in query.lower() or 'mail' in query.lower():
                    intent_tools.append('search_emails')
                if 'attachment' in query.lower() or 'download' in query.lower() or 'file' in query.lower():
                    intent_tools.append('download_attachment')
                if 'drive' in query.lower() or 'document' in query.lower():
                    intent_tools.append('search_drive')
                if 'process' in query.lower() or 'analyze' in query.lower() or 'summarize' in query.lower():
                    intent_tools.append('perform_rag')
                
                # Always include search_emails and perform_rag as fallback
                if not intent_tools or len(intent_tools) == 0:
                    intent_tools = ['search_emails', 'perform_rag']
                
                logger.info(f"Predicted tools via fallback: {intent_tools}")
            
            result = {
                'status': 'success',
                'data': {},
                'query': query,
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
                        emails = await tool_func(ctx, query=query, services=self.services, deps=deps)
                        result['data']['emails'] = emails
                        deps.state['processed_emails'] = emails
                    
                    elif tool == 'download_attachment':
                        attachment_results = []
                        
                        # Try to find emails if not already in state
                        if not deps.state.get('processed_emails'):
                            logger.info("No emails in state, running search_emails first")
                            if 'search_emails' in self.available_tools:
                                emails = await self.available_tools['search_emails'](
                                    ctx, query=query, services=self.services, deps=deps
                                )
                                deps.state['processed_emails'] = emails
                            else:
                                logger.warning("Cannot search emails: tool not available")
                        
                        if deps.state.get('processed_emails'):
                            for email in deps.state['processed_emails']:
                                if email.get('attachments'):
                                    for attachment in email.get('attachments', []):
                                        logger.info(f"Downloading attachment {attachment.get('filename', 'unknown')} from email {email.get('id', 'unknown')}")
                                        try:
                                            download_result = await tool_func(
                                                ctx,
                                                query=query,
                                                services=self.services,
                                                deps=deps
                                            )
                                            attachment_results.append(download_result)
                                        except Exception as e:
                                            logger.error(f"Error downloading attachment: {str(e)}")
                                            attachment_results.append({
                                                "status": "error",
                                                "error": str(e),
                                                "filename": attachment.get('filename', 'unknown')
                                            })
                        
                        result['data']['attachments'] = attachment_results
                        deps.state['downloaded_attachments'] = attachment_results
                    
                    elif tool == 'search_drive':
                        try:
                            drive_files = await tool_func(ctx, query=query, services=self.services, deps=deps)
                            result['data']['drive_files'] = drive_files
                            deps.state['drive_files'] = drive_files
                        except Exception as e:
                            logger.error(f"Error searching drive: {str(e)}")
                            result['data']['drive_files'] = {"status": "error", "error": str(e)}
                    
                    elif tool == 'perform_rag':
                        # Gather documents from various sources
                        documents = []
                        
                        # From emails
                        if 'processed_emails' in deps.state and deps.state['processed_emails']:
                            for email in deps.state['processed_emails']:
                                if email.get('body'):
                                    documents.append(email.get('body'))
                        
                        # From attachments
                        if 'attachments_content' in deps.state and deps.state['attachments_content']:
                            for filename, content in deps.state['attachments_content'].items():
                                if isinstance(content, str):
                                    documents.append(content)
                                elif isinstance(content, bytes):
                                    # Try to decode bytes to string
                                    try:
                                        text_content = content.decode('utf-8', errors='ignore')
                                        documents.append(f"ATTACHMENT: {filename}\n\n{text_content}")
                                    except:
                                        logger.warning(f"Could not decode attachment {filename} as text")
                        
                        # From drive files
                        if 'drive_contents' in deps.state and deps.state['drive_contents']:
                            for file_id, content in deps.state['drive_contents'].items():
                                if content:
                                    documents.append(content)
                        
                        # If we don't have any documents yet, try to get some
                        if not documents and 'search_emails' in self.available_tools:
                            logger.info("No documents for RAG, trying to search emails first")
                            try:
                                emails = await self.available_tools['search_emails'](
                                    ctx, query=query, services=self.services, deps=deps
                                )
                                deps.state['processed_emails'] = emails
                                
                                # Add email bodies to documents
                                for email in emails:
                                    if email.get('body'):
                                        documents.append(email.get('body'))
                            except Exception as e:
                                logger.error(f"Error searching emails for RAG: {str(e)}")
                        
                        if documents:
                            try:
                                rag_result = await tool_func(ctx, query=query, documents=documents)
                                result['data']['rag_result'] = rag_result
                                deps.state['rag_results'] = rag_result
                            except Exception as e:
                                logger.error(f"Error performing RAG: {str(e)}")
                                result['data']['rag_result'] = {
                                    'status': 'error',
                                    'error': str(e),
                                    'query': query
                                }
                        else:
                            # No documents to process
                            logger.warning("No documents available for RAG processing")
                            result['data']['rag_result'] = {
                                'status': 'no_documents',
                                'message': 'No documents available for RAG processing'
                            }
                    else:
                        # For any other tools
                        try:
                            tool_result = await tool_func(ctx, query=query, services=self.services, deps=deps)
                            result['data'][tool] = tool_result
                            deps.state[tool] = tool_result
                        except Exception as e:
                            logger.error(f"Error executing tool {tool}: {str(e)}")
                            result['data'][tool] = {"status": "error", "error": str(e)}
                
                else:
                    logger.warning(f"Tool {tool} not available and could not be created")
                    result['data'][tool] = {"status": "unavailable", "message": f"Tool {tool} not available"}
            
            # Track progress
            try:
                progress_record = {
                    'email_id': 'system',
                    'file_hash': hashlib.md5(query.encode()).hexdigest(),
                    'status': 'completed',
                    'filename': f'query_{int(time.time())}'
                }
                await agent_tools.track_progress(ctx, **progress_record)
            except Exception as e:
                logger.error(f"Failed to track progress: {e}")
            
            # Add execution time
            execution_time = time.time() - started_at
            result['execution_time'] = f"{execution_time:.2f} seconds"
            logger.info(f"Workflow executed in {execution_time:.2f} seconds")
            
            return result
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Ensure we have a valid response even in case of error
            execution_time = time.time() - started_at
            return {
                'status': 'error',
                'error': str(e),
                'query': query,
                'execution_time': f"{execution_time:.2f} seconds",
                'timestamp': datetime.now().isoformat()
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