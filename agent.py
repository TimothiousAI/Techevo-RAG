"""
Main agent implementation for Techevo-RAG.

This module contains the primary agent implementation, workflow orchestration,
and setup functions for Gmail, Drive, and FAISS.
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
import hashlib
from datetime import datetime
import copy

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
from pydantic_ai.agent import Agent, Deps
from pydantic_ai.agent.context import AgentContext
from pydantic_ai.agent.retry import ModelRetry
from pydantic_ai.agent.parallel import parallel

# Logging with logfire
import logfire

# Local imports
from agent_prompts import SYSTEM_PROMPT, INTENT_PROMPT
import agent_tools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure logfire
logfire.configure(
    app_name="techevo-rag",
    level="INFO",
    capture_stdout=True,
    capture_stderr=True
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
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly',
          'https://www.googleapis.com/auth/drive']

# For MCP-based communication with Archon
class ArchonClient:
    """Client for interacting with Archon MCP."""
    
    def __init__(self):
        self.session = None
        self.thread_id = None
        self._prev_states = {}  # Store previous states to revert if needed
    
    async def ensure_session(self):
        """Ensure an aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def create_thread(self):
        """
        Create a thread in Archon via MCP.
        
        Returns:
            str: Thread ID for the conversation
        """
        if self.thread_id is not None:
            return self.thread_id
            
        try:
            # Direct MCP call through Cursor
            import cursor.mcp.archon as archon_mcp
            
            result = await archon_mcp.create_thread(random_string="init")
            self.thread_id = result
            logger.info(f"Created Archon thread: {self.thread_id}")
            logfire.info(f"Created Archon thread: {self.thread_id}")
            return self.thread_id
        except ImportError:
            logger.warning("Cursor MCP not available, falling back to placeholder thread ID")
            logfire.warning("Cursor MCP not available, falling back to placeholder thread ID")
            # Fallback to a placeholder ID for testing without Cursor
            self.thread_id = f"thread_{datetime.now().timestamp()}"
            return self.thread_id
        except Exception as e:
            error_msg = f"Error creating Archon thread: {str(e)}"
            logger.error(error_msg)
            logfire.error(error_msg)
            # Fallback to a placeholder ID for testing
            self.thread_id = f"thread_{datetime.now().timestamp()}"
            return self.thread_id
    
    async def run_agent(self, thread_id: str, user_input: str):
        """
        Run the Archon agent with the given input via MCP.
        
        Args:
            thread_id: Thread ID for the conversation
            user_input: User message to process
            
        Returns:
            str: The agent's response
        """
        try:
            # Direct MCP call through Cursor
            import cursor.mcp.archon as archon_mcp
            
            result = await archon_mcp.run_agent(
                thread_id=thread_id,
                user_input=user_input
            )
            
            # Verify response isn't empty or invalid
            if not result or len(result.strip()) < 10:
                logger.warning("Empty or very short response from Archon")
                logfire.warning("Empty or very short response from Archon")
                raise ValueError("Empty or invalid response from Archon")
                
            return result
        except ImportError:
            logger.warning("Cursor MCP not available, falling back to OpenAI")
            logfire.warning("Cursor MCP not available, falling back to OpenAI")
            # For testing purposes without Cursor
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"Error running Archon agent: {str(e)}"
            logger.error(error_msg)
            logfire.error(error_msg)
            # For testing purposes only
            return f"Simulated response to: {user_input}"
    
    async def generate(self, model: str, prompt: str, **kwargs):
        """
        Generate text using Archon MCP.
        
        Args:
            model: Model to use (e.g., 'openai:gpt-4o')
            prompt: The prompt to send
            **kwargs: Additional parameters like temperature, max_tokens
            
        Returns:
            dict: Response containing generated text
        """
        await self.ensure_session()
        
        # Save original prompt to support retry
        original_prompt = prompt
        
        # Keep prompt size under 5k tokens to avoid Archon MCP limits
        if len(prompt) > 5000:
            logger.warning(f"Prompt size exceeds recommended limit: {len(prompt)} chars")
            logfire.warning(f"Prompt size exceeds recommended limit: {len(prompt)} chars")
            # Truncate if needed
            prompt = prompt[:4800] + "...[truncated]"
        
        try:
            # Direct MCP call through Cursor
            import cursor.mcp.archon as archon_mcp
            
            # Get or create thread
            thread_id = await self.create_thread()
            
            # Run the agent
            response_text = await self.run_agent(
                thread_id=thread_id,
                user_input=prompt
            )
            
            return {"response": response_text}
            
        except ImportError:
            logger.warning("Cursor MCP not available, falling back to OpenAI")
            logfire.warning("Cursor MCP not available, falling back to OpenAI")
            # Fallback to OpenAI for testing without Cursor
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.2),
                max_tokens=kwargs.get('max_tokens', 500)
            )
            
            return {"response": response.choices[0].message.content}
            
        except Exception as e:
            error_msg = f"Error generating with Archon: {str(e)}"
            logger.error(error_msg)
            logfire.error(error_msg)
            return {"response": "Error processing your request"}
    
    def save_state(self, key: str, state: Any):
        """Save a state to allow reverting if needed."""
        self._prev_states[key] = copy.deepcopy(state)
    
    def get_previous_state(self, key: str) -> Any:
        """Get a previously saved state."""
        return self._prev_states.get(key)

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

def create_faiss_index(dimension: int = 768) -> faiss.IndexFlatL2:
    """
    Create a FAISS index for storing document embeddings.
    
    Args:
        dimension: Dimension of embeddings (default: 768 for all-MiniLM-L6-v2)
        
    Returns:
        FAISS index
    """
    return faiss.IndexFlatL2(dimension)

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
                    
                return tools
            else:
                # Handle plaintext responses by looking for tool names
                valid_tools = ['search_emails', 'download_attachment', 'perform_rag', 'search_drive']
                found_tools = []
                
                for tool in valid_tools:
                    if tool.lower() in intent_text.lower():
                        found_tools.append(tool)
                
                if not found_tools:
                    logger.warning("No tools found in plaintext response, retrying")
                    logfire.warning("No tools found in plaintext response, retrying")
                    return await retry_predict_intent(query, archon_client)
                    
                return found_tools
        except json.JSONDecodeError:
            # Handle plain text responses
            if 'search_emails' in intent_text.lower():
                return ['search_emails']
            elif 'download' in intent_text.lower() or 'attachment' in intent_text.lower():
                return ['search_emails', 'download_attachment']
            elif 'rag' in intent_text.lower():
                return ['perform_rag']
            elif 'drive' in intent_text.lower():
                return ['search_drive']
            else:
                logger.warning("Could not parse tools from response, retrying")
                logfire.warning("Could not parse tools from response, retrying")
                return await retry_predict_intent(query, archon_client)
    
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

class TechevoRagAgent(Agent):
    """
    Main Techevo-RAG agent implementation.
    """
    
    async def run_workflow(self, query: str, deps: EnhancedDeps) -> Dict[str, Any]:
        """
        Run the workflow based on the predicted intent.
        
        Args:
            query: The user query
            deps: Agent dependencies
            
        Returns:
            The workflow result
        """
        try:
            # Initialize state for this run
            deps.state = {
                'query': query,
                'processed_emails': [],
                'downloaded_attachments': [],
                'rag_results': []
            }
            
            # Save initial state for possible reversion
            deps.archon_client.save_state('initial_state', deps.state)
            
            # Predict intent
            logger.info(f"Predicting intent for query: {query}")
            logfire.info(f"Predicting intent for query: {query}")
            intent_tools = await predict_intent(query, deps.archon_client)
            logger.info(f"Predicted tools: {intent_tools}")
            logfire.info(f"Predicted tools: {intent_tools}")
            
            # Crash protection: if no tools returned, use defaults
            if not intent_tools:
                logger.warning("No tools predicted, using default tools")
                logfire.warning("No tools predicted, using default tools")
                intent_tools = ['search_emails', 'perform_rag']
            
            result = {
                'status': 'success',
                'data': {}
            }
            
            # Execute tools based on intent
            if 'search_emails' in intent_tools:
                logger.info("Searching emails...")
                logfire.info("Searching emails...")
                try:
                    emails = await agent_tools.search_emails(
                        AgentContext(agent=self, deps=deps, log=logger),
                        query=query
                    )
                    
                    result['data']['emails'] = emails
                    deps.state['processed_emails'] = emails
                    
                    # If download_attachment is also in the intent, process attachments
                    if 'download_attachment' in intent_tools and emails:
                        attachment_results = []
                        
                        for email in emails:
                            if email.get('attachments'):
                                for attachment in email.get('attachments', []):
                                    logger.info(f"Downloading attachment {attachment.get('filename', 'unknown')} from email {email['id']}")
                                    logfire.info(f"Downloading attachment {attachment.get('filename', 'unknown')} from email {email['id']}")
                                    
                                    try:
                                        download_result = await agent_tools.download_attachment(
                                            AgentContext(agent=self, deps=deps, log=logger),
                                            email_id=email['id'],
                                            attachment_id=attachment['id'],
                                            folder_id=os.getenv('DEFAULT_DRIVE_FOLDER_ID')
                                        )
                                        
                                        attachment_results.append(download_result)
                                    except Exception as e:
                                        error_msg = f"Error downloading attachment: {str(e)}"
                                        logger.error(error_msg)
                                        logfire.error(error_msg)
                                        attachment_results.append({
                                            'status': 'error',
                                            'filename': attachment.get('filename', 'unknown'),
                                            'email_id': email['id'],
                                            'error': str(e)
                                        })
                        
                        result['data']['attachments'] = attachment_results
                        deps.state['downloaded_attachments'] = attachment_results
                except Exception as e:
                    error_msg = f"Error searching emails: {str(e)}"
                    logger.error(error_msg)
                    logfire.error(error_msg)
                    result['data']['email_error'] = str(e)
            
            if 'search_drive' in intent_tools:
                logger.info("Searching Drive...")
                logfire.info("Searching Drive...")
                folder_id = os.getenv('DEFAULT_DRIVE_FOLDER_ID')
                
                try:
                    drive_files = await agent_tools.search_drive(
                        AgentContext(agent=self, deps=deps, log=logger),
                        folder_id=folder_id
                    )
                    
                    result['data']['drive_files'] = drive_files
                    deps.state['drive_files'] = drive_files
                except Exception as e:
                    error_msg = f"Error searching Drive: {str(e)}"
                    logger.error(error_msg)
                    logfire.error(error_msg)
                    result['data']['drive_error'] = str(e)
            
            if 'perform_rag' in intent_tools:
                logger.info("Performing RAG...")
                logfire.info("Performing RAG...")
                
                # Gather documents from the workflow results so far
                documents = []
                
                # If we have emails, use their content
                if 'processed_emails' in deps.state and deps.state['processed_emails']:
                    for email in deps.state['processed_emails']:
                        # Just use the snippet as a simple example
                        # In a real system, you'd want to extract the full email body
                        if email.get('snippet'):
                            documents.append(email.get('snippet', ''))
                
                # If we have drive files, we would need to get their content
                # This is simplified - in a real system, you'd need to fetch and parse files
                
                if documents:
                    try:
                        rag_result = await agent_tools.perform_rag(
                            AgentContext(agent=self, deps=deps, log=logger),
                            query=query,
                            documents=documents
                        )
                        
                        result['data']['rag_result'] = rag_result
                        deps.state['rag_results'] = rag_result
                    except Exception as e:
                        error_msg = f"Error performing RAG: {str(e)}"
                        logger.error(error_msg)
                        logfire.error(error_msg)
                        result['data']['rag_error'] = str(e)
            
            return result
            
        except Exception as e:
            error_msg = f"Error running workflow: {str(e)}"
            logger.error(error_msg)
            logfire.error(error_msg)
            
            # Try to use previous state if available
            previous_state = deps.archon_client.get_previous_state('initial_state')
            if previous_state:
                deps.state = previous_state
            
            return {
                'status': 'error',
                'error': str(e)
            }

# Create the agent instance
primary_agent = TechevoRagAgent(
    description="Techevo-RAG Agent",
    system_prompt=SYSTEM_PROMPT
)

async def main() -> None:
    """
    Main function to run the agent.
    """
    try:
        # Configure logfire
        logfire.configure(
            app_name="techevo-rag",
            level="INFO",
            capture_stdout=True,
            capture_stderr=True
        )
        
        logfire.info("Starting Techevo-RAG agent")
        
        # Validate environment variables
        validate_env_vars()
        
        # Set up services
        logfire.info("Setting up Google services")
        gmail_service, drive_service = await setup_google_services()
        
        # Set up FAISS index with correct dimension for all-MiniLM-L6-v2 (768)
        logfire.info("Creating FAISS index")
        faiss_index = create_faiss_index(dimension=768)
        
        # Set up Supabase client
        logfire.info("Connecting to Supabase")
        supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )
        
        # Set up Archon client
        logfire.info("Setting up Archon client")
        archon_client = ArchonClient()
        
        # Create dependencies
        deps = EnhancedDeps(
            gmail_service=gmail_service,
            drive_service=drive_service,
            faiss_index=faiss_index,
            supabase=supabase,
            archon_client=archon_client,
            state={}
        )
        
        # Example query
        query = "process campaign emails"
        logfire.info(f"Running test query: {query}")
        
        # Run the workflow
        result = await primary_agent.run_workflow(query, deps)
        
        if result.get('status') == 'success':
            logfire.info(f"Workflow completed successfully with data keys: {', '.join(result.get('data', {}).keys())}")
        else:
            logfire.error(f"Workflow failed: {result.get('error', 'Unknown error')}")
        
        print("Workflow result:", result)
        
    except Exception as e:
        error_msg = f"Error in main: {str(e)}"
        logger.error(error_msg)
        logfire.error(error_msg)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 