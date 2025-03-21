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
    
    def __init__(self, api_key=None, api_base=None, session=None):
        """Initialize ArchonClient with API key and optionally a custom session.
        
        Args:
            api_key: OpenAI API key (fallback if MCP fails)
            api_base: OpenAI API base URL (fallback if MCP fails)
            session: Optional HTTPX AsyncClient session
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or "https://api.openai.com/v1"
        self.thread_id = None
        
        if session:
            self.session = session
        else:
            self.session = httpx.AsyncClient(
                base_url=self.api_base,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0
            )
        
        # Create state backup directory
        os.makedirs('backup', exist_ok=True)
    
    async def create_thread(self):
        """Create a new thread for Archon MCP."""
        try:
            import cursor.mcp.archon_mcp as archon_mcp
            result = await archon_mcp.create_thread(random_string="")
            self.thread_id = result.get('thread_id')
            logger.info(f"Created Archon thread: {self.thread_id}")
            return self.thread_id
        except Exception as e:
            logger.error(f"Failed to create Archon thread: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Generate a local thread ID as fallback
            self.thread_id = f"local-{int(time.time())}"
            logger.info(f"Using local thread ID: {self.thread_id}")
            return self.thread_id
    
    async def run_agent(self, thread_id, user_input):
        """Run the Archon agent with user input."""
        try:
            import cursor.mcp.archon_mcp as archon_mcp
            result = await archon_mcp.run_agent(thread_id=thread_id, user_input=user_input)
            return result
        except Exception as e:
            logger.error(f"Failed to run Archon agent: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"MCP error: {str(e)}"}
    
    async def generate(self, model: str, prompt: str, **kwargs):
        """Generate a response using Archon MCP or fallback to local API.
        
        Args:
            model: Model to use (gpt-4o, gpt-3.5-turbo, etc.)
            prompt: Prompt text to send to the model
            **kwargs: Additional parameters to pass to the model

        Returns:
            Dict containing the response
        """
        # Check prompt size and truncate if too large
        if len(prompt) > 5000:
            logger.warning(f"Prompt too large ({len(prompt)} chars), truncating to 5000 chars")
            prompt = prompt[:5000]
        
        # Save state before MCP call to allow recovery if it crashes
        try:
            self._save_state_backup()
        except Exception as e:
            logger.error(f"Failed to save state backup: {e}")
        
        try:
            # Try using MCP first
            if not self.thread_id:
                self.thread_id = await self.create_thread()
            
            logger.info(f"Generating with Archon MCP using thread {self.thread_id}")
            result = await self.run_agent(self.thread_id, prompt)
            
            # Check if response is valid
            if result and isinstance(result, str) and len(result) > 10:
                return {'response': result}
            elif isinstance(result, dict) and 'text' in result and result['text']:
                return {'response': result['text']}
            else:
                logger.warning(f"Invalid or empty MCP response: {result}")
                # Fall back to local API
                raise ValueError("Empty or invalid MCP response")
                
        except Exception as e:
            logger.error(f"MCP failed: {e}, falling back to local API")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            try:
                # Fallback to OpenAI API using AsyncOpenAI
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                
                # Convert model name format if needed (e.g., gpt-4o -> gpt-4o-mini)
                model_name = model.split(':')[-1] if ':' in model else model
                
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                
                return {'response': completion.choices[0].message.content}
            except Exception as fallback_error:
                logger.error(f"Fallback API also failed: {fallback_error}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Try to restore state if both methods failed
                self._restore_state_backup()
                
                return {'error': f"Generation failed: {str(e)}. Fallback error: {str(fallback_error)}"}
    
    def _save_state_backup(self):
        """Save the current state to a backup file to recover from crashes."""
        from agent_tools import load_cache
        
        # Attempt to load current state
        try:
            cache_data = load_cache()
            backup_path = Path('backup/state_backup.json')
            
            with open(backup_path, 'w') as f:
                json.dump(cache_data, f)
            
            logger.info(f"State backup saved to {backup_path}")
        except Exception as e:
            logger.error(f"Failed to save state backup: {e}")
    
    def _restore_state_backup(self):
        """Restore state from backup file after a crash."""
        from agent_tools import save_cache
        
        backup_path = Path('backup/state_backup.json')
        if backup_path.exists():
            try:
                with open(backup_path, 'r') as f:
                    cache_data = json.load(f)
                
                save_cache(cache_data)
                logger.info(f"State restored from {backup_path}")
            except Exception as e:
                logger.error(f"Failed to restore state from backup: {e}")

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
        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

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
    
    async def run_workflow(self, query: str, deps=None):
        """Run the main agent workflow based on the query.
        
        Args:
            query: User query text
            deps: Dependencies object with state management
            
        Returns:
            dict: Result of the workflow execution
        """
        try:
            # Initialize state if not provided
            if not deps or not hasattr(deps, 'state'):
                from collections import defaultdict
                deps = type('obj', (object,), {'state': defaultdict(dict)})
            
            # Create new thread if needed
            if not self.services['archon'].thread_id:
                try:
                    self.services['archon'].thread_id = await self.services['archon'].create_thread()
                except Exception as e:
                    logger.error(f"Failed to create thread: {e}")
            
            # Step 1: Predict intent
            logger.info(f"Processing query: {query}")
            from agent_prompts import INTENT_PROMPT
            
            intent_result = None
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries and not intent_result:
                try:
                    intent_prompt = INTENT_PROMPT.format(query=query)
                    intent_response = await self.services['archon'].generate(
                        model="gpt-4o", 
                        prompt=intent_prompt
                    )
                    
                    if 'error' in intent_response:
                        logger.error(f"Intent prediction error: {intent_response['error']}")
                        retry_count += 1
                        await asyncio.sleep(1)
                        continue
                    
                    intent_result = intent_response.get('response', '')
                    logger.info(f"Intent prediction: {intent_result}")
                except Exception as e:
                    logger.error(f"Intent prediction failed: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    retry_count += 1
                    await asyncio.sleep(1)
            
            if not intent_result:
                return {
                    'error': f"Failed to predict intent after {max_retries} retries",
                    'tool': 'intent_prediction'
                }
            
            # Parse intent result
            try:
                import re
                tools_match = re.search(r'tools\s*:\s*\[(.*?)\]', intent_result, re.IGNORECASE | re.DOTALL)
                if tools_match:
                    tools_str = tools_match.group(1)
                    tools = [t.strip(' "\'') for t in tools_str.split(',')]
                else:
                    tools = []
                
                logger.info(f"Selected tools: {tools}")
            except Exception as e:
                logger.error(f"Failed to parse intent: {e}")
                logger.error(f"Intent result: {intent_result}")
                return {
                    'error': f"Failed to parse intent: {str(e)}",
                    'tool': 'intent_parsing'
                }
            
            # Step 2: Execute tools based on intent
            from agent_tools import (
                search_emails, download_attachment,
                search_drive, perform_rag, track_progress
            )
            
            result = {
                'query': query,
                'intent': intent_result,
                'tools_used': tools,
                'results': {},
            }
            
            # Execute each tool
            for tool in tools:
                try:
                    if tool == 'search_emails':
                        logger.info("Executing tool: search_emails")
                        emails = await search_emails(query, self.services, deps)
                        result['results']['emails'] = emails
                        
                    elif tool == 'download_attachment':
                        logger.info("Executing tool: download_attachment")
                        attachments = await download_attachment(query, self.services, deps)
                        result['results']['attachments'] = attachments
                        
                    elif tool == 'search_drive':
                        logger.info("Executing tool: search_drive")
                        files = await search_drive(query, self.services, deps)
                        result['results']['drive_files'] = files
                        
                    elif tool == 'perform_rag':
                        logger.info("Executing tool: perform_rag")
                        rag_result = await perform_rag(query, self.services, deps)
                        result['results']['rag'] = rag_result
                except Exception as e:
                    logger.error(f"Tool execution failed for {tool}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    result['error'] = f"Tool execution failed: {str(e)}"
                    result['tool'] = tool
                    # Continue with other tools despite error
            
            # Step 3: Track progress
            try:
                progress_record = {
                    'query': query,
                    'tools_used': tools,
                    'timestamp': time.time(),
                    'success': 'error' not in result
                }
                
                await track_progress(progress_record, self.services)
            except Exception as e:
                logger.error(f"Failed to track progress: {e}")
                # Don't fail the whole operation if tracking fails
            
            return result
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'error': f"Workflow execution failed: {str(e)}",
                'query': query
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
        example_docs = ["Sample document for testing"]
        index, embeddings, model = await create_faiss_index(example_docs)
        
        # Create Archon client
        archon_client = ArchonClient()
        thread_id = await archon_client.create_thread()
        
        # Initialize Supabase
        from setup_supabase import init_supabase
        supabase = await init_supabase()
        
        # Create dependencies object
        from collections import defaultdict
        deps = type('obj', (object,), {'state': defaultdict(dict)})
        
        # Create a dictionary of services
        services = {
            'gmail': gmail_service,
            'drive': drive_service,
            'faiss_index': index,
            'embedding_model': model,
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