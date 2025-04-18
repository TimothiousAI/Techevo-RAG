"""
Streamlit web application for Techevo-RAG.

This module provides a web interface for the Techevo-RAG agent, allowing users
to interact with the system via a Streamlit interface with a chat UI.
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import time
import traceback
import uuid

import streamlit as st
from dotenv import load_dotenv
import logfire
import httpx

# Load environment variables first thing
load_dotenv(dotenv_path=".env", verbose=True)

from agent import (
    TechevoRagAgent,
    EnhancedDeps, 
    setup_google_services, 
    create_faiss_index,
    ArchonClient,
    validate_env_vars
)

from supabase import create_client

# Configure logfire with current arguments
logfire.configure(
    token=os.getenv('LOGFIRE_TOKEN'),
    service_name='techevo-rag',
    send_to_logfire=True
)

logger = logging.getLogger(__name__)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.gmail_service = None
    st.session_state.drive_service = None
    st.session_state.faiss_index = None
    st.session_state.supabase = None
    st.session_state.archon_client = None
    st.session_state.deps = None
    st.session_state.logs = []
    st.session_state.results = {}
    st.session_state.processing = False
    st.session_state.chat_history = []
    st.session_state.last_refresh = time.time()

# Create a unique session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Streamlit app setup
st.set_page_config(
    page_title="Techevo RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Debug: Check if Supabase credentials are loaded
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
creds_path = os.getenv('CREDENTIALS_JSON_PATH')
st.sidebar.title("Environment Debug")
st.sidebar.text(f"Current directory: {os.getcwd()}")
st.sidebar.text(f"Supabase URL exists: {bool(supabase_url)}")
st.sidebar.text(f"Supabase KEY exists: {bool(supabase_key)}")
st.sidebar.text(f"Credentials path: {creds_path}")
st.sidebar.text(f"Creds path exists: {os.path.exists(creds_path) if creds_path else False}")

# Show .env file contents (masked)
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
    with st.sidebar.expander("View .env file (masked)"):
        try:
            with open(env_path, 'r') as f:
                env_lines = f.readlines()
                
            masked_lines = []
            for line in env_lines:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    # Mask the value except first and last 2 chars
                    if len(value) > 6:
                        masked_value = value[:2] + '*' * (len(value) - 4) + value[-2:]
                    else:
                        masked_value = '******'
                    masked_lines.append(f"{key}={masked_value}")
                else:
                    masked_lines.append(line)
            
            st.code('\n'.join(masked_lines), language='bash')
        except Exception as e:
            st.error(f"Error reading .env: {str(e)}")
else:
    st.sidebar.error(f".env file not found at {env_path}")

def sync_run(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error in sync_run: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Define a synchronous function for service initialization
def initialize_services():
    """Initialize all required services."""
    try:
        # Validate environment variables first
        validate_env_vars()
        
        # Set up Google services
        try:
            gmail_service, drive_service = sync_run(setup_google_services())
            if not gmail_service or not drive_service:
                raise Exception("Google services failed to initialize. Check your credentials file and permissions.")
            logger.info("Google services initialized successfully")
        except Exception as google_err:
            raise Exception(f"Google services error: {str(google_err)}")
            
        # Create FAISS index
        try:
            faiss_index = create_faiss_index(dimension=768)  # Use correct dimension for all-MiniLM-L6-v2
            logger.info("FAISS index created successfully")
        except Exception as faiss_err:
            raise Exception(f"FAISS index error: {str(faiss_err)}")
        
        # Create embedding model
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as model_err:
            raise Exception(f"Embedding model error: {str(model_err)}")
        
        # Set up Supabase client
        try:
            supabase = initialize_supabase()
            if supabase:
                logger.info("Supabase client initialized successfully")
            else:
                logger.warning("Supabase client not available, continuing without Supabase")
                supabase = None
        except Exception as supabase_err:
            logger.warning(f"Supabase initialization skipped: {str(supabase_err)}")
            supabase = None
        
        # Set up Archon client
        try:
            archon_client = initialize_archon()
            logger.info("Archon MCP client initialized successfully")
        except Exception as archon_err:
            logger.warning(f"Archon client not available: {str(archon_err)}")
            archon_client = None
        
        logger.info("All services initialized successfully")
        return gmail_service, drive_service, faiss_index, embedding_model, supabase, archon_client
        
    except Exception as e:
        logger.error(f"Critical error initializing services: {str(e)}")
        logger.error(traceback.format_exc())
        # Re-raise the exception with more context
        raise Exception(f"Service initialization failed: {str(e)}")

# Async function to run the agent
async def run_agent_async(query: str, deps: EnhancedDeps):
    """Run the agent with the given query."""
    try:
        log_msg = f"Running agent with query: {query}"
        logger.info(log_msg)
        logfire.info(log_msg)
        
        # Create dictionary of services for the agent
        services = {
            'gmail': deps.gmail_service,
            'drive': deps.drive_service,
            'faiss_index': deps.faiss_index,
            'supabase': deps.supabase,
            'archon': deps.archon_client
        }
        
        # Create agent if not already in session state
        if 'agent' not in st.session_state or st.session_state.agent is None:
            st.session_state.agent = TechevoRagAgent(services=services)
        
        # Run the workflow
        result = await st.session_state.agent.run_workflow(query, deps)
        return result
    
    except Exception as e:
        error_msg = f"Error running agent: {str(e)}"
        logger.error(error_msg)
        logfire.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            'status': 'error',
            'error': str(e),
            'query': query
        }

# Function to run the agent synchronously
def sync_run_agent(query, deps):
    """
    Run the agent workflow synchronously.
    
    Args:
        query: The user's query
        deps: The dependencies object
    
    Returns:
        The result of the agent workflow
    """
    try:
        logfire.info(f"Processing query: {query}")
        if not st.session_state.agent:
            st.error("Agent not initialized")
            logfire.error("Agent not initialized")
            return {"status": "error", "error": "Agent not initialized"}
        
        # Run the agent workflow asynchronously using sync_run
        result = sync_run(st.session_state.agent.run_workflow(query, deps))
        logfire.info(f"Query processed successfully in {result.get('runtime', 0):.2f} seconds")
        return result
    except Exception as e:
        error_msg = f"Error running agent: {str(e)}"
        logfire.error(error_msg)
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"status": "error", "error": error_msg}

# Add a log message
def add_log(message: str):
    """Add a log message to the session state."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    # Ensure logs list exists and has a reasonable size (keep last 100 logs)
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    
    st.session_state.logs.append(log_entry)
    if len(st.session_state.logs) > 100:
        st.session_state.logs = st.session_state.logs[-100:]
        
    # Also log to logfire for cloud logging
    logfire.info(message)

st.title("📊 Techevo RAG System")
st.markdown("""
This system allows you to interact with AI agents for email searching, attachment downloading, Google Drive searching, and RAG processing.
Enter a request below to get started (e.g., "search emails or files for anything", "process campaign emails", "summarize quarterly reports").
If a request cannot be handled by existing agents, a new sub-agent will be created dynamically via the Archon MCP server.
""")

st.sidebar.title("📝 System Status")
status_container = st.sidebar.container()
log_container = st.sidebar.container()

# Initialize the agent and dependencies
if "initialized" not in st.session_state or not st.session_state.initialized:
    with st.spinner("Initializing services..."):
        add_log("Initializing services...")
        
        try:
            # Initialize all services
            (
                st.session_state.gmail_service,
                st.session_state.drive_service,
                st.session_state.faiss_index,
                st.session_state.embedding_model,
                st.session_state.supabase,
                st.session_state.archon_client
            ) = initialize_services()
            
            # Check if required services are available
            if not st.session_state.gmail_service:
                raise Exception("Gmail service failed to initialize. Check CREDENTIALS_JSON_PATH in .env file.")
                
            if not st.session_state.drive_service:
                raise Exception("Drive service failed to initialize. Check CREDENTIALS_JSON_PATH in .env file.")
            
            # Create dependencies object
            st.session_state.deps = EnhancedDeps(
                gmail_service=st.session_state.gmail_service,
                drive_service=st.session_state.drive_service,
                supabase=st.session_state.supabase,
                archon_client=st.session_state.archon_client,
                state=st.session_state.get('state', {})
            )
            
            # Store embedding model in state
            if st.session_state.embedding_model:
                st.session_state.deps.state['embedding_model'] = st.session_state.embedding_model
            
            # Initialize agent
            st.session_state.agent = TechevoRagAgent()
            st.session_state.initialized = True
            add_log("Services initialized successfully")
            st.success("✅ All services initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize required services: {str(e)}"
            logfire.error(error_msg)
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(error_msg)
            
            # Display more helpful information about common issues
            st.warning("""
            **Please check the following:**
            1. Ensure your .env file exists and contains all required variables
            2. Check that CREDENTIALS_JSON_PATH points to a valid Google credentials file
            3. Verify that the Archon MCP server is running at the URL specified in ARCHON_MCP_URL
            4. Check Supabase credentials if you're using Supabase
            """)
            
            # Show environment variables (with masked values)
            with st.expander("Environment Variables"):
                envs = {
                    "CREDENTIALS_JSON_PATH": os.getenv("CREDENTIALS_JSON_PATH", "Not set"),
                    "SUPABASE_URL": os.getenv("SUPABASE_URL", "Not set")[:10] + "..." if os.getenv("SUPABASE_URL") else "Not set",
                    "SUPABASE_KEY": "***" if os.getenv("SUPABASE_KEY") else "Not set",
                    "LOGFIRE_TOKEN": "***" if os.getenv("LOGFIRE_TOKEN") else "Not set",
                    "ARCHON_MCP_URL": os.getenv("ARCHON_MCP_URL", "Not set"),
                    "DEFAULT_DRIVE_FOLDER_ID": os.getenv("DEFAULT_DRIVE_FOLDER_ID", "Not set")
                }
                for key, value in envs.items():
                    st.text(f"{key}: {value}")
            
            # Add to logs
            add_log(f"Service initialization failed: {str(e)}")

if st.session_state.initialized:
    st.title("Techevo RAG Agent")
    
    # Define process function for handling user queries
    def process_query():
        if st.session_state.user_query:
            with st.spinner("Processing your query..."):
                # Run the agent
                result = sync_run_agent(st.session_state.user_query, st.session_state.deps)
                
                # Store state for persistence
                st.session_state.state = st.session_state.deps.state
                st.session_state.result = result
    
    # User input with on_change callback for Enter key functionality
    query = st.text_input(
        "Enter your query",
        placeholder="E.g., search emails or files for anything, find documents from 2025",
        key="user_query",
        on_change=process_query
    )
    
    col1, col2 = st.columns([1, 4])
    process_button = col1.button("Process Query", on_click=process_query)
    reset_button = col2.button("Reset")
    
    if reset_button:
        st.session_state.user_query = ""
        if 'result' in st.session_state:
            del st.session_state.result
        st.experimental_rerun()
    
    # Display results if available
    if 'result' in st.session_state and st.session_state.result:
        result = st.session_state.result
        
        # Display result
        if result.get("status") == "success":
            st.success("Query processed successfully")
            
            # Create tabs for different result types
            tabs = st.tabs(["RAG Results", "Emails", "Attachments", "Raw Result"])
            
            # RAG Results tab
            with tabs[0]:
                if "data" in result and "rag_result" in result["data"]:
                    rag_result = result["data"]["rag_result"]
                    st.subheader("RAG Response")
                    st.write(rag_result.get("response", "No RAG response generated"))
                elif "data" in result and "rag_results" in result["data"]:
                    rag_results = result["data"]["rag_results"]
                    st.subheader("RAG Results")
                    for i, rag_result in enumerate(rag_results):
                        with st.expander(f"Result {i+1}: {rag_result.get('filename', 'Document')}", expanded=i==0):
                            st.write(rag_result.get("summary", "No summary available"))
                else:
                    st.info("No RAG results available")
            
            # Emails tab
            with tabs[1]:
                if "data" in result and "emails" in result["data"]:
                    emails = result["data"]["emails"]
                    if emails:
                        st.subheader(f"Found {len(emails)} emails")
                        for i, email in enumerate(emails):
                            with st.expander(f"{email.get('subject', 'No Subject')} - {email.get('from', 'Unknown')}"):
                                st.write(f"**From:** {email.get('from', 'Unknown')}")
                                st.write(f"**Subject:** {email.get('subject', 'No Subject')}")
                                st.write(f"**Date:** {email.get('date', 'Unknown')}")
                                
                                # Show attachments
                                if email.get("attachments"):
                                    st.write(f"**Attachments:** {len(email['attachments'])}")
                                    for att in email["attachments"]:
                                        st.write(f"- {att.get('filename', 'Unknown')}")
                                
                                # Show body
                                st.write("**Body:**")
                                st.write(email.get("body", "No body"))
                    else:
                        st.info("No emails found")
                else:
                    st.info("No email results available")
            
            # Attachments tab
            with tabs[2]:
                if "data" in result and "attachments" in result["data"]:
                    attachments = result["data"]["attachments"]
                    if attachments:
                        st.subheader(f"Processed {len(attachments)} attachments")
                        
                        # Count by status
                        success = sum(1 for a in attachments if a.get("status") == "success")
                        skipped = sum(1 for a in attachments if a.get("status") == "skipped")
                        error = sum(1 for a in attachments if a.get("status") == "error")
                        
                        st.write(f"Success: {success}, Skipped: {skipped}, Error: {error}")
                        
                        # Show successful attachments
                        if success > 0:
                            st.subheader("Successfully Processed Attachments")
                            for att in [a for a in attachments if a.get("status") == "success"]:
                                with st.expander(f"{att.get('filename', 'Unknown')}"):
                                    st.write(f"**Filename:** {att.get('filename', 'Unknown')}")
                                    st.write(f"**MIME Type:** {att.get('mime_type', 'Unknown')}")
                                    if 'web_link' in att:
                                        st.write(f"**Drive Link:** [Open in Drive]({att['web_link']})")
                    else:
                        st.info("No attachments processed")
                else:
                    st.info("No attachment results available")
            
            # Raw Result tab
            with tabs[3]:
                st.subheader("Raw Result")
                st.json(result)
        else:
            st.error(f"Error: {result.get('error', 'Unknown error')}")
    
    # Show sample queries
    with st.expander("Sample Queries"):
        st.write("Click on any sample query to use it:")
        sample_queries = [
            "Find all emails with attachments from a specific sender",
            "Download attachments from emails sent by john.doe@example.com",
            "Summarize all attachments about project updates",
            "Analyze content from spreadsheets related to finance"
        ]
        
        for sample in sample_queries:
            if st.button(sample):
                st.session_state.user_query = sample
                st.experimental_rerun()

# Display logs in sidebar
log_container.text_area(
    "System Logs",
    value="\n".join(st.session_state.logs[-50:]),  # Show only last 50 logs
    height=400,
    disabled=True
)

try:
    recent_logs = logfire.get_recent_logs(max_entries=50)
    if recent_logs:
        with log_container.expander("Detailed Logs"):
            st.code(recent_logs)
except Exception:
    pass

st.markdown("---")
st.markdown("""
**Instructions:** Run with `streamlit run streamlit_app.py --server.port 8502`

**GitHub:** [Techevo-RAG](https://github.com/TimothiousAI/Techevo-RAG) | **Version:** 1.0.0
""")

def on_exit():
    """Close connections and perform cleanup when app exits."""
    if st.session_state.initialized and hasattr(st.session_state, 'archon_client') and st.session_state.archon_client is not None:
        if hasattr(st.session_state.archon_client, 'session') and st.session_state.archon_client.session is not None:
            asyncio.run(st.session_state.archon_client.session.close())
    logfire.info("Application shutting down")

import atexit
atexit.register(on_exit)

# Define functions to initialize services
def initialize_supabase():
    """Initialize Supabase client."""
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        logger.error("SUPABASE_URL or SUPABASE_KEY is missing in .env")
        raise Exception("Supabase credentials missing. Check SUPABASE_URL and SUPABASE_KEY in your .env file.")
    
    try:
        # Create synchronous Supabase client
        logger.info(f"Creating Supabase client with URL starting with: {supabase_url[:10]}...")
        supabase = create_client(supabase_url, supabase_key)
        
        # Validate connection by making a simple query
        try:
            result = supabase.table('processed_items').select('id').limit(1).execute()
            logger.info("Supabase connection validated successfully")
            return supabase
        except Exception as query_err:
            raise Exception(f"Supabase query failed: {str(query_err)}")
    except Exception as e:
        logger.error(f"Supabase connection failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to connect to Supabase: {str(e)}")

def initialize_archon():
    """Initialize Archon MCP client using httpx."""
    try:
        # Try with host.docker.internal first (standard Docker hostname)
        archon_url = os.getenv('ARCHON_MCP_URL', 'http://host.docker.internal:8100')
        logger.info(f'Initializing Archon MCP client with URL: {archon_url}')
        
        # Try creating client with the primary URL
        client = httpx.Client(base_url=archon_url, timeout=30.0)
        
        # Test connection with a health check
        try:
            response = client.get('/health')
            response.raise_for_status()
            logger.info(f"Archon MCP health check successful: {response.status_code}")
            return client
        except Exception as primary_err:
            # If the primary URL fails, try localhost:8501 as fallback
            logger.warning(f"Primary Archon URL {archon_url} failed: {str(primary_err)}")
            fallback_url = 'http://localhost:8501'
            logger.info(f"Trying fallback URL: {fallback_url}")
            
            try:
                client = httpx.Client(base_url=fallback_url, timeout=30.0)
                response = client.get('/health')
                response.raise_for_status()
                logger.info(f"Fallback Archon connection successful: {response.status_code}")
                return client
            except Exception as fallback_err:
                # If fallback also fails, try one more fallback with basic connection test
                logger.warning(f"Fallback URL also failed: {str(fallback_err)}")
                logger.info("Attempting basic connection test without health endpoint")
                
                # Try a basic connection to the root path
                try:
                    client = httpx.Client(base_url=fallback_url, timeout=30.0)
                    response = client.get('/')
                    logger.info(f"Basic connection test succeeded with status: {response.status_code}")
                    return client
                except Exception as basic_err:
                    logger.error(f"All connection attempts failed. Last error: {str(basic_err)}")
                    raise Exception(f"Could not connect to Archon MCP: Primary URL error: {str(primary_err)}, Fallback URL error: {str(fallback_err)}")
            
    except Exception as e:
        error_msg = f"Failed to initialize Archon MCP client: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise Exception(error_msg)

# Make sure TechevoRagAgent has a synchronous run_workflow method
def patch_agent_if_needed():
    """Add a synchronous wrapper method to TechevoRagAgent if it doesn't already have one."""
    if not hasattr(TechevoRagAgent, 'run_workflow_sync'):
        # Add a synchronous wrapper method
        def run_workflow_sync(self, query, deps):
            """Synchronous wrapper for run_workflow."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.run_workflow(query, deps))
                loop.close()
                return result
            except Exception as e:
                logger.error(f"Error in synchronous wrapper: {str(e)}")
                logger.error(traceback.format_exc())
                return {"status": "error", "error": str(e)}
        
        # Add the method to the class
        TechevoRagAgent.run_workflow_sync = run_workflow_sync
        logger.info("Added synchronous wrapper method to TechevoRagAgent")

# Apply the patch when the module is loaded
patch_agent_if_needed()