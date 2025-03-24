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
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv
import logfire

# Import agent components
from agent import (
    primary_agent, 
    EnhancedDeps, 
    setup_google_services, 
    create_faiss_index,
    ArchonClient,
    validate_env_vars
)

# Import Supabase client
from supabase import create_client

# Configure logging
logfire.configure()
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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

# Define an async function that gets called by a sync wrapper
async def initialize_services():
    """Initialize all required services."""
    try:
        # Validate environment variables
        validate_env_vars()
        
        # Set up Google services
        gmail_service, drive_service = await setup_google_services()
        
        # Create FAISS index
        faiss_index = create_faiss_index(dimension=768)  # Use correct dimension for all-MiniLM-L6-v2
        
        # Create embedding model
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Set up Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            logger.error("SUPABASE_URL or SUPABASE_KEY is missing in .env")
            raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in .env")
        
        try:
            supabase = create_client(supabase_url, supabase_key)
            # Validate connection by making a simple query
            result = supabase.table('processed_items').select('id').limit(1).execute()
            logger.info("Supabase connection validated")
        except Exception as e:
            logger.error(f"Supabase connection failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue without Supabase
            supabase = None
            logger.warning("Continuing without Supabase connection")
        
        # Set up Archon client
        try:
            archon_client = ArchonClient(
                api_key=os.getenv('OPENAI_API_KEY'),
                api_base=os.getenv('OPENAI_API_BASE')
            )
            thread_id = await archon_client.create_thread()
            logger.info(f"Archon thread created: {thread_id}")
        except Exception as e:
            logger.error(f"Error setting up Archon client: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue without Archon, will use OpenAI directly
            archon_client = None
            logger.warning("Continuing without Archon client")
        
        # Create dependencies object
        deps = EnhancedDeps(
            gmail_service=gmail_service,
            drive_service=drive_service,
            faiss_index=faiss_index,
            supabase=supabase,
            archon_client=archon_client,
            state={}
        )
        
        # Add embedding model to deps
        deps.state['embedding_model'] = embedding_model
        
        return gmail_service, drive_service, faiss_index, supabase, archon_client, deps
    
    except Exception as e:
        error_msg = f"Error initializing services: {str(e)}"
        logger.error(error_msg)
        logfire.error(error_msg)
        st.error(error_msg)
        return None, None, None, None, None, None

# Sync wrapper for the async initialization
def sync_initialize_services():
    """Synchronous wrapper for initialization."""
    return asyncio.run(initialize_services())

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
            from agent import TechevoRagAgent
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

# Sync wrapper for running the agent
def sync_run_agent(query: str, deps: EnhancedDeps):
    """Synchronous wrapper for running the agent."""
    return asyncio.run(run_agent_async(query, deps))

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

# Streamlit app setup
st.set_page_config(
    page_title="Techevo-RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Manual refresh button rather than auto-refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# App title and description
st.title("📊 Techevo-RAG Chat Interface")
st.markdown("""
This system allows you to interact with AI agents for email searching, attachment downloading, 
Google Drive searching, and RAG processing. Enter a request below to get started.

**Example requests:**
- "Search for emails about marketing campaigns"
- "Download attachments from finance emails"
- "Summarize all information about quarterly reports"
""")

# Sidebar
st.sidebar.title("📝 System Status")
status_container = st.sidebar.container()

# Refresh button for logs and status
if st.sidebar.button("🔄 Refresh Status"):
    st.session_state.last_refresh = time.time()
    add_log("Status refreshed manually")

# Log container in sidebar
log_container = st.sidebar.container()

# Initialize services on first run
if not st.session_state.initialized:
    with st.spinner("Initializing services..."):
        (
            st.session_state.gmail_service,
            st.session_state.drive_service,
            st.session_state.faiss_index,
            st.session_state.supabase,
            st.session_state.archon_client,
            st.session_state.deps
        ) = sync_initialize_services()
        
        if st.session_state.deps is not None:
            st.session_state.initialized = True
            add_log("Services initialized successfully")
            status_container.success("✅ All services initialized")
        else:
            st.error("Failed to initialize services. Check logs for details.")
            add_log("Failed to initialize services")
            status_container.error("❌ Service initialization failed")

# Show services status
if st.session_state.initialized:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Services")
        if st.session_state.gmail_service:
            st.success("✅ Gmail API")
        else:
            st.error("❌ Gmail API")
        if st.session_state.drive_service:
            st.success("✅ Drive API")
        else:
            st.error("❌ Drive API")
    with col2:
        st.markdown("### Database")
        if st.session_state.supabase:
            st.success("✅ Supabase")
        else:
            st.error("❌ Supabase")
        if st.session_state.archon_client:
            st.success("✅ Archon Client")
        else:
            st.error("❌ Archon Client")
    with col3:
        st.markdown("### Components")
        if st.session_state.faiss_index:
            st.success("✅ FAISS Index")
        else:
            st.error("❌ FAISS Index")

    # Chat Interface
    st.markdown("### Chat with Techevo-RAG")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if not st.session_state.processing:
        user_input = st.chat_input("Enter your request (e.g., 'process campaign emails')")
        if user_input:
            # Add user message to chat
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Process the request
            st.session_state.processing = True
            with st.spinner("Processing your request..."):
                add_log(f"Running query: {user_input}")
                status_container.info("⏳ Processing query...")
                
                # Create placeholder for streaming assistant response
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    response_placeholder.markdown("_Processing your request..._")
                
                # Run the agent
                result = sync_run_agent(user_input, st.session_state.deps)
                st.session_state.results = result
                
                # Log completion
                add_log(f"Query completed with status: {result.get('status', 'unknown')}")
                if result.get('status') == 'success':
                    status_container.success(f"✅ Query completed successfully")
                else:
                    status_container.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
            
            # Update processing flag
            st.session_state.processing = False
            
            # Format and display the response
            if result.get('status') == 'success':
                data = result.get('data', {})
                response_text = "Here are the results:\n\n"
                
                # Handle email results
                if 'emails' in data and data['emails']:
                    response_text += f"**Emails Found ({len(data['emails'])}):**\n"
                    for i, email in enumerate(data['emails'][:5]):  # Show max 5 emails
                        response_text += f"- **Subject:** {email.get('subject', 'No Subject')}\n"
                        response_text += f"  **From:** {email.get('from', 'Unknown')}\n"
                        response_text += f"  **Date:** {email.get('date', 'Unknown')}\n"
                        
                        # Show attachments if present
                        if email.get('attachments'):
                            attachment_names = [a.get('filename', 'Unnamed') for a in email.get('attachments', [])]
                            response_text += f"  **Attachments:** {', '.join(attachment_names)}\n"
                        
                        # Only show snippet of email body
                        if email.get('body'):
                            body_preview = email['body'][:200].replace('\n', ' ').strip() + "..."
                            response_text += f"  **Preview:** {body_preview}\n\n"
                        else:
                            response_text += "\n"
                            
                    if len(data['emails']) > 5:
                        response_text += f"_{len(data['emails']) - 5} more emails found..._\n\n"
                
                # Handle attachment results
                if 'attachments' in data and data['attachments']:
                    response_text += f"**Attachments Downloaded ({len(data['attachments'])}):**\n"
                    for attachment in data['attachments']:
                        if attachment.get('status') == 'error':
                            response_text += f"- ❌ **{attachment.get('filename', 'Unknown')}**: {attachment.get('error', 'Unknown error')}\n"
                        else:
                            drive_link = attachment.get('drive_link', 'N/A')
                            response_text += f"- ✅ **{attachment.get('filename', 'Unknown')}** - [View in Drive]({drive_link})\n"
                    response_text += "\n"
                
                # Handle drive files results
                if 'drive_files' in data and data['drive_files']:
                    response_text += f"**Drive Files Found ({len(data['drive_files'])}):**\n"
                    for i, file in enumerate(data['drive_files'][:5]):  # Show max 5 files
                        file_type = file.get('mimeType', 'Unknown').split('/')[-1]
                        web_link = file.get('webViewLink', '#')
                        response_text += f"- [{file.get('name', 'Unnamed')}]({web_link}) - {file_type}\n"
                        
                    if len(data['drive_files']) > 5:
                        response_text += f"_{len(data['drive_files']) - 5} more files found..._\n\n"
                
                # Handle RAG results - this is the most important part
                if 'rag_result' in data:
                    rag_result = data['rag_result']
                    if rag_result.get('status') == 'success':
                        response_text += f"**Analysis Results:**\n\n{rag_result.get('response', 'No response generated')}\n"
                    else:
                        response_text += f"**Analysis Error:** {rag_result.get('error', 'Unknown error processing your request')}\n"
                
                # Include execution time if available
                if 'execution_time' in result:
                    response_text += f"\n_Request processed in {result.get('execution_time')}_"
                
                # Update the placeholder with the final response
                response_placeholder.markdown(response_text)
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            else:
                # Error case
                error_text = f"Error: {result.get('error', 'Unknown error')}"
                response_placeholder.error(error_text)
                st.session_state.chat_history.append({"role": "assistant", "content": error_text})
    else:
        st.info("Processing your request... Please wait.")

else:
    st.warning("Services not initialized. Please check the logs and try reloading the page.")

# Display logs in sidebar
log_container.text_area(
    "System Logs",
    value="\n".join(st.session_state.logs[-50:]),  # Show only last 50 logs
    height=400,
    key=f"logs_{st.session_state.last_refresh}",  # Force refresh when button is clicked
    disabled=True
)

# Footer
st.markdown("---")
st.markdown("""
**Instructions:** Run with `streamlit run streamlit_app.py --server.port 8502`

**GitHub:** [Techevo-RAG](https://github.com/TimothiousAI/Techevo-RAG) | **Version:** 1.0.0
""")

# Cleanup on app exit
def on_exit():
    """Close connections and perform cleanup when app exits."""
    if st.session_state.initialized and hasattr(st.session_state, 'archon_client') and st.session_state.archon_client is not None:
        if hasattr(st.session_state.archon_client, 'session') and st.session_state.archon_client.session is not None:
            asyncio.run(st.session_state.archon_client.session.close())
    logfire.info("Application shutting down")

import atexit
atexit.register(on_exit) 