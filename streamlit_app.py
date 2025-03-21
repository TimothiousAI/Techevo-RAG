"""
Streamlit web application for Techevo-RAG.

This module provides a web interface for the Techevo-RAG agent, allowing users
to interact with the system via a Streamlit interface.
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

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

# Configure logging with logfire
logfire.configure(
    app_name="techevo-rag",
    level="INFO",
    capture_stdout=True,
    capture_stderr=True
)

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
        
        # Set up Supabase client
        supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )
        
        # Set up Archon client
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
        
        result = await primary_agent.run_workflow(query, deps)
        return result
    
    except Exception as e:
        error_msg = f"Error running agent: {str(e)}"
        logger.error(error_msg)
        logfire.error(error_msg)
        return {
            'status': 'error',
            'error': str(e)
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
    st.session_state.logs.append(log_entry)
    logfire.info(message)

# Streamlit app setup
st.set_page_config(
    page_title="Techevo-RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📊 Techevo-RAG System")
st.markdown("""
This system processes emails, downloads attachments, searches Google Drive, and performs RAG operations.
Enter a command below to get started.
""")

# Sidebar
st.sidebar.title("📝 System Status")
status_container = st.sidebar.container()
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
            status_container.success("✅ All services connected")
        else:
            st.error("Failed to initialize services. Check logs for details.")
            add_log("Failed to initialize services")
            status_container.error("❌ Service initialization failed")

# Main interface
if st.session_state.initialized:
    # Service status indicators
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
            st.success("✅ Archon MCP")
        else:
            st.error("❌ Archon MCP")
    
    with col3:
        st.markdown("### Components")
        if st.session_state.faiss_index:
            st.success("✅ FAISS Index")
        else:
            st.error("❌ FAISS Index")
    
    # Query input
    st.markdown("### Enter a command")
    query = st.text_input("", value="process campaign emails", placeholder="e.g., process campaign emails")
    
    # Query examples
    st.markdown("**Examples:**")
    examples_col1, examples_col2 = st.columns(2)
    
    with examples_col1:
        if st.button("Search campaign emails"):
            query = "search for campaign emails"
            st.session_state.query = query
    
    with examples_col2:
        if st.button("Analyze quarterly reports"):
            query = "analyze quarterly reports"
            st.session_state.query = query
    
    # Submit button
    if st.button("Run", type="primary", disabled=st.session_state.processing):
        st.session_state.processing = True
        
        with st.spinner("Processing your request..."):
            add_log(f"Running query: {query}")
            status_container.info("⏳ Processing query...")
            
            # Run the agent
            result = sync_run_agent(query, st.session_state.deps)
            
            # Store the result
            st.session_state.results = result
            add_log(f"Query completed with status: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'success':
                status_container.success(f"✅ Query completed successfully")
            else:
                status_container.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
        
        st.session_state.processing = False
    
    # Show results
    if st.session_state.results:
        st.markdown("---")
        st.subheader("Results")
        
        # Check status
        if st.session_state.results.get('status') == 'error':
            st.error(f"Error: {st.session_state.results.get('error', 'Unknown error')}")
        else:
            # Show successful results
            data = st.session_state.results.get('data', {})
            
            # Display emails if available
            if 'emails' in data:
                with st.expander("📧 Emails", expanded=True):
                    st.write(f"Found {len(data['emails'])} email(s)")
                    
                    email_df = []
                    for email in data['emails']:
                        email_df.append({
                            "Subject": email.get('subject', 'No Subject'),
                            "From": email.get('sender', 'Unknown'),
                            "Date": email.get('date', 'Unknown'),
                            "Attachments": len(email.get('attachments', [])),
                            "ID": email.get('id', 'Unknown')
                        })
                    
                    st.table(email_df)
            
            # Display attachments if available
            if 'attachments' in data:
                with st.expander("📎 Attachments", expanded=True):
                    attachment_df = []
                    
                    for attachment in data['attachments']:
                        attachment_df.append({
                            "Filename": attachment.get('filename', 'Unknown'),
                            "Status": attachment.get('status', 'Unknown'),
                            "Drive File ID": attachment.get('drive_file_id', 'N/A'),
                            "Email ID": attachment.get('email_id', 'Unknown')
                        })
                    
                    st.table(attachment_df)
            
            # Display Drive files if available
            if 'drive_files' in data:
                with st.expander("📁 Drive Files", expanded=True):
                    drive_df = []
                    
                    for file in data['drive_files']:
                        drive_df.append({
                            "Name": file.get('name', 'Unknown'),
                            "Type": file.get('mimeType', 'Unknown'),
                            "Modified": file.get('modifiedTime', 'Unknown'),
                            "ID": file.get('id', 'Unknown')
                        })
                    
                    st.table(drive_df)
            
            # Display RAG results if available
            if 'rag_result' in data:
                with st.expander("🔍 RAG Result", expanded=True):
                    rag_result = data['rag_result']
                    
                    st.markdown(f"**Query:** {rag_result.get('query', 'Unknown')}")
                    st.markdown(f"**Response:** {rag_result.get('response', 'No response generated')}")
                    
                    st.subheader("Source Chunks")
                    chunks = rag_result.get('chunks_used', [])
                    
                    for i, chunk in enumerate(chunks):
                        with st.expander(f"Chunk {i+1}"):
                            st.text(chunk)
else:
    st.warning("Services not initialized. Please check the logs and try reloading the page.")

# Display logs in the sidebar
log_container.text_area(
    "System Logs",
    value="\n".join(st.session_state.logs),
    height=400,
    disabled=True
)

# Also show logfire logs if available
try:
    recent_logs = logfire.get_recent_logs(max_entries=50)
    if recent_logs:
        with log_container.expander("Detailed Logs"):
            st.code(recent_logs)
except Exception:
    pass

# Footer
st.markdown("---")
st.markdown("""
**Instructions:** Run with `streamlit run streamlit_app.py --server.port 8502`

**GitHub:** [Techevo-RAG](https://github.com/yourusername/Techevo-RAG) | **Version:** 1.0.0
""")

# Exit handler
def on_exit():
    """Clean up resources when the app exits."""
    if st.session_state.initialized and hasattr(st.session_state, 'archon_client') and st.session_state.archon_client is not None:
        if hasattr(st.session_state.archon_client, 'session') and st.session_state.archon_client.session is not None:
            asyncio.run(st.session_state.archon_client.session.close())
    
    # Log exit
    logfire.info("Application shutting down")

# Register exit handler
import atexit
atexit.register(on_exit) 