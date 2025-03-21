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
import time
import sys
import traceback
from collections import defaultdict
from io import StringIO

import streamlit as st
from dotenv import load_dotenv
import logfire
from st_autorefresh import st_autorefresh

# Import agent components
from agent import (
    primary_agent, 
    EnhancedDeps, 
    setup_google_services, 
    create_faiss_index,
    ArchonClient,
    validate_env_vars,
    authenticate_google,
    TechevoRagAgent
)

# Import Supabase client
from supabase import create_client

# Import Supabase setup
from setup_supabase import init_supabase

# Import agent tools
from agent_tools import load_cache, save_cache

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
    st.session_state.error_details = {}
    st.session_state.services = {}
    st.session_state.log_output = StringIO()
    sys.stdout = st.session_state.log_output
    sys.stderr = st.session_state.log_output

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
        
        # Initialize Supabase tables
        try:
            await init_supabase()
            add_log("Supabase tables initialized successfully")
        except Exception as e:
            error_msg = f"Warning: Supabase table initialization failed: {str(e)}"
            logger.warning(error_msg)
            logfire.warning(error_msg)
            add_log(error_msg)
        
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
        
        # Initialize services
        st.session_state.services = {
            'gmail': gmail_service,
            'drive': drive_service,
            'faiss_index': faiss_index,
            'supabase': supabase,
            'archon_client': archon_client,
            'deps': deps
        }
        
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
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(initialize_services())
    finally:
        loop.close()

# Async function to run the agent
async def run_agent_async(query: str, deps: EnhancedDeps):
    """Run the agent with the given query."""
    try:
        log_msg = f"Running agent with query: {query}"
        logger.info(log_msg)
        logfire.info(log_msg)
        
        # Reset error details
        st.session_state.error_details = {}
        
        result = await primary_agent.run_workflow(query, deps)
        
        # Check for tool-specific errors
        if result.get('status') == 'success':
            data = result.get('data', {})
            
            # Check for specific error fields in data
            for error_key in ['email_error', 'drive_error', 'rag_error']:
                if error_key in data:
                    if 'tool_errors' not in st.session_state.error_details:
                        st.session_state.error_details['tool_errors'] = {}
                    
                    tool_name = error_key.split('_')[0]
                    st.session_state.error_details['tool_errors'][tool_name] = data[error_key]
                    add_log(f"Error in {tool_name} tool: {data[error_key]}")
            
            # Check for errors in attachment results
            if 'attachments' in data:
                attachment_errors = []
                for attachment in data['attachments']:
                    if attachment.get('status') == 'error':
                        attachment_errors.append({
                            'filename': attachment.get('filename', 'Unknown file'),
                            'error': attachment.get('error', 'Unknown error')
                        })
                
                if attachment_errors:
                    if 'tool_errors' not in st.session_state.error_details:
                        st.session_state.error_details['tool_errors'] = {}
                    
                    st.session_state.error_details['tool_errors']['attachments'] = attachment_errors
                    add_log(f"Errors in {len(attachment_errors)} attachment(s)")
        
        return result
    
    except Exception as e:
        error_msg = f"Error running agent: {str(e)}"
        logger.error(error_msg)
        logfire.error(error_msg)
        
        # Store the error details
        st.session_state.error_details['main_error'] = str(e)
        
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

# Set up auto-refresh for logs (every 5 seconds)
refresh_interval = 5  # seconds
st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")

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
    
    # Display tool errors if any
    if 'error_details' in st.session_state and st.session_state.error_details:
        with st.expander("⚠️ Error Details", expanded=True):
            if 'main_error' in st.session_state.error_details:
                st.error(f"Main Error: {st.session_state.error_details['main_error']}")
            
            if 'tool_errors' in st.session_state.error_details:
                for tool, error in st.session_state.error_details['tool_errors'].items():
                    if tool == 'attachments':
                        st.warning(f"Attachment Errors:")
                        for att_error in error:
                            st.warning(f"- {att_error['filename']}: {att_error['error']}")
                    else:
                        st.warning(f"{tool.capitalize()} Error: {error}")
    
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
            
            # Clear previous error details
            st.session_state.error_details = {}
            
            # Run the agent
            result = sync_run_agent(query, st.session_state.deps)
            
            # Store the result
            st.session_state.results = result
            add_log(f"Query completed with status: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'success':
                status_container.success(f"✅ Query completed successfully")
            else:
                status_container.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
                
                # Store error details for display
                if 'error' in result:
                    st.session_state.error_details['main_error'] = result['error']
        
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
                    
                    # Show email content if available
                    if st.checkbox("Show Email Content"):
                        selected_email_id = st.selectbox(
                            "Select an email to view content",
                            options=[email['id'] for email in data['emails']],
                            format_func=lambda x: next((email['subject'] for email in data['emails'] if email['id'] == x), x)
                        )
                        
                        if selected_email_id:
                            selected_email = next((email for email in data['emails'] if email['id'] == selected_email_id), None)
                            if selected_email:
                                st.markdown(f"**Subject:** {selected_email.get('subject', 'No Subject')}")
                                st.markdown(f"**From:** {selected_email.get('sender', 'Unknown')}")
                                st.markdown(f"**Date:** {selected_email.get('date', 'Unknown')}")
                                
                                # Show the body if available
                                if 'body' in selected_email and selected_email['body']:
                                    st.text_area("Body", value=selected_email['body'], height=300)
                                else:
                                    st.text_area("Snippet", value=selected_email.get('snippet', 'No content available'), height=150)
            
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
                    
                    # Show file content if available
                    if st.checkbox("Show File Content"):
                        selected_file_id = st.selectbox(
                            "Select a file to view content",
                            options=[file['id'] for file in data['drive_files'] if file.get('snippet')],
                            format_func=lambda x: next((file['name'] for file in data['drive_files'] if file['id'] == x), x)
                        )
                        
                        if selected_file_id:
                            selected_file = next((file for file in data['drive_files'] if file['id'] == selected_file_id), None)
                            if selected_file and 'snippet' in selected_file:
                                st.text_area("File content", value=selected_file['snippet'], height=300)
            
            # Display RAG results if available
            if 'rag_result' in data:
                with st.expander("🔍 RAG Result", expanded=True):
                    rag_result = data['rag_result']
                    
                    st.markdown(f"**Query:** {rag_result.get('query', 'Unknown')}")
                    st.markdown(f"**Response:** {rag_result.get('response', 'No response generated')}")
                    
                    # Show source chunks
                    chunks = rag_result.get('chunks_used', [])
                    if chunks:
                        st.subheader("Source Chunks")
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

**GitHub:** [Techevo-RAG](https://github.com/TimothiousAI/Techevo-RAG) | **Version:** 1.0.0
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

def main():
    """Main Streamlit app function."""
    st.title("Techevo RAG Agent")
    
    # Auto-refresh for logs
    st_autorefresh(interval=5000, key='logrefresh')
    
    # Initialize services
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if not st.session_state.initialized:
        if st.button("Initialize Services"):
            with st.spinner("Initializing services..."):
                st.session_state.services = sync_initialize_services()
                st.session_state.initialized = True
    else:
        # Service status display
        st.sidebar.header("Service Status")
        
        services_status = {
            "Google API": '✅' if 'gmail' in st.session_state.services else '❌',
            "FAISS Index": '✅' if 'faiss_index' in st.session_state.services else '❌',
            "Supabase": '✅' if 'supabase' in st.session_state.services else '❌',
            "Archon": '✅' if 'archon_client' in st.session_state.services else '❌'
        }
        
        for service, status in services_status.items():
            st.sidebar.write(f"{service}: {status}")
        
        # Show errors if any
        if 'errors' in st.session_state.services:
            st.sidebar.header("Initialization Errors")
            for service, error in st.session_state.services['errors'].items():
                st.sidebar.error(f"{service}: {error}")
        
        # Input section
        st.write("Enter your query below:")
        query = st.text_input("Query", key="query_input")
        
        # Use a form to handle submissions
        with st.form(key="query_form"):
            submit_button = st.form_submit_button(label="Run Agent")
            
            if submit_button and query:
                with st.spinner("Running the agent..."):
                    result = sync_run_agent(query, st.session_state.deps)
                    
                    # Display results
                    st.header("Results")
                    
                    # Display any errors
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                        if 'tool' in result:
                            st.write(f"Failed tool: {result['tool']}")
                    
                    # Show tools used
                    if 'tools_used' in result:
                        st.write(f"Tools used: {', '.join(result['tools_used'])}")
                    
                    # Display RAG results
                    if 'results' in result and 'rag' in result['results']:
                        rag = result['results']['rag']
                        st.subheader("RAG Response")
                        st.write(rag.get('answer', 'No answer generated'))
                        st.write(f"Context size: {rag.get('context_size', 0)} characters")
                        st.write(f"Context documents used: {rag.get('context_used', 0)}")
                    
                    # Display emails
                    if 'results' in result and 'emails' in result['results']:
                        emails = result['results']['emails']
                        st.subheader(f"Emails Found: {len(emails)}")
                        
                        for i, email in enumerate(emails):
                            with st.expander(f"{email.get('subject', 'No Subject')} - {email.get('from', 'Unknown')}"):
                                st.write(f"From: {email.get('from', 'Unknown')}")
                                st.write(f"Date: {email.get('date', 'Unknown')}")
                                
                                # Show full body if available, otherwise snippet
                                if 'body' in email and email['body']:
                                    st.text_area(f"Body", email['body'], height=200)
                                else:
                                    st.write(f"Snippet: {email.get('snippet', 'No content')}")
                                
                                # Show attachments
                                if 'attachments' in email and email['attachments']:
                                    st.write(f"Attachments: {len(email['attachments'])}")
                                    for attachment in email['attachments']:
                                        st.write(f"- {attachment.get('filename', 'Unnamed')} ({attachment.get('mimeType', 'Unknown type')})")
                    
                    # Display attachments
                    if 'results' in result and 'attachments' in result['results']:
                        attachments = result['results']['attachments']
                        st.subheader(f"Attachments Downloaded: {len(attachments)}")
                        
                        for i, attachment in enumerate(attachments):
                            if 'error' in attachment:
                                st.error(f"Error with {attachment.get('filename', 'unknown')}: {attachment['error']}")
                            else:
                                st.write(f"✅ {attachment.get('filename', 'File')} - [Drive Link]({attachment.get('drive_link', '#')})")
                    
                    # Display Drive files
                    if 'results' in result and 'drive_files' in result['results']:
                        files = result['results']['drive_files']
                        st.subheader(f"Drive Files Found: {len(files)}")
                        
                        for i, file in enumerate(files):
                            with st.expander(f"{file.get('name', 'Unnamed')} ({file.get('mimeType', 'Unknown type')})"):
                                st.write(f"Type: {file.get('mimeType', 'Unknown')}")
                                st.write(f"Modified: {file.get('modifiedTime', 'Unknown')}")
                                
                                if 'webViewLink' in file:
                                    st.write(f"[Open in Drive]({file['webViewLink']})")
                                
                                if 'content' in file:
                                    st.text_area(f"Content Preview", file['content'], height=200)
                                elif 'content_error' in file:
                                    st.error(f"Error fetching content: {file['content_error']}")
    
    # Display logs
    st.sidebar.header("Logs")
    logs = st.session_state.log_output.getvalue()
    st.sidebar.text_area("Latest logs", logs[-5000:] if len(logs) > 5000 else logs, height=400)

if __name__ == "__main__":
    main() 