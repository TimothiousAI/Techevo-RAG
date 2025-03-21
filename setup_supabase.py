"""
Supabase setup script for Techevo-RAG.

This script initializes the necessary tables in Supabase for tracking
processed items and RAG results.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
import logfire

# Supabase client
from supabase import create_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure logfire
logfire.configure(
    app_name="techevo-rag-setup",
    level="INFO",
    capture_stdout=True,
    capture_stderr=True
)

# Load environment variables
load_dotenv()

async def init_supabase():
    """
    Initialize the Supabase tables for Techevo-RAG.
    
    Creates the following tables if they don't exist:
    - processed_items: For tracking processed emails and attachments
    - rag_results: For storing RAG query results
    """
    logfire.info("Starting Supabase initialization")
    
    # Validate environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        error_msg = "Missing SUPABASE_URL or SUPABASE_KEY environment variables"
        logger.error(error_msg)
        logfire.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Create Supabase client
        logfire.info("Connecting to Supabase")
        supabase = create_client(supabase_url, supabase_key)
        
        # Try to use direct MCP calls if available
        try:
            import cursor.mcp.supabase as supabase_mcp
            import cursor.mcp.supabase_archon as supabase_archon
            
            # Enable unsafe mode for database operations
            logfire.info("Enabling unsafe mode for database operations")
            await supabase_archon.live_dangerously(service="database", enable=True)
            
            # Create processed_items table
            logfire.info("Creating processed_items table")
            await supabase_mcp.execute_sql_query(query="""
            CREATE TABLE IF NOT EXISTS public.processed_items (
                id SERIAL PRIMARY KEY,
                email_id TEXT,
                file_hash TEXT,
                status TEXT,
                timestamp TIMESTAMP DEFAULT NOW(),
                file_id TEXT,
                filename TEXT,
                error_message TEXT
            );
            """)
            
            # Create rag_results table
            logfire.info("Creating rag_results table")
            await supabase_mcp.execute_sql_query(query="""
            CREATE TABLE IF NOT EXISTS public.rag_results (
                id SERIAL PRIMARY KEY,
                query TEXT,
                result TEXT,
                timestamp TIMESTAMP DEFAULT NOW()
            );
            """)
            
            logfire.info("Supabase tables created successfully via MCP")
            
        except (ImportError, Exception) as mcp_error:
            logfire.warning(f"MCP not available or error: {str(mcp_error)}, using direct SQL")
            
            # Create processed_items table via REST API
            logfire.info("Creating processed_items table via REST API")
            try:
                await supabase.table('processed_items').select('id').limit(1).execute()
                logfire.info("processed_items table already exists")
            except Exception:
                # Table doesn't exist, create it
                create_processed_items_sql = """
                CREATE TABLE IF NOT EXISTS public.processed_items (
                    id SERIAL PRIMARY KEY,
                    email_id TEXT,
                    file_hash TEXT,
                    status TEXT,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    file_id TEXT,
                    filename TEXT,
                    error_message TEXT
                );
                """
                await supabase.rpc('execute_sql', {'sql': create_processed_items_sql}).execute()
                logfire.info("Created processed_items table")
            
            # Create rag_results table via REST API
            logfire.info("Creating rag_results table via REST API")
            try:
                await supabase.table('rag_results').select('id').limit(1).execute()
                logfire.info("rag_results table already exists")
            except Exception:
                # Table doesn't exist, create it
                create_rag_results_sql = """
                CREATE TABLE IF NOT EXISTS public.rag_results (
                    id SERIAL PRIMARY KEY,
                    query TEXT,
                    result TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                );
                """
                await supabase.rpc('execute_sql', {'sql': create_rag_results_sql}).execute()
                logfire.info("Created rag_results table")
        
        logfire.info("Supabase initialization completed successfully")
        
    except Exception as e:
        error_msg = f"Error initializing Supabase: {str(e)}"
        logger.error(error_msg)
        logfire.error(error_msg)
        raise
    
    return True

if __name__ == '__main__':
    # Run the initialization
    try:
        asyncio.run(init_supabase())
        print("Supabase tables initialized successfully!")
    except Exception as e:
        print(f"Error initializing Supabase: {str(e)}")
        exit(1) 