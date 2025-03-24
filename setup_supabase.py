"""
Initialize Supabase tables for the Techevo-RAG system.

This module creates the necessary tables in Supabase for tracking
processed items and RAG results.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from supabase import create_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('techevo-rag.setup-supabase')

# Load environment variables
load_dotenv()

async def init_supabase():
    """
    Initialize Supabase connection and check for required tables.
    
    This function connects to Supabase using environment variables and
    checks if the required tables exist. If they don't, it provides
    instructions for manual creation.
    """
    logger.info("Initializing Supabase connection")
    
    # Verify environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        logger.error("Missing required environment variables: SUPABASE_URL and/or SUPABASE_KEY")
        print("\nERROR: Please set SUPABASE_URL and SUPABASE_KEY in your .env file.\n")
        return
    
    # Create Supabase client
    try:
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase client created successfully")
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {str(e)}")
        print(f"\nERROR: Failed to connect to Supabase: {str(e)}\n")
        return
    
    # Check if processed_items table exists
    try:
        result = supabase.table('processed_items').select('id').limit(1).execute()
        logger.info("processed_items table already exists")
        print("✅ processed_items table exists")
    except Exception as e:
        if 'relation "public.processed_items" does not exist' in str(e):
            logger.error("processed_items table does not exist. Please create it manually in the Supabase dashboard.")
            print("\n⚠️ processed_items table does not exist.")
            print("\nPlease create it manually in the Supabase dashboard with the following SQL:")
            print("""
CREATE TABLE processed_items (
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
        else:
            logger.error(f"Error checking processed_items table: {str(e)}")
            print(f"\nERROR checking processed_items table: {str(e)}")
    
    # Check if rag_results table exists
    try:
        result = supabase.table('rag_results').select('id').limit(1).execute()
        logger.info("rag_results table already exists")
        print("✅ rag_results table exists")
    except Exception as e:
        if 'relation "public.rag_results" does not exist' in str(e):
            logger.error("rag_results table does not exist. Please create it manually in the Supabase dashboard.")
            print("\n⚠️ rag_results table does not exist.")
            print("\nPlease create it manually in the Supabase dashboard with the following SQL:")
            print("""
CREATE TABLE rag_results (
    id SERIAL PRIMARY KEY,
    query TEXT,
    result TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);
            """)
        else:
            logger.error(f"Error checking rag_results table: {str(e)}")
            print(f"\nERROR checking rag_results table: {str(e)}")
    
    logger.info("Supabase initialization complete")
    print("\nSupabase initialization complete. Please create any missing tables manually if needed.")

if __name__ == '__main__':
    asyncio.run(init_supabase()) 