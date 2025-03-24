"""
Enable Row-Level Security (RLS) on Supabase tables for the Techevo-RAG system.

This script sets up appropriate RLS policies to secure the data in Supabase.
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
logger = logging.getLogger('techevo-rag.secure-supabase')

# Load environment variables
load_dotenv()

async def setup_rls():
    """
    Set up Row-Level Security policies on Supabase tables.
    
    This function connects to Supabase using environment variables and
    sets up RLS policies for the tables.
    """
    logger.info("Setting up RLS on Supabase tables")
    
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
    
    # Enable RLS on the tables
    try:
        # Note: This requires direct SQL execution, which may not be available through the REST API
        # Typically, you would need to connect to the database directly or use the Supabase Dashboard
        
        print("""
⚠️ To enable RLS on your tables, please execute the following SQL in the Supabase Dashboard SQL Editor:

-- Enable RLS on processed_items table
ALTER TABLE processed_items ENABLE ROW LEVEL SECURITY;

-- Create policy for processed_items
CREATE POLICY "Allow full access to authenticated users" 
ON processed_items
FOR ALL
TO authenticated
USING (true);

-- Enable RLS on rag_results table
ALTER TABLE rag_results ENABLE ROW LEVEL SECURITY;

-- Create policy for rag_results
CREATE POLICY "Allow full access to authenticated users" 
ON rag_results
FOR ALL
TO authenticated
USING (true);
        """)
        
        logger.info("RLS setup instructions provided")
    except Exception as e:
        logger.error(f"Error setting up RLS: {str(e)}")
        logger.error(f"Please set up RLS manually using the Supabase Dashboard.")
    
    logger.info("RLS setup process complete")

if __name__ == '__main__':
    asyncio.run(setup_rls()) 