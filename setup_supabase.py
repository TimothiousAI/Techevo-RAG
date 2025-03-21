"""
Initialize Supabase tables for the Techevo RAG system.

This module creates the necessary tables in Supabase for tracking
processed items and RAG results.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logfire
import traceback

# Load environment variables
load_dotenv()

# Configure logging
logger = logfire.configure()

async def init_supabase():
    """Initialize Supabase tables for the RAG system.
    
    Creates the following tables if they don't exist:
    - processed_items: Tracks processed emails and attachments
    - rag_results: Stores RAG operation results
    
    Returns:
        Supabase client
    """
    try:
        # Validate environment variables
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            logger.error("Missing Supabase credentials in environment variables")
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        
        logger.info("Initializing Supabase connection")
        
        # Initialize Supabase client
        try:
            from supabase import create_client
            supabase = create_client(supabase_url, supabase_key)
            logger.info("Supabase client created")
        except Exception as e:
            logger.error(f"Error creating Supabase client: {e}")
            logger.error(traceback.format_exc())
            raise
        
        # Create tables using MCP if available
        try:
            import cursor.mcp.supabase as supabase_mcp
            
            # Enable unsafe mode for database operations
            await supabase_mcp.live_dangerously(service="database", enable=True)
            
            # Create processed_items table
            await supabase_mcp.execute_sql_query(query="""
            BEGIN;
            CREATE TABLE IF NOT EXISTS public.processed_items (
                id SERIAL PRIMARY KEY,
                query TEXT,
                tools_used JSONB,
                timestamp TIMESTAMPTZ DEFAULT now(),
                success BOOLEAN DEFAULT true
            );
            COMMIT;
            """)
            
            # Create rag_results table
            await supabase_mcp.execute_sql_query(query="""
            BEGIN;
            CREATE TABLE IF NOT EXISTS public.rag_results (
                id SERIAL PRIMARY KEY,
                query TEXT,
                context_size INTEGER,
                result TEXT,
                timestamp TIMESTAMPTZ DEFAULT now()
            );
            COMMIT;
            """)
            
            logger.info("Tables created using MCP")
            
        except Exception as mcp_error:
            logger.error(f"Error creating tables via MCP: {mcp_error}")
            logger.error(traceback.format_exc())
            logger.info("Falling back to REST API")
            
            # Fallback to using REST API
            try:
                # Create processed_items table
                result = supabase.table('processed_items').select('id').limit(1).execute()
                logger.info("processed_items table exists")
            except Exception:
                try:
                    # Create the table using SQL
                    result = supabase.rpc(
                        'execute_sql',
                        {
                            'query': """
                            CREATE TABLE IF NOT EXISTS public.processed_items (
                                id SERIAL PRIMARY KEY,
                                query TEXT,
                                tools_used JSONB,
                                timestamp TIMESTAMPTZ DEFAULT now(),
                                success BOOLEAN DEFAULT true
                            );
                            """
                        }
                    ).execute()
                    logger.info("Created processed_items table")
                except Exception as e:
                    logger.error(f"Error creating processed_items table: {e}")
                    logger.error(traceback.format_exc())
            
            try:
                # Create rag_results table
                result = supabase.table('rag_results').select('id').limit(1).execute()
                logger.info("rag_results table exists")
            except Exception:
                try:
                    # Create the table using SQL
                    result = supabase.rpc(
                        'execute_sql',
                        {
                            'query': """
                            CREATE TABLE IF NOT EXISTS public.rag_results (
                                id SERIAL PRIMARY KEY,
                                query TEXT,
                                context_size INTEGER,
                                result TEXT,
                                timestamp TIMESTAMPTZ DEFAULT now()
                            );
                            """
                        }
                    ).execute()
                    logger.info("Created rag_results table")
                except Exception as e:
                    logger.error(f"Error creating rag_results table: {e}")
                    logger.error(traceback.format_exc())
        
        logger.info("Supabase initialization complete")
        return supabase
        
    except Exception as e:
        logger.error(f"Supabase initialization failed: {e}")
        logger.error(traceback.format_exc())
        raise

async def main():
    """Run Supabase initialization when script is executed directly."""
    try:
        await init_supabase()
        print("Supabase tables initialized successfully")
    except Exception as e:
        print(f"Error initializing Supabase: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 