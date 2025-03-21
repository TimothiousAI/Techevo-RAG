"""
Test script for the Techevo-RAG agent.

This script tests the basic functionality of the agent without requiring
actual Gmail or Drive API access.
"""

import asyncio
import os
import unittest
from unittest.mock import MagicMock, patch
import json
import sys

# Ensure the parent directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent components
from agent import (
    primary_agent,
    EnhancedDeps,
    predict_intent,
    validate_env_vars,
    ArchonClient
)

class MockResponse:
    """Mock aiohttp response."""
    
    def __init__(self, data, status=200):
        self.data = data
        self.status = status
    
    async def json(self):
        return self.data
    
    async def text(self):
        return json.dumps(self.data)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class TestAgent(unittest.TestCase):
    """Test case for the agent."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock dependencies
        self.gmail_service = MagicMock()
        self.drive_service = MagicMock()
        self.supabase = MagicMock()
        self.faiss_index = MagicMock()
        self.archon_client = MagicMock()
        
        # Mock ArchonClient methods
        self.archon_client.generate = MagicMock(return_value=asyncio.Future())
        self.archon_client.generate.return_value.set_result({
            'response': '["search_emails", "download_attachment"]'
        })
        
        # Create dependencies
        self.deps = EnhancedDeps(
            gmail_service=self.gmail_service,
            drive_service=self.drive_service,
            faiss_index=self.faiss_index,
            supabase=self.supabase,
            archon_client=self.archon_client,
            state={}
        )
    
    @patch.dict(os.environ, {
        'CREDENTIALS_JSON_PATH': '/fake/path/to/credentials.json',
        'SUPABASE_URL': 'https://fake.supabase.co',
        'SUPABASE_KEY': 'fake_key',
        'OPENAI_API_KEY': 'fake_key'
    })
    def test_validate_env_vars(self):
        """Test environment variable validation."""
        # Should not raise an exception
        validate_env_vars()
    
    @patch.dict(os.environ, {
        'SUPABASE_URL': 'https://fake.supabase.co',
        'SUPABASE_KEY': 'fake_key',
        'OPENAI_API_KEY': 'fake_key'
    })
    def test_validate_env_vars_missing(self):
        """Test validation with missing env vars."""
        # Should raise a ValueError
        with self.assertRaises(ValueError):
            validate_env_vars()
    
    @patch('agent.ArchonClient.generate')
    async def test_predict_intent(self, mock_generate):
        """Test intent prediction."""
        # Mock the Archon response
        mock_generate.return_value = {'response': '["search_emails", "download_attachment"]'}
        
        # Create a client
        client = ArchonClient()
        
        # Test intent prediction
        tools = await predict_intent("process campaign emails", client)
        
        # Check the result
        self.assertEqual(tools, ["search_emails", "download_attachment"])
    
    @patch('agent_tools.search_emails')
    async def test_run_workflow(self, mock_search_emails):
        """Test workflow execution."""
        # Mock the search_emails result
        mock_search_emails.return_value = [
            {
                'id': 'email1',
                'snippet': 'This is a test email',
                'subject': 'Test Email',
                'sender': 'test@example.com',
                'date': '2023-01-01',
                'attachments': []
            }
        ]
        
        # Run the workflow
        result = await primary_agent.run_workflow("process campaign emails", self.deps)
        
        # Check the result
        self.assertEqual(result['status'], 'success')
        self.assertIn('emails', result['data'])

async def run_tests():
    """Run the test case."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAgent)
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == "__main__":
    asyncio.run(run_tests()) 