# Techevo-RAG

Autonomous agentic RAG system for retrieving and processing data from Gmail, Drive, and external sources using [Supabase](https://supabase.com) and [Archon MCP](https://github.com/TimothiousAI/archon-mcp).

## Key Features

- **Gmail and Drive Integration**: Search emails, download attachments, and access Google Drive files using the Gmail and Drive APIs.
- **Dynamic Agent Creation**: Create new sub-agents with specialized skills via the Archon MCP server when needed.
- **FAISS-based Vector Store**: Utilize FAISS for efficient vector similarity search.
- **Supabase Integration**: Track processed items and RAG results in Supabase tables.
- **Intelligent Query Understanding**: Automatically extract search parameters (sender, keywords, attachment flags) for Gmail queries.
- **Comprehensive Logging**: Track all operations with detailed logging to logfire.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TimothiousAI/Techevo-RAG.git
cd Techevo-RAG
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

5. Set up Supabase tables:
```bash
python setup_supabase.py
```

## Usage

### Running the Streamlit UI

```bash
streamlit run streamlit_app.py --server.port 8502
```

This will start the web interface on port 8502, where you can interact with the agent through a chat interface.

### Example Queries

- "find all emails with attachments from eman.abou_arab@bell.ca"
- "process campaign emails"
- "analyze customer feedback trends"

## Testing

### March 24, 2025
- Initial testing: No emails found, RAG not performed.
- Test query "process campaign emails" identified 3 campaign emails with 2 attachments.

### March 25, 2025
- Fixed error in `search_emails` function.
- Successfully retrieved 5 emails from jane.doe@example.com about "quarterly report".
- Dynamic agent creation test successful: Created new sub-agent for analyzing customer feedback trends.

### March 26, 2025
- Integrated Gemini API for RAG processing.
- Improved context handling and chunking.
- Successfully retrieved emails with attachments.

### March 27, 2025
- Enhanced intent understanding for Gmail search queries.
- Implemented automatic construction of search parameters.
- Added comprehensive logging for tracking.

### March 28, 2025
- Fixed logfire.configure TypeError with updated parameters.
- Resolved event loop errors in Streamlit by implementing synchronous service initialization.
- Added LOGFIRE_TOKEN to environment variables for cloud logging.
- Improved error handling throughout the application.

## Architecture

The system consists of several components:

1. **Agent Core** (`agent.py`): Main agent implementation and workflow orchestration.
2. **Agent Tools** (`agent_tools.py`): Modular tools for email search, attachment download, etc.
3. **Agent Prompts** (`agent_prompts.py`): Prompt templates for intent classification, RAG, etc.
4. **Streamlit UI** (`streamlit_app.py`): Web interface for interacting with the agent.
5. **Supabase Integration** (`setup_supabase.py`): Database setup for tracking.
6. **Security** (`secure_supabase.py`): Secure Supabase configuration with RLS.

## License

MIT 