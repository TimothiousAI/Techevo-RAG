# Techevo-RAG

Autonomous agentic RAG system for processing emails, searching Drive, and performing RAG operations.

## Features

- Email processing with Gmail API
- Google Drive search and document processing
- RAG (Retrieval-Augmented Generation) using FAISS and OpenAI
- Streamlit web interface
- Supabase integration for state tracking

## Setup

1. Clone this repository
2. Copy `.env.example` to `.env` and fill in your credentials
3. Install dependencies with `pip install -r requirements.txt`
4. Run the Streamlit app with `streamlit run streamlit_app.py --server.port 8502`

## MCP Integration

This project uses Cursor MCP for:
- Archon agent calls
- Supabase database interactions
- GitHub repository management

## History

- Pushed initial commits on March 21, 2025 via Cursor Git integration
- GitHub: https://github.com/techevo-user/Techevo-RAG

## Testing

- Tested on March 21, 2025: Emails processed, RAG successful

## Overview

Techevo-RAG automates processing emails, downloading attachments, searching Google Drive, and performing RAG (Retrieval Augmented Generation) operations to provide contextually relevant responses. Built with Pydantic AI, it uses embeddings from the `all-MiniLM-L6-v2` model (768 dimensions) and FAISS for vector storage.

## Key Features

- **Gmail Integration**: Search emails and download attachments
- **Google Drive Integration**: Search and process files
- **RAG Processing**: Generate contextual responses using retrieved documents
- **Supabase Integration**: Track progress and store results
- **MCP Integration**: Utilize Archon MCP for enhanced reasoning capabilities

## Requirements

- Python 3.12 or newer
- Google Cloud project with Gmail and Drive APIs enabled
- OAuth 2.0 credentials (client ID and secret)
- Supabase account and project
- Archon MCP server (via Cursor)

## Development

### Running Tests

```bash
pytest -v
```

### Project Structure

- `agent.py` - Core agent implementation
- `agent_tools.py` - Tool implementations for email, drive, and RAG
- `agent_prompts.py` - Prompts for agent communication
- `streamlit_app.py` - Web UI
- `.env.example` - Example environment variables
- `docker-compose.yml` - Docker configuration
- `Dockerfile` - Docker image definition

## Troubleshooting

### Authentication Issues

If you encounter authentication issues:
1. Delete the `token.json` file if it exists
2. Restart the application and go through the OAuth flow again

### Performance Issues

- For large amounts of data, consider increasing the FAISS index dimensions or changing the embedding model
- Limit search results to improve processing speed

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Example Queries

- "Search for emails about marketing campaigns"
- "Download attachments from finance emails"
- "Find quarterly reports in my Drive"
- "Analyze sales data from last month's emails" 