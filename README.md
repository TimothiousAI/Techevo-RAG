# Techevo-RAG

A fully autonomous agentic RAG system built with Pydantic AI and Archon MCP.

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

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Techevo-RAG.git
cd Techevo-RAG
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up Google API credentials**

- Go to the [Google Cloud Console](https://console.cloud.google.com/)
- Create a new project or select an existing one
- Enable the Gmail API and Google Drive API
- Create OAuth 2.0 credentials (Desktop application type)
- Download the credentials JSON file and save it as `credentials.json` in the project root

5. **Set up environment variables**

Create a `.env` file in the project root by copying from `.env.example`:

```bash
cp .env.example .env
```

Edit the `.env` file to include your Supabase URL and key, and any other required variables.

## Running the application

### Using Streamlit

```bash
streamlit run streamlit_app.py --server.port 8502
```

This will start the Streamlit application on [http://localhost:8502](http://localhost:8502).

### Using Docker

```bash
docker-compose up --build
```

## MCP Integration

This agent is designed to work with Cursor's Model Context Protocol (MCP) servers:

- **Archon MCP**: Used for intent classification and RAG generation
- **Supabase MCP**: Used for tracking progress and storing results

When using with Cursor, the MCP integration happens automatically. Without Cursor, the agent falls back to using OpenAI.

## Example Queries

- "Search for emails about marketing campaigns"
- "Download attachments from finance emails"
- "Find quarterly reports in my Drive"
- "Analyze sales data from last month's emails"

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

## Repository History

- Committed initial version on `2023-10-22` (Initial functional agent with OAuth, RAG, and UI) 