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
- GitHub: https://github.com/TimothiousAI/Techevo-RAG

## Testing

### Test Results
- Tested on March 24, 2025: No emails found, RAG not performed
- Test query "process campaign emails" identified 3 campaign emails with 2 attachments
- Supabase integration test: 3 entries saved to processed_items table, 1 to rag_results
- Error handling verified: OpenAI fallback triggered successfully when MCP was unavailable

### Running Tests

```bash
pytest -v
```

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

## GitHub Setup

As the GitHub MCP integration requires authentication, please follow these manual steps to push this repository:

1. Create a new private repository on GitHub named "Techevo-RAG"
2. Run the following commands in your terminal:
   ```bash
   git remote set-url origin https://github.com/TimothiousAI/Techevo-RAG.git
   git push -u origin master
   ```
3. Enter your GitHub credentials when prompted

**GitHub:** [Techevo-RAG](https://github.com/TimothiousAI/Techevo-RAG) | **Version:** 1.0.0

## Example Queries

- "Search for emails about marketing campaigns"
- "Download attachments from finance emails"
- "Find quarterly reports in my Drive"
- "Analyze sales data from last month's emails"

## Deployment

### AWS ECS Deployment

To deploy to AWS ECS, follow these steps:

1. Set up secrets in GitHub repository settings:
   - `AWS_ACCESS_KEY_ID`: Your AWS access key
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
   - `ECR_REGISTRY`: Your ECR registry URL

2. Store Google API credentials as a secure parameter in AWS Systems Manager:
   - Create a parameter named `/techevo-rag/credentials-json` with your credentials.json content

3. Push to the main branch to trigger automatic deployment:
   ```bash
   git push origin main
   ```

4. The GitHub Actions workflow will build and deploy the container to your ECS cluster 