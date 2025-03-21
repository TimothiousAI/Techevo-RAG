# Techevo-RAG Project Objective

## Goal
Develop a fully autonomous, agentic Retrieval-Augmented Generation (RAG) system as a single Pydantic AI agent, built with Archon V5 and enhanced via Cursor’s MCP servers (Archon, GitHub, Supabase, Browser Interface). The agent must replicate the capabilities of the n8n-agentic-rag-agent (reference: https://github.com/coleam00/ottomator-agents/tree/main/n8n-agentic-rag-agent), automating workflows for Gmail/Google Drive integration, document processing with RAG, progress tracking in Supabase, and user interaction via a Streamlit UI on port 8502.

## Scope
- **Autonomy**: The agent predicts intent and executes multi-step workflows (e.g., search emails, save attachments, process with RAG) without human intervention, using `openai:gpt-4o` via Archon MCP.
- **Gmail/Drive Integration**: Search emails, download attachments to Drive, and list Drive files using Google APIs, with caching and deduplication.
- **RAG Processing**: Chunk documents (500 tokens), embed with `sentence-transformers`, store in FAISS, and generate responses with `gpt-4o`, leveraging its large context window (assumed 1M tokens).
- **Tracking**: Log progress (email IDs, file hashes, status) in Supabase via MCP, with real-time `logfire` logging.
- **UI**: Streamlit interface on port 8502 for query input, live status, results, and logs.
- **MCP Usage**: Archon for intent/RAG, Supabase for tracking, GitHub for version control.
- **Deployment**: Dockerized setup for local testing and cloud deployment.

## Success Criteria
- Processes queries like "process campaign emails" end-to-end (search, save, RAG, track).
- Runs locally with `streamlit run streamlit_app.py --server.port 8502` or `docker-compose up`.
- No crashes from Archon MCP due to token overload.
- Fully functional code committed to GitHub via MCP.