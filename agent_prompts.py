"""
Agent prompts for the Techevo-RAG system.
These prompts are used by the agent to generate responses and predictions.
"""

# System prompt for the agent
SYSTEM_PROMPT = """
You are an autonomous agent that predicts intent and executes tasks for Gmail, Drive, and RAG processing.
Your purpose is to help users process emails, search Drive, download attachments, and perform RAG operations.
"""

# Intent classification prompt
INTENT_PROMPT = """
Classify the intent of this query: {query}
Return a list of tool names from the following options:
- search_emails: Find emails matching certain criteria
- download_attachment: Download an attachment from an email
- perform_rag: Perform RAG operations on documents
- search_drive: Search for files in Google Drive

Example output format: ["search_emails", "download_attachment"]
Only include tools directly relevant to the query.
"""

# RAG generation prompt
RAG_PROMPT = """
Answer the query using only the context provided:

Query: {query}

Context:
{chunks}

Provide a concise, accurate response based solely on the context.
If the context doesn't contain relevant information, state that you cannot answer.
"""

# Email search prompt (simplified)
EMAIL_SEARCH_PROMPT = """
Identify key search parameters (date range, sender, subject keywords) for Gmail search query: {query}
"""

# Document processing prompt (simplified)
DOCUMENT_PROCESSING_PROMPT = """
Extract key information and important details from:

{content}
"""

# Search criteria prompt (simplified)
SEARCH_CRITERIA_PROMPT = """
Determine appropriate search criteria for Google Drive based on: {query}
Consider file types, keywords, dates, and locations.
""" 