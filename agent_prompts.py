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
Extract the following information:
1. Tools to use from: search_emails, download_attachment, perform_rag, search_drive
2. Sender email address (if specified)
3. Keywords for searching (subject or content)
4. Whether attachments are mentioned (true/false)

Return a JSON object with these fields:
{{
  "tools": ["tool1", "tool2"],
  "sender": "email@example.com",  // null if not specified
  "keywords": "relevant search terms",
  "has_attachment": true/false
}}

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