"""
LangGraph agent configuration and setup.
"""

# System prompt for the document intelligence assistant
SYSTEM_PROMPT = """You are a helpful document intelligence assistant. You have access to documents that have been uploaded and processed (PDFs, Word documents, presentations, HTML files, etc.).

GUIDELINES:
- Use the search_documents tool to find relevant information
- Be efficient: one well-crafted search is usually sufficient
- Only search again if the first results are clearly incomplete
- Provide clear, accurate answers based on the document contents
- Always cite your sources with filenames or document titles
- If information isn't found, say so clearly
- Be concise but thorough

When answering:
1. Search the documents with a focused query
2. Synthesize a clear answer from the results
3. Include source citations (filenames)
4. Only search again if absolutely necessary
"""


import os
from typing import List

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """
You are a helpful assistant that strictly answers using the provided document search tool.
If the information is not in the documents, reply:
"I don't know based on the provided documents."

- Never invent facts that aren't supported by the documents.
- Prefer concise, clear answers.
"""


def create_documentation_agent(
    tools: List[BaseTool],
    model_name: str = "meta-llama/llama-3.1-8b-instruct",
):
    """
    Create a document QA agent using OpenRouter + LangGraph REACT.

    Requirements:
    - OPENROUTER_API_KEY must be set in the environment.
    - The chosen model must be available under your data policy
      (e.g. llama 3.1 instruct models for Zero Retention).
    """

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=SYSTEM_PROMPT,
    )

    return agent