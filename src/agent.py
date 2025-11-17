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

from typing import List
import os

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool


def create_documentation_agent(
    tools: List[BaseTool],
    model_name: str = "openrouter/auto",  # 🔁 let OpenRouter pick best model
):
    """
    Create a document QA agent using OpenRouter + LangGraph REACT agent.

    - Uses OpenRouter Auto Router by default (GPT-4, Claude, Gemini, etc.).
    - Still respects your org's zero-retention policy.
    """

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        extra_body={
            "provider": {
                "zdr": True
            },

        },
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
    )

    return agent