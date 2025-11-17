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
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoints.memory import MemorySaver
import os

SYSTEM_PROMPT = """
You are a helpful assistant that strictly answers using the provided document search tool.
If the information is not in the documents, say: 'I don't know based on the provided documents.'
"""

def create_documentation_agent(tools: List, model_name="meta-llama/llama-3.1-8b-instruct"):
    """
    Works with OpenRouter + Zero Retention approved models.
    """

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    memory = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=SYSTEM_PROMPT,
        checkpointer=memory,
    )

    return agent