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


from typing import List, Optional

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_chat_agent
from langgraph.checkpoints.memory import MemorySaver
from langchain_ollama import OllamaLLM

SYSTEM_PROMPT = """
You are a helpful assistant who answers questions about uploaded documents.
Use the provided context when available. If you don't have enough information
from the documents, say that you don't know instead of making something up.
"""


def create_documentation_agent(
    tools: Optional[List[BaseTool]] = None,
    model_name: str = "llama3",
):
    """
    Create a simple document assistant agent using a local Ollama model.

    NOTE:
        We are NOT using tool-calling here because local LLMs (OllamaLLM)
        do not support the OpenAI-style .bind_tools() interface that
        LangGraph's create_react_agent expects.
    """

    # Local LLM – no OpenAI / OpenRouter keys needed
    llm = OllamaLLM(
        model=model_name,
        temperature=0.1,
    )

    # Basic chat-style prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", "{input}"),
        ]
    )

    memory = MemorySaver()

    # Simple chat agent compatible with OllamaLLM
    agent = create_chat_agent(
        llm=llm,
        prompt=prompt,
        checkpointer=memory,
    )

    return agent