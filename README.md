📄 Document Intelligence Assistant

A Streamlit-powered application that converts documents (PDF, DOCX, PPTX, HTML) into a fully interactive AI-powered chatbot using Docling, LangChain, OpenRouter, and Chroma vector stores.

The system extracts text, structure, tables, and images from documents using OCR and advanced parsing, indexes the content into a vector database, and allows you to ask natural-language questions with responses grounded in the document.

🚀 Features

🧠 AI-powered Q&A over documents

🔍 Semantic search with vector embeddings

📄 OCR support for scanned PDFs

🗂️ Document structure viewer (tables, hierarchy, images)

💾 Persistent vector index using Chroma

🎛️ Reset button to clear index & chat history

🧩 Multiple file format support:

PDF

DOCX

PPTX

HTML

🏗️ Tech Stack

Streamlit – UI

Docling – Document & OCR processing

LangChain – LLM orchestration & agents

OpenRouter API – LLM + Embeddings

Chroma – Vector storage

Python 3.10+
