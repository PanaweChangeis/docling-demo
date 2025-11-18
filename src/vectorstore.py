# """
# Vector store management for document storage and retrieval.
# """
# from typing import List
# import os
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma


# class VectorStoreManager:
#     """Manages document chunking, embedding, and vector storage."""

#     def __init__(self):
#         """Initialize the vector store manager."""
#         #self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#                 # Initialize embeddings using OpenRouter (OpenAI embeddings via router)
#         self.embeddings = OpenAIEmbeddings(
#             model="text-embedding-3-small",
#             openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#             openai_api_base="https://openrouter.ai/api/v1",
#         )

#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=2500,
#             chunk_overlap=200,
#             length_function=len,
#         )

#     def chunk_documents(self, documents: List[Document]) -> List[Document]:
#         """
#         Split documents into smaller chunks for better retrieval.

#         Args:
#             documents: List of documents to chunk

#         Returns:
#             List of chunked documents
#         """
#         print(f"✂️ Chunking {len(documents)} documents...")
#         chunks = self.text_splitter.split_documents(documents)
#         print(f"✅ Created {len(chunks)} chunks")
#         return chunks

#     # def create_vectorstore(self, chunks: List[Document]) -> Chroma:
#     #     """
#     #     Create a Chroma vector store from document chunks.

#     #     Args:
#     #         chunks: List of document chunks

#     #     Returns:
#     #         Chroma vector store instance
#     #     """
#     #     print(f"🔢 Creating vector store with {len(chunks)} chunks...")

#     #     try:
#     #         vectorstore = Chroma.from_documents(
#     #             documents=chunks,
#     #             embedding=self.embeddings,
#     #             collection_name="documents"
#     #         )
#     #         print("✅ Vector store created successfully")
#     #         return vectorstore

#     #     except Exception as e:
#     #         print(f"❌ Error creating vector store: {str(e)}")
#     #         raise

#     def create_vectorstore(self, chunks: List[Document]) -> Chroma:
#         """
#         Create a Chroma vector store from document chunks.

#         Args:
#             chunks: List of document chunks

#         Returns:
#             Chroma vector store instance
#         """
#         print(f"🔢 Creating vector store with {len(chunks)} chunks...")

#         # 1) Filter out empty or tiny chunks (common with OCR)
#         filtered_chunks = []
#         for doc in chunks:
#             text = (doc.page_content or "")
#             # Skip only *truly empty* chunks
#             if not text.strip():
#                 continue
#             filtered_chunks.append(doc)

#         if not filtered_chunks:
#             raise ValueError(
#                 "Document appears to have no extractable text (likely a pure image scan). "
#                 "Try a clearer PDF or run OCR to convert it to searchable text first."
#             )

#         try:
#             # 2) Embed ONLY filtered chunks (this internally skips bad OCR pages)
#             print("🔄 Embedding chunks...")
#             vectorstore = Chroma.from_documents(
#                 documents=filtered_chunks,
#                 embedding=self.embeddings,   # OpenRouter embeddings already attached
#                 collection_name="documents"
#             )

#             print("✅ Vector store created successfully")
#             return vectorstore

#         except Exception as e:
#             print(f"❌ Error creating vector store: {str(e)}")
#             raise


#     def search_similar(self, vectorstore: Chroma, query: str, k: int = 4) -> List[Document]:
#         """
#         Perform semantic similarity search.

#         Args:
#             vectorstore: The Chroma vector store
#             query: Search query
#             k: Number of results to return

#         Returns:
#             List of similar documents
#         """
#         try:
#             results = vectorstore.similarity_search(query, k=k)
#             return results
#         except Exception as e:
#             print(f"❌ Error searching vector store: {str(e)}")
#             return []

"""
Vector store management for document storage and retrieval.
"""
from typing import List
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 

class VectorStoreManager:
    """Manages document chunking, embedding, and vector storage."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="openai/text-embedding-3-large",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            length_function=len,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        """
        print(f"✂️ Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """
        Create an in-memory Chroma vector store from document chunks.
        """
        print(f"🔢 Creating vector store with {len(chunks)} chunks...")

        # Filter out empty OCR chunks
        filtered_chunks = []
        for doc in chunks:
            text = (doc.page_content or "")
            if not text.strip():
                continue
            filtered_chunks.append(doc)

        if not filtered_chunks:
            raise ValueError(
                "Document appears to have no extractable text (likely a pure image scan). "
                "Try a clearer PDF or run OCR to convert it to searchable text first."
            )

        try:
            vectorstore = Chroma.from_documents(
                documents=filtered_chunks,
                embedding=self.embeddings,
                collection_name="session_documents",  # in-memory only
            )
            print("✅ Vector store created successfully")
            return vectorstore
        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            raise

    def search_similar(self, vectorstore: Chroma, query: str, k: int = 4) -> List[Document]:
        """
        Perform semantic similarity search.
        """
        try:
            return vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"❌ Error searching vector store: {str(e)}")
            return []
