"""
Document ingestion script for RAG knowledge base.

Usage:
    python ingest_docs.py

Reads PDFs from knowledge_base/, chunks them, adds metadata,
embeds with OpenAI-compatible embeddings, and stores in FAISS index.
"""

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from backend.config import settings
from backend.embeddings_factory import get_embeddings
from langchain_community.vectorstores import FAISS

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
FAISS_INDEX_DIR = Path(__file__).parent / "faiss_index"

# Document metadata mapping (filename -> metadata)
DOC_METADATA = {
    "LangGraph.pdf": {"source_type": "documentation", "framework": "LangGraph", "document": "LangGraph", "version": "latest"},
    "FastAPI.pdf": {"source_type": "documentation", "framework": "FastAPI", "document": "FastAPI", "version": "latest"},
    "Docker.pdf": {"source_type": "documentation", "framework": "Docker", "document": "Docker", "version": "latest"},
    "AWS ECS.pdf": {"source_type": "documentation", "framework": "AWS", "document": "ECS", "version": "latest"},
    "Bedrock.pdf": {"source_type": "documentation", "framework": "AWS", "document": "Bedrock", "version": "latest"},
    "Python.pdf": {"source_type": "documentation", "framework": "Python", "document": "Python", "version": "3.11"},
}


def ingest_documents():
    """
    Ingest PDFs from knowledge_base/ into FAISS vector store.
    Pipeline: PDF → chunks → metadata → embeddings → FAISS
    """
    pdf_files = list(KNOWLEDGE_BASE_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {KNOWLEDGE_BASE_DIR}")
        print("Add PDFs (LangGraph.pdf, FastAPI.pdf, etc.) and re-run.")
        return

    print(f"Found {len(pdf_files)} PDF(s) in knowledge_base/")

    all_docs = []

    for pdf_path in pdf_files:
        filename = pdf_path.name
        print(f"\nProcessing: {filename}")

        # Load PDF
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        # Add metadata
        meta = DOC_METADATA.get(filename, {
            "source_type": "documentation",
            "framework": "unknown",
            "document": filename.replace(".pdf", ""),
            "version": "unknown",
        })

        for doc in docs:
            doc.metadata.update(meta)
            doc.metadata["source_file"] = filename

        all_docs.extend(docs)
        print(f"  Loaded {len(docs)} pages")

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(all_docs)
    print(f"\nTotal chunks: {len(chunks)}")

    # Create embeddings
    embeddings = get_embeddings()

    # Build FAISS index
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index
    FAISS_INDEX_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))

    print(f"\nFAISS index saved to {FAISS_INDEX_DIR}")
    print(f"Indexed {len(chunks)} chunks from {len(pdf_files)} documents.")
    print("Ready for RAG queries.")


if __name__ == "__main__":
    ingest_documents()
