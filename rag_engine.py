"""
rag_engine.py
--------------
Handles the "Retrieval-Augmented Generation" part of the project.

Steps:
1. Load travel guide PDFs from the data/ folder (or an uploaded file)
2. Split them into chunks
3. Convert chunks into embeddings using a local HuggingFace model
4. Store/persist them in a ChromaDB collection
5. Given a query, do a semantic search and return the top matching chunks
   along with which source PDF (and page) they came from -> this is what
   lets the final answer show "source references" (Agentic RAG requirement)
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

import config


class TravelRAGEngine:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_DIR)

        # local, free embedding model (no API key needed for this part)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBED_MODEL_NAME
        )

        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=self.embed_fn,
        )

    # ---------------------------------------------------------------
    # Loading + chunking
    # ---------------------------------------------------------------
    def _read_pdf(self, path):
        """Return list of (page_number, text) tuples for a PDF file."""
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i + 1, text))
        return pages

    def _chunk_text(self, text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
        """Simple sliding-window chunker (character based, keeps things dependency-light)."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += chunk_size - overlap
        return chunks

    def ingest_pdf(self, path):
        """Load one PDF, chunk it, embed the chunks, and add them to Chroma."""
        filename = os.path.basename(path)
        pages = self._read_pdf(path)

        documents, metadatas, ids = [], [], []
        chunk_counter = 0

        for page_num, page_text in pages:
            for chunk in self._chunk_text(page_text):
                chunk_counter += 1
                documents.append(chunk)
                metadatas.append({"source": filename, "page": page_num})
                ids.append(f"{filename}-p{page_num}-c{chunk_counter}")

        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

        return len(documents)

    def ingest_folder(self, folder_path=config.DATA_DIR):
        """Ingest every PDF found in a folder. Returns a summary dict."""
        summary = {}
        if not os.path.isdir(folder_path):
            return summary

        for fname in os.listdir(folder_path):
            if fname.lower().endswith(".pdf"):
                full_path = os.path.join(folder_path, fname)
                n_chunks = self.ingest_pdf(full_path)
                summary[fname] = n_chunks
        return summary

    # ---------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------
    def search(self, query, top_k=config.TOP_K):
        """
        Semantic search over the vector DB.
        Returns a list of dicts: {text, source, page, distance}
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(query_texts=[query], n_results=top_k)

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "?"),
                "distance": dist,
            })
        return hits

    def format_context(self, hits):
        """Turn retrieved hits into a context block + a source list for citations."""
        if not hits:
            return "No relevant information found in the uploaded travel guides.", []

        context_lines = []
        sources = []
        for i, h in enumerate(hits, start=1):
            context_lines.append(f"[{i}] (Source: {h['source']}, page {h['page']})\n{h['text']}")
            sources.append(f"{h['source']} (page {h['page']})")

        return "\n\n".join(context_lines), sources


if __name__ == "__main__":
    # quick manual test: python rag_engine.py
    engine = TravelRAGEngine()
    summary = engine.ingest_folder()
    print("Ingested:", summary)
    hits = engine.search("best time to visit Goa")
    for h in hits:
        print(h["source"], h["page"], h["text"][:100])
