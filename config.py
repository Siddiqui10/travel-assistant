"""
config.py
----------
Basic settings for the project. Hugging Face is used as the LLM provider
(free-tier Inference Providers, good for students) as well as for
embeddings, so the whole stack runs on one ecosystem.

NOTE ON HUGGING FACE ROUTING (2026):
Hugging Face's old single free "serverless Inference API" endpoint has been
replaced by "Inference Providers" - a router that forwards your request to
one of several backend providers (e.g. hf-inference, together, novita, ...).
Model + provider availability changes over time, so if the default model
below stops working, pick a different chat-completion-capable model from
https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation
and/or a different HF_PROVIDER, and set them in your .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---- LLM settings (Hugging Face Inference Providers via CrewAI's built-in
#      LLM class, powered by litellm's "huggingface/" provider) ----
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Qwen/Qwen2.5-7B-Instruct")
HF_PROVIDER = os.getenv("HF_PROVIDER", "hf-inference")

# litellm expects: huggingface/<provider>/<org>/<model>
HF_MODEL = f"huggingface/{HF_PROVIDER}/{HF_MODEL_REPO}"

# Only needed if you're pointing at a dedicated/custom Inference Endpoint
# instead of the shared Inference Providers router. Leave unset otherwise.
HF_API_BASE = os.getenv("HF_API_BASE") or None

LLM_TEMPERATURE = 0.3

# ---- Embedding + vector DB settings ----
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
COLLECTION_NAME = "travel_guides"

# ---- Chunking ----
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# ---- Retrieval ----
TOP_K = 4

# ---- Folders ----
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

if not HF_API_KEY:
    print("[WARNING] HUGGINGFACEHUB_API_TOKEN not set. Add it to a .env file before running the app.")
