# AI Smart Travel Planning Assistant

Agentic AI + Agentic RAG based travel planner built with **CrewAI** and **Hugging Face**
(both the LLM and the embeddings run on the Hugging Face ecosystem).

Regenerated for: **Python 3.11**, **CrewAI 1.14.6** (latest stable at time of writing),
and current Hugging Face **Inference Providers** routing.

## What it does
- Upload travel guide PDFs, they get chunked, embedded and stored in ChromaDB
- Ask a question and a router agent decides which workflow should handle it:
  - Plain question -> single agent does Agentic RAG (retrieves + cites sources)
  - "Plan a trip" -> multi-agent crew: knowledge lookup -> weather check -> itinerary builder
  - "Compare X vs Y" -> multi-agent crew: knowledge lookup -> weather -> comparison
  - "Budget for..." -> multi-agent crew: knowledge lookup -> budget calculator tool
  - "Weather in..." -> single agent with live weather tool

## Folder structure
```
travel_assistant/
├── app.py            # Streamlit UI (main entry point)
├── router.py          # decides which agent workflow to run
├── agents.py           # CrewAI agents, tasks, crews
├── tools.py             # RAG search tool, weather tool, budget tool
├── rag_engine.py          # PDF loading, chunking, embeddings, Chroma
├── config.py               # settings / API keys
├── data/                     # put sample travel guide PDFs here
├── vectorstore/                # ChromaDB persisted data (auto-created)
├── requirements.txt
└── .env.example
```

## Setup

Requires **Python 3.11** (Python <3.14, >=3.10 also works since that's what
CrewAI requires, but this project was built/tested against 3.11).

```bash
python3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env         # Windows: copy .env.example .env
# then edit .env and add your HUGGINGFACEHUB_API_TOKEN (free token from
# https://huggingface.co/settings/tokens)

streamlit run app.py
```

After install, you can sanity-check there are no dependency conflicts with:
```bash
pip check
```

## Notes on the LLM setup
- Hugging Face retired its old single-endpoint free "serverless Inference API"
  in favor of **Inference Providers** - a router that forwards your call to one
  of several backend providers (hf-inference, together, novita, fireworks-ai, ...).
  This project talks to that router through CrewAI's `LLM` class (backed by
  litellm's `huggingface/` integration) and, for the lightweight intent
  classifier in `router.py`, directly through `huggingface_hub.InferenceClient`.
- Default model: `Qwen/Qwen2.5-7B-Instruct` via the `hf-inference` provider.
  Not every model is served by every provider - if you change `HF_MODEL_REPO`
  in `.env` and get a "model not supported by provider" error, either pick a
  different `HF_PROVIDER` or check the model's page on
  https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation
  for which providers currently serve it.
- Embeddings run locally (`all-MiniLM-L6-v2`, also a Hugging Face model) so no
  extra cost/key is needed for that part.
- Weather uses wttr.in (free, no key).
- Budget numbers are a rough starting estimate (INR), not live pricing.

## What changed from the previous version
- Upgraded to CrewAI 1.14.6 (latest stable release) and Streamlit 1.59.2.
- Removed the unused `crewai-tools` dependency - this project's custom tools
  subclass `crewai.tools.BaseTool` directly and never imported from
  `crewai_tools`, so dropping it removes a large, unnecessary dependency
  surface (and a source of potential version conflicts).
- Updated `config.py`/`router.py`/`agents.py` for Hugging Face's current
  Inference Providers routing and the current `huggingface_hub` chat
  completions API (`client.chat.completions.create(...)`) instead of the
  older `client.chat_completion(...)` call.
- `requirements.txt` pins the two frameworks that matter most for
  reproducibility (CrewAI, Streamlit) and gives sensible floor/ceiling ranges
  for the RAG stack (chromadb, sentence-transformers, pypdf, huggingface_hub)
  so pip's resolver has room to pick mutually compatible versions.
