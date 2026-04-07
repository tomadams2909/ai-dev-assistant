<div align="center">

# REX
### Repository Engineering eXpert

**A local-first AI developer assistant that understands your codebase**

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local-black?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![CI](https://github.com/tomadams2909/rex/actions/workflows/ci.yml/badge.svg)

[Features](#features) · [Architecture](#architecture) · [Getting Started](#getting-started) · [Models](#models) · [Screenshots](#screenshots)

</div>

---

## What is REX?

REX is a local AI developer assistant that indexes your codebase using RAG (Retrieval-Augmented Generation) and lets you query it in natural language. It runs entirely on your machine by default — your code never leaves your computer unless you explicitly choose a cloud provider.

Ask REX to explain a function, find where authentication is handled, review a file for bugs, or chat with a cloud model that has access to your indexed project. Switch between local and cloud models on the fly. Every conversation is saved and resumable.

```
"Where is token validation handled?"  →  REX retrieves the relevant code and explains it
"Review auth.py for security issues"  →  REX loads the full file and runs a structured analysis
"What changed in the FastAPI docs?"   →  REX searches the web and answers with citations
```

---

## Features

### Core Intelligence
- **Semantic codebase search** — indexes your project using ChromaDB vector embeddings, finds relevant code by meaning not just keywords
- **Full-file code review** — loads entire files into context and runs structured analysis across four categories: bugs, security, code quality, and improvements
- **RAG pipeline** — retrieves the most relevant chunks for each query, keeping context lean and responses accurate
- **Conversation memory** — sliding window history with compression, sessions persist across restarts and are fully resumable

### AI Provider Tier System
| Tier | Models | Cost | Strength |
|------|--------|------|----------|
| Local | qwen2.5-coder:7b, qwen3.5:9b | Free, offline | Privacy, zero cost |
| Cloud Free | Groq Llama 3.3 70B | Free | 10× local capability, 500+ tok/s |
| Cloud Paid | Gemini 2.5 Flash | ~$0.001/query | 1M token context window |
| Cloud Premium | Claude Sonnet 4.6 | ~$0.04/query | Highest quality reasoning |

### Web Search
- **Tiered web search** — each provider uses the most appropriate search method
  - Claude → native Anthropic web search tool (autonomous, cited)
  - Gemini → Google grounding (native SDK)
  - Groq / Local → DuckDuckGo fallback (free, no API key)
- Toggle per-query, automatically disabled when offline

### Chat Modes
- **Repo Mode** — full RAG pipeline with your indexed codebase
- **Chat Mode** — direct AI conversation, no codebase required

### Developer Experience
- **Streaming responses** via Server-Sent Events — tokens appear as they are generated
- **Syntax-highlighted code blocks** with copy button (VS Code Dark+ theme via Prism.js)
- **Markdown rendering** for all assistant responses
- **Smart auto-scroll** — follows generation only when already at the bottom
- **Conversation history** — browse, resume, and delete past sessions from the sidebar
- **Token usage tracking** — per-provider cost estimates with reset capability
- **Offline detection** — cloud providers automatically disabled when network unavailable

### Customisation
- **Accent colour picker** — full HSB sliders, 36 curated swatches across 6 palettes
- **Background modes** — Dark, Darker, OLED black
- **Sidebar width** — Narrow, Normal, Wide presets with clamped percentage scaling
- **Accessibility** — colour blind mode filters (Deuteranopia, Protanopia, Tritanopia, High Contrast)
- **Results per query** — configurable RAG chunk retrieval count

---

## Architecture

REX is built around a clean four-layer architecture:

```
┌─────────────────────────────────────────────────┐
│                   Browser UI                    │
│         HTML · CSS · Vanilla JS · SSE           │
└────────────────────┬────────────────────────────┘
                     │ HTTP / SSE
┌────────────────────▼────────────────────────────┐
│                  FastAPI Backend                │
│   /stream  /review  /ingest  /sessions  /usage  │
└──────┬──────────────┬────────────────┬──────────┘
       │              │                │
┌──────▼──────┐ ┌─────▼──────┐ ┌───────▼────────┐
│ Orchestrator│ │   Memory   │ │   Vector Store │
│  RAG + Tools│ │  Sessions  │ │   ChromaDB     │
└──────┬──────┘ └────────────┘ └────────────────┘
       │
┌──────▼──────────────────────────────────────────┐
│              Provider Abstraction               │
│        Ollama · Claude · Groq · Gemini          │
└─────────────────────────────────────────────────┘
```

### Key Design Decisions

**Provider abstraction** — all models implement a common `ModelProvider` interface with `chat()`, `chat_stream()`, and `embed()`. Switching providers or adding new ones requires one new class and one config line. `OpenAICompatibleProvider` serves as a base for any OpenAI-format API (currently Groq).

**Tool system** — discrete capabilities live in `tools/` as independently testable modules. The orchestrator coordinates which tools to use per request:
- `tools/file_reader.py` — safe file loading with path traversal protection
- `tools/web_search.py` — DuckDuckGo fallback for providers without native search

**Memory architecture** — conversation history uses a sliding window with compression. Code chunks retrieved for RAG are never stored in history (they are re-fetched each turn), keeping the context window lean across long sessions.

**Streaming** — the backend yields SSE events (`token`, `done`, `error`) and the frontend consumes them with a `ReadableStream` reader. Debounce delay is 0ms for Groq (500+ tok/s) and 16ms for other providers to balance responsiveness and DOM performance.

---

## Getting Started

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.com/download) installed and running
- Node.js (optional, for Electron packaging)

### 1. Clone and install

```bash
git clone https://github.com/tomadams2909/rex.git
cd rex
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Pull local models

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull qwen2.5-coder:7b   # code model
ollama pull qwen3.5:9b         # reasoning model
```

### 3. Configure environment (optional — for cloud providers)

```bash
cp .env.example .env
# Add your API keys to .env:
# ANTHROPIC_API_KEY=sk-ant-...
# GROQ_API_KEY=gsk_...
# GEMINI_API_KEY=AIza...
```

### 4. Start REX

```bash
python api.py
```

Open [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app) in your browser.

### 5. Index your first project

In the sidebar, enter the path to any local project and click **Index Project**. REX will chunk and embed every allowed file. Then start asking questions.

---

## Models

### Local (Ollama)

| Model | Role | Context | VRAM |
|-------|------|---------|------|
| `qwen2.5-coder:7b` | Code tasks | 32K | ~4.5GB |
| `qwen3.5:9b` | Reasoning, review | 128K | ~5.5GB |
| `nomic-embed-text` | Embeddings | 8K | ~300MB |

Local models run entirely on your GPU. Nothing leaves your machine.

### Cloud

| Provider | Model | Context | Web Search |
|----------|-------|---------|------------|
| Anthropic | Claude Sonnet 4.6 | 200K | Native (Anthropic) |
| Google | Gemini 2.5 Flash | 1M | Native (Google) |
| Groq | Llama 3.3 70B | 128K | DuckDuckGo |

Cloud providers require API keys in `.env`. Token usage and estimated costs are tracked per provider and visible in the sidebar.

---

## Project Structure

```
rex/
├── api.py               # FastAPI application, all HTTP endpoints
├── orchestrator.py      # Query logic, RAG pipeline, streaming
├── ingest.py            # File scanning, chunking, embedding
├── retriever.py         # Semantic search against ChromaDB
├── memory.py            # Session persistence, history trimming
├── config.py            # Model names, paths, settings
├── usage_tracker.py     # Per-provider token and cost tracking
├── models/
│   └── provider.py      # ModelProvider base + all provider implementations
├── tools/
│   ├── file_reader.py   # Safe full-file loading for review mode
│   └── web_search.py    # DuckDuckGo fallback search
├── frontend/
│   ├── index.html       # Single-page UI
│   └── style.css        # Dark theme, CSS variables, animations
└── tests/
    └── test_streaming.py
```

---

## Roadmap

- [ ] Electron wrapper — standalone desktop app, no browser required
- [ ] File and image attachment support (vision models)
- [ ] Delete indexed projects from UI
- [ ] Multi-file review mode using Gemini's 1M context window

---

## Technical Highlights

These are the engineering decisions worth discussing:

**RAG with token-aware memory** — code chunks are injected into the current turn only, never accumulated in history. This keeps the effective context window small regardless of conversation length, solving the context overflow problem without arbitrary truncation.

**Provider abstraction with native capability flags** — `HAS_NATIVE_SEARCH = True/False` on each provider class lets the orchestrator route web search correctly without conditional logic scattered through the codebase. Adding a new provider with native search is one attribute override.

**Tiered web search** — Claude and Gemini use their native search APIs (better quality, cited sources, autonomous query generation). Groq and local models use DuckDuckGo. A single user toggle controls all providers consistently.

**Streaming with per-provider debounce** — Groq generates at 500+ tokens/second. Rendering markdown on every token causes DOM thrashing. The debounce delay is set to 0ms for Groq and 16ms for other providers, detected at runtime from the active provider state.

**Safe file access** — all file operations resolve paths against the project root and reject anything that traverses outside it. The review endpoint returns a 400 with a clear message if path traversal is attempted.

---

## License

MIT — see [LICENSE](LICENSE)

---

<div align="center">
Built with FastAPI · ChromaDB · Ollama · Anthropic · Groq · Google Gemini
</div>
