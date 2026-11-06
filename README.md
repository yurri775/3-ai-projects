# 3 AI projects — from a RAG app to a served open-source model

Three hands-on AI-engineering projects, ordered **easiest → hardest**. Each one is
self-contained, ships with a `README`, and teaches a distinct, job-relevant skill.
Together they go from *"answer questions over my own docs"* all the way to
*"serve a model behind an API and measure its latency."*

> Built as a portfolio by **Ayoub AMRANI** — Junior Software Engineer (Java · Python · Generative AI).
> Proof, not just a CV.

---

## The roadmap

| # | Project | What you build | Core skills | Status |
|---|---------|----------------|-------------|--------|
| **01** | [**RAG over your own docs**](./01-rag) | A Q&A app over your PDFs/notes that answers **with sources** | chunking · embeddings · vector search · citations · evaluation | ✅ runnable |
| **02** | [**Agent for long-horizon tasks**](./02-agents) | An agent that finishes a multi-step task on its own, with guardrails | tool-use loop · sandboxing · guardrails · budgets · logging · testing | ✅ runnable |
| **03** | [**Deploy an open-source model**](./03-deploy) | Serve an open model behind an API on a free GPU and **measure** it | vLLM · GPU memory · TTFT · latency · throughput | 📦 scripts + Colab |

---

## How the projects connect to my certifications

This repo is deliberately wired to the Anthropic certifications on my profile — the code
is the proof behind the certificate:

- **Building with the Claude API** → Projects **01** and **02** both call the Claude
  Messages API (grounded answers in 01, a full tool-use agent loop in 02).
- **Introduction to Model Context Protocol (MCP)** → Project **01** ships an
  [`mcp_server.py`](./01-rag/mcp_server.py) that exposes the RAG as an MCP tool, so any
  MCP client (e.g. Claude Desktop) can query your documents.

```
01-rag ──> Claude API + MCP server
02-agents ─> Claude API tool-use loop (plan → act → check)
03-deploy ─> your own open-source model behind an OpenAI-compatible API
```

---

## Quick start

Each project has its own `requirements.txt` and `.env.example`. In general:

```bash
cd 01-rag                     # or 02-agents / 03-deploy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your own ANTHROPIC_API_KEY
```

Then follow that project's `README.md`.

> **Note on keys & secrets:** every project reads its API key from a local `.env`
> that is **git-ignored** — no secret is ever committed. Copy `.env.example` to `.env`
> and fill in your own key.

---

## Tech stack

Python · [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) ·
[ChromaDB](https://www.trychroma.com/) · [sentence-transformers](https://www.sbert.net/) ·
[MCP](https://modelcontextprotocol.io/) · [vLLM](https://docs.vllm.ai/) ·
Hugging Face `datasets`.

## License

[MIT](./LICENSE) © 2026 Ayoub AMRANI

<sub>Project roadmap inspired by a widely-shared AI-engineering learning path; all code here is my own implementation.</sub>
