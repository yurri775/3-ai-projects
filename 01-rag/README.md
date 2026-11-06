# 01 · RAG over your own docs

A question-answering app over a document corpus that answers **only from the retrieved
passages** and **cites its sources** — or says *"I don't know"* when the answer isn't
there. Retrieval runs locally and for free; only the final answer calls the Claude API.

```
load → chunk → embed (MiniLM, local) → Chroma → search (top-k) → answer (Claude, cited)
```

## What this demonstrates
- **chunking** · **embeddings** · **vector search** · **citations** · **evaluation**
- Uses the **Claude Messages API** for grounded generation.
- Ships an **MCP server** so any MCP client can query your docs → see [`mcp_server.py`](./mcp_server.py).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
```

## Use it
```bash
# 1) Build the index (starter: a ready-made Wikipedia corpus with an eval set)
python rag.py build                 # add --limit 500 for a quick first run

# 2) Ask questions (answers come back with sources)
python rag.py ask "Was Abraham Lincoln the sixteenth President?"

# 3) Index YOUR own documents instead
python rag.py build --source local --folder ./my_docs
```

## Expose it over MCP
```bash
python mcp_server.py                 # exposes tools: ask_docs, search_docs
```
Point Claude Desktop (or any MCP client) at this server to chat with your documents.

## Evaluate it
The `rag-mini-wikipedia` dataset ships with question/answer pairs, so you get a pass
rate you can quote:
```bash
python eval.py --n 50                # fast substring grader
python eval.py --n 50 --judge        # stricter: Claude-as-judge
```

## Results

> Run `python eval.py --n 50` and paste your number here.

| Config | Questions | Grader | Pass rate |
|--------|-----------|--------|-----------|
| MiniLM + Chroma + `claude-haiku-5-5`, top-5 | 50 | substring | `__%` |
| same | 50 | Claude judge | `__%` |

## "Done when" checklist
- [ ] Every answer cites the document it came from
- [ ] It says "I don't know" when the answer isn't in the docs
- [ ] `eval.py` prints a pass rate you can quote

## Start here
DeepLearning.AI short course — *Building and Evaluating Advanced RAG*.
