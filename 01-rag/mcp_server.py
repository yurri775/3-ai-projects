"""MCP server exposing the RAG as a tool.

Any MCP client (Claude Desktop, an IDE, your own agent) can now call `ask_docs`
and get a grounded, cited answer from your indexed documents.

Run the index first:   python rag.py build
Then run the server:    python mcp_server.py

Register it in an MCP client (e.g. Claude Desktop config):
    {
      "mcpServers": {
        "rag-docs": { "command": "python", "args": ["/abs/path/to/mcp_server.py"] }
      }
    }
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

import rag

rag._load_env()
mcp = FastMCP("rag-docs")


@mcp.tool()
def ask_docs(question: str) -> str:
    """Answer a question strictly from the indexed documents, with sources.

    Args:
        question: a natural-language question about the indexed corpus.
    """
    text, hits = rag.answer(question)
    sources = ", ".join(sorted({str(h["source"]) for h in hits}))
    return f"{text}\n\nSources: {sources}"


@mcp.tool()
def search_docs(question: str, k: int = 5) -> str:
    """Return the top-k raw passages for a question (no LLM answer)."""
    hits = rag.search(question, k=k)
    return "\n\n".join(
        f"[{h['source']}] (score {h['score']})\n{h['text']}" for h in hits
    )


if __name__ == "__main__":
    mcp.run()
