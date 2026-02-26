"""RAG Agent for knowledge-grounded merchant responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:  # pragma: no cover
    try:
        from langchain.prompts import ChatPromptTemplate
    except Exception:  # pragma: no cover
        ChatPromptTemplate = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:  # pragma: no cover
        RecursiveCharacterTextSplitter = None

try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # pragma: no cover
    FAISS = None
    ChatOpenAI = None
    OpenAIEmbeddings = None


class RAGAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1) if ChatOpenAI else None
        self.embeddings = OpenAIEmbeddings() if OpenAIEmbeddings else None
        self.vectorstore = None
        if FAISS and self.embeddings and Path("data/vector_store").exists():
            try:
                self.vectorstore = FAISS.load_local(
                    "data/vector_store",
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                self.vectorstore = None

    @classmethod
    def build_vector_store(cls, documents_dir: str, output_dir: str) -> None:
        if not (FAISS and OpenAIEmbeddings and RecursiveCharacterTextSplitter):
            raise RuntimeError("RAG dependencies are missing")

        from langchain_community.document_loaders import DirectoryLoader

        loader = DirectoryLoader(documents_dir, glob="**/*.md")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(output_dir)

    async def execute(self, query: str, merchant_id: str) -> dict[str, Any]:
        _ = merchant_id
        if not self.vectorstore:
            return {
                "response": "Knowledge base index is not initialized. Build vector_store first.",
                "sources": [],
                "confidence": 0.0,
            }

        relevant_docs = self.vectorstore.similarity_search_with_score(query, k=5)
        context = "\n\n".join(
            [
                f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc, score in relevant_docs
                if score < 0.8
            ]
        )

        if not (self.llm and ChatPromptTemplate):
            sources = [doc.metadata.get("source", "unknown") for doc, _ in relevant_docs[:3]]
            return {
                "response": "Retrieved relevant policy documents. LLM response disabled in local mode.",
                "sources": sources,
                "confidence": 0.5,
            }

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer using only the provided context. If insufficient, say so.",
                ),
                ("human", "Context:\n{context}\n\nQuestion: {query}"),
            ]
        )

        chain = prompt | self.llm
        response = await chain.ainvoke({"context": context, "query": query})

        sources = [doc.metadata.get("source", "unknown") for doc, _ in relevant_docs[:3]]
        confidence = 1.0 - min([score for _, score in relevant_docs[:1]] + [1.0])

        return {
            "response": str(response.content),
            "sources": sources,
            "confidence": float(max(min(confidence, 1.0), 0.0)),
        }
