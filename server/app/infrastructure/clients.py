"""
infrastructure/clients.py — Lazy-loaded ML/infra clients via functools.lru_cache.
No globals, no classes, no DI container.
"""

import functools
import logging
from pathlib import Path

from config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder

log = logging.getLogger("default")


@functools.lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    log.info("Loading embedding model %s ...", settings.embed_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@functools.lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
    )


@functools.lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
    )


@functools.lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    log.info("Loading reranker %s ...", settings.rerank_model)
    reranker = CrossEncoder(
        settings.rerank_model,
        max_length=1024,
        device=settings.rerank_device,
    )
    log.info("Reranker loaded")
    return reranker


@functools.lru_cache(maxsize=1)
def get_bm25_index():
    """Lazy-load BM25 index from disk. Returns None if not found."""
    from infrastructure.ml.hybrid import load_bm25_index

    bm25_path = Path(settings.data_dir) / "bm25_index.json"
    index = load_bm25_index(bm25_path)
    if index is None:
        log.info("No BM25 index found at %s — hybrid search disabled for this run", bm25_path)
    return index
