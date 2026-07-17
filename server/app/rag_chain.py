"""
rag_chain.py — LangChain LCEL цепочка с историей диалога.
"""

import logging
from collections.abc import AsyncIterator

from config import settings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder

log = logging.getLogger("rag_chain")

# ---------------------------------------------------------------------------
# Синглтоны (инициализируются один раз при старте сервера)
# ---------------------------------------------------------------------------

_embeddings: HuggingFaceEmbeddings | None = None
_vector_store: QdrantVectorStore | None = None
_llm: ChatOllama | None = None
_reranker: CrossEncoder | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store() -> QdrantVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore.from_existing_collection(
            embedding=get_embeddings(),
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
        )
    return _vector_store


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
        )
    return _llm


def get_reranker() -> CrossEncoder:
    """BAAI/bge-reranker-v2-m3 — кросс-энкодер, лицензия MIT."""
    global _reranker
    if _reranker is None:
        log.info("Загружаю реранкер %s ...", settings.rerank_model)
        _reranker = CrossEncoder(
            settings.rerank_model,
            max_length=1024,
            device=settings.rerank_device,
        )
        log.info("Реранкер загружен")
    return _reranker


def rerank_documents(question: str, docs: list, top_n: int) -> list:
    """
    Переранжирует кандидатов из векторного поиска кросс-энкодером
    и возвращает top_n самых релевантных документов.
    """
    if not docs:
        return docs

    reranker = get_reranker()
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _score in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Промпт
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты — корпоративный ассистент. Отвечай только на основе предоставленного контекста из документов компании.
Если ответа в контексте нет — честно скажи об этом. Не придумывай факты.
Отвечай на том же языке, на котором задан вопрос.

Контекст из документов:
{context}
"""


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def format_docs(docs) -> str:
    """Форматирует найденные чанки в строку для промпта."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def history_to_messages(history: list[dict]):
    """Конвертирует историю из БД в LangChain-сообщения."""
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


def extract_sources(docs) -> list[dict]:
    """Извлекает метаданные источников для сохранения в БД."""
    sources = []
    seen = set()
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(
                {
                    "source": src,
                    "page": doc.metadata.get("page", None),
                }
            )
    return sources


# ---------------------------------------------------------------------------
# Основная функция — стриминг ответа
# ---------------------------------------------------------------------------


async def rag_stream(
    question: str,
    history: list[dict],
) -> AsyncIterator[str]:
    """
    Yields chunks of the answer string.
    После полного ответа — финальный чанк с источниками: "__sources__:{json}"
    """
    import json

    retriever = get_vector_store().as_retriever(
        search_type="similarity",
        # Берём с запасом (fetch_k) — реранкер сам отберёт top_k лучших
        search_kwargs={"k": settings.retriever_fetch_k},
    )
    llm = get_llm()
    prompt = build_prompt()

    # 1. Ретривим кандидатов из Qdrant (быстрый bi-encoder поиск, широкий k)
    candidates = retriever.invoke(question)

    # 2. Реранкируем кросс-энкодером bge-reranker-v2-m3 и оставляем top_k лучших
    docs = rerank_documents(question, candidates, top_n=settings.retriever_top_k)

    context = format_docs(docs)
    sources = extract_sources(docs)
    history_messages = history_to_messages(history)

    # Собираем промпт и стримим
    messages = prompt.format_messages(
        context=context,
        history=history_messages,
        question=question,
    )

    full_answer = ""
    async for chunk in llm.astream(messages):
        text = chunk.content
        if text:
            full_answer += text
            yield text

    # Отдаём источники отдельным маркером
    yield f"\n__sources__:{json.dumps(sources, ensure_ascii=False)}"


async def rag_invoke(question: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Не-стриминговый вариант — для тестов."""
    import json

    answer_parts = []
    sources = []

    async for chunk in rag_stream(question, history):
        if chunk.startswith("\n__sources__:"):
            sources = json.loads(chunk.replace("\n__sources__:", ""))
        else:
            answer_parts.append(chunk)

    return "".join(answer_parts), sources
