"""
benchmark.py — оценка качества RAG-системы.

Что измеряет:
  Retriever:
    hit_rate       — есть ли среди top-k чанков хотя бы один с правильным источником
    mrr            — Mean Reciprocal Rank (насколько высоко стоит правильный чанк)
    avg_similarity — средний similarity score найденных чанков

  Generator (LLM-судья через Ollama):
    faithfulness   — ответ основан на контексте или модель придумала? (0-10)
    relevancy      — ответ по существу вопроса? (0-10)
    correctness    — совпадает с эталонным ответом? (0-10, только если задан expected_answer)

Запуск:
    # Запускать из папки api/ пока работает docker-compose
    python benchmark.py --questions test_questions.json
    python benchmark.py --questions test_questions.json --out results/
    python benchmark.py --questions test_questions.json --judge-model qwen2.5:14b --top-k 8
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain.schema import Document

# ---------------------------------------------------------------------------
# Зависимости — те же что у основного проекта
# ---------------------------------------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore

from config import settings


# ---------------------------------------------------------------------------
# Цвета для терминала
# ---------------------------------------------------------------------------
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    GRAY   = "\033[90m"
    BLUE   = "\033[94m"

def ok(s):    return f"{C.GREEN}✓{C.RESET} {s}"
def warn(s):  return f"{C.YELLOW}~{C.RESET} {s}"
def fail(s):  return f"{C.RED}✗{C.RESET} {s}"
def info(s):  return f"{C.CYAN}i{C.RESET} {s}"
def score_color(v: float, low=4.0, high=7.0) -> str:
    if v >= high:  return f"{C.GREEN}{v:.1f}{C.RESET}"
    if v >= low:   return f"{C.YELLOW}{v:.1f}{C.RESET}"
    return f"{C.RED}{v:.1f}{C.RESET}"


# ---------------------------------------------------------------------------
# Загрузка тестовых вопросов
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    {
        "id": "q1",
        "question": "Какие товары подлежат обязательной маркировке?",
        "expected_answer": None,        # если нет — correctness не считается
        "source_hint": None             # часть имени файла где должен быть ответ
    },
    {
        "id": "q2",
        "question": "Каков порядок электронного документооборота?",
        "expected_answer": None,
        "source_hint": "электронном документе"
    },
]


def load_questions(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"{C.YELLOW}[WARN]{C.RESET} Файл {path} не найден — создаю пример test_questions.json")
        example_path = Path(path)
        example_path.write_text(
            json.dumps(EXAMPLE_QUESTIONS, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"       Отредактируй {path} и запусти снова.\n")
        sys.exit(0)

    data = json.loads(p.read_text(encoding="utf-8"))
    print(info(f"Загружено вопросов: {len(data)}"))
    return data


# ---------------------------------------------------------------------------
# Компоненты RAG (переиспользуем из основного проекта)
# ---------------------------------------------------------------------------

def build_retriever(top_k: int):
    print(info(f"Загружаю эмбеддинг-модель {settings.embed_model} ..."))
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=settings.qdrant_url,
        collection_name=settings.collection_name,
    )
    return vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": 0.0},
    ), vs


def retrieve_with_scores(vs: QdrantVectorStore, question: str, top_k: int) -> list[tuple[Document, float]]:
    """Возвращает [(doc, score), ...] — нужно для MRR и avg_similarity."""
    results = vs.similarity_search_with_score(question, k=top_k)
    return results  # (Document, float)


def build_llm(model: str, base_url: str) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.0,   # судья должен быть детерминирован
    )


# ---------------------------------------------------------------------------
# RAG: получить ответ
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """\
Ты — корпоративный ассистент. Отвечай только на основе предоставленного контекста.
Если ответа в контексте нет — напиши ровно: "Информация в документах не найдена."
Не придумывай факты.

Контекст:
{context}

Вопрос: {question}
"""

def get_rag_answer(llm: ChatOllama, docs_with_scores: list[tuple[Document, float]], question: str) -> str:
    context = "\n\n---\n\n".join(
        f"[Источник: {d.metadata.get('filename', 'unknown')}]\n{d.page_content}"
        for d, _ in docs_with_scores
    )
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content.strip()


# ---------------------------------------------------------------------------
# LLM-судья: оценки
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """\
Ты — строгий эксперт по оценке качества ответов AI-ассистентов.

Контекст из документов:
{context}

Вопрос: {question}

Ответ ассистента: {answer}

Задача: оцени FAITHFULNESS (достоверность) — насколько ответ основан на предоставленном контексте.
Ответ полностью из контекста = 10. Ответ содержит выдуманные факты = 0.

Ответь СТРОГО в формате JSON (только JSON, без пояснений):
{{"score": <число от 0 до 10>, "reason": "<одно предложение>"}}
"""

RELEVANCY_PROMPT = """\
Ты — строгий эксперт по оценке качества ответов AI-ассистентов.

Вопрос: {question}

Ответ ассистента: {answer}

Задача: оцени RELEVANCY (релевантность) — насколько ответ отвечает на поставленный вопрос.
Точный полный ответ = 10. Ответ не по теме = 0.

Ответь СТРОГО в формате JSON (только JSON, без пояснений):
{{"score": <число от 0 до 10>, "reason": "<одно предложение>"}}
"""

CORRECTNESS_PROMPT = """\
Ты — строгий эксперт по оценке качества ответов AI-ассистентов.

Вопрос: {question}

Эталонный ответ: {expected}

Ответ ассистента: {answer}

Задача: оцени CORRECTNESS (правильность) — насколько ответ совпадает по смыслу с эталонным.
Полное совпадение по смыслу = 10. Противоречит эталону = 0.

Ответь СТРОГО в формате JSON (только JSON, без пояснений):
{{"score": <число от 0 до 10>, "reason": "<одно предложение>"}}
"""


def parse_judge_response(raw: str, metric: str) -> tuple[float, str]:
    """Парсит JSON-ответ судьи. Устойчив к мусору вокруг JSON."""
    import re
    # Ищем JSON объект в ответе
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if not match:
        return 0.0, f"[Ошибка парсинга ответа судьи: {raw[:100]}]"
    try:
        data = json.loads(match.group())
        score = float(data.get("score", 0))
        score = max(0.0, min(10.0, score))  # клампим в [0, 10]
        reason = str(data.get("reason", ""))
        return score, reason
    except (json.JSONDecodeError, ValueError) as e:
        return 0.0, f"[JSON parse error: {e}]"


def judge_answer(
    judge_llm: ChatOllama,
    question: str,
    answer: str,
    context: str,
    expected_answer: str | None = None,
) -> dict:
    scores = {}

    # Faithfulness
    raw = judge_llm.invoke(
        FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)
    ).content
    scores["faithfulness"], scores["faithfulness_reason"] = parse_judge_response(raw, "faithfulness")

    # Relevancy
    raw = judge_llm.invoke(
        RELEVANCY_PROMPT.format(question=question, answer=answer)
    ).content
    scores["relevancy"], scores["relevancy_reason"] = parse_judge_response(raw, "relevancy")

    # Correctness (только если есть эталон)
    if expected_answer:
        raw = judge_llm.invoke(
            CORRECTNESS_PROMPT.format(question=question, expected=expected_answer, answer=answer)
        ).content
        scores["correctness"], scores["correctness_reason"] = parse_judge_response(raw, "correctness")
    else:
        scores["correctness"] = None
        scores["correctness_reason"] = "Эталонный ответ не задан"

    return scores


# ---------------------------------------------------------------------------
# Retriever метрики
# ---------------------------------------------------------------------------

def compute_retriever_metrics(
    docs_with_scores: list[tuple[Document, float]],
    source_hint: str | None,
) -> dict:
    """
    hit_rate  — есть ли хотя бы один чанк с source_hint в имени файла
    mrr       — позиция первого релевантного / rank
    avg_sim   — средний similarity score
    """
    scores = [s for _, s in docs_with_scores]
    avg_sim = sum(scores) / len(scores) if scores else 0.0

    if source_hint is None:
        return {
            "hit_rate": None,
            "mrr": None,
            "avg_similarity": round(avg_sim, 4),
            "retrieved_sources": [
                d.metadata.get("filename", "?") for d, _ in docs_with_scores
            ],
        }

    hit_rate = 0
    mrr = 0.0
    for rank, (doc, _) in enumerate(docs_with_scores, 1):
        filename = doc.metadata.get("filename", "") or doc.metadata.get("source", "")
        if source_hint.lower() in filename.lower():
            hit_rate = 1
            if mrr == 0.0:
                mrr = 1.0 / rank
            break

    return {
        "hit_rate": hit_rate,
        "mrr": round(mrr, 4),
        "avg_similarity": round(avg_sim, 4),
        "retrieved_sources": [
            d.metadata.get("filename", "?") for d, _ in docs_with_scores
        ],
    }


# ---------------------------------------------------------------------------
# Форматирование прогресса
# ---------------------------------------------------------------------------

def print_question_result(idx: int, total: int, q: dict, result: dict):
    print(f"\n{C.BOLD}[{idx}/{total}] {q['question']}{C.RESET}")

    # Retriever
    rm = result["retriever_metrics"]
    sim_str = f"avg_sim={rm['avg_similarity']:.3f}"
    if rm["hit_rate"] is not None:
        hr_str = ok("hit") if rm["hit_rate"] else fail("miss")
        mrr_str = f"mrr={rm['mrr']:.2f}"
        print(f"  Retriever: {hr_str}  {mrr_str}  {sim_str}")
    else:
        print(f"  Retriever: {sim_str}  {C.GRAY}(source_hint не задан){C.RESET}")

    src_list = ", ".join(result["retriever_metrics"]["retrieved_sources"][:3])
    print(f"  Источники: {C.GRAY}{src_list}{C.RESET}")

    # Generator
    gm = result["generator_metrics"]
    f_str = score_color(gm["faithfulness"])
    r_str = score_color(gm["relevancy"])
    print(f"  Faithfulness: {f_str}/10  — {C.GRAY}{gm['faithfulness_reason']}{C.RESET}")
    print(f"  Relevancy:    {r_str}/10  — {C.GRAY}{gm['relevancy_reason']}{C.RESET}")
    if gm["correctness"] is not None:
        c_str = score_color(gm["correctness"])
        print(f"  Correctness:  {c_str}/10  — {C.GRAY}{gm['correctness_reason']}{C.RESET}")

    # Ответ (первые 200 символов)
    answer_preview = result["answer"][:200].replace("\n", " ")
    if len(result["answer"]) > 200:
        answer_preview += "..."
    print(f"  Ответ: {C.GRAY}{answer_preview}{C.RESET}")
    print(f"  Время: {result['latency_sec']:.1f}s")


# ---------------------------------------------------------------------------
# Итоговый отчёт
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], total_time: float):
    n = len(results)
    print(f"\n{'='*60}")
    print(f"{C.BOLD}ИТОГОВЫЙ ОТЧЁТ  ({n} вопросов, {total_time:.1f}s){C.RESET}")
    print(f"{'='*60}")

    # Retriever aggregates
    hit_rates = [r["retriever_metrics"]["hit_rate"] for r in results
                 if r["retriever_metrics"]["hit_rate"] is not None]
    mrrs      = [r["retriever_metrics"]["mrr"] for r in results
                 if r["retriever_metrics"]["mrr"] is not None]
    sims      = [r["retriever_metrics"]["avg_similarity"] for r in results]

    print(f"\n{C.BOLD}Retriever:{C.RESET}")
    if hit_rates:
        avg_hr = sum(hit_rates) / len(hit_rates)
        avg_mrr = sum(mrrs) / len(mrrs)
        print(f"  Hit Rate:        {score_color(avg_hr*10)}/10  ({avg_hr:.0%} вопросов нашли нужный источник)")
        print(f"  MRR:             {C.CYAN}{avg_mrr:.3f}{C.RESET}  (1.0 = нужный чанк всегда первый)")
    print(f"  Avg Similarity:  {C.CYAN}{sum(sims)/len(sims):.3f}{C.RESET}")

    # Generator aggregates
    faiths = [r["generator_metrics"]["faithfulness"] for r in results]
    rels   = [r["generator_metrics"]["relevancy"] for r in results]
    corrs  = [r["generator_metrics"]["correctness"] for r in results
              if r["generator_metrics"]["correctness"] is not None]

    print(f"\n{C.BOLD}Generator:{C.RESET}")
    print(f"  Faithfulness:    {score_color(sum(faiths)/len(faiths))}/10  (достоверность — нет ли выдуманных фактов)")
    print(f"  Relevancy:       {score_color(sum(rels)/len(rels))}/10  (ответ по существу вопроса)")
    if corrs:
        print(f"  Correctness:     {score_color(sum(corrs)/len(corrs))}/10  (совпадение с эталоном)")

    # Провальные кейсы
    bad = [r for r in results
           if r["generator_metrics"]["faithfulness"] < 5
           or r["generator_metrics"]["relevancy"] < 5]
    if bad:
        print(f"\n{C.BOLD}{C.RED}Проблемные вопросы ({len(bad)}):{C.RESET}")
        for r in bad:
            gm = r["generator_metrics"]
            print(f"  [{r['id']}] {r['question'][:60]}")
            print(f"        faith={gm['faithfulness']:.1f}  rel={gm['relevancy']:.1f}")
            print(f"        {C.GRAY}{gm['faithfulness_reason']}{C.RESET}")

    print(f"\n{C.BOLD}Время:{C.RESET} {total_time:.1f}s  ({total_time/n:.1f}s на вопрос)")
    print("="*60)


# ---------------------------------------------------------------------------
# Сохранение результатов
# ---------------------------------------------------------------------------

def save_results(results: list[dict], out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Полный JSON
    json_path = out / f"benchmark_{ts}.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Краткий CSV для Excel/Sheets
    csv_path = out / f"benchmark_{ts}.csv"
    rows = ["id,question,faithfulness,relevancy,correctness,hit_rate,mrr,avg_sim,latency_sec"]
    for r in results:
        gm = r["generator_metrics"]
        rm = r["retriever_metrics"]
        rows.append(",".join([
            str(r["id"]),
            f'"{r["question"]}"',
            str(gm["faithfulness"]),
            str(gm["relevancy"]),
            str(gm["correctness"] if gm["correctness"] is not None else ""),
            str(rm["hit_rate"] if rm["hit_rate"] is not None else ""),
            str(rm["mrr"] if rm["mrr"] is not None else ""),
            str(rm["avg_similarity"]),
            str(round(r["latency_sec"], 2)),
        ]))
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    print(f"\n{ok('Результаты сохранены:')}")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------

def run_benchmark(
    questions_path: str,
    out_dir: str,
    top_k: int,
    judge_model: str,
):
    print(f"\n{C.BOLD}RAG Benchmark{C.RESET}")
    print(f"  questions : {questions_path}")
    print(f"  top_k     : {top_k}")
    print(f"  rag model : {settings.llm_model}")
    print(f"  judge     : {judge_model}")
    print(f"  qdrant    : {settings.qdrant_url}\n")

    questions = load_questions(questions_path)

    # Инициализируем компоненты
    retriever, vs = build_retriever(top_k)

    print(info(f"Подключаюсь к RAG LLM ({settings.llm_model}) ..."))
    rag_llm = build_llm(settings.llm_model, settings.ollama_base_url)

    print(info(f"Подключаюсь к LLM-судье ({judge_model}) ..."))
    judge_llm = build_llm(judge_model, settings.ollama_base_url)

    # Прогрев — первый вызов всегда медленнее
    print(info("Прогрев моделей ..."))
    rag_llm.invoke("Привет")
    if judge_model != settings.llm_model:
        judge_llm.invoke("Привет")

    print(f"\n{C.BOLD}Запускаю тесты...{C.RESET}")
    results = []
    total_start = time.time()

    for idx, q in enumerate(questions, 1):
        t_start = time.time()

        # 1. Retrieval
        docs_with_scores = retrieve_with_scores(vs, q["question"], top_k)

        # 2. Generate answer
        answer = get_rag_answer(rag_llm, docs_with_scores, q["question"])

        # 3. Retriever metrics
        retriever_metrics = compute_retriever_metrics(
            docs_with_scores,
            q.get("source_hint"),
        )

        # 4. LLM judge
        context_for_judge = "\n\n---\n\n".join(
            d.page_content for d, _ in docs_with_scores
        )
        generator_metrics = judge_answer(
            judge_llm,
            question=q["question"],
            answer=answer,
            context=context_for_judge,
            expected_answer=q.get("expected_answer"),
        )

        latency = time.time() - t_start

        result = {
            "id": q.get("id", str(idx)),
            "question": q["question"],
            "answer": answer,
            "expected_answer": q.get("expected_answer"),
            "source_hint": q.get("source_hint"),
            "retriever_metrics": retriever_metrics,
            "generator_metrics": generator_metrics,
            "latency_sec": round(latency, 2),
        }
        results.append(result)
        print_question_result(idx, len(questions), q, result)

    total_time = time.time() - total_start
    print_summary(results, total_time)
    save_results(results, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Бенчмарк RAG-системы — оценивает retriever и generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python benchmark.py --questions test_questions.json
  python benchmark.py --questions test_questions.json --top-k 8 --out results/
  python benchmark.py --questions test_questions.json --judge-model qwen2.5:14b

Формат test_questions.json:
  [
    {
      "id": "q1",
      "question": "Какие товары подлежат маркировке?",
      "expected_answer": "...",    // опционально — считается Correctness
      "source_hint": "Указ"        // опционально — часть имени файла с ответом
    }
  ]
        """
    )
    parser.add_argument(
        "--questions", default="test_questions.json",
        help="Путь к JSON-файлу с вопросами (default: test_questions.json)"
    )
    parser.add_argument(
        "--out", default="benchmark_results",
        help="Папка для сохранения результатов (default: benchmark_results/)"
    )
    parser.add_argument(
        "--top-k", type=int, default=settings.retriever_top_k,
        help=f"Количество чанков для retriever (default: {settings.retriever_top_k})"
    )
    parser.add_argument(
        "--judge-model", default=settings.llm_model,
        help=f"Модель Ollama для роли судьи (default: {settings.llm_model})"
    )
    args = parser.parse_args()

    run_benchmark(
        questions_path=args.questions,
        out_dir=args.out,
        top_k=args.top_k,
        judge_model=args.judge_model,
    )
