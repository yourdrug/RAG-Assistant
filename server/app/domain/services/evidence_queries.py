"""Bounded search probes derived from the question, without answer knowledge."""

import re

TABLE_MESSAGE_ALIASES = {"BLRDLN": "ЭТН", "BLRWBL": "ЭТТН"}


def message_types(text: str) -> set[str]:
    types = {
        alias for code, alias in TABLE_MESSAGE_ALIASES.items() if re.search(rf'\b{code}\b', text.upper())
    }
    types.update(re.findall(r'\b(?:ЭТТН|ЭТН)\b', text.upper()))
    return types


def listed_goods_polarities(text: str) -> set[bool]:
    return {
        not bool(match[1])
        for match in re.finditer(r"\b(не\s+)?включ[её]н\w*\s+в\s+переч", text, re.IGNORECASE)
    }


def contradicts_requested_scope(question: str, text: str) -> bool:
    requested, actual = listed_goods_polarities(question), listed_goods_polarities(text)
    return len(requested) == len(actual) == 1 and requested != actual


def table_message_matches(question: str, text: str) -> bool:
    requested = message_types(question)
    actual = set(re.findall(r"Теги\s+(ЭТТН|ЭТН)\b", text, re.IGNORECASE))
    codes = {
        alias for code, alias in TABLE_MESSAGE_ALIASES.items() if re.search(rf'\b{code}\b', text.upper())
    }
    actual.update(codes)
    requested_table = re.search(r'таблиц\w*\s+(\d+(?:\.\d+)*)', question, re.IGNORECASE)
    actual_tables = set(re.findall(r'таблиц\w*\s+(\d+(?:\.\d+)*)', text, re.IGNORECASE))
    if requested_table and actual_tables and actual_tables != {requested_table[1]}:
        return False
    return not requested or not actual or requested == {value.upper() for value in actual}


def table_scope_confirmed(question: str, text: str) -> bool:
    """Require positive evidence of the requested table and message identity."""
    if not table_message_matches(question, text):
        return False
    table = re.search(r'таблиц\w*\s+(\d+(?:\.\d+)*)', question, re.IGNORECASE)
    if table and set(re.findall(r'таблиц\w*\s+(\d+(?:\.\d+)*)', text, re.IGNORECASE)) != {table[1]}:
        return False
    requested = message_types(question)
    if requested and message_types(text) != requested:
        return False
    return True


def evidence_search_queries(question: str) -> list[str]:
    queries = [question]
    table = re.search(r"таблиц\w*\s+(\d+(?:\.\d+)*)", question, re.IGNORECASE)
    if table:
        # A long comparison query can retrieve only the more salient field.
        # Search each named field independently, retaining the original query.
        for match in re.finditer(r"пол[еяю]\s+«([^»]+)»\s*\(позици[яи]\s+(\d+)", question, re.IGNORECASE):
            queries.append(f"Таблица {table[1]}, поле {match[2]} «{match[1]}»")
    if re.search(r"режим|периодич|как часто", question, re.IGNORECASE):
        # The category's long description can dominate both dense and sparse
        # search. Search the requested action separately; the original query
        # and the answer-reading rules still carry the category and negation.
        probe = re.sub(
            r"о\s+товарах,?\s+(?:не\s+)?включ[её]н\w*\s+в\s+переч\w*[^,?]*,?",
            "",
            question,
            flags=re.IGNORECASE,
        )
        if probe != question:
            queries.append(" ".join(probe.split()))
    return list(dict.fromkeys(queries))[:4]
