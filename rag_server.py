#!/usr/bin/env python3
"""MCP-сервер индекса документов: поиск по смыслу со ссылкой на источник.

Четвёртый свой сервер проекта. Индекс собирает человек (консоль или страница:
это минуты работы эмбеддера и OCR, а не вызов на полсекунды), а сервер даёт
агенту только читать: искать фрагменты, смотреть чанк целиком, видеть, что
проиндексировано. Каждая находка несёт паспорт — файл, раздел, страницы, номер
пункта, chunk_id. Из него агент следующих дней недели соберёт ссылку на
источник, а реранкинг — второй круг отбора.

Лекция недели разводит RAG и MCP: RAG — про собственные знания, MCP — про
внешние системы. Здесь они встречаются: знания лежат в своём индексе, а
доступ к ним агент получает тем же путём, что и к трекеру и памяти, — и
маршрутизатор Дня 20 выбирает этот сервер по темам, когда вопрос о договорах
и бюджетах.

Запуск:
    python rag_server.py                   # stdio: так его запускает клиент
    python rag_server.py --http 8770       # Streamable HTTP на 127.0.0.1:8770/mcp
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Annotated, Any

from pydantic import Field

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, КОРЕНЬ)

from mcp.server import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp_types import ToolAnnotations  # noqa: E402

from agent.rag.chunking import СТРАТЕГИИ, СТРУКТУРА  # noqa: E402
from agent.rag.embed import ОшибкаЭмбеддера  # noqa: E402
from agent.rag.indexer import Индексатор  # noqa: E402
from agent.rag.store import ОшибкаИндекса  # noqa: E402

ИМЯ = "docs"
ВЕРСИЯ = "21.0"

ИНСТРУКЦИИ = """\
Индекс документов организации: договоры (Ростелеком, Вымпелком), бюджетные \
формы по ИТ, заявки на лицензии и услуги. search_docs ищет по смыслу и \
возвращает фрагменты с источником — файлом, разделом, страницами и chunk_id. \
Отвечая по документам, называйте источник из находки; если находок нет или они \
не о том, так и скажите, а не отвечайте из общих знаний."""

ЧТЕНИЕ = ToolAnnotations(read_only_hint=True, destructive_hint=False,
                         idempotent_hint=True, open_world_hint=False)

# Столько символов текста чанка отдавать в находке. Целый чанк — до 700
# токенов; пять таких в ответе инструмента — лишние тысячи токенов на каждый
# круг разговора. Полный текст — через get_chunk по chunk_id.
ПРЕДЕЛ_ТЕКСТА = 700


class ОшибкаИнструмента(ToolError):
    """Только текст ToolError доходит до модели — остальное SDK прячет."""


Запрос = Annotated[str, Field(
    description="Вопрос или слова для поиска по смыслу: «срок оплаты услуг связи».", min_length=2)]
Стратегия = Annotated[str, Field(
    description="Какой индекс: «структура» (чанк = раздел документа) или «фикс» "
                "(чанки по 500 токенов с перекрытием). По умолчанию «структура».")]
Сколько = Annotated[int, Field(description="Сколько находок вернуть, 1–20.", ge=1, le=20)]
Источник = Annotated[str, Field(
    description="Искать только в файлах, в имени которых есть эта строка, например «VPN». "
                "Пусто — во всех.")]
ИдЧанка = Annotated[str, Field(description="chunk_id из находки search_docs.", min_length=3)]
Соседи = Annotated[bool, Field(description="Добавить соседние чанки — видно, где прошла граница.")]


def создать_сервер(индексатор: Индексатор | None = None) -> MCPServer:
    """Индексатор можно подменить — так проверки работают на хэш-эмбеддере."""
    сервер = MCPServer(ИМЯ, title="Индекс документов", version=ВЕРСИЯ,
                       instructions=ИНСТРУКЦИИ, log_level="WARNING")
    сервер.индексатор = индексатор

    def рабочий() -> Индексатор:
        # Индексатор создаётся при первом вызове, а не при запуске: так сервер
        # поднимается и отвечает на рукопожатие, даже когда Ollama ещё не запущена.
        if сервер.индексатор is None:
            try:
                сервер.индексатор = Индексатор()
            except ОшибкаЭмбеддера as exc:
                raise ОшибкаИнструмента(str(exc)) from exc
        return сервер.индексатор

    def стратегия_или_ошибка(стратегия: str) -> str:
        стратегия = (стратегия or СТРУКТУРА).strip()
        if стратегия not in СТРАТЕГИИ:
            raise ОшибкаИнструмента(
                f"Нет стратегии «{стратегия}». Есть: {', '.join(СТРАТЕГИИ)}")
        return стратегия

    @сервер.tool(title="Поиск по документам", annotations=ЧТЕНИЕ)
    def search_docs(query: Запрос, strategy: Стратегия = СТРУКТУРА, limit: Сколько = 5,
                    source: Источник = "") -> dict[str, Any]:
        """Ищет в индексе документов фрагменты, близкие по смыслу к запросу.

        Каждая находка — текст фрагмента и его паспорт: source (файл), title,
        section (раздел документа), pages, clause (номер пункта), chunk_id и
        score — косинусное сходство. Ссылаясь на документ, называйте source,
        section и pages."""
        с = стратегия_или_ошибка(strategy)
        try:
            находки = рабочий().найти(query, с, limit, source)
        except (ОшибкаИндекса, ОшибкаЭмбеддера) as exc:
            raise ОшибкаИнструмента(str(exc)) from exc
        return {"query": query, "strategy": с, "found": len(находки),
                "hits": [н.кратко(ПРЕДЕЛ_ТЕКСТА) for н in находки]}

    @сервер.tool(title="Сравнить стратегии на запросе", annotations=ЧТЕНИЕ)
    def compare_strategies(query: Запрос, limit: Сколько = 3) -> dict[str, Any]:
        """Один запрос в оба индекса — «фикс» и «структура» — бок о бок: что каждая
        стратегия ставит на первые места и из каких разделов."""
        try:
            итог = рабочий().найти_везде(query, limit)
        except (ОшибкаИндекса, ОшибкаЭмбеддера) as exc:
            raise ОшибкаИнструмента(str(exc)) from exc
        return {"query": query, "strategies": {
            с: [н.кратко(300) for н in находки] for с, находки in итог.items()}}

    @сервер.tool(title="Фрагмент целиком", annotations=ЧТЕНИЕ)
    def get_chunk(chunk_id: ИдЧанка, neighbors: Соседи = False) -> dict[str, Any]:
        """Полный текст фрагмента по chunk_id и его паспорт; по желанию — соседи."""
        хранилище = рабочий().хранилище
        чанк = хранилище.чанк(chunk_id.strip())
        if чанк is None:
            raise ОшибкаИнструмента(f"Фрагмента {chunk_id} нет в индексе")
        итог: dict[str, Any] = {**чанк.паспорт(), "text": чанк.текст}
        if neighbors:
            до, после = хранилище.соседи(чанк)
            итог["previous"] = {**до.паспорт(), "text": до.текст} if до else None
            итог["next"] = {**после.паспорт(), "text": после.текст} if после else None
        return итог

    @сервер.tool(title="Документы в индексе", annotations=ЧТЕНИЕ)
    def list_documents(strategy: Стратегия = СТРУКТУРА) -> dict[str, Any]:
        """Какие файлы проиндексированы: название, страницы, сколько чанков, какие
        страницы прошли OCR и что пропущено."""
        с = стратегия_или_ошибка(strategy)
        документы = рабочий().хранилище.документы(с)
        return {"strategy": с, "documents": [
            {"source": д["источник"], "title": д["название"], "kind": д["вид"],
             "pages": д["страниц"], "chunks": д["чанков"], "ocr_pages": д["распознано"],
             "notes": д["замечания"]} for д in документы]}

    @сервер.tool(title="Состояние индекса", annotations=ЧТЕНИЕ)
    def index_stats() -> dict[str, Any]:
        """Какие индексы собраны, каким эмбеддером, когда и сколько в них чанков."""
        индексы = рабочий().хранилище.индексы()
        return {"indexes": [
            {"strategy": и["стратегия"], "embedder": и["эмбеддер"], "dim": и["размерность"],
             "params": и["параметры"], "built": и["создан"], "documents": и["документов"],
             "chunks": и["чанков"], "tokens": и["токенов"]} for и in индексы]}

    return сервер


def main() -> int:
    разбор = argparse.ArgumentParser(description="MCP-сервер индекса документов.")
    разбор.add_argument("--http", type=int, default=0, metavar="ПОРТ",
                        help="слушать Streamable HTTP на 127.0.0.1:ПОРТ/mcp вместо stdio")
    разбор.add_argument("--индекс-в", dest="index", default="", metavar="ПУТЬ",
                        help="каталог индекса (по умолчанию index/)")
    разбор.add_argument("--эмбеддер", dest="embedder", default="", metavar="ИМЯ",
                        help="ollama[:модель] или хэш (по умолчанию ollama)")
    аргументы = разбор.parse_args()
    if аргументы.index:
        os.environ["RAG_DIR"] = os.path.abspath(аргументы.index)
    if аргументы.embedder:
        os.environ["RAG_EMBEDDER"] = аргументы.embedder
    сервер = создать_сервер()
    if аргументы.http:
        сервер.run("streamable-http", host="127.0.0.1", port=аргументы.http)
    else:
        сервер.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
