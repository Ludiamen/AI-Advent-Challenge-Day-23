#!/usr/bin/env python3
"""Свой MCP-сервер: состояние агента миграции, только для чтения.

Этот сервер отдаёт по протоколу MCP то, что агент хранит в memory/: задачи и
их стадии, инварианты, условия переходов, журнал решений. Подключить его может
не только наш агент, но и любой MCP-клиент — Claude Code, Claude Desktop,
MCP Inspector, — и тогда любая модель сможет спросить «на какой стадии перенос
схемы и что мешает перейти к реализации», не зная, как устроены наши файлы.

Все инструменты только читают, и это сказано в их пометках (readOnlyHint).
Пометка — обещание, а не защита, поэтому защита сделана кодом: сервер не
создаёт файлов и каталогов, а имена задачи и пользователя проверяются тем же
правилом, что и в самой памяти. Имя приходит от модели, и «../../.env» вместо
имени пользователя — ровно та «логическая бомба», о которой говорили в лекции.

Запуск:
    python mcp_server.py                       # stdio — так его запускает клиент
    python mcp_server.py --http 8765           # Streamable HTTP на 127.0.0.1:8765/mcp
    python mcp_server.py --память-в ПУТЬ       # другой каталог памяти

В mcp-servers.json он описан как {"command": "${PYTHON}", "args": [".../mcp_server.py"]}.
Руками запускать его обычно незачем: stdio-сервер молча ждёт JSON на входе.
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

from agent.invariants import Invariant, InvariantStore, merge as merge_invariants  # noqa: E402
from agent.memory.long import SAFE_NAME, DecisionsStore, ProfileStore  # noqa: E402
from agent.memory.working import TaskState, WorkingMemory, WorkingMemoryError  # noqa: E402
from agent.transitions import БАЗОВЫЕ, ConditionStore, Ворота, merge as merge_conditions  # noqa: E402

ИМЯ = "agent-state"
ВЕРСИЯ = "16.0"

ИНСТРУКЦИИ = """\
Сервер агента миграции ГИС газовых сетей (PHP5/CodeIgniter/MapServer → \
Django/GeoDjango/PostGIS/OpenLayers). Отдаёт состояние агента только на чтение: \
задачи рабочей памяти и их стадии (planning → execution → validation → done), \
инварианты проекта, условия переходов между стадиями, журнал решений. \
Начинайте с list_tasks; подробности задачи — get_task."""

# Пометки одни на все инструменты: читают локальные файлы, ничего не меняют,
# повторный вызов даёт тот же ответ, во внешний мир не ходят.
ЧТЕНИЕ = ToolAnnotations(read_only_hint=True, destructive_hint=False,
                         idempotent_hint=True, open_world_hint=False)

Пользователь = Annotated[str, Field(
    description="Имя пользователя агента (каталог memory/long/<имя>). По умолчанию «инженер».")]
Задача = Annotated[str, Field(
    description="Идентификатор задачи из list_tasks, например «перенос-моделей».")]


class ОшибкаЗапроса(ToolError):
    """Неверный аргумент инструмента.

    Наследник ToolError из SDK, и это важно: только его текст доходит до
    клиента (isError с сообщением, которое прочтёт модель). Любое другое
    исключение SDK считает падением и прячет за «Error executing tool …» —
    модель не узнала бы, что имя пользователя просто написано с ошибкой.
    """


def каталог_памяти() -> str:
    return os.path.abspath(os.getenv("MEMORY_DIR") or os.path.join(КОРЕНЬ, "memory"))


def _пользователь(имя: str) -> str:
    имя = (имя or "инженер").strip()
    if not SAFE_NAME.match(имя):
        raise ОшибкаЗапроса(
            f"Недопустимое имя пользователя «{имя}»: нужны буквы, цифры, дефис и "
            "подчёркивание. Список — list_users.")
    return os.path.join(каталог_памяти(), "long", имя)


def _задачи() -> WorkingMemory | None:
    # WorkingMemory создаёт свой каталог, а сервер только читает: если задач
    # ещё не было, каталога нет, и создавать его здесь незачем.
    путь = os.path.join(каталог_памяти(), "working")
    return WorkingMemory(путь) if os.path.isdir(путь) else None


def _условия(каталог: str) -> list:
    личные = ConditionStore(os.path.join(каталог, "transition-conditions.json")).all()
    return merge_conditions(list(БАЗОВЫЕ), личные)


сервер = MCPServer(
    ИМЯ,
    title="Состояние агента миграции",
    version=ВЕРСИЯ,
    instructions=ИНСТРУКЦИИ,
    log_level="WARNING",
)


@сервер.tool(title="Пользователи агента", annotations=ЧТЕНИЕ)
def list_users() -> dict[str, Any]:
    """Пользователи, у которых есть долговременная память: профиль, решения, условия."""
    корень = os.path.join(каталог_памяти(), "long")
    имена = sorted(и for и in os.listdir(корень)
                   if os.path.isdir(os.path.join(корень, и))) if os.path.isdir(корень) else []
    return {"память": каталог_памяти(), "пользователи": имена}


@сервер.tool(title="Задачи рабочей памяти", annotations=ЧТЕНИЕ)
def list_tasks() -> dict[str, Any]:
    """Все задачи агента со сводкой: стадия, шаг сценария, пауза и чего ждёт задача.

    Свежие сверху. Задача живёт в рабочей памяти, пока не завершена.
    """
    хранилище = _задачи()
    return {"задачи": хранилище.tasks() if хранилище else []}


@сервер.tool(title="Состояние задачи", annotations=ЧТЕНИЕ)
def get_task(task_id: Задача, user: Пользователь = "инженер") -> dict[str, Any]:
    """Полное состояние задачи: стадия, план, шаги, пауза, подпись под планом,
    отчёт валидации, история переходов и — по каждой стадии — открыт ли туда
    переход и каких условий не хватает."""
    хранилище = _задачи()
    if хранилище is None:
        raise ОшибкаЗапроса("Задач ещё нет: рабочая память пуста.")
    try:
        з: TaskState = хранилище.load(task_id)
    except WorkingMemoryError as exc:
        raise ОшибкаЗапроса(str(exc)) from None
    ворота = Ворота(lambda: _условия(_пользователь(user)))
    return {
        "task_id": з.task_id,
        "название": з.title,
        "стадия": з.stage,
        "стадия_словами": з.stage_label,
        "словами": з.состояние_словами,
        "план": list(з.plan),
        "шаги": [ш.to_dict() for ш in з.шаги],
        "пауза": з.пауза,
        "причина_паузы": з.причина_паузы,
        "ожидание": з.ожидание,
        "план_утверждён": dict(з.план_утверждён),
        "утверждение_актуально": з.утверждение_актуально,
        "отчёт_валидации": dict(з.отчёт_валидации),
        "переходы": [
            {"стадия": п["стадия"], "можно": п["можно"], "причина": п["причина"],
             "чего_не_хватает": п.get("чего_не_хватает", [])}
            for п in ворота.обзор(з) if not п["текущая"]
        ],
        "история_переходов": list(з.transitions),
        "обновлена": з.updated_at,
    }


@сервер.tool(title="Инварианты", annotations=ЧТЕНИЕ)
def list_invariants(user: Пользователь = "инженер") -> dict[str, Any]:
    """Инварианты проекта и личные инварианты пользователя: правило, вид, почему и
    что делать вместо. Нарушать их агенту нельзя ни в каком ответе."""
    проектные = InvariantStore(os.path.join(каталог_памяти(), "invariants.json")).all()
    профиль = ProfileStore(os.path.join(_пользователь(user), "profile.json")).load()
    личные = [Invariant.from_dict(и) for и in (профиль.get("инварианты") or [])]
    return {"инварианты": [и.to_dict() for и in merge_invariants(проектные, личные)]}


@сервер.tool(title="Условия переходов", annotations=ЧТЕНИЕ)
def list_transition_conditions(user: Пользователь = "инженер") -> dict[str, Any]:
    """Условия, без которых задача не перейдёт со стадии на стадию: базовые из кода
    и личные пользователя."""
    return {"условия": [у.to_dict() for у in _условия(_пользователь(user))]}


@сервер.tool(title="Журнал решений", annotations=ЧТЕНИЕ)
def list_decisions(
    user: Пользователь = "инженер",
    limit: Annotated[int, Field(description="Сколько последних решений вернуть, 1–100.",
                                ge=1, le=100)] = 20,
) -> dict[str, Any]:
    """Последние решения из журнала пользователя: что решили, почему и какие
    были альтернативы."""
    журнал = DecisionsStore(os.path.join(_пользователь(user), "decisions.jsonl"))
    return {"решения": журнал.recent(limit)}


def main() -> int:
    разбор = argparse.ArgumentParser(description="MCP-сервер состояния агента (только чтение).")
    разбор.add_argument("--http", type=int, default=0, metavar="ПОРТ",
                        help="слушать Streamable HTTP на 127.0.0.1:ПОРТ/mcp вместо stdio")
    разбор.add_argument("--память-в", dest="memory", default="", metavar="ПУТЬ",
                        help="каталог памяти агента (по умолчанию memory/ рядом со скриптом)")
    аргументы = разбор.parse_args()
    if аргументы.memory:
        os.environ["MEMORY_DIR"] = os.path.abspath(аргументы.memory)
    if аргументы.http:
        сервер.run("streamable-http", host="127.0.0.1", port=аргументы.http)
    else:
        сервер.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
