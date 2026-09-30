"""Инструмент MCP в том виде, в каком с ним работает агент.

Сервер отдаёт инструмент как запись протокола: имя, заголовок, описание,
JSON-схема входа (и, если есть, выхода) и пометки поведения. Здесь эта запись
превращается в то, что нужно показывать и проверять:

  * параметры списком — имя, тип, обязательность, описание;
  * доступ словами — «только чтение», «меняет данные» — по пометкам сервера;
  * решение фильтра из mcp-servers.json — разрешён ли инструмент агенту;
  * цена в токенах — сколько места займёт его описание в запросе к модели.

Про пометки. readOnlyHint, destructiveHint и прочие — подсказки, а не
гарантия: спецификация прямо требует не доверять им, если сервер не проверен.
Поэтому решает фильтр в файле, а пометки только объясняют, что закрывать. И
если пометок нет вовсе, инструмент считается способным менять данные — так
велит спецификация (destructiveHint по умолчанию истинно).

Про цену. Модель видит инструменты через поле tools запроса, и описание каждого
уходит в КАЖДЫЙ запрос, пока инструмент подключён. Отсюда и «квадратичный рост»
из лекции: схема не разовый расход, а налог на всю переписку. Оценка делается
по тому же формату, в котором инструменты отдаются модели (function calling в
стиле OpenAI — его понимают и DeepSeek, и Groq, и z.ai).

Публичное API:
  Tool.from_sdk(tool, server)  — из записи SDK
  Tool.parameters()            — параметры списком
  Tool.for_model()             — описание для поля tools запроса к модели
  Tool.tokens()                — оценка его цены в токенах
  model_name(server, tool)     — имя функции для модели: «сервер__инструмент»
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from agent import tokens as token_counter

ТОЛЬКО_ЧТЕНИЕ = "только чтение"
МЕНЯЕТ = "меняет данные"
МОЖЕТ_УДАЛИТЬ = "может удалять"
НЕ_ЗАЯВЛЕНО = "не заявлено"

# Имя функции у OpenAI-совместимых API: латиница, цифры, «_» и «-», до 64.
_НЕДОПУСТИМОЕ = re.compile(r"[^A-Za-z0-9_-]")
РАЗДЕЛИТЕЛЬ = "__"


def model_name(server: str, tool: str) -> str:
    """Имя, под которым модель увидит инструмент.

    Имя сервера в него входит, потому что у разных серверов бывают одноимённые
    инструменты (search у GitHub и у файловой системы), а модели нужен плоский
    список без повторов.
    """
    имя = f"{server}{РАЗДЕЛИТЕЛЬ}{_НЕДОПУСТИМОЕ.sub('_', tool)}"
    return имя[:64]


def _тип(схема: dict[str, Any]) -> str:
    if not isinstance(схема, dict):
        return "?"
    тип = схема.get("type")
    if isinstance(тип, list):
        return " | ".join(str(т) for т in тип)
    if тип == "array":
        элемент = _тип(схема.get("items", {}))
        return f"array<{элемент}>" if элемент != "?" else "array"
    if тип:
        return str(тип)
    for ключ in ("anyOf", "oneOf"):
        if isinstance(схема.get(ключ), list):
            варианты = [_тип(в) for в in схема[ключ]]
            return " | ".join(в for в in варианты if в != "null") or "?"
    if "enum" in схема:
        return "enum"
    return "?"


@dataclass
class Parameter:
    имя: str
    тип: str
    обязательный: bool
    описание: str = ""
    по_умолчанию: Any = None
    варианты: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "имя": self.имя, "тип": self.тип, "обязательный": self.обязательный,
            "описание": self.описание, "по_умолчанию": self.по_умолчанию,
            "варианты": list(self.варианты),
        }


@dataclass
class Tool:
    """Инструмент одного сервера вместе с решением фильтра."""

    сервер: str
    имя: str
    описание: str = ""
    заголовок: str = ""
    входная_схема: dict[str, Any] = field(default_factory=dict)
    выходная_схема: dict[str, Any] | None = None
    пометки: dict[str, Any] = field(default_factory=dict)
    разрешён: bool = True
    почему_закрыт: str = ""

    @classmethod
    def from_sdk(cls, tool: Any, server: Any) -> "Tool":
        """Из записи SDK (mcp_types.Tool) и сервера, чей фильтр к ней применить."""
        пометки = {}
        if getattr(tool, "annotations", None) is not None:
            пометки = tool.annotations.model_dump(by_alias=True, exclude_none=True)
        разрешён, почему = server.доступ(tool.name)
        return cls(
            сервер=server.имя,
            имя=tool.name,
            описание=(tool.description or "").strip(),
            заголовок=(tool.title or пометки.get("title") or "").strip(),
            входная_схема=dict(tool.input_schema or {}),
            выходная_схема=dict(tool.output_schema) if tool.output_schema else None,
            пометки=пометки,
            разрешён=разрешён,
            почему_закрыт=почему,
        )

    # --- что показывать -------------------------------------------------------

    def parameters(self) -> list[Parameter]:
        свойства = self.входная_схема.get("properties") or {}
        обязательные = set(self.входная_схема.get("required") or [])
        итог = []
        for имя, схема in свойства.items():
            схема = схема if isinstance(схема, dict) else {}
            итог.append(Parameter(
                имя=имя,
                тип=_тип(схема),
                обязательный=имя in обязательные,
                описание=str(схема.get("description", "")).strip(),
                по_умолчанию=схема.get("default"),
                варианты=list(схема.get("enum") or []),
            ))
        # Обязательные — вперёд: при чтении списка их ищут в первую очередь.
        итог.sort(key=lambda п: not п.обязательный)
        return итог

    @property
    def доступ(self) -> str:
        """Что инструмент делает с данными — по пометкам сервера."""
        if not self.пометки:
            return НЕ_ЗАЯВЛЕНО
        if self.пометки.get("readOnlyHint") is True:
            return ТОЛЬКО_ЧТЕНИЕ
        # По спецификации destructiveHint по умолчанию истинно: пока сервер не
        # сказал обратного, инструмент, который пишет, может и удалить.
        if self.пометки.get("destructiveHint", True) is False:
            return МЕНЯЕТ
        return МОЖЕТ_УДАЛИТЬ

    @property
    def полное_имя(self) -> str:
        return model_name(self.сервер, self.имя)

    # --- что отдавать модели --------------------------------------------------

    def for_model(self) -> dict[str, Any]:
        """Описание в формате function calling — ровно то, что уйдёт в запрос."""
        описание = self.описание
        if self.заголовок and self.заголовок not in описание:
            описание = f"{self.заголовок}. {описание}".strip(". ").strip()
        return {
            "type": "function",
            "function": {
                "name": self.полное_имя,
                "description": описание,
                "parameters": self.входная_схема or {"type": "object", "properties": {}},
            },
        }

    def tokens(self) -> int:
        return token_counter.estimate(json.dumps(self.for_model(), ensure_ascii=False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "сервер": self.сервер,
            "имя": self.имя,
            "полное_имя": self.полное_имя,
            "заголовок": self.заголовок,
            "описание": self.описание,
            "параметры": [п.to_dict() for п in self.parameters()],
            "входная_схема": self.входная_схема,
            "выходная_схема": self.выходная_схема,
            "пометки": self.пометки,
            "доступ": self.доступ,
            "разрешён": self.разрешён,
            "почему_закрыт": self.почему_закрыт,
            "токенов": self.tokens(),
        }
