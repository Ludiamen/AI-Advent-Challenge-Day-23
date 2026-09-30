#!/usr/bin/env python3
"""MCP-сервер вокруг API трекера задач: задачи миграции глазами модели.

Это и есть «первый инструмент MCP» этого дня. Сервер ничего не хранит: он
ходит в чужой HTTP API и переводит его ответы в инструменты, которыми может
пользоваться модель. API — Яндекс.Трекер версии v3; под тем же кодом работают
два адреса:

  * мок из этого репозитория (tracker_api.py) — с ним день проверяется офлайн,
    его не жалко испортить и пересоздать;
  * настоящий https://api.tracker.yandex.net — те же пути и те же поля.

Разницы в коде между ними нет вовсе: меняются три переменные окружения
(TRACKER_URL, TRACKER_TOKEN, TRACKER_ORG), и они задаются в mcp-servers.json
для каждого сервера отдельно. Это не случайность, а проверка честности: если бы
сервер «знал» про мок, он не был бы сервером вокруг API.

Три вещи, которые требует задание дня, живут здесь:

  1. Регистрация инструмента — декоратор @сервер.tool с заголовком и пометками
     поведения (readOnlyHint, destructiveHint): по ним клиент понимает, что
     инструмент делает с данными, ещё до вызова.
  2. Описание входных параметров — аннотации Annotated[...] + Field(description=…),
     из которых SDK строит JSON-схему. Модель выбирает инструмент по описанию,
     поэтому описание — это код, а не комментарий.
  3. Возврат результата — словарь, который SDK отдаёт как structuredContent.
     Отдаётся не весь ответ API, а выжимка: ответ Трекера на одну задачу — это
     под три тысячи символов, из которых модели нужны полторы сотни. Лишнее
     здесь — деньги в каждом следующем запросе.

Запись отделена от чтения. Инструменты add_comment и move_issue регистрируются
только когда разрешено явно (TRACKER_WRITE=1 или ключ --запись). Поэтому у
настоящего Трекера список инструментов по умолчанию короче: испортить чужие
задачи случайным вызовом нельзя. Подтверждение человеком — вторым рубежом, уже
на стороне агента (agent/mcp/toolbox.py и заявки в памяти).

Запуск:
    python tracker_server.py                 # stdio: так его запускает клиент
    python tracker_server.py --http 8767     # Streamable HTTP на 127.0.0.1:8767/mcp
    python tracker_server.py --адрес https://api.tracker.yandex.net --только-чтение
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Annotated, Any

import httpx
from pydantic import Field

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, КОРЕНЬ)

from mcp.server import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp_types import ToolAnnotations  # noqa: E402

ИМЯ = "tracker"
ВЕРСИЯ = "17.0"

АДРЕС_МОКА = "http://127.0.0.1:8766"
ТОКЕН_МОКА = "mock-token"
ОРГАНИЗАЦИЯ_МОКА = "mock-org"
ЗАГОЛОВОК_ОРГАНИЗАЦИИ = "X-Cloud-Org-ID"
ТАЙМАУТ = 20.0

# Сколько символов описания задачи отдавать модели. Описания в трекере бывают на
# страницу; модели для выбора действия хватает начала, а платит она за всё.
ОПИСАНИЕ = 700
КОММЕНТАРИЙ = 400

# Ключ задачи: очередь латиницей, дефис, номер. По этому же правилу их выдаёт и
# мок, и настоящий Трекер.
_КЛЮЧ = re.compile(r"^[A-Z][A-Z0-9_]{0,20}-\d{1,9}$")

ИНСТРУКЦИИ = """\
Трекер задач проекта миграции ГИС газовых сетей (PHP5/CodeIgniter/MapServer → \
Django/GeoDjango/PostGIS/OpenLayers). Задачи лежат в очередях; у задачи есть \
ключ вида MIG-2, тема, статус (open, inProgress, testing, closed), исполнитель \
и комментарии. Начинайте с list_issues: он отдаёт короткие карточки. \
Подробности одной задачи, её комментарии и доступные статусы — get_issue. \
Меняющие инструменты, если они есть в списке, вызывайте только по явной \
просьбе человека."""

ЧТЕНИЕ = ToolAnnotations(read_only_hint=True, destructive_hint=False,
                         idempotent_hint=True, open_world_hint=True)
# Комментарий добавляется, ничего не затирая; перевод статуса меняет состояние
# задачи, и повторный вызов уже не пройдёт — значит, не идемпотентен.
ЗАПИСЬ_МЯГКАЯ = ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                idempotent_hint=False, open_world_hint=True)
ЗАПИСЬ_СТАТУСА = ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                 idempotent_hint=False, open_world_hint=True)


# Аннотация общая для нескольких инструментов, и объявлена она на уровне модуля
# намеренно: из-за «from __future__ import annotations» SDK разбирает подписи
# функций по именам в глобальной области, и локальный синоним он бы не нашёл.
Ключ = Annotated[str, Field(description="Ключ задачи в трекере, например «MIG-2».")]


class ОшибкаТрекера(ToolError):
    """Понятная причина вместо стека.

    Наследник ToolError из SDK: только его текст доходит до модели. Любое другое
    исключение клиент получит как «Error executing tool», и модель не узнает,
    что задачи просто нет или что кончился лимит запросов.
    """


def _да(значение: str) -> bool:
    return значение.strip().lower() in ("1", "да", "true", "yes", "on")


def _ключ(значение: str) -> str:
    """Проверяет ключ задачи до обращения к API.

    Ключ приходит от модели, а она может прислать что угодно — русские буквы,
    пробел, кусок пути. Без проверки это уходит прямо в адрес запроса: чужой
    сервер ответит невнятно, а в нашем случае HTTP-клиент ещё и падает на
    кириллице в пути. Ключ в трекере всегда вида «MIG-2»: латиница и номер.
    """
    очищенный = (значение or "").strip().upper()
    if not _КЛЮЧ.match(очищенный):
        raise ОшибкаТрекера(
            f"«{значение}» не похоже на ключ задачи. Ключ пишется латиницей: "
            "очередь, дефис, номер — например «MIG-2». Список задач — list_issues.")
    return очищенный


def _обрезать(текст: str, предел: int) -> str:
    текст = (текст or "").strip()
    if len(текст) <= предел:
        return текст
    return текст[:предел].rstrip() + f"… (всего {len(текст)} символов)"


class Трекер:
    """Клиент к API трекера: один метод на каждую нужную ручку.

    Здесь же перевод сбоев HTTP в причины, понятные человеку и модели. Это
    главная работа обёртки вокруг чужого API: без неё модель получает
    «Error executing tool» и начинает гадать, а с ней — «трекер не принял
    токен» и повод сказать об этом человеку.
    """

    def __init__(self, адрес: str = "", токен: str = "", организация: str = "",
                 заголовок_организации: str = "", таймаут: float = ТАЙМАУТ,
                 http: httpx.Client | None = None) -> None:
        self.адрес = (адрес or АДРЕС_МОКА).rstrip("/")
        self.токен = токен or ТОКЕН_МОКА
        self.организация = организация or ОРГАНИЗАЦИЯ_МОКА
        self.заголовок_организации = заголовок_организации or ЗАГОЛОВОК_ОРГАНИЗАЦИИ
        self.таймаут = таймаут
        self._свой_http = http
        self._http: httpx.Client | None = http

    @classmethod
    def из_окружения(cls) -> "Трекер":
        return cls(
            адрес=os.getenv("TRACKER_URL", ""),
            токен=os.getenv("TRACKER_TOKEN", ""),
            организация=os.getenv("TRACKER_ORG", ""),
            заголовок_организации=os.getenv("TRACKER_ORG_HEADER", ""),
            таймаут=float(os.getenv("TRACKER_TIMEOUT", ТАЙМАУТ)),
        )

    @property
    def настоящий(self) -> bool:
        return "api.tracker.yandex.net" in self.адрес

    @property
    def заголовки(self) -> dict[str, str]:
        return {
            "Authorization": f"OAuth {self.токен}",
            self.заголовок_организации: self.организация,
            "Content-Type": "application/json",
        }

    def _клиент(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self.таймаут)
        return self._http

    def _запрос(self, метод: str, путь: str, **аргументы: Any) -> tuple[Any, httpx.Headers]:
        адрес = путь if путь.startswith("http") else f"{self.адрес}{путь}"
        try:
            ответ = self._клиент().request(метод, адрес, headers=self.заголовки,
                                           timeout=self.таймаут, **аргументы)
        except httpx.ConnectError as exc:
            подсказка = ("Если это мок из репозитория, запустите его: "
                         "python tracker_api.py" if not self.настоящий else
                         "Проверьте сеть и адрес TRACKER_URL.")
            raise ОшибкаТрекера(f"Трекер не отвечает по адресу {self.адрес}. {подсказка}") from exc
        except httpx.TimeoutException as exc:
            raise ОшибкаТрекера(
                f"Трекер не ответил за {self.таймаут:g} с ({метод} {путь}).") from exc
        except UnicodeEncodeError as exc:
            # Токен и идентификатор организации уходят в HTTP-заголовки, а те
            # допускают только ASCII. Кириллица попадает туда из .env, когда
            # вместо токена вписали пояснение или промахнулись раскладкой.
            raise ОшибкаТрекера(
                "В токене или идентификаторе организации есть не-латинские символы. "
                "HTTP-заголовки допускают только ASCII: проверьте TRACKER_TOKEN и "
                "TRACKER_ORG в .env.") from exc
        except httpx.InvalidURL as exc:
            raise ОшибкаТрекера(f"Неверный адрес трекера «{self.адрес}»: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ОшибкаТрекера(f"Сбой связи с трекером: {exc}") from exc

        if ответ.is_success:
            данные = ответ.json() if ответ.content else None
            return данные, ответ.headers
        raise self._ошибка(ответ)

    def _ошибка(self, ответ: httpx.Response) -> ОшибкаТрекера:
        # Трекер объясняет отказ в errorMessages — это лучшее, что можно
        # показать: там и «нет такого перехода», и «поле обязательно».
        пояснение = ""
        try:
            тело = ответ.json()
            сообщения = тело.get("errorMessages") or []
            пояснение = "; ".join(str(с) for с in сообщения)
            if not пояснение and тело.get("errors"):
                пояснение = "; ".join(f"{к}: {з}" for к, з in тело["errors"].items())
        except ValueError:
            пояснение = ответ.text[:200]

        код = ответ.status_code
        если = {
            401: (f"Трекер не принял токен (401). Проверьте TRACKER_TOKEN в .env: "
                  f"он задан, не истёк и выдан для этой организации."),
            403: (f"Трекер отказал в доступе (403). Проверьте TRACKER_ORG и заголовок "
                  f"организации: сейчас «{self.заголовок_организации}», у Яндекс 360 — X-Org-ID."),
            404: "В трекере нет такого объекта (404).",
            422: "Трекер отклонил запрос (422).",
        }.get(код)
        if код == 429:
            пауза = ответ.headers.get("Retry-After", "")
            return ОшибкаТрекера(
                "Трекер ограничил частоту запросов (429)."
                + (f" Повторите через {пауза} с." if пауза else " Повторите позже."))
        if если is None:
            если = f"Трекер ответил ошибкой {код}."
        return ОшибкаТрекера(f"{если} {пояснение}".strip())

    # --- ручки API ------------------------------------------------------------

    def профиль(self) -> dict[str, Any]:
        данные, _ = self._запрос("GET", "/v3/myself")
        return данные or {}

    def очереди(self, сколько: int = 50) -> list[dict[str, Any]]:
        данные, _ = self._запрос("GET", "/v3/queues", params={"perPage": сколько})
        return данные or []

    def поиск(self, фильтр: dict[str, Any], запрос: str = "",
              сколько: int = 20, страница: int = 1) -> tuple[list[dict[str, Any]], int]:
        тело: dict[str, Any] = {"filter": фильтр}
        if запрос:
            тело["query"] = запрос
        данные, заголовки = self._запрос(
            "POST", "/v3/issues/_search", json=тело,
            params={"perPage": сколько, "page": страница})
        всего = заголовки.get("X-Total-Count")
        return (данные or []), int(всего) if всего and всего.isdigit() else len(данные or [])

    def задача(self, ключ: str) -> dict[str, Any]:
        данные, _ = self._запрос("GET", f"/v3/issues/{ключ}")
        return данные or {}

    def комментарии(self, ключ: str) -> list[dict[str, Any]]:
        данные, _ = self._запрос("GET", f"/v3/issues/{ключ}/comments")
        return данные or []

    def добавить_комментарий(self, ключ: str, текст: str) -> dict[str, Any]:
        данные, _ = self._запрос("POST", f"/v3/issues/{ключ}/comments", json={"text": текст})
        return данные or {}

    def переходы(self, ключ: str) -> list[dict[str, Any]]:
        данные, _ = self._запрос("GET", f"/v3/issues/{ключ}/transitions")
        return данные or []

    def перевести(self, ключ: str, переход: str, примечание: str = "") -> list[dict[str, Any]]:
        тело = {"comment": примечание} if примечание else {}
        данные, _ = self._запрос(
            "POST", f"/v3/issues/{ключ}/transitions/{переход}/_execute", json=тело)
        return данные or []

    def close(self) -> None:
        if self._http is not None and self._свой_http is None:
            self._http.close()
        self._http = self._свой_http


# --- выжимки: что из ответа API вообще нужно модели -------------------------------

def _поле(значение: Any, ключ: str = "display") -> str:
    """Поле-ссылку API отдаёт объектом: {"key": …, "display": …}."""
    if isinstance(значение, dict):
        return str(значение.get(ключ) or значение.get("display") or значение.get("key") or "")
    return str(значение or "")


def _карточка(задача: dict[str, Any]) -> dict[str, Any]:
    """Короткая карточка задачи — то, чем отвечает список."""
    return {
        "ключ": задача.get("key", ""),
        "тема": задача.get("summary", ""),
        "статус": _поле(задача.get("status")),
        "статус_код": _поле(задача.get("status"), "key"),
        "очередь": _поле(задача.get("queue"), "key"),
        "исполнитель": _поле(задача.get("assignee")) or "не назначен",
        "приоритет": _поле(задача.get("priority")),
        "срок": задача.get("deadline") or "",
        "обновлена": задача.get("updatedAt", ""),
    }


def _подробно(задача: dict[str, Any]) -> dict[str, Any]:
    карточка = _карточка(задача)
    карточка.update({
        "описание": _обрезать(задача.get("description", ""), ОПИСАНИЕ),
        "тип": _поле(задача.get("type")),
        "метки": list(задача.get("tags") or []),
        "создана": задача.get("createdAt", ""),
        "ссылка": задача.get("self", ""),
    })
    return карточка


def _комментарий(запись: dict[str, Any]) -> dict[str, Any]:
    return {
        "автор": _поле(запись.get("createdBy")),
        "когда": запись.get("createdAt", ""),
        "текст": _обрезать(запись.get("text", ""), КОММЕНТАРИЙ),
    }


def _статусы_перехода(переходы: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"статус": _поле(п.get("to"), "key"), "называется": _поле(п.get("to")),
             "переход": str(п.get("id", ""))} for п in переходы]


def _найти_переход(переходы: list[dict[str, Any]], куда: str) -> dict[str, Any] | None:
    """Переход по ключу статуса, по его названию или по идентификатору самого перехода."""
    цель = (куда or "").strip().lower()
    for п in переходы:
        варианты = {_поле(п.get("to"), "key").lower(), _поле(п.get("to")).lower(),
                    str(п.get("id", "")).lower()}
        if цель in варианты:
            return п
    return None


# --- сервер и инструменты ---------------------------------------------------------

def создать_сервер(трекер: Трекер | None = None, запись: bool | None = None) -> MCPServer:
    """Собирает сервер. Меняющие инструменты появляются только с разрешением.

    Разрешение спрашивается один раз, при сборке, а не при вызове: инструмента,
    которого нет в tools/list, модель не попросит вовсе. Это дешевле любой
    проверки внутри вызова — и видно снаружи, обычным списком инструментов.
    """
    трекер = трекер or Трекер.из_окружения()
    писать = _да(os.getenv("TRACKER_WRITE", "")) if запись is None else запись
    очередь_по_умолчанию = os.getenv("TRACKER_QUEUE", "")

    сервер = MCPServer(ИМЯ, title="Трекер задач миграции", version=ВЕРСИЯ,
                       instructions=ИНСТРУКЦИИ, log_level="WARNING")
    сервер.трекер = трекер          # для тестов и для main(): чем сервер ходит в API

    @сервер.tool(title="Очереди трекера", annotations=ЧТЕНИЕ)
    def list_queues() -> dict[str, Any]:
        """Очереди (проекты) трекера: ключ, название и описание.

        С этого начинают, когда неизвестно, где искать задачу."""
        return {"очереди": [{"ключ": о.get("key", ""), "название": о.get("name", ""),
                             "описание": _обрезать(о.get("description", ""), 200)}
                            for о in трекер.очереди()]}

    @сервер.tool(title="Список задач", annotations=ЧТЕНИЕ)
    def list_issues(
        queue: Annotated[str, Field(
            description="Ключ очереди, например «MIG». Пусто — искать во всех очередях.")] = "",
        status: Annotated[str, Field(
            description="Статус: open (открыта), inProgress (в работе), testing "
                        "(тестируется), closed (закрыта). Пусто — любой.")] = "",
        assignee: Annotated[str, Field(
            description="Исполнитель: имя как в трекере. Пусто — любой.")] = "",
        text: Annotated[str, Field(
            description="Искать эти слова в теме и описании задачи.")] = "",
        limit: Annotated[int, Field(
            description="Сколько задач вернуть, 1–50.", ge=1, le=50)] = 20,
    ) -> dict[str, Any]:
        """Задачи трекера короткими карточками: ключ, тема, статус, исполнитель, срок.

        Свежие сверху. Отвечает на вопросы «что в работе», «что просрочено»,
        «чем занят такой-то». Подробности одной задачи — get_issue."""
        фильтр: dict[str, Any] = {}
        очередь = queue or очередь_по_умолчанию
        if очередь:
            фильтр["queue"] = очередь
        if status:
            фильтр["status"] = status
        if assignee:
            фильтр["assignee"] = assignee
        задачи, всего = трекер.поиск(фильтр, запрос=text, сколько=limit)
        return {
            "найдено": всего,
            "показано": len(задачи),
            "фильтр": {"очередь": очередь or "все", "статус": status or "любой",
                       "исполнитель": assignee or "любой", "текст": text or ""},
            "задачи": [_карточка(з) for з in задачи],
        }

    @сервер.tool(title="Задача целиком", annotations=ЧТЕНИЕ)
    def get_issue(
        key: Ключ,
        comments: Annotated[int, Field(
            description="Сколько последних комментариев приложить, 0–20.", ge=0, le=20)] = 3,
    ) -> dict[str, Any]:
        """Одна задача целиком: описание, метки, сроки, последние комментарии и
        статусы, в которые её можно перевести из текущего."""
        key = _ключ(key)
        задача = трекер.задача(key)
        итог = _подробно(задача)
        if comments:
            последние = трекер.комментарии(key)[-comments:]
            итог["комментарии"] = [_комментарий(к) for к in последние]
        итог["можно_перевести_в"] = _статусы_перехода(трекер.переходы(key))
        return итог

    if not писать:
        return сервер

    @сервер.tool(title="Комментарий к задаче", annotations=ЗАПИСЬ_МЯГКАЯ)
    def add_comment(
        key: Ключ,
        text: Annotated[str, Field(
            description="Текст комментария. Пишите по делу: он останется в трекере "
                        "и его прочтут коллеги.", min_length=1)],
    ) -> dict[str, Any]:
        """Добавляет комментарий к задаче трекера. Меняет данные: вызывать только
        по явной просьбе человека."""
        key = _ключ(key)
        запись = трекер.добавить_комментарий(key, text)
        return {"задача": key, "комментарий": _комментарий(запись),
                "номер": запись.get("id", ""), "добавлен": True}

    @сервер.tool(title="Перевод задачи в статус", annotations=ЗАПИСЬ_СТАТУСА)
    def move_issue(
        key: Ключ,
        status: Annotated[str, Field(
            description="В какой статус перевести: open, inProgress, testing, closed. "
                        "Допустимые для задачи статусы показывает get_issue.")],
        comment: Annotated[str, Field(
            description="Необязательное пояснение — оно станет комментарием к задаче.")] = "",
    ) -> dict[str, Any]:
        """Переводит задачу в другой статус. Меняет данные: вызывать только по
        явной просьбе человека. Недопустимый для текущего статуса переход
        трекер отклонит."""
        key = _ключ(key)
        доступные = трекер.переходы(key)
        переход = _найти_переход(доступные, status)
        if переход is None:
            куда = ", ".join(с["статус"] for с in _статусы_перехода(доступные)) or "никуда"
            raise ОшибкаТрекера(
                f"Из текущего статуса задачу «{key}» можно перевести только в: {куда}. "
                f"Статус «{status}» недоступен.")
        трекер.перевести(key, str(переход.get("id", "")), comment)
        стало = трекер.задача(key)
        return {"задача": key, "статус": _поле(стало.get("status")),
                "статус_код": _поле(стало.get("status"), "key"),
                "переведена": True,
                "можно_перевести_в": _статусы_перехода(трекер.переходы(key))}

    return сервер


сервер = создать_сервер()


def main() -> int:
    global сервер
    разбор = argparse.ArgumentParser(
        description="MCP-сервер вокруг API трекера задач (мок или Яндекс.Трекер).")
    разбор.add_argument("--http", type=int, default=0, metavar="ПОРТ",
                        help="слушать Streamable HTTP на 127.0.0.1:ПОРТ/mcp вместо stdio")
    разбор.add_argument("--адрес", default="", metavar="URL",
                        help=f"адрес API трекера (по умолчанию {АДРЕС_МОКА} — мок)")
    разбор.add_argument("--запись", action="store_true",
                        help="разрешить меняющие инструменты (то же, что TRACKER_WRITE=1)")
    разбор.add_argument("--только-чтение", dest="readonly", action="store_true",
                        help="запретить меняющие инструменты, даже если TRACKER_WRITE=1")
    аргументы = разбор.parse_args()

    if аргументы.адрес:
        os.environ["TRACKER_URL"] = аргументы.адрес
    запись = None
    if аргументы.запись:
        запись = True
    if аргументы.readonly:
        запись = False
    сервер = создать_сервер(запись=запись)

    if аргументы.http:
        сервер.run("streamable-http", host="127.0.0.1", port=аргументы.http)
    else:
        сервер.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
