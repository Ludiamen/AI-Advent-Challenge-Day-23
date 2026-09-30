"""Соединение с MCP-сервером: рукопожатие, список инструментов, закрытие.

SDK асинхронный, а весь проект — синхронный: консоль, Flask, сценарии. Чтобы не
переписывать агента под asyncio, соединение живёт в отдельном потоке со своим
циклом событий (BlockingPortal из anyio), а наружу у него обычные методы.

Устройство соединения — одна долгоживущая задача, которая входит в контекст
клиента SDK, сообщает «готово» и ждёт команды закрыться. Это не прихоть: в
anyio контекст с группой задач обязан открываться и закрываться в одной и той
же задаче, и если открыть его из одного вызова, а закрыть из другого, SDK
падает на выходе. Зато соединение переживает любое число запросов: рукопожатие,
tools/list и сколько угодно вызовов инструментов подряд. Это не роскошь — на
запуск stdio-сервера уходит секунда-две, и поднимать его заново на каждый
вызов означало бы платить эту секунду за каждый ответ модели.

Что происходит при открытии, по протоколу:

  1. Транспорт. stdio — запускается процесс, сообщения идут строками JSON через
     его stdin/stdout, а stderr — это журнал сервера. HTTP — POST на адрес,
     ответ приходит JSON или потоком SSE, сессию держит заголовок Mcp-Session-Id.
  2. Рукопожатие. Клиент SDK сначала пробует server/discover (так устроена
     ревизия спецификации 2026-07-28), а если сервер его не знает — классическое
     initialize и уведомление notifications/initialized. Сервер отвечает версией
     протокола, своим именем и возможностями (tools, resources, prompts…).
  3. tools/list — постранично, пока сервер отдаёт nextCursor.
  4. tools/call — вызов инструмента по имени и аргументам. Ответ приходит
     списком блоков (текст, ресурс) и, если у инструмента объявлена выходная
     схема, ещё и структурой. Неудача инструмента — не исключение, а ответ с
     пометкой isError: модель должна её увидеть и исправиться.

Сбои приходят из SDK исключениями разной природы — FileNotFoundError, группы
исключений anyio, MCPError с текстом «Server returned an error response», —
которые человеку ничего не говорят. Здесь они переводятся в причину и подсказку:
«не найдена команда npx — установите Node.js», «сервер отклонил токен». Для
HTTP для этого ведётся журнал ответов: статус и заголовок WWW-Authenticate
говорят о причине больше, чем текст ошибки.

Публичное API:
  Session(server)            — соединение; with Session(s) as сеанс: …
  Session.tools()            — все инструменты сервера (с решением фильтра)
  Session.call_tool(имя, …)  -> ToolResult  — вызвать инструмент (tools/call)
  inspect(server)            -> Inspection  — открыть, получить список, закрыть
  Handshake, Inspection, ToolResult — что узнали и что получили
  MCPClientError             — соединение не удалось, с этапом и подсказкой
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import anyio
import httpx2
from anyio.from_thread import start_blocking_portal
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

from agent.mcp.config import HTTP, STDIO, Server
from agent.mcp.tools import Tool, model_name

log = logging.getLogger("agent.mcp")

# Этапы, на которых соединение может сорваться. По этапу понятно, куда смотреть:
# «запуск» — в команду из конфигурации, «рукопожатие» — в сеть и в токен,
# «список» — в сам сервер.
ЗАПУСК = "запуск"
РУКОПОЖАТИЕ = "рукопожатие"
СПИСОК = "список"
ВЫЗОВ = "вызов"
ЗАКРЫТИЕ = "закрытие"

# Сколько последних строк журнала сервера показывать при сбое.
СТРОК_ЖУРНАЛА = 12
# Сколько символов ответа инструмента доходит до модели. Ответ приходит от
# чужого сервера, и его длину мы не выбираем: один tools/call к файловому
# серверу способен вернуть мегабайт. В запрос к модели это уходит целиком и
# оплачивается как вход, поэтому предел ставится здесь, а не «когда-нибудь».
ПРЕДЕЛ_ОТВЕТА = 6000
# Защита от сервера, который отдаёт nextCursor бесконечно.
ПРЕДЕЛ_СТРАНИЦ = 100
# Сколько ждать, пока сервер закроется. Процесс stdio SDK гасит сам: сначала
# закрывает stdin, потом шлёт сигнал, — на это уходит до пары секунд.
ЖДАТЬ_ЗАКРЫТИЯ = 15.0


class MCPClientError(RuntimeError):
    """Соединение с сервером не удалось. Несёт этап, причину и подсказку."""

    def __init__(self, этап: str, причина: str, подсказка: str = "", журнал: str = "") -> None:
        super().__init__(причина)
        self.этап = этап
        self.причина = причина
        self.подсказка = подсказка
        self.журнал = журнал


@dataclass
class Handshake:
    """Что сервер сообщил о себе при рукопожатии."""

    протокол: str
    способ: str                      # server/discover или initialize
    имя: str = ""
    версия: str = ""
    заголовок: str = ""
    возможности: list[str] = field(default_factory=list)
    инструкции: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "протокол": self.протокол, "способ": self.способ, "имя": self.имя,
            "версия": self.версия, "заголовок": self.заголовок,
            "возможности": list(self.возможности), "инструкции": self.инструкции,
        }


@dataclass
class Inspection:
    """Итог осмотра одного сервера: соединение и список инструментов."""

    сервер: Server
    ок: bool = False
    пропущен: bool = False           # не подключались: сервер не настроен
    этап: str = ""                   # где сорвалось, если сорвалось
    ошибка: str = ""
    подсказка: str = ""
    журнал: str = ""                 # stderr сервера stdio (хвост)
    рукопожатие: Handshake | None = None
    инструменты: list[Tool] = field(default_factory=list)
    страниц: int = 0
    подключение_с: float = 0.0
    список_с: float = 0.0
    всего_с: float = 0.0

    @property
    def разрешено(self) -> list[Tool]:
        return [и for и in self.инструменты if и.разрешён]

    @property
    def токенов(self) -> int:
        """Сколько стоили бы в каждом запросе к модели разрешённые инструменты."""
        return sum(и.tokens() for и in self.разрешено)

    @property
    def токенов_всех(self) -> int:
        return sum(и.tokens() for и in self.инструменты)

    def to_dict(self) -> dict[str, Any]:
        return {
            "сервер": self.сервер.to_dict(),
            "ок": self.ок,
            "пропущен": self.пропущен,
            "этап": self.этап,
            "ошибка": self.ошибка,
            "подсказка": self.подсказка,
            "журнал": self.журнал,
            "рукопожатие": self.рукопожатие.to_dict() if self.рукопожатие else None,
            "инструменты": [и.to_dict() for и in self.инструменты],
            "всего_инструментов": len(self.инструменты),
            "разрешено": len(self.разрешено),
            "страниц": self.страниц,
            "токенов": self.токенов,
            "токенов_всех": self.токенов_всех,
            "подключение_с": round(self.подключение_с, 2),
            "список_с": round(self.список_с, 2),
            "всего_с": round(self.всего_с, 2),
        }


@dataclass
class ToolResult:
    """Что вернул вызов инструмента — в виде, пригодном и модели, и человеку.

    Сервер отвечает списком блоков (текст, картинка, ссылка на ресурс) и,
    если у инструмента есть выходная схема, ещё и структурой. Модели нужен
    один кусок текста, человеку — то же самое плюс время и аргументы, а коду
    проверок — структура. Здесь всё это лежит рядом.

    Отдельно про «ок». Сбой инструмента — не сбой программы: «нет такой
    задачи» приходит как обычный ответ с пометкой isError, и модель обязана
    его увидеть, чтобы исправиться. Поэтому неудачный вызов — такой же
    результат, просто с ок=False.
    """

    сервер: str
    инструмент: str
    полное_имя: str = ""
    аргументы: dict[str, Any] = field(default_factory=dict)
    ок: bool = True
    текст: str = ""
    данные: Any = None
    секунд: float = 0.0
    обрезан: bool = False
    блоков: int = 0

    def для_модели(self) -> str:
        """Текст, который уйдёт в сообщение роли tool."""
        return self.текст or ("(пусто)" if self.ок else "(ошибка без пояснения)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "сервер": self.сервер,
            "инструмент": self.инструмент,
            "полное_имя": self.полное_имя or self.инструмент,
            "аргументы": self.аргументы,
            "ок": self.ок,
            "текст": self.текст,
            "данные": self.данные,
            "секунд": round(self.секунд, 2),
            "обрезан": self.обрезан,
            "блоков": self.блоков,
        }


def _текст_ответа(итог: Any) -> tuple[str, int]:
    """Склеивает блоки ответа в текст. Не-текстовые блоки называются словами."""
    куски: list[str] = []
    блоки = list(getattr(итог, "content", None) or [])
    for блок in блоки:
        вид = getattr(блок, "type", "")
        if вид == "text":
            куски.append(getattr(блок, "text", ""))
        elif вид == "resource_link":
            куски.append(f"[ссылка на ресурс: {getattr(блок, 'uri', '')}]")
        elif вид == "resource":
            вложение = getattr(блок, "resource", None)
            куски.append(getattr(вложение, "text", None) or f"[ресурс {getattr(вложение, 'uri', '')}]")
        else:
            куски.append(f"[{вид or 'неизвестный блок'}]")
    return "\n".join(к for к in куски if к).strip(), len(блоки)


# --- перевод сбоев на человеческий язык ------------------------------------------

def _лист(exc: BaseException) -> BaseException:
    """Настоящая причина из-под групп исключений anyio."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _цепочка(exc: BaseException) -> list[BaseException]:
    итог, текущее = [], _лист(exc)
    while текущее is not None and текущее not in итог:
        итог.append(текущее)
        текущее = текущее.__cause__ or текущее.__context__
        if текущее is not None:
            текущее = _лист(текущее)
    return итог


def _объяснить(exc: BaseException, сервер: Server, ответы: list[dict[str, Any]],
               журнал: str) -> tuple[str, str]:
    """Причина и подсказка для исключения, вылетевшего из SDK."""
    цепочка = _цепочка(exc)
    корень = цепочка[0]

    if сервер.транспорт == STDIO:
        if isinstance(корень, FileNotFoundError):
            подсказка = ("Для npx нужен Node.js (https://nodejs.org)."
                         if сервер.команда in ("npx", "node") else
                         "Проверьте «command» в mcp-servers.json и что программа установлена.")
            return f"не найдена команда «{сервер.команда}»", подсказка
        if isinstance(корень, PermissionError):
            return f"нет прав на запуск «{сервер.команда}»", "Проверьте права на файл (chmod +x)."
        if isinstance(корень, MCPError) and "Connection closed" in str(корень):
            подсказка = ("Процесс завершился раньше, чем ответил. Причина обычно в "
                         "журнале сервера ниже." if журнал else
                         "Процесс завершился раньше, чем ответил, и ничего не написал в stderr.")
            return "сервер закрыл соединение, не ответив на рукопожатие", подсказка

    if сервер.транспорт == HTTP:
        for звено in цепочка:
            if isinstance(звено, UnicodeEncodeError):
                # HTTP-заголовки — только ASCII. Кириллица попадает туда из .env,
                # когда вместо токена вписали пояснение или опечатались в раскладке.
                return ("в заголовках запроса есть не-латинские символы",
                        "HTTP-заголовки допускают только ASCII: проверьте значение "
                        "в .env и «headers» в mcp-servers.json.")
            if isinstance(звено, httpx2.ConnectError):
                хост = urlsplit(сервер.адрес).netloc
                return (f"нет соединения с {хост}",
                        "Сервер не запущен, адрес неверен или нет сети.")
            if isinstance(звено, httpx2.TimeoutException):
                return (f"{urlsplit(сервер.адрес).netloc} не ответил вовремя",
                        "Сервер перегружен или недоступен из этой сети.")
        # Статус и WWW-Authenticate из журнала ответов — самое точное, что есть.
        отказы = [о for о in ответы if о["статус"] >= 400]
        if отказы:
            последний = отказы[-1]
            статус, вход = последний["статус"], последний["www_authenticate"]
            if статус in (401, 403) or "invalid_token" in вход:
                описание = ""
                if "error_description=" in вход:
                    описание = вход.split("error_description=", 1)[1].split('"')[1]
                return (f"сервер отклонил авторизацию (HTTP {статус}"
                        + (f": {описание}" if описание else "") + ")",
                        "Проверьте токен в .env: он задан, не истёк и выдан с нужными правами.")
            if статус == 404:
                return ("по этому адресу нет MCP-сервера (HTTP 404)",
                        "Проверьте «url»: у многих серверов путь заканчивается на /mcp.")
            return (f"сервер ответил HTTP {статус}",
                    "Сервер отказал в запросе — подробности в его документации.")

    if isinstance(корень, MCPError):
        код = getattr(getattr(корень, "error", None), "code", None)
        return (f"сервер ответил ошибкой протокола: {корень}" + (f" (код {код})" if код else ""),
                "Возможно, сервер поддерживает другую версию протокола.")
    if isinstance(корень, (TimeoutError, anyio.EndOfStream, anyio.ClosedResourceError)):
        return "сервер перестал отвечать", ""
    return f"{type(корень).__name__}: {корень}", ""


# --- соединение ------------------------------------------------------------------

async def _все_инструменты(клиент: Any) -> tuple[list[Any], int]:
    """tools/list постранично: сервер вправе отдавать список частями."""
    инструменты: list[Any] = []
    курсор: str | None = None
    страниц = 0
    while True:
        страница = await клиент.list_tools(cursor=курсор)
        страниц += 1
        инструменты.extend(страница.tools)
        курсор = страница.next_cursor
        if курсор is None or страниц >= ПРЕДЕЛ_СТРАНИЦ:
            return инструменты, страниц


class Session:
    """Открытое соединение с одним сервером. Синхронное снаружи."""

    def __init__(self, server: Server) -> None:
        self.server = server
        self.рукопожатие: Handshake | None = None
        self.подключение_с = 0.0
        self._портал_cm = None
        self._портал = None
        self._клиент: Any = None
        self._готово = threading.Event()
        self._стоп: anyio.Event | None = None
        self._область: anyio.CancelScope | None = None
        self._задача = None
        self._ошибка: BaseException | None = None
        self._журнал_файл = None
        self._ответы: list[dict[str, Any]] = []

    # --- жизнь соединения внутри цикла событий -----------------------------------

    def _цель(self) -> Any:
        """То, что получает Client из SDK: транспорт stdio или HTTP."""
        if self.server.транспорт == STDIO:
            # stderr сервера — его журнал. Пускать его в консоль нельзя: он
            # перемешается с выводом агента. Файл нужен настоящий, с fileno —
            # процесс пишет в него напрямую.
            self._журнал_файл = tempfile.TemporaryFile("w+", encoding="utf-8")
            параметры = StdioServerParameters(
                command=self.server.команда,
                args=self.server.аргументы,
                env=self.server.окружение or None,
                cwd=self.server.каталог or None,
            )
            return stdio_client(параметры, errlog=self._журнал_файл)

        async def записать(ответ: httpx2.Response) -> None:
            self._ответы.append({
                "метод": ответ.request.method,
                "статус": ответ.status_code,
                "www_authenticate": ответ.headers.get("www-authenticate", ""),
            })

        http = httpx2.AsyncClient(
            headers=self.server.заголовки or None,
            timeout=httpx2.Timeout(self.server.таймаут, read=max(self.server.таймаут, 60.0)),
            event_hooks={"response": [записать]},
        )
        self._http = http
        return streamable_http_client(self.server.адрес, http_client=http)

    async def _жить(self) -> None:
        self._стоп = anyio.Event()
        with anyio.CancelScope() as область:
            self._область = область
            try:
                цель = self._цель()
                if self.server.транспорт == HTTP:
                    async with self._http:
                        await self._в_контексте(цель)
                else:
                    await self._в_контексте(цель)
            except Exception as exc:  # отмена (BaseException) уходит в область
                self._ошибка = exc
            finally:
                self._готово.set()

    async def _в_контексте(self, цель: Any) -> None:
        async with Client(цель, client_info=_о_клиенте()) as клиент:
            self._клиент = клиент
            self.рукопожатие = _рукопожатие(клиент)
            self._готово.set()
            await self._стоп.wait()

    # --- синхронная сторона --------------------------------------------------------

    def open(self) -> Handshake:
        if self._клиент is not None:
            return self.рукопожатие
        начало = time.monotonic()
        self._портал_cm = start_blocking_portal()
        self._портал = self._портал_cm.__enter__()
        self._задача = self._портал.start_task_soon(self._жить)
        дождались = self._готово.wait(self.server.таймаут)
        self.подключение_с = time.monotonic() - начало

        if not дождались:
            self._погасить()
            подсказка = ("Первый запуск npx скачивает пакет и может занять минуты: "
                         "повторите или увеличьте «таймаут» в mcp-servers.json."
                         if self.server.команда == "npx" else
                         "Сервер запущен, но молчит. Увеличьте «таймаут» или проверьте сервер.")
            raise MCPClientError(РУКОПОЖАТИЕ,
                                 f"сервер не ответил за {self.server.таймаут:g} с",
                                 подсказка, self.журнал())
        if self._ошибка is not None or self._клиент is None:
            ошибка = self._ошибка or RuntimeError("соединение не установлено")
            журнал = self.журнал()
            self._погасить()
            причина, подсказка = _объяснить(ошибка, self.server, self._ответы, журнал)
            этап = (ЗАПУСК if isinstance(_лист(ошибка), (FileNotFoundError, PermissionError))
                    else РУКОПОЖАТИЕ)
            log.info("MCP %s: %s (%r)", self.server.имя, причина, ошибка)
            raise MCPClientError(этап, причина, подсказка, журнал) from ошибка
        return self.рукопожатие

    def _вызвать(self, функция: Any, *аргументы: Any, этап: str) -> Any:
        """Выполнить корутину клиента с таймаутом и понятной ошибкой."""
        if self._клиент is None:
            raise MCPClientError(этап, "соединение не открыто")

        async def с_таймаутом() -> Any:
            with anyio.fail_after(self.server.таймаут):
                return await функция(*аргументы)

        try:
            return self._портал.call(с_таймаутом)
        except TimeoutError:
            raise MCPClientError(этап, f"сервер не ответил за {self.server.таймаут:g} с",
                                 журнал=self.журнал()) from None
        except Exception as exc:
            причина, подсказка = _объяснить(exc, self.server, self._ответы, self.журнал())
            raise MCPClientError(этап, причина, подсказка, self.журнал()) from exc

    def tools(self) -> tuple[list[Tool], int]:
        """Все инструменты сервера и число страниц, которыми они пришли."""
        записи, страниц = self._вызвать(_все_инструменты, self._клиент, этап=СПИСОК)
        return [Tool.from_sdk(з, self.server) for з in записи], страниц

    def call_tool(self, имя: str, аргументы: dict[str, Any] | None = None) -> ToolResult:
        """Вызывает инструмент сервера — tools/call — и разбирает ответ.

        Соединение при этом переиспользуется: сервер уже запущен, рукопожатие
        уже было. Поэтому второй вызов в том же разговоре стоит миллисекунды, а
        не секунду на запуск процесса.
        """
        if self._клиент is None:
            raise MCPClientError(ВЫЗОВ, "соединение не открыто",
                                 "Откройте сеанс: with Session(server) as сеанс.")
        аргументы = dict(аргументы or {})
        начало = time.monotonic()
        итог = self._вызвать(self._клиент.call_tool, имя, аргументы, этап=ВЫЗОВ)
        текст, блоков = _текст_ответа(итог)
        обрезан = len(текст) > ПРЕДЕЛ_ОТВЕТА
        if обрезан:
            текст = (текст[:ПРЕДЕЛ_ОТВЕТА].rstrip()
                     + f"\n… ответ обрезан, всего {len(текст)} символов")
        результат = ToolResult(
            сервер=self.server.имя,
            инструмент=имя,
            полное_имя=model_name(self.server.имя, имя),
            аргументы=аргументы,
            ок=not bool(getattr(итог, "is_error", False)),
            текст=текст,
            данные=getattr(итог, "structured_content", None),
            секунд=time.monotonic() - начало,
            обрезан=обрезан,
            блоков=блоков,
        )
        log.info("MCP %s: вызов %s за %.2f с, %s", self.server.имя, имя,
                 результат.секунд, "ок" if результат.ок else "с ошибкой")
        return результат

    def журнал(self) -> str:
        """Хвост stderr сервера stdio — то, что он сам написал о себе."""
        if self._журнал_файл is None:
            return ""
        try:
            self._журнал_файл.flush()
            self._журнал_файл.seek(0)
            строки = self._журнал_файл.read().strip().splitlines()
        except (OSError, ValueError):
            return ""
        return "\n".join(строки[-СТРОК_ЖУРНАЛА:])

    def _погасить(self) -> None:
        """Остановить задачу соединения и цикл событий, что бы ни случилось."""
        if self._портал is not None:
            try:
                if self._стоп is not None:
                    self._портал.call(self._стоп.set)
                if not self._готово.is_set() and self._область is not None:
                    self._портал.call(self._область.cancel)
                if self._задача is not None:
                    self._задача.result(timeout=ЖДАТЬ_ЗАКРЫТИЯ)
            except Exception as exc:  # закрытие не должно ронять вызывающего
                log.info("MCP %s: закрытие с ошибкой %r", self.server.имя, exc)
            try:
                self._портал_cm.__exit__(None, None, None)
            except Exception as exc:
                log.info("MCP %s: цикл событий закрылся с ошибкой %r", self.server.имя, exc)
        self._портал = self._портал_cm = self._задача = None
        self._клиент = None

    def close(self) -> None:
        self._погасить()
        if self._журнал_файл is not None:
            try:
                self._журнал_файл.close()
            except OSError:
                pass
            self._журнал_файл = None

    def __enter__(self) -> "Session":
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _о_клиенте() -> Any:
    from mcp_types import Implementation

    return Implementation(name="ai-advent-day18-agent", version="18.0",
                          title="Агент миграции ГИС (День 18)")


def _рукопожатие(клиент: Any) -> Handshake:
    сессия = клиент.session
    способ = "server/discover" if getattr(сессия, "discover_result", None) else "initialize"
    о_сервере = клиент.server_info
    возможности = клиент.server_capabilities
    есть = [имя for имя in ("tools", "resources", "prompts", "logging", "completions", "tasks")
            if getattr(возможности, имя, None) is not None] if возможности else []
    return Handshake(
        протокол=клиент.protocol_version,
        способ=способ,
        имя=getattr(о_сервере, "name", "") or "",
        версия=getattr(о_сервере, "version", "") or "",
        заголовок=getattr(о_сервере, "title", "") or "",
        возможности=есть,
        инструкции=(клиент.instructions or "").strip(),
    )


def inspect(server: Server) -> Inspection:
    """Подключиться, получить список инструментов и закрыться. Не бросает.

    Все сбои — внутри итога: осмотр нескольких серверов не должен обрываться
    на первом неработающем, а интерфейсу нужно показать, что именно сломалось.
    """
    итог = Inspection(сервер=server)
    if not server.готов:
        итог.пропущен = True
        итог.ошибка = server.почему_не_готов
        return итог

    начало = time.monotonic()
    сеанс = Session(server)
    try:
        итог.рукопожатие = сеанс.open()
        итог.подключение_с = сеанс.подключение_с
        список = time.monotonic()
        итог.инструменты, итог.страниц = сеанс.tools()
        итог.список_с = time.monotonic() - список
        итог.ок = True
    except MCPClientError as exc:
        итог.этап, итог.ошибка, итог.подсказка = exc.этап, exc.причина, exc.подсказка
        итог.подключение_с = итог.подключение_с or сеанс.подключение_с
    finally:
        итог.журнал = сеанс.журнал()
        сеанс.close()
        итог.всего_с = time.monotonic() - начало
    return итог
