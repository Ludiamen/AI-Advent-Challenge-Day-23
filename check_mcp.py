#!/usr/bin/env python3
"""Проверка подключения MCP — задание Дня 16, в Дне 17 прогоняется как регрессия.

Задание требует кода, который устанавливает MCP-соединение и получает список
доступных инструментов, и проверки двух вещей: соединение устанавливается,
список инструментов корректно возвращается. Скрипт устроен по ним.

  Часть 1  СОЕДИНЕНИЕ. Каждый сервер из mcp-servers.json: рукопожатие, версия
           протокола, имя сервера, заявленные возможности, время.
  Часть 2  СПИСОК КОРРЕКТЕН. «Вернулся какой-то список» — ещё не доказательство,
           поэтому у каждого подключённого сервера список проверяется четырьмя
           независимыми способами:
             а) сверка без SDK: тот же tools/list запрашивается сырыми
                сообщениями JSON-RPC — initialize, notifications/initialized,
                tools/list по курсору — своим кодом поверх subprocess и httpx.
                Имена и схемы должны совпасть с тем, что отдал SDK. Так ошибка
                в обёртке или в разборе не может спрятаться за самой обёрткой;
             б) каждая inputSchema — корректная JSON Schema типа object: иначе
                модель не сможет вызвать инструмент;
             в) имена уникальны, и имена для модели («сервер__инструмент»)
                допустимы и не повторяются между серверами;
             г) у своего сервера список совпадает с тем, что объявлено в коде.
  Часть 3  СБОИ. Команда, которой нет; закрытый порт; настоящий удалённый
           сервер GitHub с заведомо негодным токеном; сервер, который молчит.
           Каждый — понятная причина и подсказка, и ни один не вешает агента.
  Часть 4  ЦЕНА. Сколько токенов займут описания инструментов в каждом запросе
           к модели — задел на сравнение MCP и «скилл + CLI» из лекции.

Модель не вызывается ни разу: подключение к MCP токенов не тратит, их тратит
только описание инструментов, когда его отдают модели.

Запуск:
    python check_mcp.py                   # все серверы из mcp-servers.json
    python check_mcp.py --сервер deepwiki # один сервер
    python check_mcp.py --без-сети        # только свой сервер и местные сбои
    python check_mcp.py --в RESULTS.md    # дописать отчёт в файл
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

print = functools.partial(print, flush=True)

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, КОРЕНЬ)

import httpx  # noqa: E402
from jsonschema.exceptions import SchemaError  # noqa: E402
from jsonschema.validators import validator_for  # noqa: E402
from mcp.client.stdio import get_default_environment  # noqa: E402

from agent import tokens  # noqa: E402
import agent.mcp as amcp  # noqa: E402

LINE = "-" * 78
ПРОТОКОЛ_СВЕРКИ = "2025-06-18"
СВОИ = {"agent-state"}
СЕТЬ = {amcp.HTTP}


# --- сырой JSON-RPC: сверка без SDK -------------------------------------------------

class СыройКлиент:
    """Минимальный клиент MCP без SDK — ровно столько, чтобы получить tools/list.

    Нужен как независимый свидетель: если SDK или наша обёртка где-то теряет
    или искажает инструменты, списки разойдутся.
    """

    def __init__(self, сервер: amcp.Server, таймаут: float) -> None:
        self.сервер = сервер
        self.таймаут = таймаут
        self.номер = 0
        self.сессия = ""
        self.процесс: subprocess.Popen | None = None
        self.строки: queue.Queue = queue.Queue()
        self.http: httpx.Client | None = None

    # --- транспорт ---
    def открыть(self) -> None:
        if self.сервер.транспорт == amcp.STDIO:
            self.процесс = subprocess.Popen(
                [self.сервер.команда, *self.сервер.аргументы],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                cwd=self.сервер.каталог or None, text=True, encoding="utf-8",
                env={**get_default_environment(), **self.сервер.окружение},
            )
            threading.Thread(target=self._читать, daemon=True).start()
        else:
            self.http = httpx.Client(timeout=self.таймаут, headers={
                **self.сервер.заголовки,
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            })

    def _читать(self) -> None:
        for строка in self.процесс.stdout:
            self.строки.put(строка)

    def закрыть(self) -> None:
        if self.процесс is not None:
            try:
                self.процесс.stdin.close()
                self.процесс.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.процесс.kill()
                self.процесс.wait()
        if self.http is not None:
            if self.сессия:
                try:
                    self.http.delete(self.сервер.адрес, headers={"Mcp-Session-Id": self.сессия})
                except httpx.HTTPError:
                    pass
            self.http.close()

    def _отправить(self, сообщение: dict) -> dict | None:
        """Отправить сообщение; для запроса — дождаться ответа с тем же id."""
        ждать = "id" in сообщение
        if self.процесс is not None:
            self.процесс.stdin.write(json.dumps(сообщение, ensure_ascii=False) + "\n")
            self.процесс.stdin.flush()
            if not ждать:
                return None
            предел = time.monotonic() + self.таймаут
            while time.monotonic() < предел:
                try:
                    строка = self.строки.get(timeout=max(0.1, предел - time.monotonic()))
                except queue.Empty:
                    break
                ответ = json.loads(строка)
                if ответ.get("id") == сообщение["id"]:
                    return ответ
            raise TimeoutError("сервер не ответил")

        заголовки = {"Mcp-Session-Id": self.сессия} if self.сессия else {}
        if self.сессия:
            заголовки["MCP-Protocol-Version"] = ПРОТОКОЛ_СВЕРКИ
        ответ = self.http.post(self.сервер.адрес, json=сообщение, headers=заголовки)
        ответ.raise_for_status()
        self.сессия = ответ.headers.get("mcp-session-id", self.сессия)
        if not ждать:
            return None
        if ответ.headers.get("content-type", "").startswith("text/event-stream"):
            for строка in ответ.text.splitlines():
                if строка.startswith("data:"):
                    данные = json.loads(строка[5:].strip())
                    if данные.get("id") == сообщение["id"]:
                        return данные
            raise ValueError("в потоке SSE нет ответа на запрос")
        return ответ.json()

    def запрос(self, метод: str, параметры: dict | None = None) -> dict:
        self.номер += 1
        ответ = self._отправить({"jsonrpc": "2.0", "id": self.номер, "method": метод,
                                 "params": параметры or {}})
        if "error" in ответ:
            raise RuntimeError(f"{метод}: {ответ['error']}")
        return ответ["result"]

    def инструменты(self) -> tuple[dict, list[dict]]:
        self.открыть()
        try:
            приветствие = self.запрос("initialize", {
                "protocolVersion": ПРОТОКОЛ_СВЕРКИ, "capabilities": {},
                "clientInfo": {"name": "check-mcp-raw", "version": "16"},
            })
            self._отправить({"jsonrpc": "2.0", "method": "notifications/initialized"})
            итог, курсор = [], None
            for _ in range(100):
                страница = self.запрос("tools/list", {"cursor": курсор} if курсор else {})
                итог.extend(страница.get("tools", []))
                курсор = страница.get("nextCursor")
                if not курсор:
                    break
            return приветствие, итог
        finally:
            self.закрыть()


# --- отчёт -------------------------------------------------------------------------

class Отчёт:
    def __init__(self) -> None:
        self.строки: list[str] = []
        self.провалов = 0
        self.проверок = 0

    def __call__(self, текст: str = "") -> None:
        print(текст)
        self.строки.append(текст)

    def пункт(self, ок: bool, текст: str) -> bool:
        self.проверок += 1
        if not ок:
            self.провалов += 1
        self(f"  {'✓' if ок else '✗'} {текст}")
        return ок


def _секунды(значение: float) -> str:
    return f"{значение:.1f}".replace(".", ",") + " с"


# --- части -------------------------------------------------------------------------

def часть_1(отчёт: Отчёт, осмотры: list) -> None:
    отчёт(LINE)
    отчёт("ЧАСТЬ 1. Соединение устанавливается")
    отчёт(LINE)
    for о in осмотры:
        с = о.сервер
        if о.пропущен:
            отчёт(f"  ○ {с.имя} ({с.транспорт}) — пропущен: {о.ошибка}")
            continue
        if not о.ок:
            отчёт.пункт(False, f"{с.имя} ({с.транспорт}) — этап «{о.этап}»: {о.ошибка}")
            if о.подсказка:
                отчёт(f"      подсказка: {о.подсказка}")
            continue
        р = о.рукопожатие
        отчёт.пункт(True, f"{с.имя} ({с.транспорт}) — соединение за {_секунды(о.подключение_с)}, "
                          f"{р.способ}, протокол {р.протокол}")
        отчёт(f"      сервер {р.имя} {р.версия}".rstrip()
              + (f" («{р.заголовок}»)" if р.заголовок else "")
              + f"; умеет: {', '.join(р.возможности) or '—'}")


def часть_2(отчёт: Отчёт, осмотры: list) -> None:
    отчёт(LINE)
    отчёт("ЧАСТЬ 2. Список инструментов возвращается корректно")
    отчёт(LINE)
    имена_для_модели: dict[str, str] = {}
    повторы_между: list[str] = []
    for о in осмотры:
        if not о.ок:
            continue
        с = о.сервер
        имена = [и.имя for и in о.инструменты]
        отчёт(f"  {с.имя}: инструментов {len(имена)}, страниц {о.страниц}, "
              f"получены за {_секунды(о.список_с)}")

        # а) сверка без SDK
        try:
            приветствие, сырые = СыройКлиент(с, с.таймаут).инструменты()
            сырые_схемы = {и["name"]: и.get("inputSchema") for и in сырые}
            наши_схемы = {и.имя: и.входная_схема for и in о.инструменты}
            отчёт.пункт(
                сырые_схемы == наши_схемы,
                f"сверка без SDK (сырой JSON-RPC, initialize {приветствие.get('protocolVersion')}): "
                f"{len(сырые)} инструментов, имена и схемы "
                + ("совпадают" if сырые_схемы == наши_схемы else
                   f"РАСХОДЯТСЯ: только у SDK {sorted(set(наши_схемы) - set(сырые_схемы))}, "
                   f"только в сыром {sorted(set(сырые_схемы) - set(наши_схемы))}"))
        except Exception as exc:  # сверка — свидетель, её сбой не должен ронять прогон
            отчёт.пункт(False, f"сверка без SDK не удалась: {type(exc).__name__}: {exc}")

        # б) схемы
        негодные = []
        for и in о.инструменты:
            схема = и.входная_схема
            try:
                validator_for(схема).check_schema(схема)
                if схема.get("type") != "object":
                    негодные.append(f"{и.имя}: type={схема.get('type')!r}")
            except SchemaError as exc:
                негодные.append(f"{и.имя}: {exc.message}")
        отчёт.пункт(not негодные, "каждая inputSchema — корректная JSON Schema типа object"
                    + (f": {негодные}" if негодные else ""))

        # в) имена
        повторы = sorted({и for и in имена if имена.count(и) > 1})
        отчёт.пункт(not повторы, "имена инструментов уникальны" + (f": {повторы}" if повторы else ""))
        for и in о.инструменты:
            if и.полное_имя in имена_для_модели and имена_для_модели[и.полное_имя] != с.имя:
                повторы_между.append(и.полное_имя)
            имена_для_модели[и.полное_имя] = с.имя

        # г) свой сервер — против объявленного в коде
        if с.имя in СВОИ:
            import anyio
            from mcp import Client
            import mcp_server

            async def объявлено():
                async with Client(mcp_server.сервер) as клиент:
                    return (await клиент.list_tools()).tools

            в_коде = {и.name for и in anyio.run(объявлено)}
            отчёт.пункт(в_коде == set(имена),
                        f"совпадает с объявленным в mcp_server.py: {', '.join(sorted(в_коде))}")

        закрыто = [и.имя for и in о.инструменты if not и.разрешён]
        if закрыто:
            отчёт(f"      фильтр закрыл: {', '.join(закрыто)}")

    import re
    негодные = [и for и in имена_для_модели if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", и)]
    отчёт.пункт(not негодные and not повторы_между,
                f"имена для модели («сервер__инструмент») допустимы и не повторяются: "
                f"{len(имена_для_модели)} шт."
                + (f" Негодные: {негодные}" if негодные else "")
                + (f" Повторы: {повторы_между}" if повторы_между else ""))


def часть_3(отчёт: Отчёт, без_сети: bool) -> None:
    отчёт(LINE)
    отчёт("ЧАСТЬ 3. Сбои называются понятно и не вешают агента")
    отчёт(LINE)
    каталог = tempfile.mkdtemp(prefix="check-mcp-")
    import socket
    with socket.socket() as с:
        с.bind(("127.0.0.1", 0))
        закрытый = с.getsockname()[1]
    случаи = [
        ("нет команды", {"command": "no-such-mcp-server"}, "не найдена команда"),
        ("закрытый порт", {"type": "http", "url": f"http://127.0.0.1:{закрытый}/mcp", "таймаут": 5},
         "нет соединения"),
        ("молчащий сервер", {"command": sys.executable,
                             "args": ["-c", "import time; time.sleep(60)"], "таймаут": 3},
         "не ответил за 3 с"),
    ]
    if not без_сети:
        случаи.append(("GitHub с негодным токеном",
                       {"type": "http", "url": "https://api.githubcopilot.com/mcp/readonly",
                        "headers": {"Authorization": "Bearer ghp_invalid-token-for-check"}, "таймаут": 20},
                       "отклонил авторизацию"))
    for название, описание, ожидаем in случаи:
        путь = os.path.join(каталог, "mcp-servers.json")
        with open(путь, "w", encoding="utf-8") as файл:
            json.dump({"mcpServers": {"probe": описание}}, файл, ensure_ascii=False)
        сервер, = amcp.load(путь)
        начало = time.monotonic()
        о = amcp.inspect(сервер)
        прошло = time.monotonic() - начало
        отчёт.пункт(not о.ок and ожидаем in о.ошибка and прошло < сервер.таймаут + 15,
                    f"{название}: «{о.ошибка}» за {_секунды(прошло)}")
        if о.подсказка:
            отчёт(f"      подсказка: {о.подсказка}")


def часть_4(отчёт: Отчёт, осмотры: list) -> None:
    отчёт(LINE)
    отчёт("ЧАСТЬ 4. Цена описаний инструментов в каждом запросе к модели")
    отчёт(LINE)
    итог = amcp.summary(осмотры)
    for о in осмотры:
        if о.ок:
            отчёт(f"  {о.сервер.имя:<14} разрешено {len(о.разрешено):>3} из {len(о.инструменты):>3}  "
                  f"≈ {tokens.format_tokens(о.токенов):>6} ток."
                  + (f" (без фильтра {tokens.format_tokens(о.токенов_всех)})"
                     if о.токенов != о.токенов_всех else ""))
    отчёт(f"  {'всего':<14} разрешено {итог['разрешено']:>3} из {итог['инструментов']:>3}  "
          f"≈ {tokens.format_tokens(итог['токенов']):>6} ток.")
    отчёт("  Эти токены уходят в КАЖДЫЙ запрос, пока инструменты подключены: десять реплик "
          f"диалога — ≈ {tokens.format_tokens(итог['токенов'] * 10)} ток. только на описания.")


def main() -> int:
    разбор = argparse.ArgumentParser(description="Проверка задания Дня 16: подключение MCP.")
    разбор.add_argument("--сервер", dest="server", default="", help="проверить один сервер")
    разбор.add_argument("--без-сети", dest="offline", action="store_true",
                        help="только свои серверы и местные сбои")
    разбор.add_argument("--в", dest="out", default="", metavar="ФАЙЛ",
                        help="дописать отчёт в markdown-файл")
    аргументы = разбор.parse_args()

    реестр = amcp.Registry()
    имена = [аргументы.server] if аргументы.server else [
        с.имя for с in реестр.servers
        if not (аргументы.offline and (с.транспорт in СЕТЬ or с.команда == "npx"))
    ]
    отчёт = Отчёт()
    отчёт(f"Проверка Дня 16: MCP. Серверы из {реестр.path}")
    отчёт(f"Время: {time.strftime('%Y-%m-%d %H:%M')}; серверов: {len(имена)}")
    начало = time.monotonic()
    осмотры = реестр.inspect(имена)
    часть_1(отчёт, осмотры)
    часть_2(отчёт, осмотры)
    часть_3(отчёт, аргументы.offline)
    часть_4(отчёт, осмотры)
    отчёт(LINE)
    итог = amcp.summary(осмотры)
    отчёт(f"ИТОГ: проверок {отчёт.проверок}, провалов {отчёт.провалов}. Подключено "
          f"{итог['подключено']} из {итог['серверов']}"
          + (f", пропущено {итог['пропущено']} (не настроены)" if итог["пропущено"] else "")
          + f". Прогон {_секунды(time.monotonic() - начало)}.")

    if аргументы.out:
        with open(аргументы.out, "a", encoding="utf-8") as файл:
            файл.write("\n```text\n" + "\n".join(отчёт.строки) + "\n```\n")
        print(f"Отчёт дописан в {аргументы.out}")
    return 1 if отчёт.провалов else 0


if __name__ == "__main__":
    sys.exit(main())
