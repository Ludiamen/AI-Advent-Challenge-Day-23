"""Реестр MCP-серверов: все серверы из файла и общий список их инструментов.

Сегодня реестр умеет одно — осмотреть серверы: подключиться к каждому, получить
список инструментов и сложить итог. Осмотр идёт параллельно, по потоку на
сервер: удалённые отвечают за секунду-две, npx стартует дольше, и ждать их
друг за другом — это сумма всех задержек вместо самой долгой.

Дальше реестр станет местом, через которое агент видит инструменты: отдаст
модели разрешённые (Tool.for_model) под именами «сервер__инструмент» и по
такому имени найдёт, к какому серверу идти с вызовом. Поэтому имя для модели
считается уже сейчас и проверяется на повторы.

Публичное API:
  Registry(path)             — серверы из mcp-servers.json
  .server(name)              — один сервер по имени
  .inspect(names)            -> list[Inspection]
  summary(inspections)       -> dict   — сводка: подключено, инструментов, токенов
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent.mcp import client as mcp_client
from agent.mcp.client import Inspection
from agent.mcp.config import MCPConfigError, Server, config_path, load


class Registry:
    def __init__(self, path: str = "") -> None:
        self.path = config_path(path)
        self.servers: list[Server] = load(self.path)

    def names(self) -> list[str]:
        return [с.имя for с in self.servers]

    def server(self, name: str) -> Server:
        for сервер in self.servers:
            if сервер.имя == name:
                return сервер
        известные = ", ".join(self.names()) or "список пуст"
        raise MCPConfigError(f"Сервера «{name}» нет в {self.path}. Есть: {известные}.")

    def inspect(self, names: list[str] | None = None, parallel: bool = True) -> list[Inspection]:
        """Осмотреть серверы (все или названные). Порядок — как в файле."""
        серверы = [self.server(н) for н in names] if names else list(self.servers)
        if not parallel or len(серверы) < 2:
            return [mcp_client.inspect(с) for с in серверы]
        with ThreadPoolExecutor(max_workers=min(8, len(серверы))) as пул:
            return list(пул.map(mcp_client.inspect, серверы))


def summary(inspections: list[Inspection]) -> dict[str, Any]:
    """Сводка осмотра — одна строка итога в консоли и шапка панели на странице."""
    подключены = [о for о in inspections if о.ок]
    инструменты = [и for о in подключены for и in о.инструменты]
    разрешённые = [и for и in инструменты if и.разрешён]
    имена = [и.полное_имя for и in разрешённые]
    повторы = sorted({и for и in имена if имена.count(и) > 1})
    return {
        "серверов": len(inspections),
        "подключено": len(подключены),
        "пропущено": sum(1 for о in inspections if о.пропущен),
        "сбоев": sum(1 for о in inspections if not о.ок and not о.пропущен),
        "инструментов": len(инструменты),
        "разрешено": len(разрешённые),
        "закрыто": len(инструменты) - len(разрешённые),
        "токенов": sum(и.tokens() for и in разрешённые),
        "токенов_всех": sum(и.tokens() for и in инструменты),
        "повторы_имён": повторы,
    }
