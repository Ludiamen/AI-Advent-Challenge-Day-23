"""MCP: подключение агента к внешним инструментам по Model Context Protocol.

  config    — mcp-servers.json в формате mcpServers: какие серверы и как до них добраться
  client    — соединение с сервером: рукопожатие, tools/list, tools/call, причины сбоев
  tools     — инструмент глазами агента: параметры, доступ, фильтр, цена в токенах
  registry  — все серверы сразу: параллельный осмотр и сводка
  toolbox   — то, чем агент пользуется: открытые соединения и вызов по имени
"""

from agent.mcp.client import (
    Handshake, Inspection, MCPClientError, Session, ToolResult, inspect,
)
from agent.mcp.config import DEFAULT_PATH, HTTP, STDIO, MCPConfigError, Server, load
from agent.mcp.registry import Registry, summary
from agent.mcp.tools import Tool, model_name
from agent.mcp.toolbox import ВСЕ, Toolbox, ToolboxError

__all__ = [
    "ВСЕ", "DEFAULT_PATH", "HTTP", "STDIO",
    "Handshake", "Inspection", "MCPClientError", "MCPConfigError",
    "Registry", "Server", "Session", "Tool", "ToolResult", "Toolbox", "ToolboxError",
    "inspect", "load", "model_name", "summary",
]
