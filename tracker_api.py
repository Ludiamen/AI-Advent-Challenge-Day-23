"""Трекер задач миграции: небольшой HTTP-сервис в форме API Яндекс.Трекера.

Это не часть агента и не MCP-сервер. Это «чужое приложение» — тот самый API,
вокруг которого в этот день строится MCP-сервер (tracker_server.py). Он живёт
отдельным процессом, слушает свой порт, требует токен в заголовке и отвечает
так же, как отвечает настоящий Трекер: те же пути (/v3/issues/…), те же имена
полей (key, summary, status.key), тот же формат ошибок и постраничность через
X-Total-Count.

Зачем мок, если есть настоящий Трекер. Настоящий нужен токена, организации и
сети — на нём нельзя ни прогнать тесты, ни показать ошибку 429, ни откатить
случайную запись. Мок даёт то же API под рукой: он пустой при старте, данные
берёт из tracker-seed.json, а живёт в tracker-data.json, который в любой момент
не жалко удалить. Переключение MCP-сервера между ними — одна переменная
TRACKER_URL, кода это не касается.

Запуск:
    python tracker_api.py                      # http://127.0.0.1:8766
    python tracker_api.py --порт 8800 --сброс  # другой порт, данные заново
    python tracker_api.py --лимит 5            # каждый шестой запрос в минуту → 429
    python tracker_api.py --медленно 5         # каждый ответ с задержкой: проверка таймаутов

Токен и организация задаются ключами --токен и --орг (по умолчанию
«mock-token» и «mock-org»); без верных заголовков сервис отвечает 401 и 403 —
как настоящий.

Что поддерживается (подмножество API Трекера, версия v3):
    GET  /v3/myself
    GET  /v3/queues
    POST /v3/issues/_search        {"filter": {...}, "query": "текст"}
    GET  /v3/issues/<key>
    GET  /v3/issues/<key>/comments
    POST /v3/issues/<key>/comments        {"text": "..."}
    GET  /v3/issues/<key>/transitions
    POST /v3/issues/<key>/transitions/<transition>/_execute
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any

from flask import Blueprint, Flask, current_app, g, jsonify, request

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
ЗЕРНО = os.path.join(КОРЕНЬ, "tracker-seed.json")
ДАННЫЕ = os.path.join(КОРЕНЬ, "tracker-data.json")

ТОКЕН = "mock-token"
ОРГАНИЗАЦИЯ = "mock-org"
ПОРТ = 8766
# Заголовок организации у Яндекс.Трекера зависит от того, где заведена
# организация: X-Cloud-Org-ID у Яндекс Облака, X-Org-ID у Яндекс 360. Мок
# принимает оба — MCP-серверу не приходится гадать.
ЗАГОЛОВКИ_ОРГАНИЗАЦИИ = ("X-Cloud-Org-ID", "X-Org-ID")

ЛИМИТ = 0          # запросов в минуту; 0 — без ограничения
ЗАДЕРЖКА = 0.0     # секунд перед каждым ответом

# Приложение собирается фабрикой «создать», а не лежит готовым в модуле: так у
# каждого запуска свой файл данных и свои настройки. Прежде настройки жили в
# одном общем объекте, и два сервиса в одном процессе (а в тестах их несколько)
# переписывали конфигурацию друг другу.
трекер = Blueprint("трекер", __name__)

_замок = threading.Lock()

log = logging.getLogger("tracker")


def _теперь() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "+0000")


# --- хранилище -------------------------------------------------------------------

def путь_данных() -> str:
    return current_app.config.get("ДАННЫЕ", ДАННЫЕ)


def читать() -> dict[str, Any]:
    """Данные трекера. Файла нет — создаётся из tracker-seed.json."""
    путь = путь_данных()
    if not os.path.exists(путь):
        shutil.copyfile(ЗЕРНО, путь)
    with open(путь, encoding="utf-8") as файл:
        return json.load(файл)


def писать(данные: dict[str, Any]) -> None:
    # Через временный файл: иначе оборванная запись оставит битый JSON, и
    # сервис перестанет отвечать вообще.
    путь = путь_данных()
    временный = путь + ".tmp"
    with open(временный, "w", encoding="utf-8") as файл:
        json.dump(данные, файл, ensure_ascii=False, indent=2)
    os.replace(временный, путь)


# --- ответы и ошибки --------------------------------------------------------------

def ошибка(код: int, сообщение: str, поля: dict[str, str] | None = None):
    """Ошибка в формате Трекера: errorMessages + statusCode."""
    тело = {"errors": поля or {}, "errorMessages": [сообщение], "statusCode": код}
    return jsonify(тело), код


@трекер.before_request
def проверить_доступ():
    """Токен, организация, частота — всё то, чем настоящий API встречает клиента."""
    задержка = current_app.config.get("ЗАДЕРЖКА", ЗАДЕРЖКА)
    if задержка:
        time.sleep(задержка)

    предел = current_app.config.get("ЛИМИТ", ЛИМИТ)
    if предел:
        минута = int(time.time() // 60)
        счётчик = current_app.config["СЧЁТЧИК"]
        with _замок:
            for прошлая in [м for м in счётчик if м != минута]:
                del счётчик[прошлая]
            счётчик[минута] = счётчик.get(минута, 0) + 1
            сколько = счётчик[минута]
        if сколько > предел:
            ответ = ошибка(429, "Too many requests")
            ответ[0].headers["Retry-After"] = "60"
            return ответ

    разрешение = request.headers.get("Authorization", "")
    ожидаемый = current_app.config.get("ТОКЕН", ТОКЕН)
    if разрешение != f"OAuth {ожидаемый}":
        # Настоящий Трекер на плохой токен отвечает именно 401 и ничего не
        # рассказывает о том, какой токен он ждал.
        return ошибка(401, "Unauthorized")

    организация = ""
    for заголовок in ЗАГОЛОВКИ_ОРГАНИЗАЦИИ:
        организация = организация or request.headers.get(заголовок, "")
    if организация != current_app.config.get("ОРГАНИЗАЦИЯ", ОРГАНИЗАЦИЯ):
        return ошибка(403, "Organization is not specified or not accessible")

    g.данные = читать()
    return None


# --- пользователь и очереди -------------------------------------------------------

@трекер.get("/v3/myself")
def я():
    кто = g.данные["пользователь"]
    return jsonify({"self": f"{request.host_url}v3/users/{кто['id']}", **кто})


@трекер.get("/v3/queues")
def очереди():
    страница, на_странице = _страницы()
    все = g.данные["очереди"]
    кусок = все[(страница - 1) * на_странице: страница * на_странице]
    ответ = jsonify([
        {"self": f"{request.host_url}v3/queues/{о['key']}", "id": номер + 1, **о}
        for номер, о in enumerate(кусок)
    ])
    ответ.headers["X-Total-Count"] = str(len(все))
    return ответ


# --- задачи -----------------------------------------------------------------------

def _страницы() -> tuple[int, int]:
    try:
        страница = max(1, int(request.args.get("page", 1)))
    except ValueError:
        страница = 1
    try:
        на_странице = min(100, max(1, int(request.args.get("perPage", 50))))
    except ValueError:
        на_странице = 50
    return страница, на_странице


def _задача(ключ: str) -> dict[str, Any] | None:
    for з in g.данные["задачи"]:
        if з["key"].upper() == ключ.upper():
            return з
    return None


def _вид(задача: dict[str, Any]) -> dict[str, Any]:
    """Задача в том виде, в каком её отдаёт API: со ссылкой self и вложенными полями."""
    return {"self": f"{request.host_url}v3/issues/{задача['key']}", **задача}


@трекер.post("/v3/issues/_search")
def поиск():
    """Поиск задач: фильтр по полям и «query» — свободный текст по теме и описанию."""
    тело = request.get_json(silent=True) or {}
    фильтр = тело.get("filter") or {}
    if not isinstance(фильтр, dict):
        return ошибка(400, "filter must be an object")
    запрос = str(тело.get("query") or "").strip().lower()

    найденные = []
    for задача in g.данные["задачи"]:
        if not _подходит(задача, фильтр):
            continue
        if запрос and запрос not in (задача["summary"] + " " + задача.get("description", "")).lower():
            continue
        найденные.append(задача)

    найденные.sort(key=lambda з: з["updatedAt"], reverse=True)
    страница, на_странице = _страницы()
    кусок = найденные[(страница - 1) * на_странице: страница * на_странице]
    ответ = jsonify([_вид(з) for з in кусок])
    ответ.headers["X-Total-Count"] = str(len(найденные))
    ответ.headers["X-Total-Pages"] = str(max(1, (len(найденные) + на_странице - 1) // на_странице))
    return ответ


def _подходит(задача: dict[str, Any], фильтр: dict[str, Any]) -> bool:
    for поле, значение in фильтр.items():
        ожидаемые = [str(з).lower() for з in (значение if isinstance(значение, list) else [значение])]
        текущее = задача.get(поле)
        # Поля-ссылки приходят объектами: {"key": "open", "display": "Открыта"}.
        # Фильтруют по ключу — так же, как в настоящем Трекере.
        имеющееся = текущее.get("key", текущее.get("display", "")) if isinstance(текущее, dict) else текущее
        if str(имеющееся).lower() not in ожидаемые:
            return False
    return True


@трекер.get("/v3/issues/<key>")
def задача(key: str):
    найдена = _задача(key)
    if найдена is None:
        return ошибка(404, f"Issue {key} does not exist or you do not have access to it")
    return jsonify(_вид(найдена))


@трекер.get("/v3/issues/<key>/comments")
def комментарии(key: str):
    if _задача(key) is None:
        return ошибка(404, f"Issue {key} does not exist or you do not have access to it")
    свои = [к for к in g.данные["комментарии"] if к["issue"].upper() == key.upper()]
    свои.sort(key=lambda к: к["createdAt"])
    return jsonify([
        {"self": f"{request.host_url}v3/issues/{key}/comments/{к['id']}",
         "id": к["id"], "text": к["text"], "createdBy": к["createdBy"],
         "createdAt": к["createdAt"]}
        for к in свои
    ])


@трекер.post("/v3/issues/<key>/comments")
def добавить_комментарий(key: str):
    если_нет = _задача(key)
    if если_нет is None:
        return ошибка(404, f"Issue {key} does not exist or you do not have access to it")
    тело = request.get_json(silent=True) or {}
    текст = str(тело.get("text") or "").strip()
    if not текст:
        return ошибка(422, "Comment text is required", {"text": "may not be empty"})

    данные = g.данные
    номер = max([к["id"] for к in данные["комментарии"]], default=0) + 1
    запись = {"id": номер, "issue": если_нет["key"], "text": текст,
              "createdBy": данные["пользователь"], "createdAt": _теперь()}
    данные["комментарии"].append(запись)
    если_нет["updatedAt"] = запись["createdAt"]
    писать(данные)
    return jsonify({"self": f"{request.host_url}v3/issues/{key}/comments/{номер}",
                    "id": номер, "text": текст, "createdBy": запись["createdBy"],
                    "createdAt": запись["createdAt"]}), 201


@трекер.get("/v3/issues/<key>/transitions")
def переходы(key: str):
    найдена = _задача(key)
    if найдена is None:
        return ошибка(404, f"Issue {key} does not exist or you do not have access to it")
    return jsonify(_переходы(найдена))


def _переходы(задача: dict[str, Any]) -> list[dict[str, Any]]:
    """Куда задачу можно перевести из текущего статуса — как в жизненном цикле очереди."""
    статусы = g.данные["статусы"]
    текущий = задача["status"]["key"]
    можно = g.данные["жизненный_цикл"].get(текущий, [])
    return [
        {"id": f"to_{куда}", "self": "",
         "display": f"{статусы[текущий]['display']} → {статусы[куда]['display']}",
         "to": {"key": куда, **статусы[куда]}}
        for куда in можно
    ]


@трекер.post("/v3/issues/<key>/transitions/<transition>/_execute")
def выполнить_переход(key: str, transition: str):
    найдена = _задача(key)
    if найдена is None:
        return ошибка(404, f"Issue {key} does not exist or you do not have access to it")
    возможные = {п["id"]: п for п in _переходы(найдена)}
    if transition not in возможные:
        # Настоящий Трекер на недопустимый transition отвечает 422: статус есть, но
        # из текущего в него нельзя.
        доступные = ", ".join(возможные) or "нет ни одного"
        return ошибка(422, f"Transition {transition} is not available for issue {key}. "
                           f"Available: {доступные}")

    данные = g.данные
    куда = возможные[transition]["to"]
    для_записи = next(з for з in данные["задачи"] if з["key"] == найдена["key"])
    для_записи["status"] = {"key": куда["key"], "display": куда["display"]}
    для_записи["statusType"] = {"key": куда["type"], "display": куда["type_display"]}
    для_записи["updatedAt"] = _теперь()

    тело = request.get_json(silent=True) or {}
    примечание = str(тело.get("comment") or "").strip()
    if примечание:
        номер = max([к["id"] for к in данные["комментарии"]], default=0) + 1
        данные["комментарии"].append({
            "id": номер, "issue": для_записи["key"], "text": примечание,
            "createdBy": данные["пользователь"], "createdAt": для_записи["updatedAt"]})
    писать(данные)
    return jsonify(_переходы(для_записи))


@трекер.app_errorhandler(404)
def нет_пути(_):
    return ошибка(404, f"Not found: {request.path}")


@трекер.app_errorhandler(405)
def не_тот_метод(_):
    return ошибка(405, f"Method {request.method} is not allowed for {request.path}")


def создать(данные: str = "", токен: str = "", орг: str = "", лимит: int = 0,
            сброс: bool = False, задержка: float = 0.0) -> Flask:
    """Новое приложение сервиса. Им пользуются и запуск из консоли, и тесты."""
    приложение = Flask(__name__)
    приложение.json.ensure_ascii = False
    приложение.json.sort_keys = False
    приложение.config.update({
        "ДАННЫЕ": os.path.abspath(данные or os.getenv("TRACKER_DATA") or ДАННЫЕ),
        "ТОКЕН": токен or os.getenv("TRACKER_MOCK_TOKEN") or ТОКЕН,
        "ОРГАНИЗАЦИЯ": орг or os.getenv("TRACKER_MOCK_ORG") or ОРГАНИЗАЦИЯ,
        "ЛИМИТ": лимит,
        "ЗАДЕРЖКА": задержка,
        "СЧЁТЧИК": {},
    })
    приложение.register_blueprint(трекер)
    if сброс and os.path.exists(приложение.config["ДАННЫЕ"]):
        os.remove(приложение.config["ДАННЫЕ"])
    return приложение


def main() -> int:
    разбор = argparse.ArgumentParser(
        description="Мок-трекер задач миграции в форме API Яндекс.Трекера.")
    разбор.add_argument("--порт", type=int, default=int(os.getenv("TRACKER_PORT", ПОРТ)))
    разбор.add_argument("--данные", default="", metavar="ПУТЬ",
                        help=f"файл с задачами (по умолчанию {os.path.basename(ДАННЫЕ)})")
    разбор.add_argument("--сброс", action="store_true",
                        help="начать с нуля: удалить файл данных и взять tracker-seed.json")
    разбор.add_argument("--токен", default="", help=f"ожидаемый OAuth-токен (по умолчанию {ТОКЕН})")
    разбор.add_argument("--орг", default="", help=f"ожидаемая организация (по умолчанию {ОРГАНИЗАЦИЯ})")
    разбор.add_argument("--лимит", type=int, default=0, metavar="N",
                        help="сколько запросов в минуту пропускать, дальше 429")
    разбор.add_argument("--медленно", type=float, default=0.0, metavar="СЕК",
                        help="задержка перед каждым ответом — для проверки таймаутов")
    аргументы = разбор.parse_args()

    приложение = создать(аргументы.данные, аргументы.токен, аргументы.орг,
                         аргументы.лимит, аргументы.сброс, аргументы.медленно)
    print(f"Трекер (мок) на http://127.0.0.1:{аргументы.порт}, данные: "
          f"{приложение.config['ДАННЫЕ']}", file=sys.stderr, flush=True)
    приложение.run(host="127.0.0.1", port=аргументы.порт, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    sys.exit(main())
