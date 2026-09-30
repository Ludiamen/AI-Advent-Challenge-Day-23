#!/usr/bin/env python3
"""Работник планировщика: процесс, который держит агента живым 24/7.

Запускается отдельно от консоли и страницы и работает, пока его не остановят:

    python worker.py                      # бесконечно, тик 5 секунд
    python worker.py --тик 1              # чаще заглядывать в расписание
    python worker.py --раз                # один обход и выйти (для проверок)
    python worker.py --сколько 60         # поработать минуту и выйти
    python worker.py --опоздание 300      # шире окно «опоздал, но выполним»
    python worker.py --без-модели         # не тратить ключ на текст сводок
    python worker.py --сводка сутки       # собрать сводку прямо сейчас и выйти
    python worker.py --состояние          # что в расписании и жив ли работник

Работник — такой же клиент MCP, как и модель: он не знает, что такое трекер, а
знает, что задание велит позвать «tracker__list_issues» и сохранить ответ.
Поэтому в расписание можно поставить любой инструмент любого сервера из
mcp-servers.json — в том числе целый конвейер Дня 19
(«pipeline__run_pipeline» с именем цепочки в аргументах). После такого вызова
работник дописывает в прогон, каким заданием тот запущен: сам сервер конвейеров
этого знать не может, он видит только вызов инструмента.

Остановка — Ctrl+C или SIGTERM: работник дорабатывает текущий вызов, отмечает
последний пульс и выходит. Задания при этом не теряются — они в базе, и
следующий запуск подберёт их, пересчитав сроки от «сейчас» (пропущенное не
догоняется, см. agent/scheduler/schedule.py).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Any

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, КОРЕНЬ)

from dotenv import load_dotenv  # noqa: E402

from agent.scheduler.store import ScheduleStore, ScheduleStoreError  # noqa: E402
from agent.scheduler.worker import ОКНО_ОПОЗДАНИЯ, ТИК, Работник, сводка_сейчас  # noqa: E402
from scheduler_server import путь_базы  # noqa: E402

load_dotenv(os.path.join(КОРЕНЬ, ".env"))

ЛОГ = logging.getLogger("worker")


def _время(момент: str) -> str:
    """Время без даты, если это сегодня: в журнале работника так читается лучше."""
    if not момент:
        return "—"
    try:
        когда = datetime.fromisoformat(момент)
    except ValueError:
        return момент
    сегодня = datetime.now().date()
    return когда.strftime("%H:%M:%S") if когда.date() == сегодня else момент


def показать_состояние(хранилище: ScheduleStore) -> None:
    цифры = хранилище.счётчики()
    работник = цифры["работник"]
    если_жив = (f"жив, отмечался {_время(работник['отмечался'])} "
                f"({работник['секунд_назад']} с назад), тик {работник['тик']} с"
                if работник["запущен"] else
                "не запущен — задания ждут, ничего не выполняется")
    print(f"Работник: {если_жив}")
    print(f"База: {хранилище.path}")
    print(f"Заданий: {цифры['заданий']} (активных {цифры['активных']}, "
          f"периодических {цифры['периодических']})")
    if цифры["ближайший_срок"]:
        print(f"Ближайший срок: {цифры['ближайший_срок']}")
    print(f"Запусков: {цифры['запусков']} (сбоев {цифры['сбоев']}), "
          f"сводок {цифры['сводок']}, новых напоминаний {цифры['новых_напоминаний']}")

    задания = хранилище.задания(только_активные=True, сколько=20)
    if задания:
        print("\nВ расписании:")
        for задание in задания:
            print(f"  {задание.словами()} · срок {_время(задание.срок)}"
                  f" · запусков {задание.запусков}"
                  + (f", сбоев {задание.сбоев}" if задание.сбоев else ""))


def main(argv: list[str] | None = None) -> int:
    разбор = argparse.ArgumentParser(
        description="Работник планировщика: выполняет задания по расписанию.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Примеры:\n"
               "  python worker.py                  работать, пока не остановят\n"
               "  python worker.py --раз            один обход (для проверок)\n"
               "  python worker.py --сводка сутки   собрать сводку сейчас\n"
               "  python worker.py --состояние      что в расписании\n")
    разбор.add_argument("--тик", dest="tick", type=float, default=ТИК, metavar="СЕК",
                        help=f"как часто заглядывать в расписание (по умолчанию {ТИК} с)")
    разбор.add_argument("--опоздание", dest="grace", type=float, default=ОКНО_ОПОЗДАНИЯ,
                        metavar="СЕК",
                        help=f"на сколько секунд задание может опоздать и всё-таки "
                             f"выполниться (по умолчанию {ОКНО_ОПОЗДАНИЯ:g}); что опоздало "
                             "сильнее — помечается пропуском и не догоняется")
    разбор.add_argument("--раз", dest="once", action="store_true",
                        help="сделать один обход и выйти")
    разбор.add_argument("--сколько", dest="seconds", type=float, default=0.0, metavar="СЕК",
                        help="поработать столько секунд и выйти")
    разбор.add_argument("--база", dest="db", default="", metavar="ПУТЬ",
                        help="файл хранилища (по умолчанию memory/scheduler.db)")
    разбор.add_argument("--память-в", dest="memory", default="", metavar="ПУТЬ",
                        help="каталог памяти агента")
    разбор.add_argument("--mcp-файл", dest="mcp_config", default="", metavar="ПУТЬ",
                        help="другой mcp-servers.json")
    разбор.add_argument("--модель", dest="model", default="", metavar="КЛЮЧ",
                        help="какой моделью писать текст сводок (по умолчанию — роль «сжатие»)")
    разбор.add_argument("--без-модели", dest="silent", action="store_true",
                        help="не звать модель: у сводок останутся только цифры")
    разбор.add_argument("--сводка", dest="digest", nargs="?", const="сутки", default="",
                        metavar="ПЕРИОД", help="собрать сводку сейчас и выйти")
    разбор.add_argument("--состояние", dest="state", action="store_true",
                        help="показать расписание и пульс работника, ничего не выполняя")
    разбор.add_argument("--логи", action="store_true", help="подробный журнал работы")
    аргументы = разбор.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if аргументы.логи else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    if аргументы.memory:
        os.environ["MEMORY_DIR"] = os.path.abspath(аргументы.memory)
    if аргументы.db:
        os.environ["SCHEDULER_DB"] = os.path.abspath(аргументы.db)

    try:
        хранилище = ScheduleStore(путь_базы())
    except ScheduleStoreError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2

    if аргументы.state:
        показать_состояние(хранилище)
        return 0

    работник = Работник(хранилище, mcp_config=аргументы.mcp_config, тик=аргументы.tick,
                        модель=аргументы.model, озвучивать=not аргументы.silent,
                        опоздание=аргументы.grace)

    if аргументы.digest:
        сводка = сводка_сейчас(хранилище, за=аргументы.digest,
                               озвучить=not аргументы.silent, работник=работник)
        print(f"Сводка №{сводка['номер']} за {аргументы.digest} собрана "
              f"({сводка['создана']}).")
        print(сводка["текст"] or сводка_словами(сводка))
        работник.close()
        return 0

    for сигнал in (signal.SIGINT, signal.SIGTERM):
        signal.signal(сигнал, lambda *_: работник.остановить())

    предел = "" if not (аргументы.once or аргументы.seconds) else (
        "один обход" if аргументы.once else f"{аргументы.seconds:g} с")
    print(f"Работник планировщика запущен: тик {аргументы.tick:g} с"
          + (f", {предел}" if предел else ", остановка — Ctrl+C"))
    print(f"База: {хранилище.path}")
    try:
        обходов = работник.работать(сколько=аргументы.seconds,
                                    обходов=1 if аргументы.once else 0)
    except KeyboardInterrupt:  # pragma: no cover — сигнал уже обработан выше
        обходов = 0
    finally:
        работник.close()
    print(f"Обходов сделано: {обходов}. Запусков всего: "
          f"{хранилище.счётчики()['запусков']}.")
    return 0


def сводка_словами(сводка: dict[str, Any]) -> str:
    from agent.scheduler import digest as _digest
    return _digest.словами(сводка.get("цифры", {}))


if __name__ == "__main__":
    sys.exit(main())
