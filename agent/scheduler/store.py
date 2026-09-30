"""Хранилище планировщика: задания, запуски, сводки и напоминания в SQLite.

Почему SQLite, а не JSON, как у заявок Дня 17. Здесь в один и тот же файл
пишут сразу несколько процессов: работник (worker.py) выполняет задания,
MCP-сервер планировщика заводит и отменяет их по просьбе модели, консоль и
страница читают. JSON пришлось бы переписывать целиком, и два одновременных
сохранения затёрли бы друг друга — так теряются именно те записи, ради которых
всё и затевалось. SQLite в режиме WAL разводит читателей и писателя сам.

Время хранится местное и без часового пояса. Это сознательно: расписание
«ежедневно в 09:00» человек задаёт по своим часам, и перевод в UTC туда-сюда
добавил бы ошибок ровно там, где их труднее всего заметить.

Что где лежит:

  задания      — что вызывать, с какими аргументами и когда; срок следующего
                 запуска хранится готовым, чтобы работник не пересчитывал его
                 для каждого задания на каждом обходе;
  запуски      — история: когда сработало, сколько заняло, что вернуло; по ней
                 строится сводка, поэтому результат хранится целиком (усечённо);
  сводки       — агрегированный результат: цифры и текст, написанный моделью;
  напоминания  — то, что работник должен сказать человеку словами;
  состояние    — пульс работника: жив ли он и когда был последний обход.

Публичное API:
  ScheduleStore(path)          — открыть (и создать) базу
  .добавить_задание(...)       -> Задание
  .задания(...) / .задание(n)  — список и одно задание
  .к_сроку(сейчас)             — чему пора сработать
  .отменить(номер) / .записать_запуск(...) / .запуски(...)
  .добавить_сводку(...) / .сводки(...) / .последняя_сводка() / .озвучить(...)
  .напомнить(текст) / .напоминания(...) / .прочитать_напоминания()
  .пульс(...) / .работник()    — жив ли работник
  .счётчики(сейчас)            — короткая сводка состояния для консоли и страницы
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from agent.scheduler.schedule import (ЕЖЕДНЕВНО, ИНТЕРВАЛ, ОДНОКРАТНО, ОшибкаРасписания,
                                      Расписание, разобрать)

ЖДЁТ = "ждёт"
ИСПОЛНЕНО = "исполнено"
ОТМЕНЕНО = "отменено"
СОСТОЯНИЯ = (ЖДЁТ, ИСПОЛНЕНО, ОТМЕНЕНО)

# Сколько символов результата оставлять в истории. Ответ инструмента бывает на
# мегабайт, а в сводку идут числа и первые строки; хранить остальное значит
# растить файл, который никто не прочитает.
ПРЕДЕЛ_РЕЗУЛЬТАТА = 4000
ПРЕДЕЛ_ИТОГА = 300

# Сколько секунд без пульса считать работника мёртвым. Три тика с запасом:
# один пропущенный обход бывает от долгого вызова, три подряд — это остановка.
ПУЛЬС_ЖИВ = 30.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS задания (
    номер      INTEGER PRIMARY KEY AUTOINCREMENT,
    инструмент TEXT NOT NULL,
    сервер     TEXT NOT NULL DEFAULT '',
    аргументы  TEXT NOT NULL DEFAULT '{}',
    расписание TEXT NOT NULL,
    вид        TEXT NOT NULL,
    секунд     INTEGER NOT NULL DEFAULT 0,
    час        INTEGER NOT NULL DEFAULT 0,
    минута     INTEGER NOT NULL DEFAULT 0,
    зачем      TEXT NOT NULL DEFAULT '',
    создано    TEXT NOT NULL,
    срок       TEXT NOT NULL DEFAULT '',
    состояние  TEXT NOT NULL DEFAULT 'ждёт',
    запусков   INTEGER NOT NULL DEFAULT 0,
    сбоев      INTEGER NOT NULL DEFAULT 0,
    последний  TEXT NOT NULL DEFAULT '',
    итог       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_задания_срок ON задания(состояние, срок);

CREATE TABLE IF NOT EXISTS запуски (
    номер      INTEGER PRIMARY KEY AUTOINCREMENT,
    задание    INTEGER NOT NULL,
    инструмент TEXT NOT NULL DEFAULT '',
    начат      TEXT NOT NULL,
    секунд     REAL NOT NULL DEFAULT 0,
    ок         INTEGER NOT NULL DEFAULT 0,
    текст      TEXT NOT NULL DEFAULT '',
    данные     TEXT NOT NULL DEFAULT '',
    пропущено  INTEGER NOT NULL DEFAULT 0,
    пропуск    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_запуски_начат ON запуски(начат);

CREATE TABLE IF NOT EXISTS сводки (
    номер      INTEGER PRIMARY KEY AUTOINCREMENT,
    создана    TEXT NOT NULL,
    окно_с     TEXT NOT NULL DEFAULT '',
    окно_по    TEXT NOT NULL DEFAULT '',
    цифры      TEXT NOT NULL DEFAULT '{}',
    текст      TEXT NOT NULL DEFAULT '',
    модель     TEXT NOT NULL DEFAULT '',
    прочитана  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS напоминания (
    номер      INTEGER PRIMARY KEY AUTOINCREMENT,
    создано    TEXT NOT NULL,
    текст      TEXT NOT NULL,
    задание    INTEGER NOT NULL DEFAULT 0,
    прочитано  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS состояние (
    ключ      TEXT PRIMARY KEY,
    значение  TEXT NOT NULL
);
"""


class ScheduleStoreError(RuntimeError):
    """Хранилище планировщика не читается или его просят о невозможном."""


def _сейчас() -> datetime:
    return datetime.now().replace(microsecond=0)


def _строка(момент: datetime | None) -> str:
    return момент.replace(microsecond=0).isoformat(sep=" ") if момент else ""


def _момент(текст: str) -> datetime | None:
    текст = (текст or "").strip()
    if not текст:
        return None
    try:
        return datetime.fromisoformat(текст)
    except ValueError:
        return None


def _json(значение: Any) -> str:
    try:
        return json.dumps(значение, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(значение), ensure_ascii=False)


def _из_json(текст: str, по_умолчанию: Any) -> Any:
    if not текст:
        return по_умолчанию
    try:
        return json.loads(текст)
    except (TypeError, ValueError):
        return по_умолчанию


def _обрезать(текст: str, предел: int) -> str:
    текст = текст or ""
    return текст if len(текст) <= предел else текст[:предел] + f"… (всего {len(текст)} симв.)"


@dataclass
class Задание:
    """Одно задание: какой инструмент MCP звать и когда."""

    номер: int
    инструмент: str
    аргументы: dict[str, Any] = field(default_factory=dict)
    расписание: str = ""
    вид: str = ИНТЕРВАЛ
    секунд: int = 0
    час: int = 0
    минута: int = 0
    сервер: str = ""
    зачем: str = ""
    создано: str = ""
    срок: str = ""
    состояние: str = ЖДЁТ
    запусков: int = 0
    сбоев: int = 0
    последний: str = ""
    итог: str = ""

    @property
    def правило(self) -> Расписание:
        return Расписание(как=self.вид, секунд=self.секунд, час=self.час,
                          минута=self.минута, текст=self.расписание)

    @property
    def активно(self) -> bool:
        return self.состояние == ЖДЁТ

    @property
    def повторяется(self) -> bool:
        return self.вид != ОДНОКРАТНО

    def словами(self) -> str:
        когда = self.правило.словами()
        аргументы = ", ".join(f"{к}={_коротко(з)}" for к, з in self.аргументы.items())
        хвост = f" ({аргументы})" if аргументы else ""
        return f"№{self.номер} {self.инструмент}{хвост} — {когда}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "номер": self.номер, "инструмент": self.инструмент,
            "аргументы": dict(self.аргументы), "расписание": self.расписание,
            "когда": self.правило.словами(), "вид": self.вид, "секунд": self.секунд,
            "сервер": self.сервер, "зачем": self.зачем, "создано": self.создано,
            "срок": self.срок, "состояние": self.состояние,
            "активно": self.активно, "повторяется": self.повторяется,
            "запусков": self.запусков, "сбоев": self.сбоев,
            "последний": self.последний, "итог": self.итог,
        }


@dataclass
class Запуск:
    """След одного срабатывания задания."""

    номер: int
    задание: int
    инструмент: str = ""
    начат: str = ""
    секунд: float = 0.0
    ок: bool = True
    текст: str = ""
    данные: Any = None
    # Сколько сроков прошло мимо перед этим запуском.
    пропущено: int = 0
    # Эта запись — не вызов, а отметка о пропущенном сроке: работника не было
    # слишком долго, и задание решено не догонять.
    пропуск: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"номер": self.номер, "задание": self.задание, "инструмент": self.инструмент,
                "начат": self.начат, "секунд": round(self.секунд, 3), "ок": self.ок,
                "текст": self.текст, "данные": self.данные, "пропущено": self.пропущено,
                "пропуск": self.пропуск}


@dataclass
class Сводка:
    """Агрегированный результат за окно времени."""

    номер: int
    создана: str = ""
    окно_с: str = ""
    окно_по: str = ""
    цифры: dict[str, Any] = field(default_factory=dict)
    текст: str = ""
    модель: str = ""
    прочитана: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"номер": self.номер, "создана": self.создана, "окно_с": self.окно_с,
                "окно_по": self.окно_по, "цифры": dict(self.цифры), "текст": self.текст,
                "модель": self.модель, "прочитана": self.прочитана}


@dataclass
class Напоминание:
    """Строка, которую работник должен передать человеку."""

    номер: int
    создано: str = ""
    текст: str = ""
    задание: int = 0
    прочитано: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"номер": self.номер, "создано": self.создано, "текст": self.текст,
                "задание": self.задание, "прочитано": self.прочитано}


def _коротко(значение: Any, предел: int = 40) -> str:
    текст = значение if isinstance(значение, str) else _json(значение)
    return текст if len(текст) <= предел else текст[:предел] + "…"


class ScheduleStore:
    """Задания, история запусков, сводки и напоминания в одном файле SQLite."""

    def __init__(self, path: str) -> None:
        self.path = path
        каталог = os.path.dirname(os.path.abspath(path)) if path != ":memory:" else ""
        if каталог:
            os.makedirs(каталог, exist_ok=True)
        try:
            # check_same_thread=False: страница обслуживает запросы в разных
            # потоках. isolation_level=None — автофиксация: работник и сервер
            # держат соединение открытым часами, и незакрытая транзакция одного
            # заперла бы базу для другого.
            self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None,
                                       timeout=10.0)
            self._db.row_factory = sqlite3.Row
            if path != ":memory:":
                # WAL: читатели не ждут писателя. Ради него база и выбрана.
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA busy_timeout=10000")
            self._db.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise ScheduleStoreError(f"База планировщика «{path}» не открывается: {exc}") from exc

    # --- задания --------------------------------------------------------------

    def добавить_задание(self, инструмент: str, расписание: str,
                         аргументы: dict[str, Any] | None = None, сервер: str = "",
                         зачем: str = "", сейчас: datetime | None = None) -> Задание:
        """Заводит задание. Расписание разбирается сразу: ошибка — здесь, а не в 3 часа ночи."""
        инструмент = (инструмент or "").strip()
        if not инструмент:
            raise ScheduleStoreError("Не сказано, какой инструмент вызывать.")
        if аргументы is not None and not isinstance(аргументы, dict):
            raise ScheduleStoreError("Аргументы задания — объект JSON.")
        правило = разобрать(расписание)
        сейчас = сейчас or _сейчас()
        срок = правило.первый(сейчас)
        сервер = сервер or (инструмент.split("__")[0] if "__" in инструмент else "")
        курсор = self._db.execute(
            "INSERT INTO задания (инструмент, сервер, аргументы, расписание, вид, секунд, "
            "час, минута, зачем, создано, срок, состояние) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (инструмент, сервер, _json(аргументы or {}), правило.текст or расписание,
             правило.как, правило.секунд, правило.час, правило.минута, зачем,
             _строка(сейчас), _строка(срок), ЖДЁТ))
        задание = self.задание(int(курсор.lastrowid or 0))
        if задание is None:  # pragma: no cover — вставка без строки означала бы битую базу
            raise ScheduleStoreError("Задание не сохранилось.")
        return задание

    def задание(self, номер: int) -> Задание | None:
        строка = self._db.execute("SELECT * FROM задания WHERE номер = ?", (номер,)).fetchone()
        return self._задание(строка) if строка else None

    def задания(self, только_активные: bool = False, сколько: int = 100) -> list[Задание]:
        запрос = "SELECT * FROM задания"
        параметры: tuple[Any, ...] = ()
        if только_активные:
            запрос += " WHERE состояние = ?"
            параметры = (ЖДЁТ,)
        запрос += " ORDER BY номер DESC LIMIT ?"
        строки = self._db.execute(запрос, (*параметры, max(1, сколько))).fetchall()
        return [self._задание(с) for с in строки]

    def к_сроку(self, сейчас: datetime | None = None) -> list[Задание]:
        """Задания, чей срок наступил. Порядок — по сроку: раньше срок, раньше запуск."""
        сейчас = сейчас or _сейчас()
        строки = self._db.execute(
            "SELECT * FROM задания WHERE состояние = ? AND срок <> '' AND срок <= ? "
            "ORDER BY срок, номер", (ЖДЁТ, _строка(сейчас))).fetchall()
        return [self._задание(с) for с in строки]

    def отменить(self, номер: int, почему: str = "") -> Задание:
        задание = self.задание(номер)
        if задание is None:
            raise ScheduleStoreError(f"Задания №{номер} нет.")
        if not задание.активно:
            raise ScheduleStoreError(f"Задание №{номер} уже {задание.состояние}.")
        итог = почему or задание.итог
        self._db.execute("UPDATE задания SET состояние = ?, срок = '', итог = ? WHERE номер = ?",
                         (ОТМЕНЕНО, итог, номер))
        обновлённое = self.задание(номер)
        assert обновлённое is not None
        return обновлённое

    def перенести(self, номер: int, срок: datetime | None) -> None:
        """Ставит следующий срок. None — задание исполнено и больше не сработает."""
        if срок is None:
            self._db.execute("UPDATE задания SET срок = '', состояние = ? WHERE номер = ?",
                             (ИСПОЛНЕНО, номер))
        else:
            self._db.execute("UPDATE задания SET срок = ? WHERE номер = ?",
                             (_строка(срок), номер))

    # --- запуски --------------------------------------------------------------

    def записать_запуск(self, задание: int, ок: bool, текст: str = "", данные: Any = None,
                        секунд: float = 0.0, инструмент: str = "", пропущено: int = 0,
                        пропуск: bool = False, сейчас: datetime | None = None) -> Запуск:
        """Сохраняет результат срабатывания и обновляет счётчики задания.

        Отметка о пропущенном сроке (пропуск=True) — такая же запись истории,
        но в счётчик запусков задания она не идёт: вызова не было.
        """
        сейчас = сейчас or _сейчас()
        текст = _обрезать(текст, ПРЕДЕЛ_РЕЗУЛЬТАТА)
        курсор = self._db.execute(
            "INSERT INTO запуски (задание, инструмент, начат, секунд, ок, текст, данные, "
            "пропущено, пропуск) VALUES (?,?,?,?,?,?,?,?,?)",
            (задание, инструмент, _строка(сейчас), float(секунд), 1 if ок else 0, текст,
             _обрезать(_json(данные) if данные is not None else "", ПРЕДЕЛ_РЕЗУЛЬТАТА),
             int(пропущено), 1 if пропуск else 0))
        self._db.execute(
            "UPDATE задания SET запусков = запусков + ?, сбоев = сбоев + ?, последний = ?, "
            "итог = ? WHERE номер = ?",
            (0 if пропуск else 1, 1 if (not ок and not пропуск) else 0, _строка(сейчас),
             _обрезать(текст, ПРЕДЕЛ_ИТОГА), задание))
        запуск = self.запуск(int(курсор.lastrowid or 0))
        if запуск is None:  # pragma: no cover
            raise ScheduleStoreError("Запуск не сохранился.")
        return запуск

    def запуск(self, номер: int) -> Запуск | None:
        строка = self._db.execute("SELECT * FROM запуски WHERE номер = ?", (номер,)).fetchone()
        return self._запуск(строка) if строка else None

    def запуски(self, задание: int = 0, сколько: int = 20,
                с: datetime | None = None) -> list[Запуск]:
        запрос, параметры = "SELECT * FROM запуски WHERE 1=1", []
        if задание:
            запрос += " AND задание = ?"
            параметры.append(задание)
        if с is not None:
            запрос += " AND начат >= ?"
            параметры.append(_строка(с))
        запрос += " ORDER BY номер DESC LIMIT ?"
        параметры.append(max(1, сколько))
        строки = self._db.execute(запрос, параметры).fetchall()
        return [self._запуск(с) for с in строки]

    # --- сводки ---------------------------------------------------------------

    def добавить_сводку(self, цифры: dict[str, Any], окно_с: datetime | None = None,
                        окно_по: datetime | None = None, текст: str = "", модель: str = "",
                        сейчас: datetime | None = None) -> Сводка:
        сейчас = сейчас or _сейчас()
        курсор = self._db.execute(
            "INSERT INTO сводки (создана, окно_с, окно_по, цифры, текст, модель) "
            "VALUES (?,?,?,?,?,?)",
            (_строка(сейчас), _строка(окно_с), _строка(окно_по or сейчас), _json(цифры),
             текст, модель))
        сводка = self.сводка(int(курсор.lastrowid or 0))
        if сводка is None:  # pragma: no cover
            raise ScheduleStoreError("Сводка не сохранилась.")
        return сводка

    def сводка(self, номер: int) -> Сводка | None:
        строка = self._db.execute("SELECT * FROM сводки WHERE номер = ?", (номер,)).fetchone()
        return self._сводка(строка) if строка else None

    def сводки(self, сколько: int = 10) -> list[Сводка]:
        строки = self._db.execute("SELECT * FROM сводки ORDER BY номер DESC LIMIT ?",
                                  (max(1, сколько),)).fetchall()
        return [self._сводка(с) for с in строки]

    def последняя_сводка(self) -> Сводка | None:
        строки = self.сводки(1)
        return строки[0] if строки else None

    def непрочитанная_сводка(self) -> Сводка | None:
        """Самая свежая сводка, которую человеку ещё не показывали."""
        строка = self._db.execute(
            "SELECT * FROM сводки WHERE прочитана = 0 ORDER BY номер DESC LIMIT 1").fetchone()
        return self._сводка(строка) if строка else None

    def озвучить(self, номер: int, текст: str, модель: str = "") -> Сводка:
        """Дописывает к цифрам текст, написанный моделью."""
        self._db.execute("UPDATE сводки SET текст = ?, модель = ? WHERE номер = ?",
                         (текст, модель, номер))
        сводка = self.сводка(номер)
        if сводка is None:
            raise ScheduleStoreError(f"Сводки №{номер} нет.")
        return сводка

    def прочитать_сводку(self, номер: int) -> None:
        self._db.execute("UPDATE сводки SET прочитана = 1 WHERE номер = ?", (номер,))

    # --- напоминания ----------------------------------------------------------

    def напомнить(self, текст: str, задание: int = 0,
                  сейчас: datetime | None = None) -> Напоминание:
        текст = (текст or "").strip()
        if not текст:
            raise ScheduleStoreError("Пустое напоминание.")
        курсор = self._db.execute(
            "INSERT INTO напоминания (создано, текст, задание) VALUES (?,?,?)",
            (_строка(сейчас or _сейчас()), _обрезать(текст, ПРЕДЕЛ_ИТОГА), int(задание)))
        строка = self._db.execute("SELECT * FROM напоминания WHERE номер = ?",
                                  (int(курсор.lastrowid or 0),)).fetchone()
        return self._напоминание(строка)

    def напоминания(self, сколько: int = 20, только_новые: bool = False) -> list[Напоминание]:
        запрос = "SELECT * FROM напоминания"
        if только_новые:
            запрос += " WHERE прочитано = 0"
        запрос += " ORDER BY номер DESC LIMIT ?"
        строки = self._db.execute(запрос, (max(1, сколько),)).fetchall()
        return [self._напоминание(с) for с in строки]

    def прочитать_напоминания(self, номера: list[int] | None = None) -> int:
        """Помечает напоминания показанными. Возвращает, сколько было новых."""
        if номера is None:
            курсор = self._db.execute("UPDATE напоминания SET прочитано = 1 WHERE прочитано = 0")
        else:
            если = ",".join("?" for _ in номера) or "NULL"
            курсор = self._db.execute(
                f"UPDATE напоминания SET прочитано = 1 WHERE прочитано = 0 AND номер IN ({если})",
                номера)
        return курсор.rowcount if курсор.rowcount and курсор.rowcount > 0 else 0

    # --- пульс работника ------------------------------------------------------

    def пульс(self, тик: float = 0.0, pid: int = 0, сейчас: datetime | None = None) -> None:
        """Работник отмечается, что он жив. По этой отметке видно «24/7»."""
        значения = {"жив": _строка(сейчас or _сейчас())}
        if тик:
            значения["тик"] = str(тик)
        if pid:
            значения["pid"] = str(pid)
        for ключ, значение in значения.items():
            self._db.execute(
                "INSERT INTO состояние (ключ, значение) VALUES (?,?) "
                "ON CONFLICT(ключ) DO UPDATE SET значение = excluded.значение",
                (f"работник_{ключ}", значение))

    def работник(self, сейчас: datetime | None = None) -> dict[str, Any]:
        """Что известно о работнике: когда отмечался и считается ли живым."""
        строки = self._db.execute(
            "SELECT ключ, значение FROM состояние WHERE ключ LIKE 'работник_%'").fetchall()
        данные = {с["ключ"].removeprefix("работник_"): с["значение"] for с in строки}
        жив_в = _момент(данные.get("жив", ""))
        сейчас = сейчас or _сейчас()
        секунд = (сейчас - жив_в).total_seconds() if жив_в else None
        return {
            "запущен": bool(жив_в) and секунд is not None and секунд <= ПУЛЬС_ЖИВ,
            "отмечался": данные.get("жив", ""),
            "секунд_назад": round(секунд, 1) if секунд is not None else None,
            "тик": float(данные.get("тик", 0) or 0),
            "pid": int(данные.get("pid", 0) or 0),
        }

    # --- сводные цифры --------------------------------------------------------

    def счётчики(self, сейчас: datetime | None = None) -> dict[str, Any]:
        """Короткая сводка состояния — для страницы и для консоли."""
        сейчас = сейчас or _сейчас()
        задания = self.задания(сколько=1000)
        активные = [з for з in задания if з.активно]
        ближайший = min((з.срок for з in активные if з.срок), default="")
        сводка = self.последняя_сводка()
        return {
            "заданий": len(задания),
            "активных": len(активные),
            "периодических": sum(1 for з in активные if з.повторяется),
            "ближайший_срок": ближайший,
            "запусков": self._одно("SELECT COUNT(*) FROM запуски WHERE пропуск = 0"),
            "сбоев": self._одно("SELECT COUNT(*) FROM запуски WHERE ок = 0 AND пропуск = 0"),
            "пропусков": self._одно("SELECT COUNT(*) FROM запуски WHERE пропуск = 1"),
            "сводок": self._одно("SELECT COUNT(*) FROM сводки"),
            "новых_напоминаний": self._одно(
                "SELECT COUNT(*) FROM напоминания WHERE прочитано = 0"),
            "последняя_сводка": сводка.создана if сводка else "",
            "работник": self.работник(сейчас),
        }

    def _одно(self, запрос: str) -> int:
        строка = self._db.execute(запрос).fetchone()
        return int(строка[0]) if строка else 0

    def close(self) -> None:
        try:
            self._db.close()
        except sqlite3.Error:  # pragma: no cover — закрытие уже закрытой базы
            pass

    # --- разбор строк ---------------------------------------------------------

    @staticmethod
    def _задание(строка: sqlite3.Row) -> Задание:
        return Задание(
            номер=int(строка["номер"]), инструмент=строка["инструмент"],
            аргументы=_из_json(строка["аргументы"], {}), расписание=строка["расписание"],
            вид=строка["вид"], секунд=int(строка["секунд"]), час=int(строка["час"]),
            минута=int(строка["минута"]), сервер=строка["сервер"], зачем=строка["зачем"],
            создано=строка["создано"], срок=строка["срок"],
            состояние=строка["состояние"], запусков=int(строка["запусков"]),
            сбоев=int(строка["сбоев"]), последний=строка["последний"], итог=строка["итог"])

    @staticmethod
    def _запуск(строка: sqlite3.Row) -> Запуск:
        return Запуск(
            номер=int(строка["номер"]), задание=int(строка["задание"]),
            инструмент=строка["инструмент"], начат=строка["начат"],
            секунд=float(строка["секунд"]), ок=bool(строка["ок"]), текст=строка["текст"],
            данные=_из_json(строка["данные"], None), пропущено=int(строка["пропущено"]),
            пропуск=bool(строка["пропуск"]))

    @staticmethod
    def _сводка(строка: sqlite3.Row) -> Сводка:
        return Сводка(
            номер=int(строка["номер"]), создана=строка["создана"], окно_с=строка["окно_с"],
            окно_по=строка["окно_по"], цифры=_из_json(строка["цифры"], {}),
            текст=строка["текст"], модель=строка["модель"],
            прочитана=bool(строка["прочитана"]))

    @staticmethod
    def _напоминание(строка: sqlite3.Row) -> Напоминание:
        return Напоминание(
            номер=int(строка["номер"]), создано=строка["создано"], текст=строка["текст"],
            задание=int(строка["задание"]), прочитано=bool(строка["прочитано"]))


def окно(за: str, сейчас: datetime | None = None) -> tuple[datetime, datetime]:
    """Границы окна сводки по человеческому слову: «час», «сутки», «неделя»."""
    сейчас = сейчас or _сейчас()
    слово = " ".join((за or "сутки").lower().replace("ё", "е").split())
    сколько = {"час": 3600, "часа": 3600, "сутки": 86400, "день": 86400, "суток": 86400,
               "неделя": 7 * 86400, "неделю": 7 * 86400, "все": 0, "всё": 0}.get(слово)
    if сколько is None:
        try:
            правило = разобрать(f"через {слово}")
        except ОшибкаРасписания as exc:
            raise ScheduleStoreError(
                f"Не понял окно сводки «{за}». Бывает: час, сутки, неделя, всё, "
                "либо промежуток вроде «30м».") from exc
        сколько = правило.секунд
    начало = datetime(1970, 1, 1) if сколько == 0 else сейчас - timedelta(seconds=сколько)
    return начало, сейчас


__all__ = ["ScheduleStore", "ScheduleStoreError", "Задание", "Запуск", "Сводка",
           "Напоминание", "ЖДЁТ", "ИСПОЛНЕНО", "ОТМЕНЕНО", "окно", "ПУЛЬС_ЖИВ",
           "ЕЖЕДНЕВНО", "ИНТЕРВАЛ", "ОДНОКРАТНО"]
