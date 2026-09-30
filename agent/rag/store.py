"""Локальный индекс в SQLite: чанки, их паспорта и векторы в одном файле.

Почему SQLite, а не FAISS. Весь проект хранит своё в SQLite (диалог, планировщик,
конвейеры, флоу), и индекс ложится рядом: один файл, который читается любым
клиентом, переживает перезапуск и не требует сервера. Векторы лежат в BLOB как
float32, а поиск — это одно умножение матрицы на вектор в numpy. Для тысяч
чанков это миллисекунды; FAISS окупается на сотнях тысяч, и тогда его можно
поставить за тот же метод `искать`, не трогая остальное.

Обе стратегии живут в одном файле, но в своих строках (поле `стратегия`):
пересборка одной не задевает другую, а сравнение читает обе разом.

Отдельная таблица — кэш векторов по хэшу текста и имени модели. Повторная
индексация, другое перекрытие, вторая стратегия с теми же разделами — всё, что
уже считалось, берётся из кэша. На CPU это разница между минутами и секундами.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from agent.rag.chunking import Чанк
from agent.rag.extract import Документ

СХЕМА = """
CREATE TABLE IF NOT EXISTS индексы (
    стратегия   TEXT PRIMARY KEY,
    эмбеддер    TEXT NOT NULL,
    размерность INTEGER NOT NULL,
    параметры   TEXT NOT NULL,
    каталог     TEXT NOT NULL,
    создан      TEXT NOT NULL,
    секунд      REAL NOT NULL DEFAULT 0,
    секунд_векторов REAL NOT NULL DEFAULT 0,
    из_кэша     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS документы (
    стратегия   TEXT NOT NULL,
    источник    TEXT NOT NULL,
    название    TEXT NOT NULL,
    вид         TEXT NOT NULL,
    sha1        TEXT NOT NULL,
    страниц     INTEGER NOT NULL,
    распознано  TEXT NOT NULL,
    символов    INTEGER NOT NULL,
    замечания   TEXT NOT NULL,
    PRIMARY KEY (стратегия, источник)
);
CREATE TABLE IF NOT EXISTS чанки (
    chunk_id    TEXT PRIMARY KEY,
    стратегия   TEXT NOT NULL,
    источник    TEXT NOT NULL,
    номер       INTEGER NOT NULL,
    название    TEXT NOT NULL,
    раздел      TEXT NOT NULL,
    разделы     TEXT NOT NULL,
    страница_с  INTEGER NOT NULL,
    страница_по INTEGER NOT NULL,
    пункт       TEXT NOT NULL,
    токенов     INTEGER NOT NULL,
    распознан   INTEGER NOT NULL,
    контекст    TEXT NOT NULL,
    текст       TEXT NOT NULL,
    вектор      BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS чанки_по_стратегии ON чанки(стратегия, источник, номер);
CREATE TABLE IF NOT EXISTS кэш_векторов (
    модель      TEXT NOT NULL,
    хэш         TEXT NOT NULL,
    вектор      BLOB NOT NULL,
    PRIMARY KEY (модель, хэш)
);
"""


class ОшибкаИндекса(RuntimeError):
    """Индекса нет, он собран другим эмбеддером или запрос не к чему применить."""


@dataclass
class Находка:
    чанк: Чанк
    оценка: float          # косинусное сходство с запросом, от −1 до 1
    место: int             # 1 — самый похожий
    # Второй этап (День 23): оценка реранкера в [0, 1] и место до него.
    реранк: float | None = None
    место_до: int = 0

    def кратко(self, символов: int = 400) -> dict:
        данные = self.чанк.паспорт()
        данные.update({"rank": self.место, "score": round(self.оценка, 4),
                       "text": _обрезать(self.чанк.текст, символов)})
        if self.реранк is not None:
            данные.update({"rerank": round(self.реранк, 4), "rank_before": self.место_до})
        return данные


def _обрезать(текст: str, предел: int) -> str:
    return текст if len(текст) <= предел else текст[:предел].rstrip() + "…"


def хэш_текста(текст: str) -> str:
    return hashlib.sha1(текст.encode("utf-8")).hexdigest()


def _в_blob(вектор) -> bytes:
    return np.asarray(вектор, dtype=np.float32).tobytes()


class ХранилищеИндекса:
    def __init__(self, путь: str) -> None:
        self.путь = os.path.abspath(путь)
        os.makedirs(os.path.dirname(self.путь), exist_ok=True)
        self._замок = threading.Lock()
        # Матрица векторов стратегии держится в памяти до следующей записи:
        # поиск на каждый вопрос не должен перечитывать BLOB-ы с диска.
        self._матрицы: dict[str, tuple[int, np.ndarray, list[str]]] = {}
        self._версия = 0
        with self._соединение() as с:
            с.executescript(СХЕМА)

    def _соединение(self) -> sqlite3.Connection:
        с = sqlite3.connect(self.путь, timeout=30)
        с.row_factory = sqlite3.Row
        return с

    # --- кэш векторов ---------------------------------------------------------

    def из_кэша(self, модель: str, тексты: list[str]) -> dict[str, list[float]]:
        хэши = list({хэш_текста(т) for т in тексты})
        найдено: dict[str, list[float]] = {}
        with self._соединение() as с:
            for начало in range(0, len(хэши), 500):
                часть = хэши[начало:начало + 500]
                метки = ",".join("?" * len(часть))
                for ряд in с.execute(
                        f"SELECT хэш, вектор FROM кэш_векторов WHERE модель=? AND хэш IN ({метки})",
                        [модель, *часть]):
                    найдено[ряд["хэш"]] = np.frombuffer(ряд["вектор"], dtype=np.float32).tolist()
        return найдено

    def в_кэш(self, модель: str, пары: list[tuple[str, list[float]]]) -> None:
        with self._замок, self._соединение() as с:
            с.executemany("INSERT OR REPLACE INTO кэш_векторов VALUES (?, ?, ?)",
                          [(модель, хэш_текста(т), _в_blob(в)) for т, в in пары])

    # --- запись индекса -------------------------------------------------------

    def записать(self, стратегия: str, эмбеддер: str, размерность: int, параметры: dict,
                 каталог: str, документы: list[Документ], чанки: list[Чанк],
                 векторы: list[list[float]], секунд: float = 0.0,
                 секунд_векторов: float = 0.0, из_кэша: int = 0) -> None:
        """Заменить индекс стратегии целиком — одной транзакцией.

        Половинчатого индекса не бывает: если запись упала посреди, остаётся
        прежний. Иначе поиск нашёл бы чанки одной версии документа рядом с
        чанками другой.
        """
        if len(чанки) != len(векторы):
            raise ОшибкаИндекса(f"Чанков {len(чанки)}, а векторов {len(векторы)}")
        with self._замок, self._соединение() as с:
            с.execute("DELETE FROM чанки WHERE стратегия=?", (стратегия,))
            с.execute("DELETE FROM документы WHERE стратегия=?", (стратегия,))
            с.execute(
                "INSERT OR REPLACE INTO индексы VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (стратегия, эмбеддер, размерность, json.dumps(параметры, ensure_ascii=False),
                 каталог, datetime.now().isoformat(timespec="seconds"), round(секунд, 2),
                 round(секунд_векторов, 2), из_кэша))
            с.executemany(
                "INSERT INTO документы VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(стратегия, д.источник, д.название, д.вид, д.sha1, д.страниц,
                  json.dumps(д.распознано), д.символов,
                  json.dumps(д.замечания, ensure_ascii=False)) for д in документы])
            с.executemany(
                "INSERT INTO чанки VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(ч.chunk_id, стратегия, ч.источник, ч.номер, ч.название, ч.раздел,
                  json.dumps(ч.разделы, ensure_ascii=False), ч.страница_с, ч.страница_по,
                  ч.пункт, ч.токенов, int(ч.распознан), ч.контекст, ч.текст, _в_blob(в))
                 for ч, в in zip(чанки, векторы)])
            self._версия += 1
            self._матрицы.pop(стратегия, None)

    def удалить(self, стратегия: str) -> int:
        with self._замок, self._соединение() as с:
            удалено = с.execute("DELETE FROM чанки WHERE стратегия=?", (стратегия,)).rowcount
            с.execute("DELETE FROM документы WHERE стратегия=?", (стратегия,))
            с.execute("DELETE FROM индексы WHERE стратегия=?", (стратегия,))
            self._матрицы.pop(стратегия, None)
        return удалено

    # --- чтение ---------------------------------------------------------------

    def индекс(self, стратегия: str) -> dict | None:
        with self._соединение() as с:
            ряд = с.execute("SELECT * FROM индексы WHERE стратегия=?", (стратегия,)).fetchone()
            if not ряд:
                return None
            счёт = с.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(токенов), 0) AS t, "
                "COUNT(DISTINCT источник) AS d FROM чанки WHERE стратегия=?",
                (стратегия,)).fetchone()
        данные = dict(ряд)
        данные["параметры"] = json.loads(данные["параметры"])
        данные.update({"чанков": счёт["n"], "токенов": счёт["t"], "документов": счёт["d"]})
        return данные

    def индексы(self) -> list[dict]:
        with self._соединение() as с:
            имена = [р[0] for р in с.execute("SELECT стратегия FROM индексы ORDER BY стратегия")]
        return [и for и in (self.индекс(н) for н in имена) if и]

    def документы(self, стратегия: str) -> list[dict]:
        with self._соединение() as с:
            ряды = с.execute("SELECT * FROM документы WHERE стратегия=? ORDER BY источник",
                             (стратегия,)).fetchall()
            чанков = dict(с.execute(
                "SELECT источник, COUNT(*) FROM чанки WHERE стратегия=? GROUP BY источник",
                (стратегия,)).fetchall())
        итог = []
        for р in ряды:
            д = dict(р)
            д["распознано"] = json.loads(д["распознано"])
            д["замечания"] = json.loads(д["замечания"])
            д["чанков"] = чанков.get(д["источник"], 0)
            итог.append(д)
        return итог

    def чанки(self, стратегия: str, источник: str = "") -> list[Чанк]:
        запрос = "SELECT * FROM чанки WHERE стратегия=?"
        значения: list = [стратегия]
        if источник:
            запрос += " AND источник=?"
            значения.append(источник)
        with self._соединение() as с:
            ряды = с.execute(запрос + " ORDER BY источник, номер", значения).fetchall()
        return [_чанк(р) for р in ряды]

    def чанк(self, chunk_id: str) -> Чанк | None:
        with self._соединение() as с:
            ряд = с.execute("SELECT * FROM чанки WHERE chunk_id=?", (chunk_id,)).fetchone()
        return _чанк(ряд) if ряд else None

    def соседи(self, чанк: Чанк) -> tuple[Чанк | None, Чанк | None]:
        """Чанк до и после — чтобы показать, где проходит граница резки."""
        with self._соединение() as с:
            ряды = {р["номер"]: р for р in с.execute(
                "SELECT * FROM чанки WHERE стратегия=? AND источник=? AND номер IN (?, ?)",
                (чанк.стратегия, чанк.источник, чанк.номер - 1, чанк.номер + 1))}
        до, после = ряды.get(чанк.номер - 1), ряды.get(чанк.номер + 1)
        return (_чанк(до) if до else None, _чанк(после) if после else None)

    # --- поиск ----------------------------------------------------------------

    def _матрица(self, стратегия: str) -> tuple[np.ndarray, list[str]]:
        готово = self._матрицы.get(стратегия)
        if готово and готово[0] == self._версия:
            return готово[1], готово[2]
        with self._соединение() as с:
            ряды = с.execute("SELECT chunk_id, вектор FROM чанки WHERE стратегия=? "
                             "ORDER BY источник, номер", (стратегия,)).fetchall()
        if not ряды:
            raise ОшибкаИндекса(f"Индекса стратегии «{стратегия}» нет — сначала проиндексируйте")
        матрица = np.vstack([np.frombuffer(р["вектор"], dtype=np.float32) for р in ряды])
        ключи = [р["chunk_id"] for р in ряды]
        self._матрицы[стратегия] = (self._версия, матрица, ключи)
        return матрица, ключи

    def искать(self, стратегия: str, вектор: list[float], сколько: int = 5,
               источник: str = "") -> list[Находка]:
        """Ближайшие по косинусу чанки. Векторы нормированы — косинус равен скалярному."""
        матрица, ключи = self._матрица(стратегия)
        запрос = np.asarray(вектор, dtype=np.float32)
        if запрос.shape[0] != матрица.shape[1]:
            raise ОшибкаИндекса(
                f"Размерность запроса {запрос.shape[0]}, а индекса {матрица.shape[1]}: "
                "индекс собран другим эмбеддером — пересоберите его")
        оценки = матрица @ запрос
        порядок = np.argsort(-оценки)
        итог: list[Находка] = []
        for i in порядок:
            чанк = self.чанк(ключи[int(i)])
            if чанк is None or (источник and источник.lower() not in чанк.источник.lower()):
                continue
            итог.append(Находка(чанк, float(оценки[int(i)]), len(итог) + 1))
            if len(итог) >= сколько:
                break
        return итог


def _чанк(ряд: sqlite3.Row) -> Чанк:
    return Чанк(
        chunk_id=ряд["chunk_id"], стратегия=ряд["стратегия"], источник=ряд["источник"],
        название=ряд["название"], номер=ряд["номер"], текст=ряд["текст"],
        раздел=ряд["раздел"], разделы=json.loads(ряд["разделы"]),
        страница_с=ряд["страница_с"], страница_по=ряд["страница_по"], пункт=ряд["пункт"],
        токенов=ряд["токенов"], распознан=bool(ряд["распознан"]), контекст=ряд["контекст"],
    )
