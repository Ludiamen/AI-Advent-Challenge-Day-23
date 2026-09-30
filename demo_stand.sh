#!/usr/bin/env bash
# Стенд для показа Дня 23 в браузере — одной командой.
#
# Зачем: сценарий DEMO-CHROME.md проходит браузерное расширение, а ему нужен
# живой стенд в известном состоянии: мок-трекер с исходными задачами, память
# агента с несколькими знаниями и решениями (иначе поиску нечего находить) и
# страница на порту, который не занят. Руками это четыре команды и три
# терминала, а перед каждым дублем записи — заново.
#
# С Дня 21 стенд ещё и держит копию индекса документов (demo-index/): показ
# пересобирает индекс с другими параметрами, и делать это на рабочем index/
# незачем. Копия берётся из готового index/ — со всеми векторами и OCR в кэше,
# поэтому показ не ждёт распознавания сканов заново.
#
# Имена переменных латиницей: bash не понимает кириллицу в идентификаторах.
#
# Запуск:
#   bash demo_stand.sh                 # мок + память + страница на 5001
#   bash demo_stand.sh --tunnel        # плюс публичный HTTPS-адрес для расширения
#   bash demo_stand.sh --tunnel-only --port 5000   # вывести наружу уже работающий сервер
#   bash demo_stand.sh --stop          # погасить всё, что этот скрипт поднял
#   bash demo_stand.sh --port 5002 --key мой-секрет
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

PORT=5001                       # 5000 не трогаем: там обычно сервер человека
TRACKER_PORT=8766
MEM="$ROOT/demo-memory"
INDEX="$ROOT/index"
DEMO_INDEX="$ROOT/demo-index"
REPORTS="$ROOT/demo-reports"
PIDS="$ROOT/demo-memory/.стенд-pids"
KEY=""
TUNNEL=0
TUNNEL_ONLY=0
STOP=0
KEEP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --tunnel|--туннель) TUNNEL=1 ;;
    --tunnel-only|--только-туннель) TUNNEL=1; TUNNEL_ONLY=1 ;;
    --stop|--стоп) STOP=1 ;;
    --keep|--без-сброса) KEEP=1 ;;
    --port|--порт) PORT="$2"; shift ;;
    --key|--ключ) KEY="$2"; shift ;;
    -h|--help|--помощь)
      sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Не знаю ключа «$1». Список: --tunnel, --tunnel-only, --stop, --keep, --port, --key"
       exit 1 ;;
  esac
  shift
done

stop_stand() {
  if [ -f "$PIDS" ]; then
    while read -r pid; do
      [ -n "$pid" ] && kill "$pid" 2>/dev/null && echo "  остановлен процесс $pid"
    done < "$PIDS"
    rm -f "$PIDS"
  fi
  # Только свои процессы: чужой Chrome и чужой сервер на 5000 не трогаем.
  pkill -f "tracker_api.py --сброс --порт $TRACKER_PORT" 2>/dev/null
}

if [ "$STOP" = 1 ]; then
  echo "Гашу стенд…"
  stop_stand
  echo "Готово. Порты $PORT и $TRACKER_PORT свободны:"
  ss -lntH "sport = :$PORT" 2>/dev/null | head -2
  exit 0
fi

busy() { ss -lntH "sport = :$1" 2>/dev/null | grep -q .; }

if [ "$TUNNEL_ONLY" = 1 ]; then
  # Сервер уже работает — свой, на своём порту. Ничего не поднимаем и не гасим:
  # только выводим его наружу. Такой стенд секретом не закрыт (DEMO_KEY читается
  # при запуске сервера), поэтому предупреждаем прямо.
  if ! busy "$PORT"; then
    echo "На порту $PORT никто не слушает — нечего выводить наружу."
    echo "Либо поднимите свой сервер, либо запустите стенд целиком: bash demo_stand.sh"
    exit 1
  fi
  mkdir -p "$MEM"
  echo "Вывожу наружу то, что уже работает на порту $PORT."
  echo "ВНИМАНИЕ: этот сервер запущен без DEMO_KEY, значит публичный адрес будет"
  echo "открыт всем, кто его узнает, — вместе с ключами провайдера и доступом к"
  echo "чужим системам агента. Для показа это терпимо, если адрес никому не"
  echo "давать и погасить туннель сразу после записи; надёжнее — отдельный стенд"
  echo "с секретом: bash demo_stand.sh --tunnel"
  if command -v cloudflared >/dev/null 2>&1; then
    cloudflared tunnel --url "http://localhost:$PORT" > "$MEM/туннель.log" 2>&1 &
    echo $! >> "$PIDS"
    PATTERN='https://[a-z0-9-]+\.trycloudflare\.com'
  else
    ssh -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -R "80:localhost:$PORT" nokey@localhost.run > "$MEM/туннель.log" 2>&1 &
    echo $! >> "$PIDS"
    PATTERN='https://[a-z0-9.-]+\.lhr\.life'
  fi
  for _ in $(seq 1 40); do
    URL="$(grep -oE "$PATTERN" "$MEM/туннель.log" | head -1)"
    [ -n "${URL:-}" ] && break
    sleep 1
  done
  if [ -n "${URL:-}" ]; then
    echo "ПУБЛИЧНЫЙ: $URL"
    echo "Погасить туннель: bash demo_stand.sh --stop"
  else
    echo "Туннель не поднялся. Смотрите $MEM/туннель.log"
    exit 1
  fi
  exit 0
fi

stop_stand
sleep 1
for p in "$PORT" "$TRACKER_PORT"; do
  if busy "$p"; then
    echo "Порт $p занят кем-то ещё. Это не мой процесс, и я его не трогаю."
    echo "Посмотрите: ss -lptn 'sport = :$p'   и либо освободите порт, либо"
    echo "запустите стенд на другом: bash demo_stand.sh --port 5002"
    exit 1
  fi
done

if [ "$KEEP" = 0 ]; then
  rm -rf "$MEM" "$REPORTS"
fi
mkdir -p "$MEM" "$REPORTS"

echo "1/4 Мок-трекер на порту $TRACKER_PORT (задачи MIG-1…MIG-8 заново)…"
"$PY" "$ROOT/tracker_api.py" --сброс --порт "$TRACKER_PORT" \
  > "$MEM/трекер.log" 2>&1 &
echo $! >> "$PIDS"

echo "2/4 Память демо: знания и решения, чтобы поиску было что находить…"
MEMORY_DIR="$MEM" "$PY" - <<'PYCODE'
import os
from agent.memory.long import LongTermMemory

каталог = os.path.join(os.environ["MEMORY_DIR"], "long")
долгая = LongTermMemory(каталог, "инженер")
долгая.knowledge.add("f-печать", "Печать схем в старой системе — mapserv + PDF через "
                     "GD; в новой печатаем через OpenLayers и серверный рендер",
                     topic="Печать", tags=["печать", "миграция", "pdf"])
долгая.knowledge.add("f-postgis", "Геометрию газопроводов храним в EPSG:3857, "
                     "индекс GIST обязателен", topic="PostGIS",
                     tags=["postgis", "миграция"])
долгая.knowledge.add("f-легенда", "Легенду и штамп для печати собираем на сервере: "
                     "в браузере шрифты разъезжаются", topic="Печать",
                     tags=["печать", "отчёты"])
долгая.decisions.add("Порядок миграции",
                     "Сначала справочники, потом геометрия газопроводов",
                     reason="Геометрия ссылается на справочники")
долгая.decisions.add("Печать через сервер",
                     "PDF собирает сервер, браузер только показывает",
                     reason="Одинаковый результат у всех операторов")
print("   знаний 3, решений 2")
PYCODE

if [ -z "$KEY" ] && [ "$TUNNEL" = 1 ]; then
  KEY="$("$PY" -c 'import secrets; print(secrets.token_urlsafe(9))')"
fi

echo "3/4 Индекс документов: копия index/ → demo-index/…"
if [ ! -f "$INDEX/docs.db" ]; then
  echo "   index/docs.db нет — сначала соберите индекс: python cli.py --индексировать"
  exit 1
fi
rm -rf "$DEMO_INDEX"
cp -r "$INDEX" "$DEMO_INDEX"
echo "   $(du -sh "$DEMO_INDEX" | cut -f1), OCR-кэш: $(ls "$DEMO_INDEX/ocr" 2>/dev/null | wc -l) стр."
# С Дня 23 — и настройки второго этапа: кнопка «Подобрать заново» на стенде не
# должна переписывать подобранный файл проекта. Кэш оценок реранкера и
# переписанных запросов лежит в docs.db и приезжает вместе с индексом.
if [ -f "$ROOT/rag-settings.json" ]; then
  cp "$ROOT/rag-settings.json" "$DEMO_INDEX/rag-settings.json"
  echo "   настройки поиска: $(grep -o '"реранкер": "[^"]*"' "$DEMO_INDEX/rag-settings.json" | head -1)"
else
  echo "   rag-settings.json нет — режимы на значениях по умолчанию (python cli.py --подобрать)"
fi

echo "4/4 Страница на порту $PORT…"
MEMORY_DIR="$MEM" REPORTS_DIR="$REPORTS" PORT="$PORT" DEMO_KEY="$KEY" \
  RAG_DIR="$DEMO_INDEX" RAG_SETTINGS="$DEMO_INDEX/rag-settings.json" \
  TRACKER_URL="http://127.0.0.1:$TRACKER_PORT" \
  "$PY" "$ROOT/web.py" > "$MEM/веб.log" 2>&1 &
echo $! >> "$PIDS"

for _ in $(seq 1 30); do
  busy "$PORT" && break
  sleep 0.5
done
if ! busy "$PORT"; then
  echo "Страница не поднялась. Смотрите $MEM/веб.log"
  exit 1
fi

LOCAL="http://localhost:$PORT"
[ -n "$KEY" ] && LOCAL="$LOCAL/?key=$KEY"

echo "Адреса стенда:"
echo "   местный:  $LOCAL"

if [ "$TUNNEL" = 1 ]; then
  echo "   поднимаю туннель…"
  # Сначала ssh-туннель: ничего не надо ставить, а ssh есть всюду. Проверено
  # вживую — localhost.run отдаёт публичный HTTPS-адрес за десяток секунд.
  # cloudflared лучше держит долгие сессии, поэтому если он есть — берём его.
  if command -v cloudflared >/dev/null 2>&1; then
    cloudflared tunnel --url "http://localhost:$PORT" > "$MEM/туннель.log" 2>&1 &
    echo $! >> "$PIDS"
    PATTERN='https://[a-z0-9-]+\.trycloudflare\.com'
  else
    ssh -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -R "80:localhost:$PORT" nokey@localhost.run > "$MEM/туннель.log" 2>&1 &
    echo $! >> "$PIDS"
    PATTERN='https://[a-z0-9.-]+\.lhr\.life'
  fi
  for _ in $(seq 1 40); do
    URL="$(grep -oE "$PATTERN" "$MEM/туннель.log" | head -1)"
    [ -n "${URL:-}" ] && break
    sleep 1
  done
  if [ -n "${URL:-}" ]; then
    echo "   ПУБЛИЧНЫЙ: $URL/?key=$KEY"
    echo "   Этот адрес и давайте расширению. Он живёт, пока работает стенд."
  else
    echo "   Туннель не поднялся. Смотрите $MEM/туннель.log."
    echo "   Установить cloudflared (один файл, без аккаунта):"
    echo "     curl -L -o ~/.local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x ~/.local/bin/cloudflared"
  fi
fi

cat <<INFO

Стенд готов. Дальше:
  1) откройте адрес в Chrome, где стоит расширение Claude in Chrome;
  2) дайте расширению разрешение на этот сайт (значок → Extension settings);
  3) скопируйте расширению промпт из DEMO-CHROME.md («Промпт для расширения»).

Запись экрана:   bash record_demo.sh
Погасить стенд:  bash demo_stand.sh --stop
Логи стенда:     $MEM/веб.log, $MEM/трекер.log
INFO
