#!/usr/bin/env bash
# Показ Дня 23 одной командой: стенд, видимый Chrome, запись экрана и проход по
# семи сценам сценария DEMO-CHROME.md.
#
# Зачем: сценарий рассчитан на браузерное расширение, но расширение может не
# подключиться — тогда показывать всё равно надо. Здесь те же семь сцен
# проходит скрипт-водитель (browser.mjs, режим «показ»): он печатает подписи
# внизу страницы, делает паузы для камеры и ничего не торопит. С экрана в это
# время идёт запись, и на выходе получается готовый mp4.
#
# Имена переменных латиницей: bash не понимает кириллицу в идентификаторах.
#
# Запуск:
#   bash demo_show.sh                 # стенд + Chrome + запись + показ
#   bash demo_show.sh --no-record     # без записи (посмотреть глазами)
#   bash demo_show.sh --port 5002 --pause 4000
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
# Каталог прогона — со своим номером процесса. С общим каталогом уборка одного
# прогона убивала Chrome другого (профиль-то один и тот же), и водитель терял
# соединение посреди сцены. Проверено на живом показе: сцена 4 обрывалась.
SCRATCH="${TMPDIR:-/tmp}/day23-показ-$$"
PORT=5001
PAUSE=3000
RECORD=1
MODEL="ds-flash"
# Порт отладки — свободный, а не фиксированный. С фиксированным вторым запуском
# Chrome его не занимает (порт уже чей-то), и водитель молча подключается к
# ЧУЖОМУ, оставшемуся от прошлого прогона: на экране одно окно, а сцены идут в
# другом. Так и случилось при первых прогонах показа.
DEBUG_PORT="$("${PY}" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"

while [ $# -gt 0 ]; do
  case "$1" in
    --no-record|--без-записи) RECORD=0 ;;
    --port|--порт) PORT="$2"; shift ;;
    --pause|--пауза) PAUSE="$2"; shift ;;
    --model|--модель) MODEL="$2"; shift ;;
    -h|--help|--помощь) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Не знаю ключа «$1». Список: --no-record, --port, --pause, --model"; exit 1 ;;
  esac
  shift
done

command -v google-chrome >/dev/null 2>&1 || {
  echo "Не нашёл google-chrome. Показ водит именно его."; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Нет node — им запускается browser.mjs"; exit 1; }

cleanup() {
  # TERM, а не INT: SIGINT в фоновом задании неинтерактивной оболочки
  # игнорируется, и запись оставалась незакрытой — mp4 без moov-атома.
  # record_demo.sh ловит TERM и уже сам посылает ffmpeg именно INT.
  [ -n "${REC_PID:-}" ] && kill -TERM "$REC_PID" 2>/dev/null && wait "$REC_PID" 2>/dev/null
  [ -n "${CHROME_PID:-}" ] && kill "$CHROME_PID" 2>/dev/null
  # Chrome отпускает не всех детей, а прерванный прогон вообще не доходит до
  # уборки. Поэтому добиваем по признакам, которые бывают только у нашего
  # показа. Квадратные скобки — чтобы шаблон не совпал с самой этой командой:
  # иначе pkill убивает собственную оболочку.
  pkill -f "user-data-dir=$SCRATCH/chrom[e]" 2>/dev/null
  pkill -INT -f "x11grab.*dem[o]/показ" 2>/dev/null
}
# TERM и INT тоже: прогон часто прерывают, и после него не должно оставаться ни
# записывающего ffmpeg, ни лишнего окна Chrome.
trap cleanup EXIT INT TERM

echo "1/5 Стенд…"
bash "$ROOT/demo_stand.sh" --port "$PORT" | tail -3

echo "2/5 Видимый Chrome со своим профилем (ваш профиль и вкладки не трогаю)…"
# Сначала уборка за прошлым прогоном: иначе старое окно держит профиль, а новое
# не получает порт отладки.
pkill -f "user-data-dir=$SCRATCH/chrom[e]" 2>/dev/null
pkill -f "x11grab.*dem[o]/показ" 2>/dev/null
sleep 1
rm -rf "$SCRATCH/chrome"
mkdir -p "$SCRATCH/chrome"
google-chrome --new-window \
  --remote-debugging-port="$DEBUG_PORT" \
  --user-data-dir="$SCRATCH/chrome" \
  --no-first-run --no-default-browser-check \
  --window-size=1600,980 --window-position=40,40 \
  "http://localhost:$PORT" > "$SCRATCH/chrome.log" 2>&1 &
CHROME_PID=$!

for _ in $(seq 1 40); do
  curl -s -o /dev/null "http://127.0.0.1:$DEBUG_PORT/json/version" && break
  sleep 0.5
done
if ! curl -s -o /dev/null "http://127.0.0.1:$DEBUG_PORT/json/version"; then
  echo "Chrome не отдал порт отладки $DEBUG_PORT. Смотрите $SCRATCH/chrome.log"
  exit 1
fi
sleep 2

if [ "$RECORD" = 1 ]; then
  echo "3/5 Запись экрана…"
  # --pid: пишем именно то окно, которое сами открыли, а не весь экран и не
  # чужое окно браузера с тем же заголовком.
  bash "$ROOT/record_demo.sh" --name показ --pid "$CHROME_PID" \
    > "$SCRATCH/запись.log" 2>&1 &
  REC_PID=$!
  sleep 2
else
  echo "3/5 Запись выключена ключом --no-record"
fi

echo "4/5 Прохожу семь сцен (это 10–15 минут: модель, реранкер и судья работают по-настоящему)…"
# PAUSE латиницей: bash не умеет присваивать переменные окружения с
# кириллическими именами — «ПАУЗА=1200 node …» он считает командой.
CDP="http://127.0.0.1:$DEBUG_PORT" PAUSE="$PAUSE" \
  node "$ROOT/browser.mjs" "http://localhost:$PORT" "$MODEL" "" показ
SHOW_CODE=$?

if [ "$RECORD" = 1 ]; then
  echo "5/5 Останавливаю запись…"
  kill -TERM "$REC_PID" 2>/dev/null
  wait "$REC_PID" 2>/dev/null
  REC_PID=""
  FILE="$(ls -t "$ROOT/demo"/показ-*.mp4 2>/dev/null | head -1)"
  [ -n "$FILE" ] && echo "Запись: $FILE ($(du -h "$FILE" | cut -f1))"
else
  echo "5/5 Готово"
fi

rm -rf "$SCRATCH/chrome"

echo
echo "Показ окончен (код водителя $SHOW_CODE). Стенд ещё работает:"
echo "  страница:        http://localhost:$PORT"
echo "  погасить стенд:  bash demo_stand.sh --stop"
