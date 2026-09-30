#!/usr/bin/env bash
# Запись экрана для показа Дня 20: ffmpeg, X11, без звука.
#
# В кадре — только окно браузера. Рабочий стол, панели окружения и окно
# редактора в запись не попадают: показывают работу агента, а не рабочее место.
# Поэтому область берётся по геометрии самого окна, а весь экран пишется лишь по
# явной просьбе (--screen).
#
# Зачем скриптом, а не руками: у записи должны быть одинаковые параметры от
# дубля к дублю (размер, частота, кодек), файл должен попадать в одно и то же
# место с понятным именем, а остановка — не портить контейнер mp4.
#
# Имена переменных латиницей: bash не понимает кириллицу в идентификаторах.
#
# Запуск:
#   bash record_demo.sh                 # окно Chrome, пока не нажмёте Ctrl+C
#   bash record_demo.sh --pid 12345      # окно вот этого процесса браузера
#   bash record_demo.sh --window "Агент миграции"   # окно с таким заголовком
#   bash record_demo.sh --area 1280x800+320+140     # область руками
#   bash record_demo.sh --screen         # весь экран (по умолчанию — нет)
#   bash record_demo.sh --seconds 600 --name маршрут-и-флоу
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
OUT_DIR="$ROOT/demo"
AREA=""
WINDOW="Google Chrome"          # по умолчанию пишем окно браузера
PID=""
WHOLE=0
SECONDS_LIMIT=""
NAME="day20"
FPS=25

while [ $# -gt 0 ]; do
  case "$1" in
    --area|--область) AREA="$2"; shift ;;
    --window|--окно) WINDOW="$2"; shift ;;
    --pid|--процесс) PID="$2"; shift ;;
    --screen|--весь-экран) WHOLE=1 ;;
    --seconds|--секунд) SECONDS_LIMIT="$2"; shift ;;
    --name|--имя) NAME="$2"; shift ;;
    --fps) FPS="$2"; shift ;;
    -h|--help|--помощь) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Не знаю ключа «$1». Список: --area, --window, --pid, --screen, --seconds, --name, --fps"
       exit 1 ;;
  esac
  shift
done

command -v ffmpeg >/dev/null 2>&1 || {
  echo "Нет ffmpeg. Поставить: sudo apt install ffmpeg"; exit 1; }

if [ "${XDG_SESSION_TYPE:-x11}" != "x11" ]; then
  echo "Сессия не X11 (${XDG_SESSION_TYPE:-?}), x11grab не сработает."
  echo "Под Wayland записывайте встроенным рекордером: Ctrl+Alt+Shift+R."
  exit 1
fi

# Геометрия окна: ищем среди окон, которыми управляет оконный менеджер, и берём
# то, что принадлежит нужному процессу (точный путь) или чей заголовок содержит
# искомое (запасной). xdotool в системе может не стоять, а xprop и xwininfo
# есть всегда, где есть X. Размеры округляются до чётных: libx264 с yuv420p
# нечётную ширину или высоту не принимает.
window_area() {   # печатает WxH+X+Y окна браузера
  "$PY" - "${PID:-}" "${WINDOW:-}" <<'PYCODE'
import re
import subprocess
import sys

нужный_pid, заголовок = (sys.argv + ["", ""])[1:3]


def запустить(команда):
    готово = subprocess.run(команда, capture_output=True, text=True)
    return готово.stdout


def потомки(pid):
    """Сам процесс и все его потомки: окно создаёт не обёртка, а сам браузер."""
    свои = {pid}
    вывод = запустить(["ps", "-eo", "pid=,ppid="])
    дети = {}
    for строка in вывод.splitlines():
        части = строка.split()
        if len(части) == 2:
            дети.setdefault(части[1], []).append(части[0])
    очередь = [pid]
    while очередь:
        текущий = очередь.pop()
        for ребёнок in дети.get(текущий, []):
            if ребёнок not in свои:
                свои.add(ребёнок)
                очередь.append(ребёнок)
    return свои


окна = re.findall(r"0x[0-9a-f]+", запустить(["xprop", "-root", "_NET_CLIENT_LIST"]))
семья = потомки(нужный_pid) if нужный_pid else set()
лучшее = None

for окно in окна:
    свойства = запустить(["xprop", "-id", окно, "_NET_WM_PID", "WM_NAME"])
    pid_окна = re.search(r"_NET_WM_PID\(CARDINAL\) = (\d+)", свойства)
    имя = re.search(r'WM_NAME\(\w+\) = "(.*)"', свойства)
    подходит = False
    if семья:
        подходит = bool(pid_окна) and pid_окна.group(1) in семья
    elif заголовок:
        подходит = bool(имя) and заголовок in имя.group(1)
    if not подходит:
        continue
    сведения = запустить(["xwininfo", "-id", окно])
    числа = dict(re.findall(r"(Absolute upper-left X|Absolute upper-left Y|Width|Height):\s+(-?\d+)",
                            сведения))
    if len(числа) < 4:
        continue
    ш, в = int(числа["Width"]), int(числа["Height"])
    x, y = int(числа["Absolute upper-left X"]), int(числа["Absolute upper-left Y"])
    # Chrome рисует рамку сам, и вокруг видимого окна у него лежит прозрачная
    # тень. В геометрию X она входит, на экране её нет — без поправки по краям
    # кадра оказывается полоска рабочего стола. Размеры тени лежат в
    # _GTK_FRAME_EXTENTS: слева, справа, сверху, снизу.
    тень = re.search(r"_GTK_FRAME_EXTENTS\(CARDINAL\) = (\d+), (\d+), (\d+), (\d+)",
                     запустить(["xprop", "-id", окно, "_GTK_FRAME_EXTENTS"]))
    if тень:
        слева, справа, сверху, снизу = (int(з) for з in тень.groups())
        x, y = x + слева, y + сверху
        ш, в = ш - слева - справа, в - сверху - снизу
    if ш < 300 or в < 300:          # служебные окошки Chrome бывают крошечными
        continue
    if лучшее is None or ш * в > лучшее[0] * лучшее[1]:
        лучшее = (ш, в, x, y)

if лучшее:
    ш, в, x, y = лучшее
    x, y = max(x, 0), max(y, 0)
    экран = re.search(r"(\d+)x(\d+)\+0\+0", запустить(["xrandr"]))
    if экран:
        ш = min(ш, int(экран.group(1)) - x)
        в = min(в, int(экран.group(2)) - y)
    print(f"{ш - ш % 2}x{в - в % 2}+{x}+{y}")
PYCODE
}

if [ "$WHOLE" = 1 ]; then
  AREA=""
  SIZE="$(xrandr | grep -oP '\d+x\d+(?=\+0\+0)' | head -1)"
  SIZE="${SIZE:-1920x1080}"
  OFFSET="+0,0"
  echo "Пишу весь экран — так просили ключом --screen."
elif [ -n "$AREA" ]; then
  SIZE="${AREA%%+*}"
  REST="${AREA#*+}"
  OFFSET="+${REST//+/,}"
else
  GEOM="$(window_area)"
  if [ -z "$GEOM" ]; then
    if [ -n "$PID" ]; then
      echo "Не нашёл окна браузера у процесса $PID и его потомков."
    else
      echo "Не нашёл окна браузера с заголовком «$WINDOW»."
    fi
    echo "Окно должно быть открыто до начала записи. Варианты:"
    echo "  bash record_demo.sh --window \"часть заголовка\""
    echo "  bash record_demo.sh --area 1600x980+40+40"
    echo "  bash record_demo.sh --screen        # весь экран, если так и надо"
    exit 1
  fi
  SIZE="${GEOM%%+*}"
  REST="${GEOM#*+}"
  OFFSET="+${REST//+/,}"
fi

mkdir -p "$OUT_DIR"
FILE="$OUT_DIR/${NAME}-$(date +%Y%m%d-%H%M).mp4"
LIMIT=()
[ -n "$SECONDS_LIMIT" ] && LIMIT=(-t "$SECONDS_LIMIT")

cat <<INFO
Пишу $( [ "$WHOLE" = 1 ] && echo "весь экран" || echo "окно браузера"): $SIZE, сдвиг $OFFSET, $FPS кадров в секунду
Файл: $FILE
Остановить: Ctrl+C (файл закроется корректно) $( [ -n "$SECONDS_LIMIT" ] && echo "или само через $SECONDS_LIMIT с")

Совет для показа: разверните окно браузера пошире и откройте панель
«Оркестр» — в кадре должно быть видно и переписку, и карточки флоу. Окно
двигать во время записи не нужно: область зафиксирована в начале.
INFO

# -draw_mouse 1: курсор виден — по нему понятно, что нажимает расширение.
# veryfast/crf 26: текст остаётся читаемым, а файл не разрастается.
#
# ffmpeg запускается фоном, а сигналы ему пересылаются вручную. Прямой запуск
# выглядит проще, но тогда «kill -INT» от вызывающего скрипта попадает в эту
# оболочку, а не в ffmpeg: ffmpeg умирает от SIGTERM, не успев записать moov, и
# файл получается нечитаемым («moov atom not found»). Проверено на живой записи.
ffmpeg -hide_banner -loglevel warning -stats \
  -f x11grab -framerate "$FPS" -draw_mouse 1 \
  -video_size "$SIZE" -i "${DISPLAY:-:0}${OFFSET}" \
  "${LIMIT[@]}" \
  -c:v libx264 -preset veryfast -crf 26 -pix_fmt yuv420p \
  -movflags +faststart "$FILE" &
FF=$!
# INT (Ctrl+C) и TERM пересылаем как INT: только по нему ffmpeg закрывает файл.
trap 'kill -INT "$FF" 2>/dev/null' INT TERM
wait "$FF" 2>/dev/null
trap - INT TERM

echo
echo "Готово: $FILE"
ls -lh "$FILE" | awk '{print "   размер:", $5}'
echo "Посмотреть: xdg-open \"$FILE\""
