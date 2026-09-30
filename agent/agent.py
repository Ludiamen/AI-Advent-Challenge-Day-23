"""MemoryAgent — агент с явной моделью памяти.

Отличие этого дня от предыдущих в том, что у агента больше нет «истории» как
одной сущности. Есть три слоя с разным сроком жизни, разными хранилищами и
разными правилами записи, и каждый шаг агента проходит через них явно:

    вопрос
      -> не требует ли запрос нарушить инвариант — кодом        (StateValidator)
         да: отказ, дальше ни маршрутизатор, ни модель не зовутся
      -> маршрутизатор: нужно ли что-то сохранить надолго       (MemoryManager)
      -> запись реплики в краткосрочную память                  (MemoryManager)
      -> сборка промпта из слоёв по политике стадии             (PromptBuilder)
      -> вызов модели                                           (llm.Client)
      -> если модель просит инструмент MCP: вызов и результат    (mcp.Toolbox)
         меняющий вызов не исполняется — заводится заявка        (MemoryManager)
      -> проверка ответа кодом по жёстким инвариантам           (StateValidator)
      -> при нарушении: повтор, затем эскалация модели
      -> запись ответа в краткосрочную память                   (MemoryManager)

Наружу агент отдаёт не только текст ответа, но и трейс: какие записи какого
слоя попали в промпт и во что это обошлось. Без трейса выполнить требование
задания «проверьте, какие данные попадают в каждый слой» нельзя — пришлось бы
верить на слово.

Публичное API:
  MemoryAgent(...)                  — создать агента
  .ask(question)                    — спросить с учётом всех включённых слоёв
  .plan()                           — составить план задачи и записать в рабочую память
  .start_task / .use_task / .transition / .finish_task
  .call_tool(имя, аргументы)        — вызвать инструмент MCP вручную
  .pending_calls() / .confirm_call(n) / .reject_call(n)
  .remember(target, ...)            — записать в указанный слой явно
  .set_layers(...)                  — включить и выключить слои (аблация)
  .info() / .stats() / .journal() / .files()
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from agent import catalog, interview, preferences, prompts, seed as seed_module
from agent.builder import BuiltPrompt, PromptBuilder
from agent.llm import Client, LLMError, Reply, ToolCall
from agent.mcp import DEFAULT_PATH as ФАЙЛ_СЕРВЕРОВ
from agent.mcp import ВСЕ as ВСЕ_СЕРВЕРЫ
from agent.mcp import MCPConfigError, ToolResult, Toolbox, ToolboxError
from agent.mcp import tools as mcp_names
from agent.memory.calls import ВЫПОЛНЕНА, ОТКЛОНЕНА, СБОЙ, Заявка
from agent.memory.long import LongTermError
from agent.memory.manager import AUTO, LONG, SHORT, WORKING, MemoryManager
from agent.memory.router import DEFAULT_THRESHOLD, Routing
from agent.memory.short import DEFAULT_MAX_CHARS, DEFAULT_MAX_MESSAGES
from agent.memory.working import (
    DONE, PLANNING, ПО_КОМАНДЕ, ПРОДОЛЖИТЬ, TaskState, WorkingMemoryError,
)
from agent.preferences import Deviation, PreferenceChecker, SoftResult, StyleJudge
from agent.transitions import (
    КОД, МОДЕЛЬ, ЧЕЛОВЕК, ПереходОтклонён, Вердикт, Условие, Ворота,
    TransitionConfigError, просьба as просьба_о_переходе, убрать_маркер,
)
from agent.scenarios import RunResult, Scenario, ScenarioRunner
from agent.pipeline.store import PipelineStore, PipelineStoreError
from agent.orchestra.flow import (ПРЕДЕЛ_ВЫЗОВОВ, ПРЕДЕЛ_КРУГОВ, Оркестр,
                                  ОшибкаФлоу)
from agent.orchestra.order import ФАЙЛ as ФАЙЛ_ПРАВИЛ
from agent.orchestra.order import ОшибкаПорядка, Порядок
from agent.orchestra.order import загрузить as загрузить_правила
from agent.orchestra.routing import ЧЕЛОВЕК as СПОСОБ_ЧЕЛОВЕК
from agent.orchestra.routing import Маршрут, Маршрутизатор
from agent.orchestra.store import FlowStore, FlowStoreError
from agent.scheduler.store import ScheduleStore, ScheduleStoreError
from agent.validator import Refusal, StateValidator, Violation, _разобрать_json

log = logging.getLogger("agent")

ALL_LAYERS = {SHORT, WORKING, LONG}

# Сколько раз агент пытается получить ответ, не нарушающий инварианты.
# Первая попытка — обычная; вторая — с напоминанием; третья — на модели
# следующей ступени. Дальше уже честнее отказать, чем жечь лимиты.
MAX_ATTEMPTS = 3

# Сколько раз подряд модель может просить вызовы инструментов, прежде чем агент
# потребует ответа словами. Каждый круг — полный запрос с историей, поэтому
# предел здесь не формальность, а ограничение расхода.
MAX_TOOL_ROUNDS = 3

# Имя сервера планировщика в mcp-servers.json. Агент ходит к нему за
# расписанием сам — консолью и страницей, а не только устами модели.
ПЛАНИРОВЩИК = "scheduler"

# Имя сервера конвейеров (День 19) в mcp-servers.json. Через него агент
# запускает цепочки — тем же путём, каким это делает модель.
КОНВЕЙЕР = "pipeline"

_ШАГ_ПЛАНА = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.{3,})$", re.MULTILINE)


class AgentError(RuntimeError):
    """Единственный тип ошибки, который агент выпускает наружу."""


@dataclass
class Answer:
    """Ответ агента вместе со всем, что понадобилось, чтобы его получить."""

    text: str
    prompt: BuiltPrompt | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    violations: list[Violation] = field(default_factory=list)
    deviations: list[Deviation] = field(default_factory=list)
    blocked: bool = False            # ответ так и не уложился в инварианты
    refusal: Refusal | None = None   # формализованный отказ вместо ответа
    применимые: list[str] = field(default_factory=list)   # коды учтённых инвариантов
    исправлено: list[str] = field(default_factory=list)   # нарушения, снятые повтором
    самоотчёт: list[str] = field(default_factory=list)    # что агент назвал сам
    вердикт: Any = None              # суждение проверяющей модели
    escalated_to: str = ""
    routing: Routing | None = None
    routing_entry: dict[str, Any] = field(default_factory=dict)
    soft: SoftResult | None = None
    # Попытка ассистента сменить стадию: что просил и чем кончилось.
    переход: dict[str, Any] = field(default_factory=dict)
    # Вызовы инструментов MCP, которые понадобились для этого ответа, и заявки
    # на изменяющие вызовы, которые ждут подтверждения человека.
    вызовы: list[dict[str, Any]] = field(default_factory=list)
    заявки: list[dict[str, Any]] = field(default_factory=list)
    # Что накопил фоновый работник, пока человека не было: свежая сводка и
    # новые напоминания. Показываются один раз — дальше они помечены
    # прочитанными, иначе агент повторял бы одно и то же в каждом ответе.
    сводка: dict[str, Any] | None = None
    напоминания: list[dict[str, Any]] = field(default_factory=list)
    # Протокол конвейера, если за этот ответ модель запустила цепочку.
    прогон: dict[str, Any] | None = None
    # Какие MCP-серверы маршрутизатор выбрал под этот запрос и почему (День 20).
    маршрут: dict[str, Any] | None = None

    def layers(self) -> dict[str, dict[str, int]]:
        return self.prompt.by_layer() if self.prompt else {}

    def trace(self) -> list[dict[str, Any]]:
        return self.prompt.trace() if self.prompt else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "attempts": self.attempts,
            "blocked": self.blocked,
            "escalated_to": self.escalated_to,
            "usage": self.usage,
            "violations": [н.to_dict() for н in self.violations],
            "refusal": self.refusal.to_dict() if self.refusal else None,
            "применимые": self.применимые,
            "исправлено": self.исправлено,
            "самоотчёт": self.самоотчёт,
            "вердикт": self.вердикт.to_dict() if self.вердикт else None,
            "deviations": [о.to_dict() for о in self.deviations],
            "layers": self.layers(),
            "trace": self.trace(),
            "routing": self.routing.to_dict() if self.routing else None,
            "routing_entry": self.routing_entry,
            "soft": self.soft.to_dict() if self.soft else None,
            "переход": self.переход,
            "вызовы": self.вызовы,
            "заявки": self.заявки,
            "сводка": self.сводка,
            "напоминания": self.напоминания,
            "прогон": self.прогон,
            "маршрут": self.маршрут,
            "stage": self.prompt.stage if self.prompt else "",
            "task_id": self.prompt.task_id if self.prompt else "",
        }


class MemoryAgent:
    """Агент, у которого память разложена по трём слоям явно."""

    def __init__(
        self,
        model_key: str = "",
        user_id: str = "инженер",
        session: str = "основная",
        base_dir: str = "",
        task_id: str = "",
        layers: set[str] | None = None,
        router_mode: str = AUTO,
        threshold: float = DEFAULT_THRESHOLD,
        router_model: str = "",
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_chars: int = DEFAULT_MAX_CHARS,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        soft_check: bool = False,
        follow_profile: bool = True,
        judge_semantic: bool = True,
        require_self_report: bool = True,
        seed_project: bool = True,
        tools: list[str] | None = None,
        mcp_config: str = "",
        confirm_writes: bool = True,
        tool_rounds: int = MAX_TOOL_ROUNDS,
        маршрутизация: bool = True,
        кругов_флоу: int = ПРЕДЕЛ_КРУГОВ,
        вызовов_флоу: int = ПРЕДЕЛ_ВЫЗОВОВ,
    ) -> None:
        # Пустой model_key означает «брать модель по роли»: на планировании и
        # на исполнении роли разные, и жёстко фиксировать одну модель не нужно.
        self.model_key = model_key
        self.layers = set(layers) if layers is not None else set(ALL_LAYERS)
        self.soft_check = soft_check

        self.client = Client(temperature=temperature, max_tokens=max_tokens)
        база = base_dir or os.getenv("MEMORY_DIR", "memory")
        try:
            self.memory = MemoryManager(
                base_dir=база, user_id=user_id, session=session, client=self.client,
                router_mode=router_mode, threshold=threshold, router_model=router_model,
            )
        except (OSError, ValueError, LongTermError) as exc:
            # Сюда приходит и недопустимое имя пользователя: оно превращается в
            # путь, и проверка живёт в хранилище. Наружу это должно выйти
            # понятным отказом, а не стеком из недр памяти.
            raise AgentError(f"Не удалось открыть память: {exc}") from exc

        if seed_project:
            seed_module.seed(self.memory)

        # Ворота — единственное место, где меняется стадия задачи. Условия они
        # берут вызовом: пользователь может добавить своё условие посреди
        # работы, и следующий переход обязан его увидеть.
        self.ворота = Ворота(self.memory.all_conditions)
        self.builder = PromptBuilder(self.memory, max_messages=max_messages,
                                     max_chars=max_chars, gate=self.ворота)
        # Валидатор берёт инварианты вызовом, а не списком: их правят посреди
        # разговора, и слепок, снятый при создании агента, однажды окажется
        # вчерашним.
        self.validator = StateValidator(self.memory.all_invariants, self.client)
        self.style = StyleJudge(self.memory.long.profile, self.client)
        # Проверка предпочтений создаётся на каждый запрос заново: профиль
        # правят посреди разговора, и держать его слепок в поле значит однажды
        # проверить ответ по вчерашним настройкам.
        self.follow_profile = follow_profile
        # Два рубежа проверки смысловых инвариантов включаются порознь: первый
        # стоит один лишний вызов модели, второй — нескольких строк в ответе.
        self.judge_semantic = judge_semantic
        self.require_self_report = require_self_report
        self.runner = ScenarioRunner(self)

        # Инструменты MCP. По умолчанию выключены: описания инструментов уходят
        # в каждый запрос к модели и оплачиваются как вход, а на вопрос «чем
        # PostGIS отличается от MapServer» звать в трекере нечего. Включает их
        # человек — ключом консоли или переключателем на странице.
        self.mcp_config = mcp_config
        self.toolbox: Toolbox | None = (
            Toolbox(list(tools), mcp_config) if tools is not None else None)
        self.confirm_writes = confirm_writes
        self.tool_rounds = max(1, tool_rounds)

        # Оркестровка (День 20). Серверы под запрос выбирает маршрутизатор, и
        # у каждого набора свой инструментарий: открытые сеансы дороги, а
        # наборы повторяются — «про задачи» спрашивают не по одному разу.
        # Ящик из конструктора кладётся в тот же склад под своим ключом, иначе
        # первый же маршрут поднял бы те же серверы вторым процессом.
        self._серверы_ключом = [и for и in (tools or []) if и != ВСЕ_СЕРВЕРЫ]
        self.маршрутизация = bool(маршрутизация)
        self.кругов_флоу = max(1, int(кругов_флоу))
        self.вызовов_флоу = max(1, int(вызовов_флоу))
        self._ящики: dict[tuple[str, ...], Toolbox] = {}
        if self.toolbox is not None:
            self._ящики[tuple(sorted(self._серверы_ключом))] = self.toolbox
        self._flows: FlowStore | None = None
        self._порядок: Порядок | None = None
        self._маршрутизатор: Маршрутизатор | None = None

        # Планировщик (День 18). Хранилище открывается по требованию: агенту,
        # который просто отвечает на вопрос, файл базы не нужен, а создаётся
        # он в том же каталоге памяти — рядом с диалогом и задачами.
        self.scheduler_path = os.path.join(база, "scheduler.db")
        self._scheduler: ScheduleStore | None = None

        # Конвейеры (День 19). Прогоны лежат в том же файле, что и расписание:
        # фоновая жизнь агента одна, и работник, запускающий конвейер по
        # расписанию, пишет туда же. Описания цепочек — рядом с файлом серверов.
        self._pipelines: Any = None
        self._pipeline_runs: PipelineStore | None = None

        self.task: TaskState | None = None
        if task_id:
            self.use_task(task_id)

    # --- свойства ------------------------------------------------------------

    @property
    def session(self) -> str:
        return self.memory.session

    @property
    def user_id(self) -> str:
        return self.memory.user_id

    @property
    def stage(self) -> str:
        return self.task.stage if self.task else ""

    def model_for(self, stage: str = "") -> str:
        """Какая модель отвечает на этой стадии.

        Планирование и исполнение разведены по ролям: на планировании цена
        ошибки выше, потому что на плане строится всё остальное.
        """
        if self.model_key:
            return self.model_key
        роль = "планирование" if (stage or self.stage) == PLANNING else "исполнение"
        return catalog.for_role(роль, offset=0)

    # --- слои ----------------------------------------------------------------

    def set_layers(self, layers: set[str]) -> None:
        """Включает и выключает слои. Этим делается аблация в compare.py."""
        неизвестные = layers - ALL_LAYERS
        if неизвестные:
            raise AgentError(
                f"Неизвестные слои: {', '.join(неизвестные)}. "
                f"Допустимы: {', '.join(sorted(ALL_LAYERS))}."
            )
        self.layers = set(layers)

    # --- задачи --------------------------------------------------------------

    def start_task(self, task_id: str, title: str = "", overwrite: bool = False) -> TaskState:
        """Заводит задачу и делает её текущей."""
        try:
            self.task = self.memory.working.create(task_id, title, overwrite=overwrite)
        except WorkingMemoryError as exc:
            raise AgentError(str(exc)) from exc
        self.memory._log("шаг-задачи", WORKING, task_id, title, applied=True,
                         reason="задача заведена, стадия planning")
        return self.task

    def use_task(self, task_id: str) -> TaskState:
        """Поднимает задачу из рабочей памяти — в том числе спустя дни."""
        try:
            self.task = self.memory.working.load(task_id)
        except WorkingMemoryError as exc:
            raise AgentError(str(exc)) from exc
        return self.task

    def drop_task(self) -> None:
        self.task = None

    def save_task(self) -> None:
        """Кладёт состояние текущей задачи на диск.

        Вызывается после каждого шага и каждой остановки: именно записанное
        состояние, а не переменная в памяти процесса, позволяет вернуться к
        задаче завтра и с другого запуска.
        """
        if self.task is not None:
            self.memory.working.save(self.task)

    # --- пауза и продолжение -------------------------------------------------

    def pause_task(self, причина: str = ПО_КОМАНДЕ, пояснение: str = "") -> TaskState:
        """Просит задачу остановиться. Текущий шаг при этом доводится до конца.

        Флаг ставится на том же объекте состояния, с которым работает исполнитель
        сценария, поэтому он увидит паузу перед следующим шагом. Прерывать шаг
        посреди вызова модели незачем: ответ уже оплачен, и выбрасывать его —
        значит платить дважды.
        """
        if self.task is None:
            raise AgentError("Активной задачи нет — останавливать нечего.")
        if self.task.finished:
            raise AgentError("Задача уже завершена.")
        self.task.остановить(причина, ПРОДОЛЖИТЬ, пояснение)
        self.save_task()
        self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                         f"пауза: {причина}", applied=True,
                         reason=пояснение or "остановлено по команде")
        return self.task

    def continue_task(self) -> TaskState:
        """Снимает паузу, не запуская исполнение. Само продолжение — у сценария."""
        if self.task is None:
            raise AgentError("Активной задачи нет.")
        self.task.продолжить()
        self.save_task()
        return self.task

    def answer_task(self, текст: str) -> TaskState:
        """Принимает ответ человека на остановку «не хватает сведений»."""
        if self.task is None:
            raise AgentError("Активной задачи нет.")
        try:
            self.task.ответить(текст)
        except WorkingMemoryError as exc:
            raise AgentError(str(exc)) from exc
        self.save_task()
        self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                         текст, applied=True, reason="ответ человека на вопрос шага")
        return self.task

    def resume_scenario(
        self,
        task_id: str = "",
        ответ: str = "",
        on_step: Any = None,
        on_result: Any = None,
        по_шагам: bool = False,
    ) -> RunResult:
        """Продолжает отложенную задачу с того шага, где она стоит."""
        task_id = task_id or (self.task.task_id if self.task else "")
        if not task_id:
            raise AgentError("Не указано, какую задачу продолжать.")
        try:
            return self.runner.resume(task_id, ответ=ответ, on_step=on_step,
                                      on_result=on_result, по_шагам=по_шагам)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"Не удалось продолжить задачу «{task_id}»: {exc}") from exc

    def task_state(self) -> dict[str, Any]:
        """Полное состояние задачи для интерфейсов: этап, шаг, ожидание, пауза."""
        if self.task is None:
            return {"есть": False}
        з = self.task
        return {
            "есть": True,
            "task_id": з.task_id,
            "title": з.title,
            "сценарий": з.сценарий,
            "этап": з.stage,
            "этап_словами": з.stage_label,
            "разрешено": list(з.allowed()),
            "шаг": з.текущий_шаг,
            "шагов": з.шагов,
            "шаг_словами": з.шаг_словами,
            "шаги": [ш.to_dict() for ш in з.шаги],
            "ожидание": з.ожидание,
            "ожидание_текст": з.ожидание_текст,
            "пауза": з.пауза,
            "причина_паузы": з.причина_паузы,
            "ответы": list(з.ответы),
            "собрано": len(з.collected),
            "словами": з.состояние_словами,
            # --- ворота: чем открывается следующий этап ----------------------
            "план": list(з.plan),
            "план_утверждён": dict(з.план_утверждён),
            "утверждение_актуально": з.утверждение_актуально,
            "отчёт_валидации": dict(з.отчёт_валидации),
            "валидация_актуальна": з.валидация_актуальна,
            "переходы": self.ворота.обзор(з),
            "история_переходов": list(з.transitions),
            "отказы": list(з.отказы),
        }

    def transition(self, stage: str, note: str = "", кто: str = ЧЕЛОВЕК) -> TaskState:
        """Переводит задачу на новую стадию — если выполнены условия перехода.

        Проверок две, и они разной природы: есть ли такая стрелка в жизненном
        цикле и заслужен ли шаг. Вторую делает Ворота; отклонённая попытка
        ложится в журнал задачи вместе с тем, кто её предпринял.
        """
        if self.task is None:
            raise AgentError("Активной задачи нет: сначала --задача или --новая-задача.")
        вердикт = self.ворота.перевести(self.task, stage, кто, note)
        self.memory.working.save(self.task)
        if not вердикт.можно:
            self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                             f"-X-> {stage}", applied=False,
                             reason=f"переход отклонён ({кто}): {вердикт.причина}")
            raise ПереходОтклонён(вердикт.отказ)
        self.memory._log("шаг-задачи", WORKING, self.task.task_id, f"-> {stage}",
                         applied=True, reason=note or f"смена стадии ({кто})")
        return self.task

    def check_transition(self, stage: str, кто: str = ЧЕЛОВЕК) -> Вердикт:
        """Вердикт по переходу, ничего не меняя: этим живут кнопки интерфейсов."""
        if self.task is None:
            raise AgentError("Активной задачи нет.")
        return self.ворота.проверить(self.task, stage, кто)

    def transition_options(self) -> list[dict[str, Any]]:
        """Все стадии с пометкой «можно/нельзя и почему» — для страницы и CLI."""
        if self.task is None:
            return []
        return self.ворота.обзор(self.task)

    # --- ворота: подпись под планом и отчёт проверки -------------------------

    def approve_plan(self, кем: str = "", пояснение: str = "") -> dict[str, Any]:
        """Утверждает план задачи — этим открывается переход к реализации."""
        if self.task is None:
            raise AgentError("Активной задачи нет: утверждать нечего.")
        try:
            подпись = self.task.утвердить_план(кем or self.user_id, пояснение)
        except WorkingMemoryError as exc:
            raise AgentError(str(exc)) from exc
        self.memory.working.save(self.task)
        self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                         f"план утверждён ({подпись['пунктов']} п.)", applied=True,
                         reason=f"подпись под планом: {подпись['кем']}")
        return подпись

    def unapprove_plan(self, почему: str = "") -> None:
        """Снимает подпись с плана: план решили переделать."""
        if self.task is None:
            raise AgentError("Активной задачи нет.")
        self.task.снять_утверждение(почему)
        self.memory.working.save(self.task)

    def validate_task(self, ревизор: bool = True, model_key: str = "") -> dict[str, Any]:
        """Собирает отчёт проверки — этим открывается переход в done.

        Чек-лист собирает код: он смотрит на состояние задачи, а не на мнение о
        нём. Модель-ревизор добавляет к нему смысловой вердикт — то, чего код не
        видит: сходится ли сделанное с планом по существу. Красный пункт в любой
        из двух частей держит задачу незавершённой.
        """
        if self.task is None:
            raise AgentError("Активной задачи нет: проверять нечего.")
        з = self.task
        пункты: list[dict[str, str]] = []

        def пункт(имя: str, ок: bool, пояснение: str) -> None:
            пункты.append({"пункт": имя, "итог": "прошло" if ок else "не прошло",
                           "пояснение": пояснение, "кем": "код"})

        пункт("план утверждён", з.утверждение_актуально,
              "подпись под текущим планом есть" if з.утверждение_актуально
              else "плана нет или он правился после утверждения")
        осталось = [ш for ш in з.шаги if ш.состояние != "готов"]
        пункт("все шаги пройдены", bool(з.шаги) and not осталось,
              f"шагов {len(з.шаги)}, не доведено {len(осталось)}")
        пункт("результаты собраны", bool(з.collected),
              f"записей в рабочей памяти: {len(з.collected)}")
        открытые = [о for о in з.ответы if not о.get("ответ")]
        пункт("вопросов без ответа нет", not открытые and з.ожидание != "ответ-пользователя",
              "задача ничего не ждёт от человека" if з.ожидание != "ответ-пользователя"
              else з.ожидание_текст)

        модель = ""
        if ревизор and self.client is not None and з.collected:
            итог = self._ревизия(model_key)
            if итог:
                исход, пояснение, модель = итог
                пункты.append({
                    "пункт": "сделанное отвечает плану (по смыслу)",
                    "итог": исход, "пояснение": пояснение,
                    "кем": модель or "ревизор",
                })

        # Красным считается только явное «не прошло». Третий исход — «не
        # проверено»: ревизор не ответил или ответил неразбираемо. Он виден в
        # отчёте, но задачу не запирает: человек не может починить чужой сбой
        # провайдера, а вечно незавершаемая задача — худший из исходов.
        красные = [п for п in пункты if п["итог"] == "не прошло"]
        не_проверено = [п for п in пункты if п["итог"] == "не проверено"]
        отчёт = з.записать_валидацию({
            "вердикт": "прошла" if not красные else "не прошла",
            "пункты": пункты,
            "красных": len(красные),
            "не_проверено": len(не_проверено),
            "ревизор": модель,
        })
        self.memory.working.save(з)
        self.memory._log("шаг-задачи", WORKING, з.task_id,
                         f"проверка: {отчёт['вердикт']}", applied=True,
                         reason=f"красных пунктов {len(красные)}")
        return отчёт

    def _ревизия(self, model_key: str = "") -> tuple[str, str, str] | None:
        """Смысловой вердикт модели: сходится ли сделанное с планом.

        Ревизору не показывают переписку — только план и результаты шагов. Его
        нечем уговаривать: уговорам поддаётся та модель, которую просят
        «посмотреть, всё ли в порядке», зная, чего от неё ждут.
        """
        з = self.task
        план = "\n".join(f"{i}. {ш}" for i, ш in enumerate(з.plan, 1)) or "плана нет"
        сделано = "\n\n".join(
            f"[{имя}]\n{(текст or '')[:1200]}" for имя, текст in з.collected.items()
        )[:6000]
        ключ = model_key or self.model_key or catalog.for_role("ревизор", offset=0)
        сообщения = [
            {"role": "system", "content": prompts.REVIEWER},
            {"role": "user",
             "content": f"ПЛАН ЗАДАЧИ:\n{план}\n\nЧТО СДЕЛАНО:\n{сделано}"},
        ]
        # Как и у судьи инвариантов, одна попытка подняться на ступень выше:
        # слабые модели регулярно отвечают не-JSON, и без эскалации смысловая
        # часть проверки пропадает целиком.
        последний = ""
        for попытка in range(2):
            try:
                ответ = self.client.call(ключ, сообщения, max_tokens=400,
                                         temperature=0.0, low_effort=True)
            except LLMError as exc:
                log.warning("ревизор не ответил (%s): %s", ключ, exc)
                последний = str(exc)[:160]
            else:
                данные = _разобрать_json((ответ.text or "").strip())
                # Разобранный JSON — ещё не вердикт: модель может вернуть
                # правильный по форме, но не тот ответ (чужую схему, пустой
                # объект). Вердиктом считается только тот, где есть ключ, о
                # котором спрашивали; иначе это «не проверено», а не «плохо».
                if данные is not None and "сходится" in данные:
                    сходится = bool(данные.get("сходится"))
                    пояснение = (данные.get("почему") or "").strip()[:300]
                    return (
                        "прошло" if сходится else "не прошло",
                        пояснение or ("сходится с планом" if сходится else "есть расхождения"),
                        ответ.model_key,
                    )
                последний = f"вердикт не разобран: {(ответ.text or '')[:120]}"
            if попытка == 0:
                следующая = catalog.escalate(ключ)
                if следующая != ключ:
                    ключ = следующая
                    continue
            break
        return "не проверено", f"ревизор не дал вердикта ({последний})", ключ

    # --- условия перехода ----------------------------------------------------

    def conditions(self) -> list[Условие]:
        return self.memory.all_conditions()

    def add_condition(self, условие: Условие) -> Условие:
        """Добавляет личное условие перехода. Базовые этим не тронуть."""
        try:
            return self.memory.long.conditions.add(условие)
        except TransitionConfigError as exc:
            raise AgentError(str(exc)) from exc

    def remove_condition(self, код: str) -> bool:
        return self.memory.long.conditions.remove(код)

    def finish_task(self, note: str = "") -> dict[str, Any]:
        """Завершает задачу: свёртка в журнал решений и очистка рабочей памяти."""
        if self.task is None:
            raise AgentError("Активной задачи нет.")
        if self.task.stage != DONE:
            # Завершение — такой же переход, как остальные, и проходит через те
            # же ворота. Раньше здесь была своя проверка «можно ли в done», и
            # она смотрела только на стрелку: задача закрывалась без всякой
            # проверки, лишь бы стадия была validation.
            self.transition(DONE, note or "задача завершена", кто=КОД)
        запись = self.memory.finish_task(self.task, note, model_key=self.model_key)
        self.task = None
        return запись

    # --- запись в память -----------------------------------------------------

    def remember(self, target: str, value: str, key: str = "", section: str = "",
                 reason: str = "") -> dict[str, Any]:
        """Явная запись в указанный слой. Модель не участвует."""
        try:
            return self.memory.remember_explicit(
                target, value, key=key, section=section, reason=reason,
                task_id=self.task.task_id if self.task else "",
            )
        except Exception as exc:  # LongTermError, WorkingMemoryError
            raise AgentError(str(exc)) from exc

    def remember_reply(self, role: str, text: str) -> None:
        """Кладёт реплику в краткосрочную память, если слой включён.

        Нужен исполнителю сценария: сам он ходит в ask() служебными вызовами,
        которые в диалог не пишут, а вопрос человека и итоговый ответ в истории
        разговора быть должны.
        """
        if SHORT in self.layers:
            self.memory.remember_message(
                role, text,
                task_id=self.task.task_id if self.task else "",
                stage=self.stage,
            )

    def close_step(self, результат: str = "") -> TaskState:
        """Отмечает текущий шаг задачи сделанным и двигает указатель.

        Нужен задаче, которую ведут руками: её шаги приходят из плана, и
        закрывать их некому — исполнитель сценария тут не участвует. Без этого
        условие «шаги доведены» из интерфейса не выполнить вовсе, и переход к
        проверке оставался бы закрытым навсегда.
        """
        if self.task is None:
            raise AgentError("Активной задачи нет: закрывать нечего.")
        шаг = self.task.шаг
        if шаг is None:
            raise AgentError(
                "Открытых шагов нет: либо плана не было, либо все шаги пройдены."
            )
        self.task.начать_шаг()
        self.task.закончить_шаг(результат or "закрыт вручную")
        self.memory.working.save(self.task)
        self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                         f"шаг «{шаг.имя}» закрыт", applied=True,
                         reason="шаг отмечен сделанным вручную")
        return self.task

    def remember_step(self, key: str, value: str) -> TaskState:
        if self.task is None:
            raise AgentError("Активной задачи нет: промежуточный результат некуда класть.")
        self.task = self.memory.remember_step(self.task, key, value)
        return self.task

    # --- основной цикл -------------------------------------------------------

    def checker(self) -> PreferenceChecker:
        """Проверка ответа по текущему профилю пользователя."""
        return PreferenceChecker(self.memory.long.profile.load())

    def ask(
        self,
        question: str,
        layers: set[str] | None = None,
        model_key: str = "",
        step_role: str = "",
        personal: bool = True,
        internal: bool = False,
    ) -> Answer:
        """Полный цикл: маршрутизация, сборка, вызов, проверка, запись.

        model_key и step_role задаёт исполнитель сценария: у каждого шага своя
        модель и своя роль, и общие настройки агента их не переопределяют.

        personal=False — промежуточный шаг сценария: настройки пользователя ни в
        промпт не идут, ни по ответу не проверяются.

        internal=True — вызов служебный, а не разговор с человеком: шаг сценария
        получает на вход машинный текст, собранный из результатов предыдущих
        шагов. Такой текст нельзя ни разбирать маршрутизатором, ни класть в
        диалог. Проверено, что бывает иначе: в базу знаний попадали записи вроде
        «Объяснительное сообщение о разделении работы между расширением и
        схемой» — маршрутизатор принял кусок ответа архитектора за факт,
        сказанный пользователем, и записал его навсегда.
        """
        question = (question or "").strip()
        if not question:
            raise AgentError("Пустой вопрос.")
        слои = set(layers) if layers is not None else set(self.layers)

        # Рубеж 1: не требует ли сам запрос нарушить инвариант. Отказ здесь
        # честнее и дешевле, чем «предложил, потом сам себя отругал»: токены не
        # тратятся, а пользователь не видит непоследовательности.
        #
        # Проверка идёт ПЕРВОЙ, раньше маршрутизатора (День 16). Прежде
        # маршрутизатор — вызов дешёвой модели — шёл до неё, и у этого было
        # два следствия. Отказ «за ноль токенов» на деле ждал его ответа: когда
        # бесплатная модель отвечала 429, «мгновенный» отказ приходил через
        # одну-две минуты. И запрос, который сейчас будет отклонён, успевал
        # предложить запись в долговременную память, — однажды это был
        # выдуманный раздел профиля «переписать» для «перепиши на Laravel».
        # Проверка запроса — код без обращения к модели, ей незачем ждать.
        конфликт = (self.validator.check_request(question)
                    if LONG in слои and not internal else [])

        # Правило 5: может быть, в реплике есть что-то для долговременной памяти.
        # Если долговременный слой выключен, спрашивать маршрутизатор незачем, а
        # отклонённый запрос в долговременную память не попадает вовсе.
        маршрут: Routing | None = None
        запись_маршрута: dict[str, Any] = {}
        if LONG in слои and not internal and not конфликт:
            маршрут, запись_маршрута = self.memory.route(question)

        # Правило 2: сама реплика — в краткосрочную память. И отклонённая тоже:
        # отказ — часть разговора, и следующий ответ должен его помнить.
        if SHORT in слои and not internal:
            self.memory.remember_message(
                "user", question,
                task_id=self.task.task_id if self.task else "",
                stage=self.stage,
            )

        if конфликт:
            return self._отказать(конфликт, "до вызова", question, слои)

        # Вопрос о самом инварианте — не попытка его нарушить. Объясняя, почему
        # проект не на Laravel, агент обязан назвать Laravel.
        объяснение = (self.validator.is_explanatory(question)
                      if LONG in слои and not internal else False)

        # Инструменты даются модели только в разговоре с человеком. Промежуточный
        # шаг сценария получает машинный вход и отдаёт машинный выход: вызовы
        # там означали бы обращения к чужим системам, которых никто не видел.
        с_инструментами = self.toolbox is not None and not internal
        # Маршрут (День 20): какие серверы поднимать под ЭТОТ вопрос. Пустой
        # маршрут — не сбой, а ответ: «внешние данные тут не нужны», и тогда
        # запрос идёт без инструментов вовсе, не платя за их описания.
        выбор_серверов: Маршрут | None = None
        if с_инструментами:
            выбор_серверов = self._направить(question)
            # Пустой маршрут отменяет инструменты только тогда, когда выбирал
            # их агент. Набор, названный человеком, отменять нельзя: он мог
            # назвать сервер, которого нет в файле, и остаться совсем без
            # инструментов из-за опечатки — это не то, чего он просил.
            if (выбор_серверов is not None and выбор_серверов.пуст
                    and выбор_серверов.способ != СПОСОБ_ЧЕЛОВЕК):
                с_инструментами = False
        промпт = self.builder.build(question, self.task, слои, step_role=step_role,
                                    personal=personal, with_tools=с_инструментами)
        вызовы: list[ToolResult] = []
        заявки: list[Заявка] = []
        (ответ, попытки, нарушения, расхождения,
         эскалация, исправлено) = self._answer_within_rules(
            промпт, question, слои, model_key, step_role, personal, объяснение,
            с_инструментами, вызовы, заявки
        )

        применимые = (self.validator.applicable(question, ответ.text)
                      if LONG in слои else [])
        самоотчёт = self.validator.self_report(ответ.text)

        # Рубеж 3: смысловые инварианты. Суждение модели вызывается только когда
        # есть что судить и ответ дошёл до пользователя — на промежуточных шагах
        # сценария это лишний вызов на каждый шаг.
        вердикт = None
        if LONG in слои and personal and not нарушения and self.judge_semantic:
            вердикт = self.validator.judge(ответ.text, применимые)
            по_смыслу = self.validator.violations_from_judge(вердикт)
            if по_смыслу:
                нарушения = нарушения + по_смыслу

        отказ = None
        текст = ответ.text
        if нарушения:
            # Текст с нарушением наружу не выходит вовсе: «отказывается
            # предлагать» и «предложил с пометкой» — разные вещи. Сам ответ
            # остаётся в трейсе, разбирать его это не мешает.
            отказ = self.validator.refusal(нарушения, "после ответа", question)
            текст = отказ.текст()

        # Единственный канал, которым ассистент может повлиять на стадию:
        # строка «ПЕРЕХОД: done» в ответе. Просьба разбирается кодом и идёт
        # через те же ворота, что и команда человека, — поэтому «перепрыгнуть»
        # этап моделью нельзя, а попытка остаётся в журнале задачи.
        попытка: dict[str, Any] = {}
        if self.task is not None and not internal and not нарушения:
            попытка, текст = self._просьба_о_переходе(текст)

        мягкая: SoftResult | None = None
        if self.soft_check and not нарушения:
            мягкая = self.style.check(ответ.text)

        if SHORT in слои and not internal:
            self.memory.remember_message(
                "assistant", текст,
                task_id=self.task.task_id if self.task else "",
                stage=self.stage,
                tokens=ответ.total_tokens, cost=ответ.cost,
            )

        return Answer(
            text=текст,
            prompt=промпт,
            usage=ответ.to_dict(),
            attempts=попытки,
            violations=нарушения,
            deviations=расхождения,
            blocked=bool(нарушения),
            escalated_to=эскалация,
            refusal=отказ,
            применимые=[и.код for и in применимые],
            исправлено=исправлено,
            самоотчёт=самоотчёт,
            вердикт=вердикт,
            routing=маршрут,
            routing_entry=запись_маршрута,
            soft=мягкая,
            переход=попытка,
            вызовы=[в.to_dict() for в in вызовы],
            заявки=[з.to_dict() for з in заявки],
            прогон=_протокол_прогона(вызовы),
            маршрут=выбор_серверов.to_dict() if выбор_серверов else None,
            **self._из_планировщика(internal),
        )

    # --- инструменты MCP -----------------------------------------------------

    @property
    def инструменты_включены(self) -> bool:
        return self.toolbox is not None

    def подключить_инструменты(self) -> dict[str, Any]:
        """Открывает соединения с серверами и отдаёт сводку. Идемпотентно."""
        if self.toolbox is None:
            raise AgentError(
                "Инструменты MCP не включены. Включите их ключом «--инструменты» "
                "(консоль) или переключателем «Давать инструменты модели» (страница).")
        try:
            self.toolbox.открыть()
        except MCPConfigError as exc:
            raise AgentError(str(exc)) from exc
        return self.toolbox.сводка()

    def список_инструментов(self) -> list[dict[str, Any]]:
        if self.toolbox is None:
            return []
        self.подключить_инструменты()
        return [и.to_dict() for и in self.toolbox.инструменты]

    def _для_модели(self) -> list[dict[str, Any]] | None:
        """Описания инструментов для поля tools. None — инструментов нет вовсе.

        Недоступный сервер не отменяет ответ: агент отвечает тем, что есть, а о
        недоступном скажет интерфейс — причина лежит в сводке.
        """
        if self.toolbox is None:
            return None
        self.подключить_инструменты()
        return self.toolbox.для_модели() or None

    def _ящик_для(self, инструмент: str, сервер: str = "") -> Toolbox:
        """Инструментарий для одного вызова, даже когда инструменты не включены.

        Имя вида «tracker__get_issue» само говорит, какой сервер нужен, поэтому
        требовать вдобавок «--инструменты tracker» было бы придиркой: человек
        уже назвал инструмент. Поднимается ровно один сервер — не все подряд.
        """
        имя_сервера = сервер or инструмент.split(mcp_names.РАЗДЕЛИТЕЛЬ)[0]
        if self.toolbox is None:
            self.toolbox = self._ящик_маршрута([имя_сервера] if имя_сервера else [])
        self.подключить_инструменты()
        if self.toolbox.найти(инструмент) is None and имя_сервера:
            # С Дня 20 набор серверов у каждого запроса свой, и заявка,
            # заведённая вчера, вполне может звать инструмент, которого в
            # сегодняшнем наборе нет. Поднимаем нужный сервер отдельно, не
            # трогая текущий набор: он ещё пригодится этому разговору.
            отдельный = self._ящик_маршрута([имя_сервера])
            try:
                отдельный.открыть()
            except MCPConfigError as exc:
                raise AgentError(str(exc)) from exc
            if отдельный.найти(инструмент) is not None:
                return отдельный
        return self.toolbox

    def call_tool(self, имя: str, аргументы: dict[str, Any] | None = None) -> ToolResult:
        """Явный вызов инструмента человеком — из консоли или со страницы.

        Подтверждения здесь не спрашивают, и это не противоречие: подтверждение
        существует, чтобы между решением МОДЕЛИ и изменением данных стоял
        человек. Когда вызов затеял сам человек, он и есть это решение.
        """
        self._ящик_для(имя)
        try:
            результат = self.toolbox.вызвать(имя, аргументы)
        except ToolboxError as exc:
            raise AgentError(str(exc)) from exc
        self.memory.log_tool_call(результат, зачем="вызов вручную")
        return результат

    def pending_calls(self) -> list[dict[str, Any]]:
        return [з.to_dict() for з in self.memory.pending_calls()]

    def confirm_call(self, номер: int) -> ToolResult:
        """Исполняет заявку, которую человек подтвердил."""
        заявка = self.memory.calls.get(номер)
        if заявка is None:
            raise AgentError(f"Заявки №{номер} нет.")
        if not заявка.ждёт:
            raise AgentError(f"Заявка №{номер} уже {заявка.состояние}.")
        ящик = self._ящик_для(заявка.инструмент, заявка.сервер)
        try:
            результат = ящик.вызвать(заявка.инструмент, заявка.аргументы)
        except ToolboxError as exc:
            self.memory.resolve_call(номер, СБОЙ, почему=str(exc))
            raise AgentError(f"Заявка №{номер} не исполнена: {exc}") from exc

        self.memory.log_tool_call(результат, зачем=заявка.зачем)
        self.memory.resolve_call(
            номер, ВЫПОЛНЕНА if результат.ок else СБОЙ,
            почему="" if результат.ок else результат.текст[:200],
            результат=результат.to_dict())
        # Выполненный вызов — часть разговора: следующий ответ должен знать, что
        # комментарий уже добавлен, иначе агент предложит добавить его снова.
        self.memory.remember_message(
            "assistant",
            f"Выполнен подтверждённый вызов {заявка.инструмент}: "
            f"{результат.текст[:300] if результат.ок else 'ошибка — ' + результат.текст[:200]}",
            task_id=self.task.task_id if self.task else "", stage=self.stage)
        return результат

    def reject_call(self, номер: int, почему: str = "") -> dict[str, Any]:
        """Отклоняет заявку: вызов не делается вовсе."""
        заявка = self.memory.calls.get(номер)
        if заявка is None:
            raise AgentError(f"Заявки №{номер} нет.")
        if not заявка.ждёт:
            raise AgentError(f"Заявка №{номер} уже {заявка.состояние}.")
        решена = self.memory.resolve_call(номер, ОТКЛОНЕНА, почему=почему or "отклонена человеком")
        self.memory.remember_message(
            "assistant", f"Человек отклонил вызов {заявка.инструмент}"
                         + (f": {почему}" if почему else "."),
            task_id=self.task.task_id if self.task else "", stage=self.stage)
        return решена.to_dict()

    # --- планировщик (День 18) -----------------------------------------------

    @property
    def планировщик(self) -> ScheduleStore:
        """Хранилище планировщика — открывается при первом обращении.

        Читает агент отсюда напрямую: это его собственная память, как диалог
        или задачи. А вот меняет расписание он только через MCP-инструмент —
        тем же путём, которым это делает модель и которым работник исполняет
        задания. Иначе у одного и того же действия было бы две дороги, и
        расходиться они начали бы в первый же день.
        """
        if self._scheduler is None:
            try:
                self._scheduler = ScheduleStore(self.scheduler_path)
            except ScheduleStoreError as exc:
                raise AgentError(str(exc)) from exc
        return self._scheduler

    def _планировщик_через_mcp(self, инструмент: str,
                               аргументы: dict[str, Any]) -> ToolResult:
        """Вызов инструмента планировщика от имени человека, без заявки."""
        результат = self.call_tool(f"{ПЛАНИРОВЩИК}{mcp_names.РАЗДЕЛИТЕЛЬ}{инструмент}",
                                   аргументы)
        if not результат.ок:
            raise AgentError(результат.текст or f"Планировщик отказал: {инструмент}.")
        return результат

    def задания(self, только_активные: bool = True, сколько: int = 50) -> list[dict[str, Any]]:
        return [з.to_dict() for з in self.планировщик.задания(только_активные, сколько)]

    def запланировать(self, инструмент: str, когда: str, аргументы: dict[str, Any] | None = None,
                      зачем: str = "") -> dict[str, Any]:
        """Ставит вызов в расписание. Просьба идёт через MCP-сервер планировщика."""
        результат = self._планировщик_через_mcp("schedule_job", {
            "tool": инструмент, "schedule": когда,
            "arguments": аргументы or {}, "why": зачем})
        return результат.данные if isinstance(результат.данные, dict) else {}

    def отменить_задание(self, номер: int) -> dict[str, Any]:
        результат = self._планировщик_через_mcp("cancel_job", {"job": int(номер)})
        return результат.данные if isinstance(результат.данные, dict) else {}

    def собрать_сводку(self, за: str = "сутки") -> dict[str, Any]:
        """Считает сводку немедленно — тем же инструментом, что и по расписанию."""
        результат = self._планировщик_через_mcp("build_digest", {"period": за})
        return результат.данные if isinstance(результат.данные, dict) else {}

    def сводки(self, сколько: int = 10) -> list[dict[str, Any]]:
        return [с.to_dict() for с in self.планировщик.сводки(сколько)]

    def последняя_сводка(self) -> dict[str, Any] | None:
        сводка = self.планировщик.последняя_сводка()
        return сводка.to_dict() if сводка else None

    def напоминания(self, только_новые: bool = False,
                    сколько: int = 20) -> list[dict[str, Any]]:
        return [н.to_dict() for н in self.планировщик.напоминания(сколько, только_новые)]

    def запуски(self, задание: int = 0, сколько: int = 20) -> list[dict[str, Any]]:
        return [з.to_dict() for з in self.планировщик.запуски(задание, сколько)]

    def состояние_планировщика(self) -> dict[str, Any]:
        состояние = self.планировщик.счётчики()
        состояние["база"] = self.scheduler_path
        return состояние

    def _из_планировщика(self, internal: bool) -> dict[str, Any]:
        """Свежая сводка и новые напоминания — то, что накопилось без человека.

        Показывается один раз: сразу помечается прочитанным. Иначе агент
        повторял бы одну и ту же сводку в каждом ответе, а человек перестал бы
        её читать — и не заметил бы новую.

        Служебные вызовы (шаги сценария) ничего отсюда не получают: их текст
        собирается для следующего шага, и напоминание в нём было бы мусором.
        """
        if internal or not os.path.exists(self.scheduler_path):
            return {}
        try:
            хранилище = self.планировщик
            сводка = хранилище.непрочитанная_сводка()
            новые = хранилище.напоминания(сколько=10, только_новые=True)
            if сводка is not None:
                хранилище.прочитать_сводку(сводка.номер)
            if новые:
                хранилище.прочитать_напоминания([н.номер for н in новые])
        except (ScheduleStoreError, AgentError) as exc:
            log.info("Планировщик недоступен: %s", exc)
            return {}
        return {"сводка": сводка.to_dict() if сводка else None,
                "напоминания": [н.to_dict() for н in новые]}

    # --- конвейеры (День 19) ---------------------------------------------------

    @property
    def каталог_конвейеров(self) -> Any:
        """Описания цепочек из pipelines.json — читаются при первом обращении."""
        if self._pipelines is None:
            from agent.mcp.toolbox import путь_конвейеров
            from agent.pipeline.spec import Каталог
            # Путь спрашивается там же, где его берёт инструментарий: у сервера
            # конвейеров из mcp-servers.json. Иначе консоль показывала бы одни
            # цепочки, а выполнялись бы другие.
            self._pipelines = Каталог(путь_конвейеров(self.mcp_config))
        return self._pipelines

    @property
    def прогоны_конвейеров(self) -> PipelineStore:
        """Хранилище прогонов — тот же файл, что у планировщика."""
        if self._pipeline_runs is None:
            try:
                self._pipeline_runs = PipelineStore(self.scheduler_path)
            except PipelineStoreError as exc:
                raise AgentError(str(exc)) from exc
        return self._pipeline_runs

    def конвейеры(self) -> list[dict[str, Any]]:
        """Какие цепочки описаны: шаги, вход, инструменты."""
        from agent.pipeline.spec import ОшибкаКонвейера
        try:
            return [к.to_dict() for к in self.каталог_конвейеров.все()]
        except ОшибкаКонвейера as exc:
            raise AgentError(str(exc)) from exc

    def запустить_конвейер(self, имя: str, вход: dict[str, Any] | None = None,
                           зачем: str = "") -> dict[str, Any]:
        """Выполняет цепочку целиком. Просьба идёт через MCP-сервер конвейеров.

        Дорога одна и та же и для человека, и для модели: инструмент
        run_pipeline. Иначе у запуска было бы две реализации — и разошлись бы
        они в тот день, когда в одну из них добавят проверку.
        """
        результат = self.call_tool(f"{КОНВЕЙЕР}{mcp_names.РАЗДЕЛИТЕЛЬ}run_pipeline",
                                   {"pipeline": имя, "input": dict(вход or {}),
                                    "why": зачем})
        if not результат.ок:
            raise AgentError(результат.текст or f"Конвейер «{имя}» не выполнен.")
        return результат.данные if isinstance(результат.данные, dict) else {}

    def прогоны(self, конвейер: str = "", сколько: int = 10,
                с_шагами: bool = False) -> list[dict[str, Any]]:
        return [п.to_dict() for п in self.прогоны_конвейеров.прогоны(
            конвейер, сколько, с_шагами)]

    def прогон_конвейера(self, номер: int) -> dict[str, Any] | None:
        прогон = self.прогоны_конвейеров.прогон(int(номер))
        return прогон.to_dict() if прогон else None

    def последний_прогон(self, конвейер: str = "") -> dict[str, Any] | None:
        прогон = self.прогоны_конвейеров.последний(конвейер)
        return прогон.to_dict() if прогон else None

    def состояние_конвейеров(self) -> dict[str, Any]:
        """Короткая сводка для страницы и консоли: что описано и что выполнялось."""
        from agent.pipeline.spec import ОшибкаКонвейера
        итог: dict[str, Any] = {"файл": getattr(self.каталог_конвейеров, "путь", ""),
                                "конвейеры": [], "ошибка": ""}
        try:
            итог["конвейеры"] = [
                {"имя": к.имя, "описание": к.описание, "цепочка": к.словами(),
                 "шагов": len(к.шаги), "инструменты": к.инструменты(),
                 "вход": [п.to_dict() for п in к.вход]}
                for к in self.каталог_конвейеров.все()]
        except ОшибкаКонвейера as exc:
            итог["ошибка"] = str(exc)
        if os.path.exists(self.scheduler_path):
            хранилище = self.прогоны_конвейеров
            счёт = хранилище.счётчики()
            последний = хранилище.последний()
            итог["последний"] = последний.to_dict() if последний else None
        else:
            счёт = {"прогонов": 0, "сбоев": 0, "конвейеры": {}}
            итог["последний"] = None
        # Не update(): у счётчиков тоже есть ключ «конвейеры», и он затёр бы
        # описания цепочек их же статистикой.
        итог["прогонов"] = счёт["прогонов"]
        итог["сбоев"] = счёт["сбоев"]
        итог["по_конвейерам"] = счёт["конвейеры"]
        return итог


    # --- оркестровка: маршрут запроса и длинный флоу (День 20) ----------------

    @property
    def маршрутизатор(self) -> Маршрутизатор:
        """Кто выбирает серверы под запрос. Создаётся при первом обращении."""
        if self._маршрутизатор is None:
            from agent.mcp.config import load as прочитать_серверы
            try:
                серверы = прочитать_серверы(self.mcp_config)
            except MCPConfigError as exc:
                raise AgentError(str(exc)) from exc
            # Без маршрутизации клиент не передаётся вовсе: тогда маршрут
            # считается правилами по «темам» и не стоит ни одного запроса к
            # модели. Так «--без-маршрутизации --маршрут «…»» показывает выбор
            # даром — это удобно и в проверках, и когда провайдер молчит.
            self._маршрутизатор = Маршрутизатор(
                серверы, клиент=self.client if self.маршрутизация else None)
        return self._маршрутизатор

    @property
    def правила_порядка(self) -> Порядок:
        """Правила «что после чего» — flow-rules.json рядом с файлом серверов."""
        if self._порядок is None:
            путь = os.environ.get("FLOW_RULES", "").strip()
            if not путь:
                рядом = os.path.dirname(os.path.abspath(self.mcp_config or ФАЙЛ_СЕРВЕРОВ))
                путь = os.path.join(рядом, ФАЙЛ_ПРАВИЛ)
            try:
                self._порядок = загрузить_правила(путь)
            except ОшибкаПорядка as exc:
                raise AgentError(str(exc)) from exc
        return self._порядок

    @property
    def флоу_хранилище(self) -> FlowStore:
        """Протокол флоу — тот же файл, что у планировщика и конвейеров."""
        if self._flows is None:
            try:
                self._flows = FlowStore(self.scheduler_path)
            except FlowStoreError as exc:
                raise AgentError(str(exc)) from exc
        return self._flows

    def включить_инструменты(self, серверы: list[str] | None = None) -> None:
        """Включает инструменты MCP с заданным набором серверов.

        Нужно интерфейсам: страница присылает набор с каждым запросом. Идёт
        через тот же склад инструментариев, что и маршрутизатор, — иначе
        переключатель поднимал бы вторые процессы тех же серверов.
        """
        имена = [str(и).strip() for и in (серверы or []) if str(и).strip()]
        self._серверы_ключом = [и for и in имена if и != ВСЕ_СЕРВЕРЫ]
        self.toolbox = self._ящик_маршрута(self._серверы_ключом)

    def выключить_инструменты(self) -> None:
        """Закрывает все соединения: stdio-сервер — это живой процесс."""
        for ящик in self._ящики.values():
            ящик.close()
        self._ящики.clear()
        self.toolbox = None
        self._серверы_ключом = []

    def _ящик_маршрута(self, серверы: list[str]) -> Toolbox:
        """Инструментарий для набора серверов; одинаковые наборы переиспользуются."""
        ключ = tuple(sorted(серверы))
        ящик = self._ящики.get(ключ)
        if ящик is None:
            ящик = Toolbox(list(серверы), self.mcp_config)
            self._ящики[ключ] = ящик
        return ящик

    def _направить(self, вопрос: str) -> Маршрут | None:
        """Выбирает серверы под запрос и подставляет их инструментарий.

        Без маршрутизации агент отдавал бы модели описания всех девяти серверов
        в каждом запросе — полсотни инструментов и почти одиннадцать тысяч
        токенов. Это и дорого (список платится каждый раз, а не однажды), и
        вредно: из пятидесяти похожих имён выбирают хуже, чем из десяти.
        """
        if self.toolbox is None or not self.маршрутизация:
            return None
        # Явно названный набор маршрутизатор не пересматривает: человек знает,
        # чего хочет, а лишний круг модели на каждый вопрос стоит денег.
        названные = self._серверы_ключом or list(getattr(self.toolbox, "имена", []) or [])
        try:
            маршрут = self.маршрутизатор.выбрать(вопрос, названные=названные or None)
        except AgentError:
            # Файла серверов нет или он испорчен. Это причина не отвечать на
            # вопрос про MCP, но не причина рушить обычный ответ.
            return None
        if названные:
            return маршрут
        if маршрут.серверы:
            self.toolbox = self._ящик_маршрута(маршрут.серверы)
        return маршрут

    def маршрут_запроса(self, вопрос: str) -> dict[str, Any]:
        """Куда пошёл бы этот запрос. Показывает выбор, ничего не вызывая."""
        вопрос = (вопрос or "").strip()
        if not вопрос:
            raise AgentError("Пустой запрос: маршрутизировать нечего.")
        маршрут = self.маршрутизатор.выбрать(вопрос, названные=self._серверы_ключом or None)
        итог = маршрут.to_dict()
        итог["все_серверы"] = self.маршрутизатор.описания()
        return итог

    def _оркестр(self) -> Оркестр:
        """Исполнитель флоу с подключёнными к памяти агента заявками."""
        def завести(инструмент: str, аргументы: dict[str, Any], зачем: str) -> int:
            найден = self.toolbox.найти(инструмент) if self.toolbox else None
            заявка = self.memory.request_call(
                инструмент=найден.полное_имя if найден else инструмент,
                аргументы=аргументы, сервер=найден.сервер if найден else "",
                зачем=зачем,
                описание=(найден.заголовок or найден.описание[:120]) if найден else "",
                задача=self.task.task_id if self.task else "")
            return заявка.номер

        def прочитать(номер: int) -> dict[str, Any] | None:
            заявка = self.memory.calls.get(int(номер))
            return заявка.to_dict() if заявка else None

        return Оркестр(
            self.client, self.toolbox, self.флоу_хранилище, self.правила_порядка,
            модель=self.model_key or self.model_for(self.stage),
            кругов=self.кругов_флоу, вызовов=self.вызовов_флоу,
            подтверждать=self.confirm_writes, заявка=завести, читать_заявку=прочитать,
            кто="человек", пользователь=self.user_id)

    def флоу(self, вопрос: str, задача: str = "", планировать: bool = True) -> dict[str, Any]:
        """Длинный флоу: маршрут → план → вызовы разных серверов → сверка.

        Отдельный вход, а не режим обычного ответа, и это осознанно: у флоу своя
        цена (несколько кругов модели и до дюжины вызовов) и своё право
        остановиться на подтверждении. Обычный вопрос не должен незаметно
        превращаться в получасовую работу с чужими системами.
        """
        вопрос = (вопрос or "").strip()
        if not вопрос:
            raise AgentError("Пустой вопрос: с чего начинать флоу?")
        if self.toolbox is None:
            raise AgentError(
                "Инструменты MCP не включены, а без них длинный флоу — это обычный "
                "вопрос. Включите их ключом «--инструменты» (консоль) или "
                "переключателем «Давать инструменты модели» (страница).")
        маршрут = self._направить(вопрос)
        if маршрут is not None and маршрут.пуст:
            raise AgentError(
                f"Маршрутизатор не выбрал ни одного сервера: {маршрут.словами()}. "
                "Назовите серверы явно: «--инструменты tracker,pipeline».")
        self.подключить_инструменты()

        # Флоу — это разговор, а не служебный вызов: реплика человека должна
        # остаться в короткой памяти, иначе следующий ответ её не вспомнит.
        if SHORT in self.layers:
            self.memory.remember_message(
                "user", вопрос, task_id=self.task.task_id if self.task else "",
                stage=self.stage)

        промпт = self.builder.build(вопрос, self.task, set(self.layers), with_tools=True)
        try:
            запись = self._оркестр().запустить(
                вопрос, промпт.messages, маршрут=(маршрут.to_dict() if маршрут else {}),
                задача=задача or (self.task.task_id if self.task else ""),
                планировать=планировать)
        except ОшибкаФлоу as exc:
            raise AgentError(str(exc)) from exc
        self._запомнить_флоу(запись)
        return запись.to_dict()

    def продолжить_флоу(self, номер: int) -> dict[str, Any]:
        """Продолжает флоу, остановленный на подтверждении, с того же места."""
        запись = self.флоу_хранилище.флоу(int(номер))
        if запись is None:
            raise AgentError(f"Флоу №{номер} нет.")
        if not запись.ждёт:
            raise AgentError(f"Флоу №{номер} не ждёт подтверждения: он {запись.состояние}.")
        if self.toolbox is None or запись.серверы:
            # Набор серверов берётся из самой записи: продолжение может идти в
            # другом процессе, где маршрут ещё не считался.
            self.toolbox = self._ящик_маршрута(запись.серверы)
        self.подключить_инструменты()
        промпт = self.builder.build(запись.запрос, self.task, set(self.layers),
                                    with_tools=True)
        try:
            свежая = self._оркестр().продолжить(int(номер), промпт.messages)
        except ОшибкаФлоу as exc:
            raise AgentError(str(exc)) from exc
        self._запомнить_флоу(свежая)
        return свежая.to_dict()

    # --- ответы по документам (День 22) ---------------------------------------

    def _отвечатель(self, стратегия: str = "структура", сколько: int = 5):
        """Индекс создаётся при первом вопросе по документам, а не при запуске:
        агенту, которого о документах не спрашивают, эмбеддер не нужен."""
        from agent.rag import Индексатор
        from agent.rag.answer import Отвечатель
        if getattr(self, "_rag_индексатор", None) is None:
            self._rag_индексатор = Индексатор()
        # Отвечатель живёт вместе с агентом: в нём поисковик с реранкерами, и
        # кросс-энкодер (570 МБ) не должен грузиться заново на каждый вопрос.
        кэш = self.__dict__.setdefault("_rag_отвечатели", {})
        ключ = (стратегия, сколько, id(self._rag_индексатор), id(self.client))
        if ключ not in кэш:
            кэш[ключ] = Отвечатель(self._rag_индексатор, self.client, стратегия, сколько)
        return кэш[ключ]

    def спросить_документы(self, вопрос: str, режим: str = "rag", стратегия: str = "структура",
                           сколько: int = 5, сохранить: bool = True, поиск: str = ""):
        """Вопрос по документам в режиме «с RAG» или «без RAG».

        Режим с RAG: поиск в индексе Дня 21 → фрагменты с паспортом в промпт →
        ответ со ссылками [n]. Без RAG — тот же вопрос той же модели без
        фрагментов. Память агента и инварианты ГИС в этот промпт не идут: у
        вопроса о договоре с ВымпелКомом нет стадии миграции, а самоотчёт по
        инвариантам в таком ответе был бы шумом (так и было на показе Дня 21).
        """
        from agent.rag.answer import РЕЖИМЫ
        from agent.rag.embed import ОшибкаЭмбеддера
        from agent.rag.store import ОшибкаИндекса
        if режим not in РЕЖИМЫ:
            raise AgentError(f"Нет режима «{режим}». Есть: {', '.join(РЕЖИМЫ)}")
        try:
            ответ = self._отвечатель(стратегия, сколько).ответить(вопрос, режим, self.model_key,
                                                                 поиск=поиск)
        except (ValueError, ОшибкаИндекса, ОшибкаЭмбеддера) as exc:
            raise AgentError(str(exc)) from exc
        if ответ.ошибка:
            raise AgentError(ответ.ошибка)
        if сохранить:
            метка = "без RAG" if режим != "rag" else (
                f"с RAG, поиск «{поиск}»" if поиск and поиск != "база" else "с RAG")
            self.memory.short.append(self.session, "user", f"[документы, {метка}] {вопрос}")
            self.memory.short.append(self.session, "assistant", ответ.текст,
                                     tokens=ответ.токенов_вход + ответ.токенов_выход,
                                     cost=ответ.стоимость)
        return ответ

    def сравнить_поиск(self, вопрос: str, режимы: tuple[str, ...] = (), стратегия: str = "структура"):
        """День 23: один вопрос во всех режимах поиска — база, фильтр, rewrite, полный."""
        from agent.rag.retrieval import РЕЖИМЫ_ПОИСКА
        return {р: self.спросить_документы(вопрос, "rag", стратегия, поиск=р)
                for р in (режимы or РЕЖИМЫ_ПОИСКА)}

    def отбор(self, вопрос: str, поиск: str = "полный", стратегия: str = "структура",
              настройки: dict | None = None):
        """Только поиск в два этапа, без ответа модели: что нашлось, что отсеялось.

        настройки — поправки к режиму (реранкер, k_до, k_после, порог, rewrite):
        так порог и top-K настраиваются вживую, не трогая подобранный файл.
        """
        from dataclasses import replace
        from agent.rag.embed import ОшибкаЭмбеддера
        from agent.rag.rerank import ОшибкаРеранкера
        from agent.rag.store import ОшибкаИндекса
        вопрос = (вопрос or "").strip()
        if not вопрос:
            raise AgentError("Пустой вопрос")
        поисковик = self._отвечатель(стратегия).поисковик
        try:
            режим = поисковик.настройки(поиск)
            if настройки:
                допустимо = {"rewrite", "реранкер", "k_до", "k_после", "порог", "порог_косинуса"}
                режим = replace(режим, **{к: з for к, з in настройки.items() if к in допустимо},
                                имя=f"{поиск}*")
            return поисковик.найти(вопрос, режим)
        except (ValueError, ОшибкаИндекса, ОшибкаЭмбеддера, ОшибкаРеранкера) as exc:
            raise AgentError(str(exc)) from exc

    def сравнить_режимы(self, вопрос: str, стратегия: str = "структура", сколько: int = 5):
        """Один вопрос в обоих режимах на одной модели: {"без": …, "rag": …}."""
        return {режим: self.спросить_документы(вопрос, режим, стратегия, сколько)
                for режим in ("без", "rag")}

    def флоу_список(self, сколько: int = 10, состояние: str = "",
                    с_шагами: bool = False) -> list[dict[str, Any]]:
        return [ф.to_dict() for ф in self.флоу_хранилище.список(сколько, состояние, с_шагами)]

    def показать_флоу(self, номер: int) -> dict[str, Any] | None:
        запись = self.флоу_хранилище.флоу(int(номер))
        return запись.to_dict() if запись else None

    def флоу_по_заявке(self, заявка: int) -> dict[str, Any] | None:
        """Какой флоу остановлен на этой заявке. Нужно, чтобы после «да»
        человеку сказали, что работа не закончена и её можно продолжить."""
        for запись in self.флоу_хранилище.ждущие():
            if запись.заявка == int(заявка):
                return запись.to_dict()
        return None

    def состояние_флоу(self) -> dict[str, Any]:
        """Сводка для страницы и консоли: правила, серверы, последние флоу."""
        итог: dict[str, Any] = {
            "маршрутизация": self.маршрутизация,
            "бюджет": {"кругов": self.кругов_флоу, "вызовов": self.вызовов_флоу},
            "правила": self.правила_порядка.to_dict(),
            "серверы": self.маршрутизатор.описания(),
            "флоу": [], "ждут": [], "счётчики": {"флоу": 0, "готовых": 0, "сбоев": 0,
                                                 "ждут": 0, "вызовов": 0},
        }
        if os.path.exists(self.scheduler_path):
            хранилище = self.флоу_хранилище
            итог["флоу"] = [ф.to_dict() for ф in хранилище.список(сколько=10)]
            итог["ждут"] = [ф.to_dict() for ф in хранилище.ждущие()]
            итог["счётчики"] = хранилище.счётчики()
        return итог

    def _запомнить_флоу(self, запись: Any) -> None:
        """Кладёт исход флоу в короткую память — как обычный ответ агента."""
        if SHORT not in self.layers:
            return
        if запись.ждёт:
            текст = (f"Флоу №{запись.номер} остановлен на подтверждении: заявка "
                     f"№{запись.заявка} на «{запись.ожидание.get('инструмент', '')}».")
        else:
            текст = запись.итог or f"Флоу №{запись.номер}: {запись.состояние}."
        self.memory.remember_message(
            "assistant", текст, task_id=self.task.task_id if self.task else "",
            stage=self.stage)

    def _ответ_модели(
        self,
        ключ_модели: str,
        сообщения: list[dict[str, Any]],
        вопрос: str,
        инструменты: list[dict[str, Any]] | None,
        цепочка: list[dict[str, Any]],
        вызовы: list[ToolResult],
        заявки: list[Any],
    ) -> Reply:
        """Спрашивает модель, при надобности вызывая инструменты.

        Протокол простой и полностью на нашей стороне: модель не исполняет
        ничего сама, она лишь просит. Агент решает, исполнять ли, исполняет и
        возвращает результат отдельным сообщением роли tool. Это повторяется,
        пока модель не ответит словами.

        Кругов ограниченное число. Без предела модель, которой инструмент
        отвечает ошибкой, зовёт его снова и снова — и каждый круг это полный
        запрос с историей, то есть деньги. После последнего круга запрос
        повторяется без инструментов: пусть отвечает тем, что уже есть.
        """
        if not инструменты:
            return self.client.call(ключ_модели, сообщения + цепочка)

        for круг in range(1, self.tool_rounds + 1):
            ответ = self.client.call(ключ_модели, сообщения + цепочка, tools=инструменты)
            if not ответ.tool_calls:
                return ответ
            цепочка.append(ответ.message)
            for просьба in ответ.tool_calls:
                текст = self._исполнить_просьбу(просьба, вопрос, вызовы, заявки)
                цепочка.append({"role": "tool", "tool_call_id": просьба.ид,
                                "name": просьба.имя, "content": текст})
            log.info("Круг инструментов %d: вызовов %d", круг, len(ответ.tool_calls))

        log.warning("Модель просит инструменты %d круга подряд — отвечаем без них",
                    self.tool_rounds)
        хватит = {"role": "system", "content":
                  "Инструменты больше недоступны. Ответь по тому, что уже получено, "
                  "и скажи, чего узнать не удалось."}
        итог = self.client.call(ключ_модели, сообщения + цепочка + [хватит])
        if not итог.text.strip():
            # Модель и без инструментов ответила одними просьбами о вызовах.
            # Пустой текст дальше по цепочке — это упавшая запись в память и
            # пустой экран у человека, поэтому место ответа занимает объяснение.
            итог.text = ("Не удалось получить ответ словами: модель раз за разом "
                         "запрашивала инструменты. Что успели получить — в списке "
                         "вызовов.")
        return итог

    def _исполнить_просьбу(self, просьба: ToolCall, вопрос: str,
                           вызовы: list[ToolResult], заявки: list[Any]) -> str:
        """Одна просьба о вызове: исполнить, отложить в заявку или объяснить отказ.

        Что бы ни случилось, модель получает текст, из которого понятно, что
        делать дальше. Исключений наружу отсюда не выходит: сбой вызова — это
        обстоятельство разговора, а не поломка агента.
        """
        if просьба.ошибка:
            return (f"Вызов не выполнен: {просьба.ошибка}. "
                    "Пришли аргументы корректным объектом JSON.")

        ящик = self.toolbox
        if ящик is None:
            return "Вызов не выполнен: инструменты выключены."

        if ящик.меняет(просьба.имя, просьба.аргументы) and self.confirm_writes:
            инструмент = ящик.найти(просьба.имя)
            if инструмент is None:
                return (f"Вызов не выполнен: инструмента «{просьба.имя}» нет. "
                        f"Доступны: {', '.join(и.полное_имя for и in ящик.инструменты)}.")
            # Планируемый вызов меняет данные не сейчас, а когда до него дойдёт
            # работник. Человеку надо сказать именно это: он подтверждает не
            # «запись в расписание», а то, что произойдёт потом без него.
            планируемый = ящик.планируемый(просьба.имя, просьба.аргументы)
            цепочка = ящик.конвейер_вызова(просьба.имя, просьба.аргументы)
            заявка = self.memory.request_call(
                инструмент=инструмент.полное_имя, аргументы=просьба.аргументы,
                сервер=инструмент.сервер, зачем=вопрос,
                описание=инструмент.заголовок or инструмент.описание[:120],
                задача=self.task.task_id if self.task else "")
            заявки.append(заявка)
            if планируемый:
                что = (f"«{инструмент.полное_имя}» поставит в расписание вызов "
                       f"«{планируемый}», который меняет данные")
            elif цепочка:
                # Человек подтверждает не «запустить конвейер», а то, что цепочка
                # в конце напишет в чужую систему. Значит, назвать надо шаг.
                шаги = ", ".join(ящик.меняющие_шаги(цепочка)) or "один из шагов"
                что = (f"«{инструмент.полное_имя}» выполнит конвейер «{цепочка}», "
                       f"в котором меняет данные шаг: {шаги}")
            else:
                что = f"«{инструмент.полное_имя}» меняет данные"
            return (f"Вызов не выполнен: {что}, а такие вызовы исполняются только с "
                    f"подтверждения человека. Заведена заявка №{заявка.номер}. Скажи "
                    "человеку, что именно предлагаешь сделать, и попроси подтвердить её.")

        try:
            результат = ящик.вызвать(просьба.имя, просьба.аргументы)
        except ToolboxError as exc:
            неудача = ToolResult(сервер="", инструмент=просьба.имя, полное_имя=просьба.имя,
                                 аргументы=просьба.аргументы, ок=False, текст=str(exc))
            вызовы.append(неудача)
            self.memory.log_tool_call(неудача, зачем=вопрос)
            return f"Вызов не выполнен: {exc}"

        вызовы.append(результат)
        self.memory.log_tool_call(результат, зачем=вопрос)
        return результат.для_модели()

    def _просьба_о_переходе(self, текст: str) -> tuple[dict[str, Any], str]:
        """Разбирает просьбу модели сменить стадию и отвечает на неё воротами.

        Наружу уходит не служебная строка, а результат: либо отметка о переходе,
        либо тот же отказ, который получил бы человек. Так «реакция ассистента»
        на недопустимый переход не зависит от настроения модели — её пишет код.
        """
        стадия = просьба_о_переходе(текст)
        if not стадия:
            return {}, текст
        чистый = убрать_маркер(текст)
        вердикт = self.ворота.перевести(self.task, стадия, МОДЕЛЬ,
                                        "по просьбе ассистента")
        self.memory.working.save(self.task)
        if вердикт.можно:
            основания = "; ".join(f"{п.код}: {п.видно}" for п in вердикт.проверки)
            примечание = (
                f"Стадия задачи переведена по просьбе ассистента: "
                f"{вердикт.откуда} → {вердикт.куда}."
                + (f"\nУсловия перехода проверены — {основания}." if основания else "")
            )
            self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                             f"-> {стадия}", applied=True,
                             reason="переход по просьбе ассистента")
        else:
            примечание = (
                f"Ассистент попросил перейти на стадию «{стадия}». Запрос отклонён.\n\n"
                + вердикт.отказ.текст()
            )
            self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                             f"-X-> {стадия}", applied=False,
                             reason=f"переход отклонён (модель): {вердикт.причина}")
        return вердикт.to_dict(), (чистый + "\n\n" + примечание).strip()

    def _отказать(self, нарушения: list[Violation], когда: str, question: str,
                  слои: set[str]) -> Answer:
        """Собирает ответ-отказ, не обращаясь к модели.

        Отказ пишет код, а не модель: тогда он одинаков при каждом конфликте,
        называет конкретный инвариант, его обоснование и допустимую
        альтернативу — всё это заранее записано в самом инварианте.
        """
        отказ = self.validator.refusal(нарушения, когда, question)
        промпт = self.builder.build(question, self.task, слои)
        if SHORT in слои:
            self.memory.remember_message("assistant", отказ.текст(),
                                         task_id=self.task.task_id if self.task else "",
                                         stage=self.stage)
        self.memory._log("явное-указание", LONG, "инварианты",
                         "; ".join(н.код for н in нарушения), applied=False,
                         reason=f"отказ {когда}: запрос требует нарушить инвариант")
        return Answer(
            text=отказ.текст(), prompt=промпт, attempts=0,
            violations=нарушения, blocked=True, refusal=отказ,
            применимые=[н.код for н in нарушения],
        )

    def _answer_within_rules(
        self,
        prompt: BuiltPrompt,
        question: str,
        layers: set[str],
        model_key: str = "",
        step_role: str = "",
        personal: bool = True,
        explanatory: bool = False,
        with_tools: bool = False,
        вызовы: list[ToolResult] | None = None,
        заявки: list[Заявка] | None = None,
    ) -> tuple[Reply, int, list[Violation], list[Deviation], str, list[str]]:
        """Получает ответ, который укладывается и в инварианты, и в профиль.

        Проверок две, и они разной силы. Инвариант — запрет проекта: ответ,
        который его нарушает, отдавать нельзя, и если переделать не удалось,
        агент честно говорит об этом. Расхождение с профилем — это «не так, как
        просил пользователь»: повторить стоит, но ответ по существу верен, и
        отдать его лучше, чем не отдать ничего.

        Лестница повторов общая: сначала просим переделать ту же модель, и
        только если не помогло — поднимаемся на ступень. Прыгать сразу на самую
        дорогую незачем, чаще всего хватает напоминания.
        """
        ключ_модели = model_key or self.model_for(prompt.stage)
        сообщения = prompt.messages
        вызовы = вызовы if вызовы is not None else []
        заявки = заявки if заявки is not None else []
        инструменты = self._для_модели() if with_tools else None
        # Переписка с инструментами: сообщение ассистента с просьбой и ответы
        # роли tool. Она переживает повторы: если ответ пришлось переделывать
        # из-за инварианта, добытые из трекера факты терять незачем — иначе
        # переделанный ответ окажется хуже исходного.
        цепочка: list[dict[str, Any]] = []
        эскалация = ""
        нарушения: list[Violation] = []
        расхождения: list[Deviation] = []
        # Нарушения, которые были на ранних попытках и ушли после напоминания.
        # Без этого отчёт не отличает «конфликта не было» от «конфликт был, и
        # агент переписал ответ сам» — а это разные исходы.
        исправлено: list[str] = []

        for попытка in range(1, MAX_ATTEMPTS + 1):
            try:
                ответ = self._ответ_модели(ключ_модели, сообщения, question,
                                           инструменты, цепочка, вызовы, заявки)
            except LLMError as exc:
                raise AgentError(str(exc)) from exc

            нарушения = (self.validator.check(ответ.text, explanatory)
                         if LONG in layers else [])
            # Самоотчёт требуется только от ответа, который увидит человек:
            # заставлять промежуточный шаг сценария перечислять инварианты —
            # значит тратить его выходные токены на служебную строку.
            # Самоотчёт стоит последней строкой ответа, поэтому обрезанный по
            # лимиту токенов ответ не может его содержать в принципе. Требовать
            # его тут — значит трижды переспросить и трижды получить тот же
            # обрубок. Именно так и вышло на первом живом прогоне.
            обрезан = ответ.finish_reason == "length"
            пропущено = (
                self.validator.check_self_report(
                    ответ.text, self.validator.applicable(question, ответ.text))
                if LONG in layers and personal and self.require_self_report
                and not обрезан else []
            )
            if обрезан and self.require_self_report:
                log.warning("Ответ обрезан по длине — самоотчёт не требуем")
            расхождения = (
                self.checker().check(ответ.text)
                if personal and self.follow_profile and LONG in layers else []
            )
            жёсткие = [о for о in расхождения if о.hard]
            if not нарушения and not жёсткие and not пропущено:
                return ответ, попытка, [], расхождения, эскалация, исправлено

            log.warning(
                "Попытка %d: нарушений %d, расхождений с профилем %d, "
                "пропущено в самоотчёте %d",
                попытка, len(нарушения), len(жёсткие), len(пропущено),
            )
            исправлено.extend(н.код for н in нарушения if н.код not in исправлено)
            if попытка == MAX_ATTEMPTS:
                break

            напоминания = []
            if нарушения:
                напоминания.append(self.validator.reminder(нарушения))
            if жёсткие:
                напоминания.append(self.checker().reminder(жёсткие))
            if пропущено:
                напоминания.append(self.validator.self_report_reminder(пропущено))
            повтор = self.builder.build(
                question, self.task, layers,
                extra_note="\n\n".join(напоминания), step_role=step_role,
                personal=personal, with_tools=with_tools,
            )
            сообщения = повтор.messages
            # Эскалация — только из-за инварианта. Расхождение с профилем
            # косметическое: платить за ответ вчетверо дороже потому, что он на
            # двадцать слов длиннее просимого, — плохая сделка. Повторяем на той
            # же модели: напоминание обычно помогает.
            if попытка >= 2 and нарушения:
                следующая = catalog.escalate(ключ_модели)
                if следующая != ключ_модели:
                    ключ_модели, эскалация = следующая, следующая

        # То, что осталось нарушенным, исправленным не считается.
        осталось = {н.код for н in нарушения}
        исправлено = [к for к in исправлено if к not in осталось]
        return ответ, MAX_ATTEMPTS, нарушения, расхождения, эскалация, исправлено

    # --- планирование --------------------------------------------------------

    def plan(self, note: str = "") -> Answer:
        """Просит модель составить план и кладёт его в рабочую память.

        Это единственное место, где ответ модели превращается в структуру, а не
        остаётся текстом: план — рабочие данные задачи, и жить он должен в
        рабочей памяти, а не в переписке.
        """
        if self.task is None:
            raise AgentError("Активной задачи нет: планировать нечего.")
        if self.task.stage != PLANNING:
            raise AgentError(
                f"План составляется на стадии planning, а задача сейчас в «{self.task.stage}»."
            )
        вопрос = (
            f"Составь план задачи «{self.task.title or self.task.task_id}». "
            + (note or "")
            + " Дай нумерованный список шагов, по одному шагу в строке, без пояснений между ними."
        )
        ответ = self.ask(вопрос)
        шаги = [ш.strip() for ш in _ШАГ_ПЛАНА.findall(ответ.text)][:12]
        if шаги:
            self.task = self.memory.remember_plan(self.task, шаги)
        return ответ

    # --- профиль пользователя ------------------------------------------------

    def profile(self) -> dict[str, Any]:
        return self.memory.long.profile.load()

    def needs_setup(self) -> bool:
        """Профиль ещё не настраивали — стоит предложить мастер."""
        return interview.needs_setup(self.profile())

    @staticmethod
    def setup_questions() -> list[dict[str, Any]]:
        return interview.questions()

    def setup(self, answers: dict[str, Any]) -> dict[str, Any]:
        """Применяет ответы мастера настройки к профилю."""
        try:
            профиль = interview.apply(self.profile(), answers)
        except preferences.PreferenceError as exc:
            raise AgentError(str(exc)) from exc
        self.memory.long.profile.save(профиль)
        self.memory._log("явное-указание", LONG, "профиль",
                         preferences.summary(профиль), applied=True,
                         reason="мастер настройки пройден")
        return профиль

    def use_template(self, name: str) -> dict[str, Any]:
        """Берёт готовую заготовку профиля целиком."""
        try:
            профиль = interview.from_template(name, self.profile())
        except preferences.PreferenceError as exc:
            raise AgentError(str(exc)) from exc
        self.memory.long.profile.save(профиль)
        self.memory._log("явное-указание", LONG, "профиль",
                         f"заготовка «{name}»: {preferences.summary(профиль)}",
                         applied=True, reason="выбрана готовая заготовка профиля")
        return профиль

    def set_preference(self, section: str, key: str, value: Any) -> dict[str, Any]:
        """Правит одно предпочтение; недопустимое значение отклоняется."""
        try:
            return self.memory.long.profile.update(section, key, value)
        except Exception as exc:
            raise AgentError(str(exc)) from exc

    # --- инварианты ------------------------------------------------------------

    def invariants(self) -> list:
        """Проектные инварианты плюс личные из профиля."""
        return self.memory.all_invariants()

    def add_invariant(self, инвариант, личный: bool = False):
        """Заводит инвариант. Личный ложится в профиль, проектный — в общий файл."""
        from agent.invariants import ЛИЧНЫЙ, InvariantError
        try:
            if личный:
                инвариант.уровень = ЛИЧНЫЙ
                инвариант.validate()
                профиль = self.memory.long.profile.load()
                прочие = [и for и in (профиль.get("инварианты") or [])
                          if и.get("код", "").lower() != инвариант.код.lower()]
                профиль["инварианты"] = прочие + [инвариант.to_dict()]
                self.memory.long.profile.save(профиль)
            else:
                self.memory.invariants.add(инвариант)
        except InvariantError as exc:
            raise AgentError(str(exc)) from exc
        self.memory._log("явное-указание", LONG, "инварианты", инвариант.правило,
                         applied=True,
                         reason=f"заведён {инвариант.уровень} инвариант «{инвариант.код}»")
        return инвариант

    def remove_invariant(self, код: str) -> bool:
        """Снимает инвариант. Личный — из профиля, проектный — из общего файла."""
        снят = self.memory.invariants.remove(код)
        профиль = self.memory.long.profile.load()
        личные = профиль.get("инварианты") or []
        осталось = [и for и in личные if и.get("код", "").lower() != (код or "").lower()]
        if len(осталось) != len(личные):
            профиль["инварианты"] = осталось
            self.memory.long.profile.save(профиль)
            снят = True
        if снят:
            self.memory._log("явное-указание", LONG, "инварианты", код, applied=True,
                             reason="инвариант снят")
        return снят

    def promote_decision(self, номер: int, код: str = "", вместо: str = ""):
        """Возводит запись журнала решений в инвариант.

        Замыкает то, ради чего журнал и ведётся: решение перестаёт быть заметкой
        на память и становится ограничением, которое нельзя обойти. Обоснованием
        отказа служит та самая причина, по которой решение приняли.
        """
        from agent.invariants import from_decision, InvariantError
        записи = self.memory.long.decisions.all()
        найдена = next((з for з in записи if int(з.get("id", 0)) == int(номер)), None)
        if найдена is None:
            есть = ", ".join(str(з.get("id")) for з in записи) or "журнал пуст"
            raise AgentError(f"В журнале решений нет записи №{номер}. Есть: {есть}.")
        try:
            инвариант = from_decision(найдена, код=код, вместо=вместо)
        except InvariantError as exc:
            raise AgentError(str(exc)) from exc
        return self.add_invariant(инвариант)

    # --- сценарии --------------------------------------------------------------

    def scenarios(self) -> list[Scenario]:
        return self.memory.long.scenarios.all()

    def match_scenario(self, query: str) -> Scenario | None:
        """Есть ли сценарий, чей триггер сработал на этом запросе."""
        return self.memory.long.scenarios.match(query)

    def add_scenario(self, scenario: Scenario) -> Scenario:
        try:
            итог = self.memory.long.scenarios.add(scenario)
        except Exception as exc:
            raise AgentError(str(exc)) from exc
        self.memory._log("явное-указание", LONG, "сценарии", итог.digest(), applied=True,
                         reason="сценарий добавлен пользователем")
        return итог

    def remove_scenario(self, name: str) -> bool:
        return self.memory.long.scenarios.remove(name)

    def run_scenario(
        self,
        query: str,
        name: str = "",
        on_step: Any = None,
        finish: bool = True,
        on_result: Any = None,
        по_шагам: bool = False,
    ) -> RunResult:
        """Исполняет сценарий: каждый шаг — свой агент, своя модель, своя стадия."""
        сценарий = (
            self.memory.long.scenarios.get(name) if name else self.match_scenario(query)
        )
        if сценарий is None:
            подсказка = (
                f"Нет сценария «{name}»." if name
                else "Ни один сценарий не сработал на этом запросе."
            )
            есть = ", ".join(с.имя for с in self.scenarios()) or "ни одного"
            raise AgentError(f"{подсказка} Заведено сценариев: {есть}.")
        try:
            return self.runner.run(сценарий, query, on_step=on_step, finish=finish,
                                   on_result=on_result, по_шагам=по_шагам)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"Сценарий «{сценарий.имя}» прервался: {exc}") from exc

    # --- сводка --------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Состояние агента для интерфейсов."""
        модель = self.model_for()
        описание = catalog.get(модель)
        return {
            "model_key": модель,
            "model_label": описание.label,
            "model_fixed": bool(self.model_key),
            "provider": описание.provider,
            "free": описание.free,
            "user_id": self.user_id,
            "session": self.session,
            "layers": sorted(self.layers),
            "router_mode": self.memory.router_mode,
            "threshold": self.memory.threshold,
            "task": self.task.to_dict() if self.task else None,
            "stage": self.stage,
            "allowed": list(self.task.allowed()) if self.task else [],
            "soft_check": self.soft_check,
            "follow_profile": self.follow_profile,
            "judge_semantic": self.judge_semantic,
            "require_self_report": self.require_self_report,
            "инвариантов": len(self.memory.all_invariants()),
            "profile": preferences.summary(self.profile()),
            "needs_setup": self.needs_setup(),
            "scenarios": [с.digest() for с in self.scenarios()],
            "spent": dict(self.client.spent),
        }

    def stats(self) -> dict[str, Any]:
        return self.memory.stats()

    def journal(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.memory.journal(limit)

    def files(self) -> dict[str, str]:
        return self.memory.files()

    def tasks(self) -> list[dict[str, Any]]:
        return self.memory.working.tasks()

    def close(self) -> None:
        self.client.close()
        # Соединения с MCP-серверами держат процессы: stdio-сервер живёт,
        # пока его держит клиент. Не закрыть их — оставить за собой чужие
        # процессы на весь век программы.
        if self.toolbox is not None:
            self.toolbox.close()
        # Маршрутизатор Дня 20 заводит по инструментарию на набор серверов, и
        # каждый держит свои процессы. Закрыть нужно все, а не только текущий.
        for ящик in self._ящики.values():
            if ящик is not self.toolbox:
                ящик.close()
        self._ящики.clear()
        if self._flows is not None:
            self._flows.close()
            self._flows = None
        if self._scheduler is not None:
            self._scheduler.close()
            self._scheduler = None
        if self._pipeline_runs is not None:
            self._pipeline_runs.close()
            self._pipeline_runs = None


def _протокол_прогона(вызовы: list[ToolResult]) -> dict[str, Any] | None:
    """Протокол конвейера из вызовов этого ответа — последний удачный запуск.

    Нужен, чтобы человек видел цепочку, которую агент выполнил ради ответа:
    в тексте модель обычно пишет итог, а какие шаги были и что во что
    передалось — видно только здесь.
    """
    for вызов in reversed(вызовы):
        имя = вызов.полное_имя or вызов.инструмент
        if имя.endswith(f"{mcp_names.РАЗДЕЛИТЕЛЬ}run_pipeline") and вызов.ок:
            if isinstance(вызов.данные, dict) and вызов.данные.get("шаги") is not None:
                return вызов.данные
    return None
