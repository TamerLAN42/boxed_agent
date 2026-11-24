"""Runtime агента: решает, что именно увидит локальная модель на каждом шаге.

Как устроен контекст (важно понимать при правках):

* Полная история диалога живёт в памяти рантайма и НИКОГДА не обрезается
  (это `ctx.messages` внутри pydantic-ai).
* Перед каждым запросом к модели из полной истории собирается "представление"
  (view), которое уходит в модель, но НЕ записывается обратно в историю:
      [ system + исходная задача ]      — голова, всегда на месте
      [ последние N сообщений дословно ] — свежие шаги, с склейкой tool-пар
      [ напоминание о задаче + дайджест ] — хвост, приклеивается к последнему запросу
* Всё, что не влезло в окно, не выбрасывается, а сжимается в дайджест:
  "3. `python3 calc.py` → exit=1: SyntaxError ...". Так модель помнит, что уже
  делала, даже когда команды давно уехали из окна.

Раньше здесь стоял ProcessHistory(trim_history), который писал обрезанную историю
обратно в ctx.messages. Это делало обрезку необратимой: после MAX_MESSAGES агент
безвозвратно терял всё, что было раньше, и начинал повторять шаги по кругу.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# --- SSH ---
SSH_HOST = _env("SSH_HOST", "sandbox")
SSH_USER = _env("SSH_USER", "agent")
SSH_KEY = _env("SSH_KEY", "/root/.ssh/id_ed25519")
SSH_TIMEOUT = _env_int("SSH_TIMEOUT", 30)
MAX_OUTPUT = _env_int("MAX_OUTPUT", 4000)

# --- Модель ---
MODEL_NAME = _env("MODEL_NAME", "hf.co/TokenRhythm/NeoHorse-1-4B-GGUF:Q8_0")
OLLAMA_URL = _env("OLLAMA_URL", "http://ollama:11434/v1")
OLLAMA_API_KEY = _env("OLLAMA_API_KEY", "ollama")

NUM_CTX = _env_int("NUM_CTX", 16384)
MAX_TOKENS = _env_int("MAX_TOKENS", 8192)
TEMPERATURE = _env_float("TEMPERATURE", 0.3)
PRESENCE_PENALTY = _env_float("PRESENCE_PENALTY", 0.0)
TOP_P = _env_float("TOP_P", 0.95)
ENABLE_THINKING = _env_bool("ENABLE_THINKING", False)

REQUEST_LIMIT = _env_int("REQUEST_LIMIT", 100)
TOOL_CALLS_LIMIT = _env_int("TOOL_CALLS_LIMIT", 100)

# --- Контекст ---
# Сколько последних сообщений истории показывать модели дословно.
MAX_MESSAGES = _env_int("MAX_MESSAGES", 12)
# До какой длины обрезать вывод старых (уже не последних) команд в окне.
MAX_OUTPUT_CHARS = _env_int("MAX_OUTPUT_CHARS", 1000)
# Сколько символов отдаём под дайджест "что уже сделано".
CONTEXT_DIGEST_CHARS = _env_int("CONTEXT_DIGEST_CHARS", 1500)
# Приблизительное число символов на токен. Для кириллицы токенизатор ест больше
# токенов, поэтому значение занижено намеренно (2.5 симв/токен вместо 4).
CHARS_PER_TOKEN = _env_float("CHARS_PER_TOKEN", 2.5)
# Резерв контекстного окна под служебные части и "воздух".
CONTEXT_RESERVE_TOKENS = _env_int("CONTEXT_RESERVE_TOKENS", 1024)
# 0 = посчитать бюджет автоматически из NUM_CTX/MAX_TOKENS.
PROMPT_BUDGET_CHARS = _env_int("PROMPT_BUDGET_CHARS", 0)
# Печатать в лог, что уходит в модель (сколько сообщений, сколько символов).
CONTEXT_DEBUG = _env_bool("CONTEXT_DEBUG", True)
# Сколько раз одна и та же упавшая команда считается "повтором".
REPEAT_COMMAND_WARN = _env_int("REPEAT_COMMAND_WARN", 2)

TASK = _env(
    "TASK",
    "Напиши в рабочей директории калькулятор на python",
)


def _prompt_budget_chars() -> int:
    """Сколько символов промпта можно себе позволить при текущих настройках.

    Окно отдаём модели целиком, но вычитаем место под её собственный ответ.
    max_tokens ограничиваем половиной окна: 4B-модель всё равно не напишет
    больше, а место под историю нужно.
    """
    if PROMPT_BUDGET_CHARS > 0:
        return PROMPT_BUDGET_CHARS
    reserved = CONTEXT_RESERVE_TOKENS + min(MAX_TOKENS, NUM_CTX // 2)
    free_tokens = max(NUM_CTX - reserved, 1024)
    return int(free_tokens * CHARS_PER_TOKEN)


PROMPT_BUDGET = _prompt_budget_chars()

SYSTEM_PROMPT = (
    "Ты автономный агент-программист. Ты работаешь один: живого пользователя рядом нет, "
    "уточнять не у кого, поэтому все решения принимай сам и не задавай вопросов.\n"
    "\n"
    "Твои инструменты:\n"
    "- run_bash(command) — выполняет bash-команду в изолированном контейнере sandbox. "
    "Там есть python3, bash, стандартные утилиты. Сеть недоступна.\n"
    "- get_time() — текущее время UTC.\n"
    "\n"
    "Правила работы:\n"
    "1. Твоя рабочая директория — /work. Писать можно только в /work и /tmp, "
    "остальная файловая система только для чтения.\n"
    "2. Сначала осмотрись: `ls -la /work`. Затем действуй по одному шагу за раз.\n"
    "3. Каждый ответ содержит ровно один вызов run_bash. После каждого вызова смотри "
    "на `[exit=N]`: 0 — успех, иначе читай stderr и исправляй причину.\n"
    "4. Никогда не повторяй команду, которая уже упала. Меняй подход, читай ошибку.\n"
    "5. Файлы создавай через heredoc, обязательно с кавычками вокруг маркера "
    "(так bash не подставит переменные и не испортит код):\n"
    "   cat > /work/calc.py << 'EOF'\n"
    "   ...код...\n"
    "   EOF\n"
    "6. Вывод команд может быть обрезан. Если нужен кусок, который не видно, "
    "перечитай его через `head -n 40 файл`, `tail -n 40 файл` или `grep -n шаблон файл`.\n"
    "7. Не выводи длинный код в ответ пользователю — пиши его в файл.\n"
    "8. Если файл уже создан, а его содержимое не видно в истории — прочитай его "
    "`cat /work/имя.py` и правь точечно, а не перезаписывай вслепую.\n"
    "9. Задача считается выполненной, только когда результат проверен: файл существует, "
    "`python3 -m py_compile файл.py` проходит, и программа реально запускается на пробном вводе.\n"
    "10. Когда всё проверено, вызови run_bash ещё раз с финальной проверкой "
    "(`ls -la /work && python3 /work/файл.py <<< '1+2'`) и после этого напиши короткий итог "
    "без вызовов инструментов: что сделано, какие файлы, чем проверено.\n"
    "\n"
    "Не выполняй разрушительных команд (rm -rf /, mkfs, dd на устройства, shutdown): "
    "они бессмысленны, песочница одноразовая, а задачу всё равно нужно решить."
)


# ---------------------------------------------------------------------------
# SSH к sandbox
# ---------------------------------------------------------------------------

def _truncate_output(text: str, limit: int) -> str:
    """Обрезает вывод, сохраняя и начало, и конец.

    Ошибки (traceback, "command not found") обычно оказываются в конце вывода,
    поэтому резать только хвост — значит прятать от модели самое важное.
    """
    if len(text) <= limit:
        return text
    head_len = max(limit * 2 // 3, 200)
    tail_len = max(limit - head_len, 100)
    skipped = len(text) - head_len - tail_len
    return (
        text[:head_len]
        + f"\n...[обрезано {skipped} символов; перечитай нужное через head/tail/grep]...\n"
        + text[-tail_len:]
    )


def ssh_exec(command: str, timeout: int = SSH_TIMEOUT) -> tuple[int, str, str]:
    """Выполняет bash-команду в sandbox через ssh.

    Команда передаётся на stdin удалённого `bash -ls`, а не в аргументах ssh.
    Иначе её пришлось бы экранировать под shell (кавычки, `$`, бэктики) — а это
    самый частый источник "команда выполнилась не так" у небольших моделей.
    """
    ssh_cmd = [
        "ssh",
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=5",
        "-o", "BatchMode=yes",
        "-o", "LogLevel=ERROR",
        f"{SSH_USER}@{SSH_HOST}",
        "bash -ls",
    ]
    proc = subprocess.run(
        ssh_cmd,
        input=command.rstrip("\n") + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


# ---------------------------------------------------------------------------
# Контекст: разбор истории на шаги
# ---------------------------------------------------------------------------

_EXIT_RE = re.compile(r"\[exit=(-?\d+)\]")
_WRITE_RE = re.compile(
    r"(?:>>?|\btee\s+(?:-a\s+)?|\btouch\s+|\bmkdir\s+(?:-p\s+)?|\bcp\s+\S+\s+|\bmv\s+\S+\s+)([\w./~-]+)"
)


@dataclass(frozen=True)
class StepRecord:
    """Один выполненный инструмент: что запускали и чем кончилось."""

    number: int
    command: str
    exit_code: int | None
    output: str


def _part_text(part: object) -> str:
    content = getattr(part, "content", "")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def _one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 1)] + "…"


def _collect_steps(messages: list[ModelMessage]) -> list[StepRecord]:
    """Собирает пары "вызов инструмента → его результат" в хронологическом порядке."""
    calls: dict[str, tuple[str, str]] = {}
    records: list[StepRecord] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if not isinstance(part, ToolCallPart):
                    continue
                try:
                    args = part.args_as_dict()
                except Exception:
                    args = {}
                command = args.get("command") or args.get("path") or args.get("expr") or ""
                calls[part.tool_call_id] = (part.tool_name, str(command))
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if not isinstance(part, ToolReturnPart):
                    continue
                tool_name, command = calls.get(part.tool_call_id, (part.tool_name, ""))
                text = _part_text(part)
                match = _EXIT_RE.search(text[:64])
                records.append(
                    StepRecord(
                        number=len(records) + 1,
                        command=command or f"{tool_name}()",
                        exit_code=int(match.group(1)) if match else None,
                        output=text,
                    )
                )
    return records


def _error_line(record: StepRecord) -> str:
    """Самая информативная строка ошибки.

    В traceback полезно последнее: `NameError: name 'x' is not defined`,
    а не первое `Traceback (most recent call last):`.
    """
    lines = [line.strip() for line in record.output.splitlines() if line.strip()]
    for line in reversed(lines):
        if _EXIT_RE.search(line) or line.startswith("[") or line.startswith("...["):
            continue
        return _one_line(line, 140)
    return ""


def _render_digest(records: list[StepRecord], max_chars: int) -> str:
    """Сжимает выполненные шаги в компактный список для напоминания."""
    if not records:
        return ""
    lines = []
    for record in records:
        line = f"{record.number}. `{_one_line(record.command, 90)}`"
        if record.exit_code is not None:
            line += " → ok" if record.exit_code == 0 else f" → exit={record.exit_code}"
        if record.exit_code not in (0, None):
            error = _error_line(record)
            if error:
                line += f": {error}"
        lines.append(line)

    # Оставляем самые свежие шаги: они важнее для продолжения работы.
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        if used + len(line) + 1 > max_chars and kept:
            break
        kept.append(line)
        used += len(line) + 1
    kept.reverse()
    omitted = len(lines) - len(kept)
    text = "\n".join(kept)
    if omitted:
        text = f"(ещё {omitted} более ранних шагов опущено)\n" + text
    return text


def _collect_files(records: list[StepRecord]) -> list[str]:
    """Достаёт из команд имена файлов, которые агент создавал или менял."""
    seen: list[str] = []
    for record in records:
        for match in _WRITE_RE.finditer(record.command):
            path = match.group(1)
            if path in {"/dev/null", "-", ""}:
                continue
            if path not in seen:
                seen.append(path)
    return seen[-15:]


def _step_warnings(records: list[StepRecord], step: int, request_limit: int | None) -> list[str]:
    """Предупреждения, которые удерживают модель от зацикливания."""
    warnings: list[str] = []

    failed: dict[str, int] = {}
    for record in records:
        if record.exit_code not in (0, None):
            key = record.command.strip()
            failed[key] = failed.get(key, 0) + 1
    repeated = [cmd for cmd, count in failed.items() if count >= REPEAT_COMMAND_WARN]
    for cmd in repeated[-2:]:
        warnings.append(
            f"Команда `{_one_line(cmd, 90)}` уже падала {failed[cmd]} раз. "
            "Не повторяй её — измени подход или сначала прочитай ошибку."
        )

    if len(records) >= 3 and all(r.exit_code not in (0, None) for r in records[-3:]):
        warnings.append(
            "Последние три команды упали. Останови перебор: прочитай текст ошибки, "
            "проверь синтаксис и путь к файлу, затем сделай один осмысленный шаг."
        )

    if step >= 3 and not records:
        warnings.append(
            "Ты пока не выполнил ни одной команды. Начни с `ls -la /work` и действуй."
        )

    if request_limit:
        left = request_limit - step
        if left <= 3:
            warnings.append(
                f"Осталось {max(left, 0)} шагов до принудительной остановки. "
                "Прекрати исследование: сделай финальную проверку результата и напиши итог."
            )
        elif left <= max(request_limit // 5, 5):
            warnings.append(f"Осталось {left} шагов. Начинай сводить работу к проверке результата.")

    return warnings


def _message_chars(message: ModelMessage) -> int:
    total = 24
    for part in getattr(message, "parts", ()):  # type: ignore[union-attr]
        total += len(_part_text(part))
        args = getattr(part, "args", None)
        if isinstance(args, str):
            total += len(args)
    return total


def _messages_chars(messages: list[ModelMessage]) -> int:
    return sum(_message_chars(message) for message in messages)


def _has_tool_return(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, ToolReturnPart) or isinstance(part, RetryPromptPart)
        for part in message.parts
    )


def _repair_tool_pairs(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Не даёт окну разорвать пары tool_call/tool_return.

    OpenAI-совместимый сервер отвечает 400, если у assistant-сообщения с
    tool_calls нет ответа на каждый вызов, или если tool-ответ висит без вызова.
    """
    for _ in range(len(messages) + 1):
        call_ids: set[str] = set()
        for message in messages:
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        call_ids.add(part.tool_call_id)
        return_ids: set[str] = set()
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                        return_ids.add(part.tool_call_id)

        drop: set[int] = set()
        for index, message in enumerate(messages):
            if isinstance(message, ModelResponse):
                ids = {
                    part.tool_call_id
                    for part in message.parts
                    if isinstance(part, ToolCallPart)
                }
                if any(call_id not in return_ids for call_id in ids):
                    drop.add(index)
        if drop:
            messages = [m for i, m in enumerate(messages) if i not in drop]
            continue

        while messages and _has_tool_return(messages[0]):
            orphan = any(
                part.tool_call_id not in call_ids
                for part in messages[0].parts  # type: ignore[union-attr]
                if isinstance(part, (ToolReturnPart, RetryPromptPart))
            )
            if not orphan:
                break
            messages = messages[1:]
        break
    return messages


def _shrink_old_outputs(messages: list[ModelMessage], limit: int) -> list[ModelMessage]:
    """Ужимает длинные выводы старых шагов, не трогая объекты в истории."""
    result: list[ModelMessage] = []
    last_request_index = max(
        (i for i, m in enumerate(messages) if isinstance(m, ModelRequest)), default=-1
    )
    for index, message in enumerate(messages):
        if index == last_request_index:
            result.append(message)
            continue
        parts = []
        for part in getattr(message, "parts", ()):  # type: ignore[union-attr]
            text = _part_text(part)
            if isinstance(part, ToolReturnPart) and len(text) > limit:
                parts.append(replace(part, content=_truncate_output(text, limit)))
            elif isinstance(part, TextPart) and len(part.content) > 400:
                parts.append(replace(part, content=_truncate_output(part.content, 400)))
            else:
                parts.append(part)
        result.append(replace(message, parts=parts))
    return result


@dataclass(frozen=True)
class ViewStats:
    total_messages: int
    kept_messages: int
    summarised_steps: int
    chars: int
    budget: int

    @property
    def tokens_estimate(self) -> int:
        return int(self.chars / CHARS_PER_TOKEN)


def build_view(
    messages: list[ModelMessage],
    *,
    task: str,
    step: int,
    request_limit: int | None = None,
    max_messages: int = MAX_MESSAGES,
    budget: int = PROMPT_BUDGET,
    digest_chars: int = CONTEXT_DIGEST_CHARS,
    old_output_chars: int = MAX_OUTPUT_CHARS,
) -> tuple[list[ModelMessage], ViewStats]:
    """Собирает представление истории для одного запроса к модели.

    Возвращает новый список сообщений; входной список не меняется, поэтому
    полная история в рантайме остаётся целой.
    """
    if not messages:
        return [], ViewStats(0, 0, 0, 0, budget)

    # Ниже этого порога не влезают даже system prompt и само напоминание.
    budget = max(int(budget), 1500)
    # На тесном окне сначала сжимается дайджест, а не последние шаги.
    digest_chars = min(digest_chars, max(int(budget * 0.2), 250))

    head = messages[0]
    body = messages[1:]

    all_records = _collect_steps(messages)
    files = _collect_files(all_records)
    warnings = _step_warnings(all_records, step, request_limit)

    # Короткая история целиком влезает в окно — ничего не сжимаем.
    total_chars = _messages_chars(messages)
    if len(messages) <= max_messages and total_chars <= budget:
        return list(messages), ViewStats(len(messages), len(messages), 0, total_chars, budget)

    limit = max_messages
    view: list[ModelMessage] = []
    stats = ViewStats(len(messages), len(messages), 0, total_chars, budget)
    for _ in range(10):
        budget_for_body = budget - _message_chars(head) - digest_chars - 600
        kept: list[ModelMessage] = []
        used = 0
        for message in reversed(body):
            if kept and (used + _message_chars(message) > budget_for_body or len(kept) >= limit):
                break
            kept.append(message)
            used += _message_chars(message)
        kept.reverse()

        dropped = body[: len(body) - len(kept)]
        kept = _repair_tool_pairs(kept)
        # Склейка tool-пар могла выкинуть ещё пару сообщений — считаем дайджест
        # по факту того, что реально не попало в окно, иначе шаг теряется совсем.
        kept_ids = {id(message) for message in kept}
        dropped = [message for message in body if id(message) not in kept_ids]
        kept = _shrink_old_outputs(kept, old_output_chars)

        digest = _render_digest(_collect_steps(dropped), digest_chars)
        reminder = _render_reminder(task, step, digest, files, warnings)

        view = [head, *kept]
        view = _append_reminder(view, reminder)
        chars = _messages_chars(view)
        stats = ViewStats(
            total_messages=len(messages),
            kept_messages=len(view),
            summarised_steps=sum(1 for r in _collect_steps(dropped)),
            chars=chars,
            budget=budget,
        )
        if chars <= budget or limit <= 2:
            break
        limit = max(2, limit - 2)
    return view, stats


_REMINDER_HEADER = (
    "[СЛУЖЕБНОЕ НАПОМИНАНИЕ ОТ СИСТЕМЫ. Отвечать на него не нужно — "
    "просто продолжай работу.]"
)


def _render_reminder(
    task: str,
    step: int,
    digest: str,
    files: list[str],
    warnings: list[str],
) -> str:
    blocks = [
        _REMINDER_HEADER,
        f"ЗАДАЧА (не меняется с начала работы): {_one_line(task, 600)}",
        f"СЕЙЧАС ШАГ {step}.",
    ]
    if digest:
        blocks.append("[ЧТО УЖЕ СДЕЛАНО ДО ЭТОГО (сжато)]\n" + digest)
    if files:
        blocks.append("[ФАЙЛЫ, КОТОРЫЕ ТЫ УЖЕ СОЗДАВАЛ ИЛИ МЕНЯЛ]\n" + ", ".join(files))
    blocks.extend(warnings)
    blocks.append(
        "Не повторяй уже выполненные шаги. Продолжай ровно с того места, где остановился: "
        "один вызов run_bash за ответ."
    )
    return "\n\n".join(blocks)


def _append_reminder(
    messages: list[ModelMessage], reminder: str
) -> list[ModelMessage]:
    """Приклеивает напоминание в конец последнего запроса (только для этого вызова)."""
    if not reminder:
        return messages
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, ModelRequest):
            patched = replace(
                message,
                parts=[*message.parts, UserPromptPart(content=reminder)],
            )
            return [*messages[:index], patched, *messages[index + 1:]]
    return [*messages, ModelRequest(parts=[UserPromptPart(content=reminder)])]


@dataclass
class TaskKeeper(AbstractCapability[None]):
    """Подсовывает модели сжатое представление истории, не портя саму историю."""

    task: str = ""
    request_limit: int | None = None
    max_messages: int = MAX_MESSAGES
    budget: int = PROMPT_BUDGET
    digest_chars: int = CONTEXT_DIGEST_CHARS
    old_output_chars: int = MAX_OUTPUT_CHARS
    debug: bool = CONTEXT_DEBUG

    async def before_model_request(self, ctx, request_context):
        step = int(getattr(ctx, "run_step", 1) or 1)
        limit = self.request_limit
        try:  # лимит из самого запуска точнее, чем значение из окружения
            limit = ctx.usage_limits.request_limit or limit
        except Exception:  # noqa: BLE001 — не критично, берём значение из настроек
            pass
        view, stats = build_view(
            request_context.messages,
            task=self.task,
            step=step,
            request_limit=limit,
            max_messages=self.max_messages,
            budget=self.budget,
            digest_chars=self.digest_chars,
            old_output_chars=self.old_output_chars,
        )
        # Меняем ТОЛЬКО то, что уходит в этот запрос. RunContext.messages
        # (полная история) остаётся нетронутой — в этом и весь смысл.
        request_context.messages = view
        if self.debug:
            print(
                f"[context] step={step} msgs {stats.total_messages}→{stats.kept_messages} "
                f"summarised={stats.summarised_steps} chars={stats.chars}/{stats.budget} "
                f"(~{stats.tokens_estimate} tok)",
                flush=True,
            )
        return request_context


# ---------------------------------------------------------------------------
# Модель и агент
# ---------------------------------------------------------------------------

model = OpenAIChatModel(
    model_name=MODEL_NAME,
    provider=OpenAIProvider(base_url=OLLAMA_URL, api_key=OLLAMA_API_KEY),
    settings=ModelSettings(
        temperature=TEMPERATURE,
        presence_penalty=PRESENCE_PENALTY,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        extra_body={
            "num_ctx": NUM_CTX,
            "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
        },
    ),
)

agent: Agent[None, str] = Agent(
    model,
    system_prompt=SYSTEM_PROMPT,
    capabilities=[
        TaskKeeper(
            task=TASK,
            request_limit=REQUEST_LIMIT,
            max_messages=MAX_MESSAGES,
            budget=PROMPT_BUDGET,
            digest_chars=CONTEXT_DIGEST_CHARS,
            old_output_chars=MAX_OUTPUT_CHARS,
            debug=CONTEXT_DEBUG,
        )
    ],
)


@agent.tool_plain
def run_bash(command: str) -> str:
    """Выполнить bash-команду в изолированном sandbox-контейнере.

    Команда может быть многострочной, с heredoc (`<< 'EOF'`), пайпами и
    перенаправлениями — всё выполняется в /work пользователем agent.

    Args:
        command: bash-сценарий целиком. Например "ls -la /work" или
            "cat > /work/a.py << 'EOF'\\nprint(1)\\nEOF".
    """
    print(f"[run_bash] $ {command}", flush=True)
    try:
        code, out, err = ssh_exec(command)
    except subprocess.TimeoutExpired:
        return f"[timeout] команда не завершилась за {SSH_TIMEOUT}s: {command}"
    except Exception as exc:  # noqa: BLE001 — сеть/ssh может упасть как угодно
        return f"[ssh error] {exc}"

    result = f"[exit={code}]\n{out}"
    if err:
        result += f"\n[stderr]\n{err}"
    result = _truncate_output(result, MAX_OUTPUT)
    print(f"[run_bash →] {result!r}", flush=True)
    return result


@agent.tool_plain
def get_time() -> str:
    """Возвращает текущее время в UTC."""
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    result = agent.run_sync(TASK,
                            usage_limits=UsageLimits(
                                request_limit=REQUEST_LIMIT,
                                tool_calls_limit=TOOL_CALLS_LIMIT,
                              ),
                            )
    
    print(result.output)
    
