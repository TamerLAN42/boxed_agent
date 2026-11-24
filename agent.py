from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from datetime import datetime, timezone

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
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
MAX_MESSAGES = _env_int("MAX_MESSAGES", 10)
# До какой длины обрезать содержимое старых выводов в истории.
MAX_OUTPUT_CHARS = _env_int("MAX_OUTPUT_CHARS", 1000)

TASK = _env(
    "TASK",
    "Напиши в рабочей директории калькулятор на python",
)

def trim_history(
    messages: list[ModelMessage],
    max_messages: int = MAX_MESSAGES,
    max_output_chars: int = MAX_OUTPUT_CHARS,
) -> list[ModelMessage]:
    result = []
    for m in messages:
        if isinstance(m, ModelRequest):
            new_parts = []
            for p in m.parts:
                if (isinstance(p, ToolReturnPart)
                        and isinstance(p.content, str)
                        and len(p.content) > max_output_chars):
                    p = replace(
                        p,
                        content=p.content[:max_output_chars] + "...[truncated]",
                    )
                new_parts.append(p)
            m = replace(m, parts=new_parts)
        result.append(m)

    if len(result) <= max_messages:
        return result

    head = next(
        (m for m in result
         if isinstance(m, ModelRequest)
         and any(isinstance(p, UserPromptPart) for p in m.parts)),
        None,
    )
    tail = result[-(max_messages - 1):]
    while tail and isinstance(tail[0], ModelRequest) and any(
        isinstance(p, ToolReturnPart) for p in tail[0].parts
    ):
        tail = tail[1:]

    return ([head] if head else []) + tail


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

agent = Agent(
    model,
    system_prompt=(
        "Ты аккуратный агент с доступом к bash-оболочке в изолированном контейнере sandbox "
        "через инструмент run_bash. Используй его, когда нужно выполнить команду. "
        "Твоя рабочая директория - /work , ты всегда начинаешь в ней."
        "Не выполняй разрушительных команд (rm -rf /, mkfs, dd на устройства, shutdown) "
        "без явной просьбы. Если команда может быть опасной — сначала уточни у пользователя. "
        "Предпочитай простые команды и по одному шагу за раз."
    ),
    capabilities=[ProcessHistory(trim_history)]
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
    
