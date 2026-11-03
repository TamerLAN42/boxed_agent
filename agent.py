from datetime import datetime, timezone
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
import subprocess
import os

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))

def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# --- SSH ---
SSH_HOST   = _env("SSH_HOST", "sandbox")
SSH_USER   = _env("SSH_USER", "agent")
SSH_KEY    = _env("SSH_KEY", "/root/.ssh/id_ed25519")
SSH_TIMEOUT = _env_int("SSH_TIMEOUT", 30)
MAX_OUTPUT  = _env_int("MAX_OUTPUT", 8000)

# --- Модель ---
MODEL_NAME     = _env("MODEL_NAME", "hf.co/TokenRhythm/NeoHorse-1-4B-GGUF:Q8_0")
OLLAMA_URL     = _env("OLLAMA_URL", "http://ollama:11434/v1")
OLLAMA_API_KEY = _env("OLLAMA_API_KEY", "ollama")

NUM_CTX         = _env_int("NUM_CTX", 16384)
MAX_TOKENS      = _env_int("MAX_TOKENS", 16384)
TEMPERATURE     = _env_float("TEMPERATURE", 0.3)
PRESENCE_PENALTY = _env_float("PRESENCE_PENALTY", 0.0)
TOP_P           = _env_float("TOP_P", 0.95)
ENABLE_THINKING = _env_bool("ENABLE_THINKING", False)

REQUEST_LIMIT   = _env_int("REQUEST_LIMIT", 25)
TOOL_CALLS_LIMIT = _env_int("TOOL_CALLS_LIMIT", 50)

TASK = _env(
    "TASK",
    "Напиши в рабочей директории калькулятор на python",
)

model = OpenAIChatModel(
    model_name=MODEL_NAME,
    provider=OpenAIProvider(
        base_url=OLLAMA_URL,
        api_key=OLLAMA_API_KEY,
    ),
    settings=ModelSettings(
        temperature=TEMPERATURE,
        presence_penalty=PRESENCE_PENALTY,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        extra_body={"num_ctx": NUM_CTX,
                    "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
        },
    )
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
    )
)

@agent.tool_plain
def run_bash(command: str) -> str:
    """Выполнить bash-команду в удалённом sandbox-контейнере по SSH.

    Args:
        command: bash-команда для выполнения, например "ls -la /tmp".
    """
    print(f"[run_bash] $ {command}", flush=True) 
    remote_cmd = "bash -lc " + subprocess.list2cmdline([command])
    ssh_cmd = [
        "ssh",
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=5",
        "-o", "BatchMode=yes",
        "-o", "LogLevel=ERROR",
        f"{SSH_USER}@{SSH_HOST}",
        remote_cmd,
    ]
    try:
        proc = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=SSH_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"[timeout] команда не завершилась за {SSH_TIMEOUT}s: {command}"
    except Exception as e:
        return f"[ssh error] {e}"

    out = proc.stdout or ""
    err = proc.stderr or ""
    result = f"[exit={proc.returncode}]\n{out}"
    if err:
        result += f"\n[stderr]\n{err}"
    if len(result) > MAX_OUTPUT:
        result = result[:MAX_OUTPUT] + f"\n...[truncated, {len(result)} chars total]"
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
    