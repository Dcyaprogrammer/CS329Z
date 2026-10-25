"""Course-provided language model wrapper.

This module gives you exactly one primitive: a chat-completion call that takes
a list of messages and returns a string.

    lm = LM()                                  # the grading model
    text = lm([{"role": "system", "content": "..."},
               {"role": "user", "content": "..."}])

There is no tool-calling parameter and no structured-output parameter. Those
are yours to build on top of this call.

What the wrapper does for you:

* Disk cache. Every call is cached under ``.lm_cache/`` keyed on the endpoint,
  model, sampling settings, and exact messages. Re-running unchanged code
  costs nothing.
* Truncation. A call that stops because the model reached ``max_tokens``
  raises ``OutputTruncated`` instead of returning the cut-off text. The
  exception carries the text produced so far (``.text``) and the token
  counts. Truncated replies are not cached.

``ScriptedLM`` is a drop-in fake used by the deterministic tests. It replays a
fixed list of replies and never touches the network.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

Message = dict  # {"role": "system" | "user" | "assistant", "content": str}

MODEL_ROLES = ("grading", "user", "judge")
DEFAULT_BASE_URL = "https://api.openai.com/v1"

VALID_ROLES = ("system", "user", "assistant")


class ScriptExhausted(RuntimeError):
    """Raised when a ScriptedLM is called more times than its script allows."""


class OutputTruncated(RuntimeError):
    """Raised when the model stopped because it reached ``max_tokens``.

    ``text`` is what the model wrote before the cut, which is often empty:
    a reasoning model's hidden reasoning counts against the same limit, so
    the limit can be used up before any visible text is written.
    ``output_tokens`` is the whole output, ``reasoning_tokens`` the hidden
    part of it (0 when the provider does not report it). Calling again with
    a larger ``max_tokens`` is a new paid call.
    """

    def __init__(self, text: str, *, model: str, max_tokens: int, output_tokens: int, reasoning_tokens: int = 0):
        self.text = text
        self.model = model
        self.max_tokens = max_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens
        shown = f"{len(text)} characters of text" if text else "no visible text"
        super().__init__(
            f"{model} stopped at max_tokens={max_tokens} with {shown} "
            f"({output_tokens} output tokens, {reasoning_tokens} of them hidden reasoning). "
            "Ask for a shorter reply, or call again with a larger max_tokens."
        )


def resolve_model(role_or_model: str | None = None) -> str:
    """Map a role ("grading", "user", "judge") to a model name.

    ``None`` means "grading". A string that is not a role is returned
    unchanged, so you can pass a model name.
    """
    name = role_or_model or "grading"
    if name in MODEL_ROLES:
        model = os.environ.get(f"CS329Z_{name.upper()}_MODEL") or os.environ.get("OPENAI_MODEL")
        if not model:
            raise ValueError("Set OPENAI_MODEL in .env to your provider's model ID.")
        return model
    return name


def _cache_dir() -> Path:
    return Path(os.environ.get("CS329Z_CACHE_DIR", REPO_ROOT / ".lm_cache"))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(REPO_ROOT / ".env")


def _validate_messages(messages: Sequence[Message]) -> None:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a non-empty list of {'role', 'content'} dicts")
    for m in messages:
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise ValueError(
                f"each message must be a dict with exactly the keys 'role' and 'content', got {m!r}"
            )
        if m["role"] not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {m['role']!r}")
        if not isinstance(m["content"], str):
            raise ValueError("message content must be a string")


class LM:
    """A chat model: messages in, string out.

    Parameters
    ----------
    model:
        A role ("grading", "user", "judge"), a provider model ID, or ``None`` for "grading".
    temperature, max_tokens:
        Sampling settings. They are part of the cache key. Reasoning models
        spend part of ``max_tokens`` on hidden reasoning, so keep it large.
        ``temperature`` applies when reasoning effort is unset or "none".
    cache:
        Set to False to bypass the disk cache for this instance.
    salt:
        Extra string mixed into the cache key. Two instances with different
        salts get independent samples for identical messages.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        cache: bool = True,
        salt: str = "",
    ) -> None:
        _load_env()
        self.model = resolve_model(model)
        self.base_url = os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.salt = salt
        # Running totals for this instance.
        self.calls = 0
        self.cached_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.seconds = 0.0
        self._client = None

    # -- public API -------------------------------------------------------

    def __call__(self, messages: Sequence[Message]) -> str:
        _validate_messages(messages)
        messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        key = self._key(messages)
        path = _cache_dir() / key[:2] / f"{key}.json"
        self.calls += 1
        if self.cache and path.exists():
            self.cached_calls += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]

        start = time.time()
        text, usage = self._complete(messages)
        elapsed = time.time() - start
        self.input_tokens += usage["input"]
        self.output_tokens += usage["output"]
        self.seconds += elapsed
        truncated = usage.get("finish_reason") == "length"
        if truncated:  # never cache a cut-off reply
            raise OutputTruncated(
                text,
                model=self.model,
                max_tokens=self.max_tokens,
                output_tokens=usage["output"],
                reasoning_tokens=usage.get("reasoning", 0),
            )
        if self.cache and text.strip():  # an empty reply is not worth remembering
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"text": text, "model": self.model}), encoding="utf-8")
            tmp.replace(path)
        return text

    def usage(self) -> dict:
        """Totals for this instance since it was created."""
        return {
            "model": self.model,
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "seconds": self.seconds,
        }

    # -- internals --------------------------------------------------------

    def _key(self, messages: list[Message]) -> str:
        effort = os.environ.get("CS329Z_REASONING_EFFORT", "")
        blob = json.dumps(
            [self.base_url, self.model, self.temperature, self.max_tokens, effort, self.salt, messages],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _complete(self, messages: list[Message]) -> tuple[str, dict]:
        if self._client is None:
            from openai import OpenAI

            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=key,
                max_retries=6,
                timeout=120,
            )
        effort = os.environ.get("CS329Z_REASONING_EFFORT", "")
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if effort:
            kwargs["reasoning_effort"] = effort
        if not effort or effort == "none":
            kwargs["temperature"] = self.temperature
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        u = response.usage
        out_details = getattr(u, "completion_tokens_details", None)
        reasoning = getattr(out_details, "reasoning_tokens", 0) or 0
        usage = {
            "input": u.prompt_tokens,
            "output": u.completion_tokens,
            "reasoning": reasoning,
            "finish_reason": getattr(choice, "finish_reason", None),
        }
        return text, usage


class ScriptedLM:
    """A fake LM that replays a fixed script. Used by the deterministic tests.

    Each script entry is one model reply, given as one of:

    * a string, returned as is;
    * a dict ``{"text": str, "tool_call": {"name": str, "args": dict}}``,
      rendered to a string by ``formatter(text, ToolCall)``. The tests pass
      your ``run_format_tool_call`` adapter as the formatter, so scripted
      tool calls arrive in your own format;
    * a callable ``f(messages) -> str``.

    Every call's messages are recorded in ``.calls`` so tests can inspect
    exactly what your code sent to the model. When the script runs out the
    LM raises ``ScriptExhausted``, unless ``repeat_last=True``, in which case
    the last entry is replayed forever.
    """

    model = "scripted"

    def __init__(
        self,
        script: Sequence[Any],
        *,
        formatter: Callable[[str, Any], str] | None = None,
        repeat_last: bool = False,
    ) -> None:
        self.script = list(script)
        self.formatter = formatter
        self.repeat_last = repeat_last
        self.calls: list[list[Message]] = []
        self.replies: list[str] = []

    def __call__(self, messages: Sequence[Message]) -> str:
        _validate_messages(messages)
        index = len(self.calls)
        self.calls.append(copy.deepcopy(list(messages)))
        if index >= len(self.script):
            if not (self.repeat_last and self.script):
                raise ScriptExhausted(
                    f"ScriptedLM was called {index + 1} times but the script has "
                    f"{len(self.script)} entries. Your code made a model call the test "
                    "did not expect."
                )
            index = len(self.script) - 1
        entry = self.script[index]
        if callable(entry):
            reply = entry(self.calls[-1])
        elif isinstance(entry, dict):
            reply = self._render(entry)
        else:
            reply = str(entry)
        self.replies.append(reply)
        return reply

    @property
    def remaining(self) -> int:
        return max(len(self.script) - len(self.calls), 0)

    def _render(self, entry: dict) -> str:
        text = entry.get("text", "")
        call = entry.get("tool_call")
        if call is None:
            return text
        if self.formatter is None:
            raise RuntimeError("ScriptedLM needs a formatter to render scripted tool calls")
        from cs329z_hw1.types import ToolCall

        return self.formatter(text, ToolCall(name=call["name"], args=dict(call.get("args", {}))))

    def usage(self) -> dict:
        return {"model": "scripted", "calls": len(self.calls)}
