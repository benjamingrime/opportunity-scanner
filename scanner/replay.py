"""Record LLM replies, then replay them, so tests and CI evals run without calling a model.

Each reply is keyed by a hash of the tools and the conversation so far. Replay(path) only replays
and fails on anything unrecorded; Replay(path, model) calls the model for anything new and
records it (save() writes the file).
"""
import gzip
import hashlib
import json
import threading
from pathlib import Path

from langchain_core.messages import AIMessage


def _canon(m):
    d = {"type": m.type, "content": m.content}
    if getattr(m, "tool_calls", None):
        d["tool_calls"] = [{"name": c["name"], "args": c["args"], "id": c["id"]} for c in m.tool_calls]
    if getattr(m, "tool_call_id", None):
        d["tool_call_id"] = m.tool_call_id
    return d


class Replay:
    def __init__(self, path, model=None, tools=None, store=None):
        self.path, self.model, self.tools = Path(path), model, tools
        if store is None:
            store = json.loads(gzip.decompress(self.path.read_bytes())) if self.path.exists() else {}
        self.store, self._lock = store, threading.Lock()

    def bind_tools(self, tools):
        return Replay(self.path, self.model, tools, self.store)

    def invoke(self, messages):
        key = hashlib.sha256(json.dumps({"tools": self.tools, "messages": [_canon(m) for m in messages]},
                                        sort_keys=True).encode()).hexdigest()
        if key in self.store:
            r = self.store[key]
            return AIMessage(content=r["content"], tool_calls=r.get("tool_calls", []), usage_metadata=r.get("usage"))
        if self.model is None:
            raise KeyError(f"no recorded reply in {self.path.name}; record again with a live model")
        model = self.model.bind_tools(self.tools) if self.tools else self.model
        ai = model.invoke(messages)
        with self._lock:
            self.store[key] = {"content": ai.content, "usage": ai.usage_metadata,
                               "tool_calls": [{"name": c["name"], "args": c["args"], "id": c["id"]}
                                              for c in ai.tool_calls]}
        return ai

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = json.dumps(self.store, sort_keys=True).encode()
        self.path.write_bytes(gzip.compress(data, mtime=0))
