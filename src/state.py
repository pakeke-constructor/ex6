
from dataclasses import dataclass, field
from typing import Optional
import threading
import time

@dataclass
class Context:
    name: str
    model: str = "opus-4.5"
    messages: list = field(default_factory=list)
    tokens: int = 32000
    max_tokens: int = 200000
    cost: float = 0.15
    llm_running: bool = False
    llm_output: str = ""
    last_llm_time: float = 0

    def __hash__(self): return id(self)
    def __eq__(self, other): return self is other

    def call(self, text, llm_fn):
        self.messages.append({"role": "user", "content": text})
        self.llm_running = True
        self.llm_output = ""
        def run():
            for token in llm_fn(self):
                self.llm_output += token
            self.messages.append({"role": "assistant", "content": self.llm_output})
            self.llm_running = False
            self.last_llm_time = time.time()
        threading.Thread(target=run, daemon=True).start()

@dataclass
class AppState:
    contexts: set = field(default_factory=set)
    current: Optional[Context] = None
    mode: str = "selection"

state = AppState()
