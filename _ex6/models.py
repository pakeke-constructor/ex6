from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    id: str             # model id (e.g. "anthropic/claude-opus-4.6")
    context_window: int # ctx-window size
    input: float        # cost / Mtok
    output: float       # cost / Mtok
    cache_read: float   # cost / Mtok
    cache_write: float = 0 # cost / Mtok (explicit caching, e.g. Anthropic)


# Unified model registry. Each field is a ModelInfo.
# On first lookup via M.get(model_id), builds a reverse index for O(1) access.
class M:
    OPUS_46          = ModelInfo("anthropic/claude-opus-4.6",         200_000, 5,    25,   0.5,  6.25)
    OPUS_47          = ModelInfo("anthropic/claude-opus-4.7",         200_000, 5,    25,   0.5,  6.25)
    OPUS_48          = ModelInfo("anthropic/claude-opus-4.8",         200_000, 5,    25,   0.5,  6.25)
    SONNET_46        = ModelInfo("anthropic/claude-sonnet-4.6",       200_000, 3,    15,   0.3,  3.75)
    HAIKU_45         = ModelInfo("anthropic/claude-haiku-4.5",        200_000, 1,    5,    0.1,  1.25)
    GPT5             = ModelInfo("openai/gpt-5",                      400_000, 1.25, 10,   0.125)
    GPT54            = ModelInfo("openai/gpt-5.4",                  1_050_000, 2.5,  15,   0.25)
    GPT55            = ModelInfo("openai/gpt-5.5",                  1_050_000, 5,    30,   0.5)
    GPT5_MINI        = ModelInfo("openai/gpt-5-mini",                 400_000, 0.25, 2,    0.025)
    GPT5_CODEX       = ModelInfo("openai/gpt-5-codex",                400_000, 1.25, 10,   0.125)
    GPT52_CODEX      = ModelInfo("openai/gpt-5.2-codex",              400_000, 1.75, 14,   0.175)
    GPT53_CODEX      = ModelInfo("openai/gpt-5.3-codex",              400_000, 1.75, 14,   0.175)
    # GPT54_CODEX      = ModelInfo("openai/gpt-5.4-codex",              400_000, 1.75, 14,   0.175)
    GPT51_CODEX_MINI = ModelInfo("openai/gpt-5.1-codex-mini",         400_000, 0.25, 2,    0.025)
    CODEX_MINI       = ModelInfo("openai/codex-mini",                 200_000, 1.5,  6,    0.375)
    O4_MINI          = ModelInfo("openai/o4-mini",                    200_000, 1.1,  4.4,  0.275)
    GEMINI3_PRO      = ModelInfo("google/gemini-3-pro-preview",     1_048_576, 2,    12,   0.2)
    GEMINI3_FLASH    = ModelInfo("google/gemini-3-flash-preview",   1_048_576, 0.5,  3,    0.05)
    GEMINI35_FLASH    = ModelInfo("google/gemini-3.5-flash",   1_048_576, 0.5,  3,    0.05)
    GEMINI31_FLASH_LITE = ModelInfo("google/gemini-3.1-flash-lite-preview", 1_048_576, 0.25, 1.5, 0.025)
    GEMINI25_PRO     = ModelInfo("google/gemini-2.5-pro",           1_048_576, 1.25, 10,   0.125)
    GEMINI25_FLASH   = ModelInfo("google/gemini-2.5-flash",         1_048_576, 0.3,  2.5,  0.03)
    GEMINI25_FLASH_LITE = ModelInfo("google/gemini-2.5-flash-lite", 1_048_576, 0.1,  0.4,  0.01)
    GROK4            = ModelInfo("x-ai/grok-4",                      256_000, 3,    15,   0.75)
    GROK41_FAST      = ModelInfo("x-ai/grok-4.1-fast",             2_000_000, 0.2,  0.5,  0.05)
    DEEPSEEK_CHAT_V31    = ModelInfo("deepseek/deepseek-chat-v3.1",       32_768, 0.15, 0.75,  0)
    DEEPSEEK_V4_PRO  = ModelInfo("deepseek/deepseek-v4-pro",         200_000, 1.5, 3, 0.15, 0)
    DEEPSEEK_R1      = ModelInfo("deepseek/deepseek-r1",              64_000, 0.7,  2.5,   0)
    QWEN3_CODER      = ModelInfo("qwen/qwen3-coder",                 262_144, 0.22, 1,    0.022)
    KIMI_K25         = ModelInfo("moonshotai/kimi-k2.5",             262_144, 0.45, 2.2,  0.225)
    GLM_52           = ModelInfo("z-ai/glm-5.2",                   1_048_576, 1.4,  4.4,  0.26)
    GEMMA_4          = ModelInfo("google/gemma-4-31b-it",            262_144, 0.13, 0.38, 0)

    OPUS_LATEST: ModelInfo
    GPT_LATEST: ModelInfo
    CODEX_LATEST: ModelInfo
    GEMINI_LATEST: ModelInfo
    DEEPSEEK_LATEST: ModelInfo
    GLM_LATEST: ModelInfo

    _index: Optional[dict[str, "ModelInfo"]] = None

    @classmethod
    def get(cls, model_id: str) -> Optional["ModelInfo"]:
        if cls._index is None:
            cls._index = {v.id: v for v in vars(cls).values() if isinstance(v, ModelInfo)}
        return cls._index.get(model_id)

M.OPUS_LATEST = M.OPUS_48
M.GPT_LATEST = M.GPT55
M.CODEX_LATEST = M.GPT53_CODEX
M.GEMINI_LATEST = M.GEMINI35_FLASH
M.DEEPSEEK_LATEST = M.DEEPSEEK_V4_PRO
M.GLM_LATEST = M.GLM_52

