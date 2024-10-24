from collections.abc import Generator
from pathlib import Path
from time import time
import traceback
from typing import Any, Iterable, Mapping, dataclass_transform, overload

import httpx
from httpx_sse import connect_sse
from lager import log
import numpy as np
from pydantic_settings import BaseSettings
from typing_extensions import TypedDict, override  # noqa

from mwhisper.processing import audio_to_bytes

WEBSOCKET_URI = "http://localhost:5018/audio/transcriptions"


@dataclass_transform()
class State(dict[str, Any], metaclass=type[TypedDict]):
    @overload
    def update(self, **kwargs: Any) -> "State": ...
    @override
    def update(self, key: str, value: Any | None = None) -> "State":
        self[key] = value
        return self


class TaskConfig(BaseSettings):
    # agent_base_url: str = "http://localhost:3389/v1"
    # agent_token: str = "mbodi-demo-1"
    # agent_base_url: str = "http://localhost:5018"
    timeout: int = 10
    TRANSCRIPTION_ENDPOINT: str = "http://localhost:5018/audio/transcriptions"
    TRANSLATION_ENDPOINT: str = "/translations"
    TIMEOUT_SECONDS: int = 180
    WEBSOCKET_URI: str = "ws://localhost:5018/audio/transcriptions"
    CHUNK: int = 1024
    CHANNELS: int = 1
    RATE: int = 16000
    PLACE_HOLDER: str = "Loading can take 30 seconds if a new model is selected..."
    TTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    FIRST_SPEAKER: str = "Luis Moray"
    SECOND_SPEAKER: str = "Sofia Hellen"
    FIRST_LANGUAGE: str = "en"
    SECOND_LANGUAGE: str = "es"
    GPU: str = "0"
