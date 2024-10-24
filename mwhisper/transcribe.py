import logging as log
import traceback
from time import time
from typing import Generator, Iterable, Tuple
from httpx_sse import connect_sse

import numpy as np


from mwhisper.processing import audio_to_bytes
def stream_whisper(
    data: np.ndarray,
    sr: int,
    endpoint: str,
    temperature: float,
    model: str,
    http_client: httpx.Client,
) -> Iterable[str]:
    """Stream audio data to the server and yield transcriptions."""
    kwargs = {
        "files": {"file": ("audio.wav", audio_to_bytes(sr, data), "audio/wav")},
        "data": {
            "response_format": "text",
            "temperature": temperature,
            "model": model,
            "stream": True,
        },
    }
    try:
        with connect_sse(http_client, "POST", WEBSOCKET_URI, **kwargs) as event_source:
            for event in event_source.iter_sse():
                yield event.data
    except Exception as e:
        log.error(f"Error streaming audio: {e}")
        yield "Error streaming audio."


def handle_audio_file(
    file_path: str,
    state: dict,
    endpoint: str,
    temperature: float,
    model: str,
    client: httpx.Client,
) -> tuple[dict, str, str]:
    tic = time()
    with Path(file_path).open("rb") as file:
        response = client.post(
            endpoint,
            files={"file": file},
            data={
                "model": model,
                "response_format": "text",
                "temperature": temperature,
            },
        )
    result = response.text
    response.raise_for_status()
    elapsed_time = time() - tic
    total_tokens = len(result.split())
    tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
    return state, result, f"STT tok/sec: {tokens_per_sec:.4f}"


def handle_audio_stream(
    audio_source: tuple[int, np.ndarray] | None,
    audio_state: State,
    temperature: float,
    http_client: httpx.Client,
) -> Generator[tuple[dict, str, str], None, None]:
    """Handle audio data for transcription or translation tasks."""
    print(f"audio state: {audio_state}")
    endpoint = audio_state["endpoint"]
    tic = time()
    total_tokens = 0
    if not audio_source:
        yield audio_state, "", ""
        return
    sr, y = audio_source
    y = y.astype(np.float32)
    y = y.mean(axis=1) if y.ndim > 1 else y
    try:
        y /= np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1
    except Exception as e:
        log.exception("Error normalizing audio: %s", traceback.format_exc())
        return audio_state, "", ""
    stream = audio_state["stream"]
    stream = np.concatenate([stream, y]) if stream is not None else y
    if len(stream) < 16000:
        audio_state["stream"] = stream
        return audio_state, "", ""
    previous_transcription = ""
    model = audio_state["model"]
    tokens_per_sec = 0
    for transcription in stream_whisper(stream, sr, endpoint, temperature, model, http_client):
        if previous_transcription.lower().strip().endswith(transcription.lower().strip()):
            print(f"Skipping repeated transcription: {transcription}")
            continue
        total_tokens = len(previous_transcription.split())
        elapsed_time = time() - tic
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        previous_transcription += transcription
        print(f"Transcription: {previous_transcription}, State: {audio_state.update({'stream': stream})}")
        audio_state["stream"] = stream
        yield audio_state, previous_transcription, f"STT tok/sec: {tokens_per_sec:.4f}"
    print(f"Transcription: {previous_transcription}, State: {audio_state}")
    return audio_state, previous_transcription, f"STT tok/sec: {tokens_per_sec:.4f}"
