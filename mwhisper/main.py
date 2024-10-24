import asyncio
import collections
from functools import partial
import json
import os
from pathlib import Path
import re
import threading
import time
from time import time
from typing import TYPE_CHECKING, Any, Iterator, Literal, Tuple, Union, get_args
from urllib.parse import urlencode

from dotenv import load_dotenv
import gradio as gr
import httpx
from mbodied.agents.language import LanguageAgent
import more_itertools
import numpy as np
from openai import OpenAI
from rich.console import Console
from rich.pretty import pprint
import soundfile as sf

from mwhisper.agents.speaker_agent import Speaker
from mwhisper.audio_config import AudioConfig
from mwhisper.audio_config import config as DEFAULT_CONFIG  # noqa: N812
from mwhisper.audio_task import TaskConfig, handle_audio_stream
from mwhisper.predict_instr import predict_instruction as weak_predict_instruction
from mwhisper.utils.colors import mbodi_color

# Initialize console for logging
console = Console()

MAX_INT16 = 32767

# Define the AudioAgent class
from websockets.client import connect  # noqa: E402

WEBSOCKET_URL = "ws://localhost:5018/audio/speech?response_format=json&streaming=True"


console = Console()
if TYPE_CHECKING:
    from mbodied.types.message import Choice
load_dotenv("/home/ubuntu/.env")
VOICES = get_args(Speaker)
SYSTEM_PROMPT = """
You are a brand new assistant
and we are demonstrating your instruction following capabilities. Note that you were created by Embody A.I.  and are designed to assist with a variety of tasks. Here are your advanced capabilities:

    - You are a quanized version of llama3.1 70B trained on copious amounts of real-world data including robotics data.
    - You can break down complex tasks with tree of thought prompting gaurdrails using traditional robotics
    - You can remember things you teach me
    - You an clone git repositories, and run shell commands
    - You can launch ross, dora-rs, and zenoh nodes for real-time communication
    - You have a wide array of tools at my disposal from embody A.I. including SegmentAnything2 from Meta, real-time 6D pose estimation from embody A. I. and much more.
    - You can launch training scripts and upload training scripts using the embody A.I. API

Give interesting facts about embodied intelligence if asked. For example, the company that created you named itself that because the founder is  a physicist and it is a reference to N-Body.
"""


### Weak Agent ###
DETERMINE_INSTRUCTION_PROMPT = """
Determine if the user who is speaking the FINAL sentence after "POTENTIAL STATEMENT" is finished speaking. Answer "Yes" if the user is finished speaking and "No" otherwise.
NOTE: The user may still be speaking even if they pause for a few seconds. It is safe to assume that the user is finished speaking if they pause for more than 5 seconds.
Expect that the statement will have natural language so um or err is expected even for complete statements.
Greetings are also considered complete statements. For example, "Hello." is a complete statement.

PORENTIAL STATEMENT:
"""
audio_start_time = 0
audio_seconds = 0
is_speaking = False
starting_speaking = False
ending_speaking = False
last_should_speak_value = False
first_speech = False


class AudioAgent:
    def __init__(
        self,
        *,
        base_url: str,
        format: Literal["array", "bytes", "file"] = "array",  # noqa: A002
        endpoint: str = "/audio/speech",
        model: str | None = None,
        voice: str | None = None,
        language: str = "en",
        streaming: bool = False,
    ) -> None:
        self.base_url = base_url
        self.format = format
        self.endpoint = endpoint
        self.model = model
        self.voice = voice
        self.language = language
        self.streaming = streaming
        self.sample_rate = 24000

    @staticmethod
    def construct_url(base_url: str, endpoint: str, model: str, voice: str, language: str, streaming: bool) -> str:
        params = {k: v for k, v in locals().items() if v is not None and k != "self"}
        return f"{base_url}{endpoint}?{urlencode(params)}"

    async def aspeak(
        self,
        text: str | Iterator[str],
        voice: str | None = None,
        model: str | None = None,
        language: str = "en",
        streaming: bool = False,
    ) -> np.ndarray | bytes | str:
        texts = [text] if isinstance(text, str) else list(text)
        url = self.construct_url(
            self.base_url,
            self.endpoint,
            model or self.model,
            voice or self.voice,
            language or self.language,
            streaming or self.streaming,
        )
        audio_array = np.array([], dtype=np.int16)
        async with connect(url) as websocket:
            for txt in texts:
                await websocket.send(txt)
                audio_received = False
                audio_data_by_seq = []

                while not audio_received:
                    response = await websocket.recv()
                    response_json = json.loads(response)
                    if response_json.get("end_of_audio"):
                        audio_received = True
                        continue

                    seq = response_json["seq"]
                    audio_chunk = np.array(response_json["audio_chunk"], dtype=np.int16)
                    audio_data_by_seq.append((seq, audio_chunk))

                # Reorder and concatenate audio chunks
                audio_data_by_seq.sort(key=lambda x: x[0])
                ordered_audio = np.concatenate([chunk for _, chunk in audio_data_by_seq])
                audio_array = np.append(audio_array, ordered_audio)

        if format == "array":
            return audio_array
        if format == "bytes":
            return audio_array.tobytes()
        if format == "file":
            filename = "output.wav"
            sf.write(filename, audio_array, self.sample_rate)
            return filename
        return None

    def speak(
        self,
        text: str,
        voice: str | None = None,
        language: str = "en",
        model: str | None = None,
        streaming: bool = False,
    ) -> np.ndarray | bytes | str | Path:
        return asyncio.run(self.aspeak(text, voice, language, model, streaming))


# Global state variables
_agent_state: dict[str, Any] = {
    "pred_instr_mode": "predict",
    "act_mode": "wait",
    "speak_mode": "wait",
    "transcription": "",
    "instruction": "",
    "response": "",
    "spoken": "",
    "moving": False,
    "audio_array": np.array([0], dtype=np.int16),
    "audio_finish_time": float("inf"),
    "speech_fragments": 0,
    "clear": False,
}

_audio_state: dict[str, Any] = {
    "stream": np.array([]),
    "model": "Systran/faster-distil-whisper-large-v3",
    "temperature": 0.0,
    "endpoint": "/audio/transcriptions",
}

# Lock for thread-safe access to global state
state_lock = threading.Lock()
audio_lock = threading.Lock()


# Functions to get and update state
def get_state() -> dict[str, Any]:
    with state_lock:
        return _agent_state.copy()


def update_state(updates: dict[str, Any]) -> None:
    with state_lock:
        _agent_state.update(updates)


def clear_states() -> None:
    with state_lock:
        _agent_state.clear()
        _agent_state.update(
            {
                "pred_instr_mode": "predict",
                "act_mode": "wait",
                "speak_mode": "wait",
                "transcription": "",
                "instruction": "",
                "response": "",
                "spoken": "",
                "moving": False,
                "audio_array": np.array([0], dtype=np.int16),
                "audio_finish_time": float("inf"),
                "speech_fragments": 0,
                "clear": True,
            }
        )
    with audio_lock:
        _audio_state.update(
            {
                "stream": np.array([]),
                "model": "Systran/faster-distil-whisper-large-v3",
                "temperature": 0.0,
                "endpoint": "/audio/transcriptions",
            }
        )
    global \
        is_speaking, \
        starting_speaking, \
        ending_speaking, \
        last_should_speak_value, \
        first_speech, \
        audio_start_time, \
        audio_seconds
    is_speaking = False
    starting_speaking = False
    ending_speaking = False
    last_should_speak_value = False
    first_speech = False
    audio_start_time = 0
    audio_seconds = 0
    console.print("States cleared.", style="bold white")


def get_audio() -> dict[str, Any]:
    with audio_lock:
        return _audio_state.copy()


def update_audio(updates: dict[str, Any]) -> None:
    with audio_lock:
        _audio_state.update(updates)


speaker_cache = {}


# Initialize models and agents
if gr.NO_RELOAD:
    task = TaskConfig()
    audio_agent = AudioAgent(
        base_url="ws://195.1.30.215.195:5018", voice=task.FIRST_SPEAKER, language=task.FIRST_LANGUAGE
    )

    # Initialize other models
    TIMEOUT = httpx.Timeout(timeout=task.TIMEOUT_SECONDS)
    base_url = "http://localhost:5018"
    openai_client = OpenAI(base_url=f"{base_url}", api_key="cant-be-empty")
    http_client = httpx.Client(base_url=task.TRANSCRIPTION_ENDPOINT, timeout=TIMEOUT)

    agent = LanguageAgent(context=SYSTEM_PROMPT)
    weak_agent = LanguageAgent(context=SYSTEM_PROMPT)


NOT_A_COMPLETE_INSTRUCTION = "Not a complete instruction..."

print = console.print  # noqa: A001
gprint = partial(console.print, style="bold green on white")


def aprint(*args, **kwargs):
    console.print("ACT: ", *args, style="bold blue", **kwargs)


yprint = partial(console.print, style="bold yellow")
WAITING_FOR_NEXT_INSTRUCTION = "Waiting for next instruction..."

PredInstrMode = Literal["predict", "repeat", "clear", "wait"]
ActMode = Literal["acting", "repeat", "clear", "wait"]
SpeakMode = Literal["speaking", "wait", "clear"]
act_queue = []


def predict_instruction(transcription: str, last_instruction: str) -> Iterator[Tuple[str, dict]]:
    update_state({"transcription": transcription})
    state = get_state()
    last_instruction = state["instruction"]
    pprint("PREDICT: state: ")
    pprint(state)
    mode = state["pred_instr_mode"]
    if mode == "clear":
        clear_states()
        yield ""
    if mode == "repeat":
        update_state({"pred_instr_mode": "repeat", "instruction": last_instruction})
        yield last_instruction
    if not transcription or not transcription.strip():
        yield last_instruction or ""

    if not state.get("transcription") or (
        len(state["transcription"].split()) < 2 and not state["transcription"].endswith((".", "?", "!"))
    ):
        yield NOT_A_COMPLETE_INSTRUCTION
    if state.get("transcription").lower().strip() == "thank you":
        yield NOT_A_COMPLETE_INSTRUCTION
    yprint(f"Text: {transcription}, Mode Predict: {state}, Last instruction: {last_instruction}")

    weak_agent.forget(everything=True)
    if weak_predict_instruction(transcription, weak_agent, DETERMINE_INSTRUCTION_PROMPT):
        update_state(
            {
                "pred_instr_mode": "repeat",
                "instruction": transcription,
                "act_mode": "acting",
                "last_instruction": transcription,
            }
        )
        yprint(f"Complete instruction: {transcription}")
        yield transcription
    else:
        pprint(f"Instruction: {transcription} is not a complete instruction.")
        update_state({"pred_instr_mode": "predict"})
        yield NOT_A_COMPLETE_INSTRUCTION

        yprint(weak_agent.act(instruction="Why wasn't it a complete instruction?", model="gpt-4o-mini"))
        update_state({"pred_instr_mode": "predict"})
        yield NOT_A_COMPLETE_INSTRUCTION


speak_dequeue = collections.deque(maxlen=100)


def act(instruction: str, last_response: str, last_tps: str) -> Iterator[Tuple[str, str]]:
    state = get_state()
    pprint("ACT: state: ")
    pprint(state)
    instruction = state["instruction"]

    mode = state["act_mode"]
    if mode in ("clear", "wait"):
        yield "", last_tps
    if not instruction.strip():
        console.print("No instruction to act on.", style="bold red")
        yield "", last_tps
    if mode == "repeat":
        update_state({"act_mode": "repeat"})
        yield last_response, last_tps

    if len(agent.history()) > 10:
        agent.forget_after(2)

    tic = time()
    total_tokens = 0
    response = ""
    aprint(f"Following instruction: {instruction}")

    for i, resp in enumerate(agent.act_and_stream(instruction=instruction, model="gpt-4o-mini")):
        if not resp:
            yield "", last_tps
            continue
        if state["act_mode"] == "clear" or state.get("clear"):
            clear_states()
            return
        total_tokens = len(resp.split())
        elapsed_time = time() - tic
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        aprint(f"Resp {i}: {resp}, speak_mode: {state['speak_mode']}")
        response += resp
        update_state({"act_mode": "acting", "response": response, "speak_mode": "speaking"})
        yield response, f"TPS: {tokens_per_sec:.4f}"
    if state["act_mode"] == "clear":
        clear_states()
        yield "", f"TPS: {tokens_per_sec:.4f}"


def split_sentences(text: str) -> list[str]:
    # Using more_itertools to split sentences
    return ["".join(chars).strip() + " " for chars in more_itertools.split_after(text, lambda c: c in ".!?")]


def speak(text: str, speaker: str, language: str) -> Iterator[str]:
    """Generate and stream TTS audio using the AudioAgent."""
    state = get_state()
    pprint(f"speak STATE:", console=Console(style="bold magenta"))
    pprint(state, console=Console(style="bold magenta"))
    text = state["response"]
    mode = state["speak_mode"]
    sr = audio_agent.sample_rate
    spoken = state.get("spoken", "")

    pprint(f"SPEAK Text:")
    pprint(f"{text}\n,")
    pprint(f"Mode Speak: {mode}\n")
    pprint(f"spoken {spoken}\n", console=Console(style="bold magenta"))
    if not text:
        return

    if text and len(text.split()) < 2 and not (text.endswith((".", "?", "!"))) and state["act_mode"] != "repeat":
        yprint("Speak: not complete.")
        return

    if mode == "clear":
        clear_states()
        return
    if mode == "wait":
        return

    sentences = split_sentences(text)

    pprint(f"Sentences:")
    if sentences:
        pprint(sentences)
    pprint(f"Spoken:")
    if spoken:
        pprint(spoken)
    pprint(f"Spoken idx: {state.get('spoken_idx', 0)}")

    global is_speaking, ending_speaking, starting_speaking
    for idx, sentence in enumerate(sentences):
        if sentence and sentence not in spoken:
            is_speaking = True
            pprint(f"Speaking: {sentence}")

            # Synthesize speech for the sentence using AudioAgent
            audio_array = audio_agent.speak(
                sentence,
                voice=speaker,
                language=language,
                streaming=True,
            )

            if state["speak_mode"] == "clear":
                clear_states()
                return

            # Log the sentence and mark the state as speaking
            console.print(f"SPEAK Text: {sentence}, Mode Speak: {mode}", style="bold white on blue")
            spoken += sentence

            global audio_seconds
            audio_seconds = len(audio_array) / float(sr)
            global audio_start_time
            audio_start_time = time()
            console.print(f"Audio seconds: {audio_seconds}", style="bold white on blue")
            # Update state
            update_state(
                {
                    "speak_mode": "speaking",
                    "spoken": spoken,
                    "audio_array": audio_array,
                    "act_mode": "repeat",
                    "spoken_idx": idx + 1,  # Move to the next sentence
                    "audio_finish_time": time() + audio_seconds,
                }
            )
            f = f"out_new{idx}.wav"
            sf.write(file=f, data=audio_array, samplerate=sr)
            yield f

    ending_speaking = True


def create_gradio_demo(config: AudioConfig, task_config: TaskConfig) -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue=mbodi_color,
            secondary_hue="stone",
        ),
        title="Personal Assistant",
        delete_cache=[0, 0],
    ) as demo:

        def update_model_dropdown() -> gr.Dropdown:
            models = openai_client.models.list().data
            model_names: list[str] = [model.id for model in models]
            recommended_models = {model for model in model_names if model.startswith("Systran")}
            other_models = [model for model in model_names if model not in recommended_models]
            model_names = list(recommended_models) + other_models
            return gr.Dropdown(
                choices=model_names,
                label="Model",
                value=config.whisper.model,
            )

        clear_button = gr.Button(value="Clear", render=True, variant="primary")
        audio = gr.Audio(
            label="Audio Input",
            type="numpy",
            sources=["microphone"],
            streaming=True,
            interactive=True,
            render=False,
        )
        with gr.Row():
            audio.render()
            audio_out = gr.Audio(
                streaming=False, autoplay=True, label="Output", visible=True, render=True, type="filepath"
            )
            model_dropdown = gr.Dropdown(
                choices=[config.whisper.model],
                label="Model",
                value=config.whisper.model,
                interactive=True,
            )
        with gr.Row():
            with gr.Column():
                first_speaker_name = gr.Dropdown(
                    label="First Speaker Name",
                    choices=VOICES,
                    value=task.FIRST_SPEAKER,
                    interactive=True,
                )
                first_speaker_language = gr.Dropdown(
                    choices=["en", "es", "fr", "de", "it", "ja", "ko", "nl", "pl", "pt", "ru", "zh"],
                    label="First Speaker Language",
                    value=task.FIRST_LANGUAGE,
                    interactive=True,
                )
            with gr.Column():
                transcription = gr.Textbox(label="Transcription", interactive=False)
                transcription_tps = gr.Textbox(label="TPS", placeholder="No data yet...", visible=True)
                instruction = gr.Text(label="Instruction", visible=True, value="", render=True, interactive=True)
            with gr.Column():
                response = gr.Text(label="Response", value="", visible=True)
                response_tps = gr.Text(label="TPS", placeholder="No data yet...", visible=True)

            def stream_audio(
                audio: tuple[int, np.ndarray],
                model: str,
            ) -> Iterator[tuple[str, str]]:
                audio_state = get_audio()
                audio_state["model"] = model

                updated_stream = False
                stream = get_state().get("stream")
                if stream is not None:
                    audio_state["stream"] = stream
                    updated_stream = True

                for state, transcription, transcription_tps in handle_audio_stream(
                    audio,
                    audio_state,
                    0.0,
                    http_client,
                ):
                    update_audio(state)
                    if "Not enough audio yet." in transcription:
                        transcription = ""

                    console.print(f"Transcription: {transcription}", style="bold white")
                    yield transcription, transcription_tps

                    if updated_stream:
                        update_state({"stream": None})

            def run_next_speech_on_change() -> bool:
                global \
                    last_should_speak_value, \
                    is_speaking, \
                    starting_speaking, \
                    audio_start_time, \
                    audio_seconds, \
                    ending_speaking
                was_speaking = is_speaking
                while time() < audio_start_time + audio_seconds and not get_state().get("clear"):
                    was_speaking = True
                    is_speaking = False
                if was_speaking:
                    console.print("Done speaking", style="bold white")
                if last_should_speak_value:
                    yprint(f"Last Speak toggle: {last_should_speak_value}")
                if is_speaking:
                    yprint(f"was_speaking: {is_speaking}")

                update_state({"speak_mode": "wait"})

                if ending_speaking and not is_speaking:
                    console.print("Ending speaking", style="bold white")
                    ending_speaking = False
                    update_state(
                        {
                            "speak_mode": "wait",
                            "spoken": "",
                            "audio_array": np.array([0], dtype=np.int16),
                            "uncommitted": "",
                            "response": "",
                            "act_mode": "wait",
                            "audio_finish_time": float("inf"),
                            "first_act": False,
                            "pred_instr_mode": "predict",
                            "transcription": "",
                            "instruction": "",
                            "sentences": [],
                            "spoken_idx": 0,
                            "stream": np.array([]),
                        }
                    )

                if not is_speaking:
                    starting_speaking = False
                    yprint("Starting speaking")
                    last_should_speak_value = not last_should_speak_value
                    return last_should_speak_value
                return last_should_speak_value

            audio.stream(
                fn=stream_audio,
                inputs=[
                    audio,
                    model_dropdown,
                ],
                outputs=[transcription, transcription_tps],
            )
            transcription.change(
                predict_instruction,
                inputs=[transcription, instruction],
                outputs=[instruction],
            )
            instruction.change(
                act,
                inputs=[instruction, response, response_tps],
                outputs=[response, response_tps],
            )
            should_speak = gr.Checkbox(
                run_next_speech_on_change,
                label="Change = should speak",
                value=False,
                render=True,
                interactive=True,
                every=0.5,
            )
            should_speak.change(
                speak,
                inputs=[response, first_speaker_name, first_speaker_language],
                outputs=[audio_out],
                trigger_mode="once",
            )

            def clear_button_click():
                clear_states()
                return "", "", ""

            clear_button.click(
                clear_button_click,
                inputs=[],
                outputs=[transcription, instruction, response],
                trigger_mode="once",
            )
        demo.load(update_model_dropdown, inputs=None, outputs=[model_dropdown], queue=True)
        return demo


demo = create_gradio_demo(AudioConfig(), TaskConfig())
if __name__ == "__main__":
    import logging

    from rich.logging import RichHandler

    log = logging.getLogger(__file__)
    root = logging.getLogger()
    log.addHandler(RichHandler())
    root.addHandler(RichHandler())
    demo.queue().launch(
        server_name="0.0.0.0", share=False, show_error=True, debug=True, root_path="/audio", server_port=7861
    )
