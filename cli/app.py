from __future__ import annotations

import curses
import importlib
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from array import array
from pathlib import Path
from typing import Any, Callable
import pandas as pd

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app as model_app

# Alias the feature list so the batch block can reference it directly
FEATURES = model_app.FEATURES

console = Console(force_terminal=True, color_system="truecolor")
WHISPER_MODEL_PATH = Path(
    os.environ.get(
        "DRIST_WHISPER_MODEL",
        str(PROJECT_ROOT / "models" / "ggml-base.en.bin"),
    )
)
AUDIO_DEVICE = os.environ.get("DRIST_AUDIO_DEVICE")
WHISPER_CLI = os.environ.get("DRIST_WHISPER_CLI", "whisper-cli")

## features // for preprocessing
CHOICE_FEATURES = {
    "HighBP": model_app.BINARY_OPTIONS,
    "HighChol": model_app.BINARY_OPTIONS,
    "Smoker": model_app.BINARY_OPTIONS,
    "Stroke": model_app.BINARY_OPTIONS,
    "HeartDiseaseorAttack": model_app.BINARY_OPTIONS,
    "PhysActivity": model_app.BINARY_OPTIONS,
    "HvyAlcoholConsump": model_app.BINARY_OPTIONS,
    "DiffWalk": model_app.BINARY_OPTIONS,
    "GenHlth": model_app.HEALTH_OPTIONS,
    "Age": model_app.AGE_OPTIONS,
    "Education": model_app.EDUCATION_OPTIONS,
    "Income": model_app.INCOME_OPTIONS,
}
NUMERIC_LIMITS = {
    "BMI": (12.0, 98.0),
    "MentHlth": (0.0, 30.0),
    "PhysHlth": (0.0, 30.0),
}
BINARY_FEATURES = {
    "HighBP",
    "HighChol",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "HvyAlcoholConsump",
    "DiffWalk",
}

# stuff we can intut at model selection
MODEL_CHOICES = {
    "t": model_app.FT_MODEL_NAME,
    "transformer": model_app.FT_MODEL_NAME,
    "ft-transformer": model_app.FT_MODEL_NAME,
    "l": model_app.LOGREG_MODEL_NAME,
    "logreg": model_app.LOGREG_MODEL_NAME,
    "logistic regression": model_app.LOGREG_MODEL_NAME,
    "logistic_regression": model_app.LOGREG_MODEL_NAME,
    "x": model_app.XGBOOST_MODEL_NAME,
    "xgboost": model_app.XGBOOST_MODEL_NAME,
}
NUMBER_WORDS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}


class VoiceInputError(RuntimeError):
    pass


### TTS Engine

# Lightweight TTS via pyttsx3 (offline, uses macOS "say" under the hood)
# Set DRIST_TTS=0  to disable entirely.
# Set DRIST_TTS_VOICE to a voice name (or substring) e.g. "Daniel", "Samantha",
#   "Zarvox", "Rishi" — matched case-insensitively against the system voice list.
#   Run `python cli/app.py --list-voices` to see all available voices.
# Set DRIST_TTS_RATE to override words-per-minute (default 175).
# Falls back silently if pyttsx3 is not installed.

TTS_ENABLED: bool = os.environ.get("DRIST_TTS", "1").strip() not in {"0", "false", "off", "no"}
TTS_VOICE: str = os.environ.get("DRIST_TTS_VOICE", "").strip()
TTS_RATE: int = int(os.environ.get("DRIST_TTS_RATE", "175"))
_tts_engine: Any = None  # lazily initialised


def _get_tts_engine() -> Any | None:
    """Return a cached pyttsx3 engine, or None if unavailable / disabled."""
    global _tts_engine
    if not TTS_ENABLED:
        return None
    if _tts_engine is not None:
        return _tts_engine
    try:
        pyttsx3 = importlib.import_module("pyttsx3")
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", 0.9)
        if TTS_VOICE:
            voices = engine.getProperty("voices")
            needle = TTS_VOICE.lower()
            match = next(
                (v for v in voices if needle in v.name.lower() or needle in v.id.lower()),
                None,
            )
            if match:
                engine.setProperty("voice", match.id)
            else:
                console.print(
                    f"[yellow]TTS: voice {TTS_VOICE!r} not found — using default. "
                    "Run with --list-voices to see all options.[/]"
                )
        _tts_engine = engine
        return _tts_engine
    except Exception:
        return None


def list_voices() -> None:
    """Print all available TTS voices to the console and exit."""
    try:
        pyttsx3 = importlib.import_module("pyttsx3")
    except ImportError:
        console.print("[red]pyttsx3 is not installed. Run: pip install pyttsx3[/]")
        raise SystemExit(1)
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    table = Table(
        title="Available TTS voices  (set with DRIST_TTS_VOICE)",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold bright_magenta",
    )
    table.add_column("Name", style="bold bright_cyan")
    table.add_column("Language(s)", style="white")
    table.add_column("Gender", style="dim")
    for v in voices:
        langs = ", ".join(str(l) for l in v.languages) if v.languages else "—"
        gender = str(v.gender).replace("VoiceGender", "") if v.gender else "—"
        table.add_row(v.name, langs, gender)
    console.print(table)


def speak(text: str) -> None:
    """Speak *text* aloud using pyttsx3 (or native 'say' on mac). Never raises."""
    if not TTS_ENABLED:
        return
        
    if sys.platform == "darwin":
        # Native 'say' blocks perfectly on macOS; pyttsx3 runAndWait can return early.
        cmd = ["say"]
        if TTS_VOICE:
            cmd.extend(["-v", TTS_VOICE])
        cmd.extend(["-r", str(TTS_RATE), text])
        try:
            subprocess.run(cmd, check=False)
            time.sleep(0.4)
        except Exception:
            pass
        return

    engine = _get_tts_engine()
    if engine is None:
        return
    try:
        engine.say(text)
        engine.runAndWait()
        time.sleep(0.4)
    except Exception:
        pass


def transcribe_voice() -> str:
    try:
        sd = importlib.import_module("sounddevice")
    except ImportError as error:
        raise VoiceInputError(
            "Offline voice packages are missing. Install them with "
            "'pip install -r requirements-voice.txt'."
        ) from error

    whisper_executable = shutil.which(WHISPER_CLI)
    if not whisper_executable:
        raise VoiceInputError(
            f"Cannot find {WHISPER_CLI!r}. Run 'bash scripts/setup-voice.sh' "
            "or set DRIST_WHISPER_CLI to the whisper-cli executable."
        )
    if not WHISPER_MODEL_PATH.is_file():
        raise VoiceInputError(
            f"Whisper model not found at {WHISPER_MODEL_PATH}. "
            "Run 'bash scripts/setup-voice.sh' or set DRIST_WHISPER_MODEL."
        )

    try:
        silence_seconds = 0.0
        heard_speech = False
        started_at = time.monotonic()
        audio_chunks: list[bytes] = []
        with console.status(
            "[bold #A855F7]● Listening locally[/] [dim]Speak one value; pause to finish[/]",
            spinner="point",
        ):
            stream_options: dict[str, Any] = {
                "samplerate": 16000,
                "blocksize": 3200,
                "channels": 1,
                "dtype": "int16",
            }
            if AUDIO_DEVICE:
                stream_options["device"] = (
                    int(AUDIO_DEVICE) if AUDIO_DEVICE.isdigit() else AUDIO_DEVICE
                )
            with sd.InputStream(
                **stream_options,
            ) as audio_stream:
                # Flush the first 0.5 seconds of audio to clear any TTS echo
                audio_stream.read(8000)
                started_at = time.monotonic()
                while time.monotonic() - started_at < 5:
                    audio_data, _ = audio_stream.read(3200)
                    samples = array("h", audio_data.tobytes())
                    audio_chunks.append(samples.tobytes())
                    rms = math.sqrt(sum(sample * sample for sample in samples) / max(1, len(samples)))
                    if rms >= 450:
                        heard_speech = True
                        silence_seconds = 0.0
                    elif heard_speech:
                        silence_seconds += len(samples) / 16000
                    if heard_speech and silence_seconds >= 0.45:
                        break

        if not heard_speech:
            console.print("[yellow]No speech detected. You can try again or type a value.[/]")
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_path = Path(audio_file.name)
        try:
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"".join(audio_chunks))
            with console.status("[bold bright_cyan]Transcribing locally...[/]", spinner="dots"):
                result = subprocess.run(
                    [
                        whisper_executable,
                        "-m", str(WHISPER_MODEL_PATH),
                        "-f", str(audio_path),
                        "-l", "en",
                        "-t", "2",
                        "-nt",
                        "-np",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=45,
                    check=False,
                )
            if result.returncode != 0:
                details = result.stderr.strip() or result.stdout.strip()
                raise VoiceInputError(f"whisper.cpp transcription failed: {details}")
            transcript = re.sub(r"^\[[\d:.>\s\u2013-]+\]\s*", "", result.stdout.strip()).strip()
        finally:
            audio_path.unlink(missing_ok=True)

        if not transcript:
            console.print("[yellow]No speech detected. You can try again or type a value.[/]")
        return transcript
    except VoiceInputError:
        raise
    except Exception as error:
        raise VoiceInputError(f"Microphone or speech recognition error: {error}") from error


AUTO_VOICE: bool = False

def prompt_value(prompt: str, input_fn: Callable[[str], str], spoken_prompt: str = "") -> str:
    global AUTO_VOICE
    first_iteration = True
    while True:
        if AUTO_VOICE:
            if first_iteration and spoken_prompt:
                console.print(f"{prompt} ", end="")
                speak(spoken_prompt)
                console.print("[dim](listening... say 'manual' to type instead)[/]")
            else:
                console.print(f"{prompt} [dim](listening... say 'manual' to type instead)[/]")
            first_iteration = False
            try:
                transcript = transcribe_voice()
            except VoiceInputError as error:
                console.print(f"[bold red]{error}[/]")
                AUTO_VOICE = False
                continue

            if not transcript:
                AUTO_VOICE = False
                continue

            if transcript.lower().strip() in {"stop", "manual", "text", "type"}:
                console.print("[yellow]Switched to manual typing mode.[/]")
                speak("Switched to manual typing")
                AUTO_VOICE = False
                continue

            console.print(Panel.fit(
                f"[bold white]{transcript}[/] [dim](validated automatically)[/]",
                title="[bold bright_cyan]Heard locally[/]",
                border_style="bright_cyan",
            ))
            speak(f"I heard: {transcript}")
            return transcript

        console.print(f"{prompt} [dim](or type 'voice' or 'auto')[/] ", end="")
        raw_value = input_fn("").strip()
        
        if raw_value.lower() in {"voice auto", "auto"}:
            AUTO_VOICE = True
            speak("Auto voice mode enabled")
            continue
            
        if raw_value.lower() != "voice":
            return raw_value
            
        try:
            transcript = transcribe_voice()
        except VoiceInputError as error:
            console.print(f"[bold red]{error}[/]")
            continue
        if not transcript:
            continue
        console.print(Panel.fit(
            f"[bold white]{transcript}[/] [dim](validated automatically)[/]",
            title="[bold bright_cyan]Heard locally[/]",
            border_style="bright_cyan",
        ))
        speak(f"I heard: {transcript}")
        return transcript


def parse_spoken_number(text: str) -> float | None:
    cleaned = text.lower().replace("-", " ").replace(",", " ")
    try:
        return float(cleaned)
    except ValueError:
        pass

    tokens = re.findall(r"[a-z]+|\d+|\.", cleaned)
    if not tokens:
        return None
    if "point" in tokens or "." in tokens:
        point_index = tokens.index("point") if "point" in tokens else tokens.index(".")
        integer = _parse_spoken_integer(tokens[:point_index])
        fractional = tokens[point_index + 1:]
        digit_values = [
            NUMBER_WORDS[token] if token in NUMBER_WORDS and NUMBER_WORDS[token] < 10
            else int(token) if token.isdigit() and len(token) == 1
            else None
            for token in fractional
        ]
        if integer is None or not digit_values or any(value is None for value in digit_values):
            return None
        decimal_digits = "".join(str(value) for value in digit_values)
        return float(f"{integer}.{decimal_digits}")
    return _parse_spoken_integer(tokens)


def _parse_spoken_integer(tokens: list[str]) -> int | None:
    current = 0
    found = False
    for token in tokens:
        if token in {"and", "option", "number", "choice", "choose", "select", "a"}:
            continue
        if token.isdigit():
            current += int(token)
            found = True
        elif token in NUMBER_WORDS:
            current += NUMBER_WORDS[token]
            found = True
        elif token == "hundred" and found:
            current = max(1, current) * 100
        else:
            if found:
                break
            else:
                continue
    return current if found else None


## easter egg --> snake 
class SnakeGame:
    def __init__(self, width: int = 44, height: int = 42) -> None:
        self.width = width
        self.height = height
        center = (width // 2, height // 2)
        self.snake = [center, (center[0] - 1, center[1]), (center[0] - 2, center[1])]
        self.direction = (1, 0)
        self.food = self._place_food()
        self.score = 0
        self.alive = True

    def _place_food(self) -> tuple[int, int] | None:
        empty_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.snake
        ]
        return random.choice(empty_cells) if empty_cells else None

    def turn(self, direction: tuple[int, int]) -> None:
        if direction != (-self.direction[0], -self.direction[1]):
            self.direction = direction

    def step(self) -> bool:
        head_x, head_y = self.snake[0]
        next_head = (head_x + self.direction[0], head_y + self.direction[1])
        eating = next_head == self.food
        body = self.snake if eating else self.snake[:-1]
        if (
            not 0 <= next_head[0] < self.width
            or not 0 <= next_head[1] < self.height
            or next_head in body
        ):
            self.alive = False
            return False

        self.snake.insert(0, next_head)
        if eating:
            self.score += 1
            self.food = self._place_food()
        else:
            self.snake.pop()
        return True


def _play_snake_screen(screen: curses.window) -> None:
    game = SnakeGame()
    screen.nodelay(True)
    screen.keypad(True)
    delay = 0.13
    last_step = time.monotonic()
    directions = {
        curses.KEY_UP: (0, -1), ord("w"): (0, -1),
        curses.KEY_DOWN: (0, 1), ord("s"): (0, 1),
        curses.KEY_LEFT: (-1, 0), ord("a"): (-1, 0),
        curses.KEY_RIGHT: (1, 0), ord("d"): (1, 0),
    }

    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_MAGENTA, -1)
        curses.init_pair(2, curses.COLOR_CYAN, -1)
        curses.init_pair(3, curses.COLOR_GREEN, -1)

    while game.alive:
        screen.erase()
        rows, columns = screen.getmaxyx()
        board_left = max(0, (columns - game.width - 2) // 2)
        board_top = 3
        if rows < game.height + 9 or columns < game.width + 4:
            screen.addstr(0, 0, "Resize terminal to at least 30x22. Press Q to leave.")
            screen.refresh()
            key = screen.getch()
            if key in (ord("q"), ord("Q")):
                return
            time.sleep(0.1)
            continue

        title = "DRiST SNAKE"
        screen.addstr(0, max(0, (columns - len(title)) // 2), title,
                      curses.color_pair(1) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD)
        screen.addstr(1, board_left, f"Score: {game.score}    WASD / arrows: move    Q: quit")
        screen.addstr(board_top, board_left, "+" + "-" * game.width + "+")
        for y in range(game.height):
            screen.addstr(board_top + y + 1, board_left, "|")
            screen.addstr(board_top + y + 1, board_left + game.width + 1, "|")
        screen.addstr(board_top + game.height + 1, board_left, "+" + "-" * game.width + "+")

        if game.food is not None:
            food_x, food_y = game.food
            screen.addstr(board_top + food_y + 1, board_left + food_x + 1, "*",
                          curses.color_pair(2) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD)
        for index, (x, y) in enumerate(game.snake):
            glyph = "@" if index == 0 else "o"
            style = curses.color_pair(3) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD
            screen.addstr(board_top + y + 1, board_left + x + 1, glyph, style)
        screen.refresh()

        key = screen.getch()
        if key in (ord("q"), ord("Q")):
            return
        if key in directions:
            game.turn(directions[key])
        now = time.monotonic()
        if now - last_step >= delay:
            game.step()
            last_step = now

    screen.nodelay(False)
    screen.erase()
    message = f"Game over! Score: {game.score}  |  Press any key to return"
    screen.addstr(max(0, rows // 2), max(0, (columns - len(message)) // 2), message)
    screen.refresh()
    screen.getch()


def play_snake() -> None:
    curses.wrapper(_play_snake_screen)


SPOKEN_QUESTIONS: dict[str, str] = {
    "HighBP": "Have you ever been told by a doctor that you have high blood pressure? Say yes or no.",
    "HighChol": "Have you ever had your blood cholesterol checked and been told it was high? Say yes or no.",
    "BMI": "What is your Body Mass Index, or BMI? For example, twenty five.",
    "Smoker": "Have you smoked at least 100 cigarettes in your entire life? Say yes or no.",
    "Stroke": "Have you ever had a stroke? Say yes or no.",
    "HeartDiseaseorAttack": "Have you ever had coronary heart disease or a heart attack? Say yes or no.",
    "PhysActivity": "Did you do any physical activity or exercise in the past 30 days? Say yes or no.",
    "HvyAlcoholConsump": "Do you engage in heavy alcohol consumption? Say yes or no.",
    "GenHlth": "In general, how would you rate your overall health? Excellent, Very Good, Good, Fair, or Poor?",
    "MentHlth": "How many days in the past 30 days was your mental health not good? Say a number from 0 to 30.",
    "PhysHlth": "How many days in the past 30 days was your physical health not good? Say a number from 0 to 30.",
    "DiffWalk": "Do you have serious difficulty walking or climbing stairs? Say yes or no.",
    "Age": "What is your age group? For example, 18 to 24, 45 to 49, or say the number from the screen.",
    "Education": "What is your highest level of education completed? Please choose an option from 1 to 6.",
    "Income": "What is your annual household income bracket? Please choose an option from 1 to 8.",
}


def choose_model(
    input_fn: Callable[[str], str] = input,
    snake_fn: Callable[[], None] = play_snake,
) -> str:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold bright_magenta")
    table.add_column("Key", style="bold bright_cyan", width=5)
    table.add_column("Model", style="white")
    table.add_row("T", "FT-Transformer checkpoint (default)")
    table.add_row("L", "Logistic Regression")
    table.add_row("X", "XGBoost")
    console.print(table)

    spoken_prompt = "Please choose a prediction model. Say Transformer, Logistic Regression, or XGBoost."

    while True:
        raw_command = prompt_value(
            "[bold bright_magenta]Model key[/] [dim](T/L/X, default T, or Q to quit):[/]",
            input_fn,
            spoken_prompt=spoken_prompt,
        ).strip().lower()

        if raw_command in {"", "default", "first", "1", "one"}:
            model_name = model_app.FT_MODEL_NAME
            console.print(f"[green]Selected (default):[/] [bold]{model_name}[/]\n")
            return model_name
        if raw_command in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if raw_command == "snake":
            snake_fn()
            console.print("\n[bold bright_magenta]Model selection[/]")
            console.print(table)
            continue

        # Speech and text recognition for model choices
        tokens = set(re.sub(r"[^a-z0-9]", " ", raw_command).split())
        if tokens.intersection({"transformer", "ft", "t", "1", "first"}):
            model_name = model_app.FT_MODEL_NAME
        elif tokens.intersection({"logistic", "logreg", "regression", "l", "2", "second"}):
            model_name = model_app.LOGREG_MODEL_NAME
        elif tokens.intersection({"xgboost", "xgb", "boost", "x", "3", "third"}):
            model_name = model_app.XGBOOST_MODEL_NAME
        else:
            model_name = MODEL_CHOICES.get(raw_command)

        if model_name is not None:
            console.print(f"[green]Selected:[/] [bold]{model_name}[/]\n")
            return model_name
        console.print("[bold red]Enter T, L, or X (or say Transformer, Logistic, or XGBoost).[/]")


def read_numeric(
    name: str,
    label: str,
    input_fn: Callable[[str], str] = input,
) -> float:
    minimum, maximum = NUMERIC_LIMITS[name]
    spoken = SPOKEN_QUESTIONS.get(name, f"{label}. Enter a value from {minimum:g} to {maximum:g}.")
    while True:
        raw_value = prompt_value(
            f"[bold bright_cyan]{label}[/] [dim]({minimum:g}-{maximum:g})[/]",
            input_fn,
            spoken_prompt=spoken,
        )
        parsed_number = parse_spoken_number(raw_value)
        if parsed_number is None:
            console.print("[red]Enter a number.[/]")
            continue
        value = parsed_number
        if minimum <= value <= maximum:
            return value
        console.print(f"[red]Enter a value from {minimum:g} to {maximum:g}.[/]")


def read_categorical(
    name: str,
    label: str,
    input_fn: Callable[[str], str] = input,
) -> int:
    options = CHOICE_FEATURES[name]
    table = Table(title=label, box=box.SIMPLE, show_header=False, title_style="bold bright_cyan")
    table.add_column("Key", style="bold bright_magenta", width=5)
    table.add_column("Choice", style="white")
    for option_label, value in options:
        table.add_row(str(value), option_label)
    console.print(table)

    valid_values = {int(value) for _, value in options}
    spoken = SPOKEN_QUESTIONS.get(name)
    if not spoken:
        if name in BINARY_FEATURES:
            spoken = f"Do you have {label.lower()}? Say yes or no."
        else:
            option_labels = [opt[0] for opt in options]
            if len(option_labels) > 4:
                spoken = f"{label}. Please say the option number from the screen."
            else:
                spoken = f"{label}. Say 1 for {option_labels[0]}, 2 for {option_labels[1]}, or say the option number."

    while True:
        hint = "yes/no or 0/1" if name in BINARY_FEATURES else "/".join(map(str, sorted(valid_values)))
        raw_value = prompt_value(
            f"[bold bright_magenta]Choose[/] [dim]({hint})[/]",
            input_fn,
            spoken_prompt=spoken,
        ).strip().lower()
        if name in BINARY_FEATURES:
            binary_aliases = {
                "no": 0, "n": 0, "nope": 0, "zero": 0, "0": 0,
                "yes": 1, "y": 1, "yeah": 1, "yep": 1, "one": 1, "1": 1
            }
            # Tokenize to avoid substring bugs (e.g. "n" in "one")
            tokens = set(re.sub(r"[^a-z0-9]", " ", raw_value).split())
            for alias, val in binary_aliases.items():
                if alias in tokens:
                    return val
                    
        def normalize_for_match(text: str) -> str:
            words = text.lower().replace("-", " ").replace("+", " plus ").replace("/", " ").split()
            norm = []
            for w in words:
                w_clean = re.sub(r"[^a-z0-9]", "", w)
                if w_clean in NUMBER_WORDS:
                    norm.append(str(NUMBER_WORDS[w_clean]))
                elif w_clean:
                    norm.append(w_clean)
            joined = " ".join(norm)
            joined = re.sub(r"\b(to|or|and|plus|a|the|of|for|through)\b", " ", joined)
            joined = re.sub(r"s\b", "", joined)  # simple plural removal
            return re.sub(r"\s+", " ", joined).strip()

        norm_input = normalize_for_match(raw_value)
        
        # 1. Fuzzy match against the full label text
        matched_value = None
        for option_label, option_value in options:
            norm_label = normalize_for_match(option_label)
            if norm_label in norm_input or (len(norm_input) > 2 and norm_input in norm_label):
                matched_value = int(option_value)
                break
        
        if matched_value is not None:
            return matched_value
            
        # 2. Fallback: Check if they just said the option number (e.g., "Option 3")
        parsed_number = parse_spoken_number(raw_value)
        selected = int(parsed_number) if parsed_number is not None and parsed_number.is_integer() else None
        if selected in valid_values:
            return selected
            
        console.print(f"[red]Enter one of: {', '.join(str(value) for value in sorted(valid_values))}.[/]")


def collect_features(input_fn: Callable[[str], str] = input) -> dict[str, int | float]:
    values: dict[str, int | float] = {}
    console.print(Panel.fit(
        "Enter each value below. Categorical choices show their valid keys.",
        title="[bold bright_magenta]Patient details[/]",
        border_style="bright_magenta",
    ))
    for index, (name, label) in enumerate(model_app.FIELD_DEFINITIONS, start=1):
        console.print(
            Text("[*]", style="bold #A855F7")
            + Text(
                f" Feature {index:02d}/{len(model_app.FIELD_DEFINITIONS):02d}",
                style="dim",
            )
        )
        if name in NUMERIC_LIMITS:
            values[name] = read_numeric(name, label, input_fn)
        else:
            values[name] = read_categorical(name, label, input_fn)
        console.print()
    return values


def review_features(
    values: dict[str, int | float],
    input_fn: Callable[[str], str] = input,
) -> dict[str, int | float] | None:
    feature_definitions = model_app.FIELD_DEFINITIONS
    while True:
        table = Table(
            title="Review patient inputs",
            box=box.ROUNDED,
            title_style="bold bright_magenta",
            border_style="bright_magenta",
        )
        table.add_column("#", style="bold bright_cyan", justify="right")
        table.add_column("Feature", style="white")
        table.add_column("Entered value", style="bold")
        for index, (name, label) in enumerate(feature_definitions, start=1):
            value = values[name]
            if name in CHOICE_FEATURES:
                value = next(
                    (option_label for option_label, option_value in CHOICE_FEATURES[name]
                     if int(option_value) == int(value)),
                    value,
                )
            table.add_row(str(index), label, str(value))
        console.print(table)
        controls = Text()
        controls.append("[Enter]", style="bold bright_magenta")
        controls.append(" predict  ")
        controls.append("[E]", style="bold bright_cyan")
        controls.append(" edit a value  ")
        controls.append("[Q]", style="bold red")
        controls.append(" cancel")
        console.print(controls)

        spoken_review = (
            "Please review the entered patient details. "
            "Say predict to run the prediction, edit to change a value, or cancel."
        )
        command = prompt_value(
            "[bold bright_magenta]Action[/] [dim](Enter to predict, E to edit, Q to cancel):[/]",
            input_fn,
            spoken_prompt=spoken_review,
        ).strip().lower()

        if command in {"", "p", "predict", "run", "yes", "confirm", "proceed", "go", "continue", "okay", "ok"}:
            return values
        if command in {"q", "quit", "cancel", "stop", "no"}:
            return None

        # Check if user said edit, or directly asked to edit a specific feature (e.g. "edit blood pressure")
        edit_target = ""
        if "edit" in command or "change" in command:
            edit_target = re.sub(r"\b(edit|change|update|the)\b", "", command).strip()
        elif command == "e":
            edit_target = ""
        else:
            console.print("[red]Say predict, edit, or cancel.[/]")
            continue

        if not edit_target:
            spoken_edit = "Which feature would you like to edit? Say the number or feature name."
            selection = prompt_value(
                "[bright_cyan]Enter feature number or name to edit:[/] ",
                input_fn,
                spoken_prompt=spoken_edit,
            ).strip().lower()
        else:
            selection = edit_target

        selected_index: int | None = None
        parsed_num = parse_spoken_number(selection)
        if parsed_num is not None and 1 <= int(parsed_num) <= len(feature_definitions):
            selected_index = int(parsed_num) - 1
        else:
            selection_clean = re.sub(r"[^a-z0-9]", "", selection)
            selected_index = next(
                (
                    index
                    for index, (name, label) in enumerate(feature_definitions)
                    if selection_clean in re.sub(r"[^a-z0-9]", "", name.lower())
                    or selection_clean in re.sub(r"[^a-z0-9]", "", label.lower())
                    or re.sub(r"[^a-z0-9]", "", name.lower()) in selection_clean
                    or re.sub(r"[^a-z0-9]", "", label.lower()) in selection_clean
                ),
                None,
            )

        if selected_index is None:
            console.print("[red]No matching feature found. Please try again.[/]")
            speak("No matching feature found.")
            continue

        name, label = feature_definitions[selected_index]
        if name in NUMERIC_LIMITS:
            values[name] = read_numeric(name, label, input_fn)
        else:
            values[name] = read_categorical(name, label, input_fn)


def run() -> int:
    banner = Text(
        " ____   ____    _   ____   _____\n"
        "|  _ \\ |  _ \\  (_) / ___| |_   _|\n"
        "| | | || |_) | | | \\___ \\   | |\n"
        "| |_| ||  _ <  | |  ___) |  | |\n"
        "|____/ |_| \\_\\ |_| |____/   |_|\n"
        "Diabetes Risk Prediction Tool",
        style="bold #A855F7",
    )
    console.print(Panel(banner, box=box.DOUBLE, border_style="#A855F7", padding=(1, 2)))

    global AUTO_VOICE
    if any(arg in sys.argv for arg in ("--auto", "--voice", "-a", "-v")):
        AUTO_VOICE = True
        console.print("[green]Hands-free voice mode enabled via flag.[/]\n")
        speak("Hands-free voice mode enabled.")
    elif not AUTO_VOICE and sys.stdin.isatty():
        voice_input = console.input("[bold]Enable hands-free voice mode? (y/N): [/]").strip().lower()
        if voice_input in {"y", "yes"}:
            AUTO_VOICE = True
            console.print("[green]Hands-free voice mode enabled.[/]\n")
            speak("Hands-free voice mode enabled.")

    model_name = choose_model()

    # ---- batch mode prompt -------------------------------------------------
    batch_input = console.input("[bold]Batch mode? (y/N): [/]").strip().lower() if not AUTO_VOICE else "n"
    is_batch = batch_input in {"y", "yes"}
    batch_path: Path | None = None
    if is_batch:
        path_str = console.input("Enter CSV file path for batch processing: ").strip()
        batch_path = Path(path_str)
        if not batch_path.is_file():
            console.print(f"[red]File not found: {batch_path}[/]")
            return 1
    # -----------------------------------------------------------------------

    try:
        with console.status(f"[bold bright_cyan]Loading saved {model_name} model...[/]", spinner="dots"):
            fitted_model = model_app.load_model(model_name)
    except Exception as error:
        console.print(f"[bold red]Could not load model:[/] {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    if is_batch and batch_path:
        # Load CSV, drop rows with missing required features, predict each row
        df = pd.read_csv(batch_path)
        # Ensure required columns exist
        missing = set(FEATURES) - set(df.columns)
        if missing:
            console.print(f"[red]CSV missing required columns: {', '.join(missing)}[/]")
            return 1
        # Keep only needed columns and drop rows with any NaN in those columns
        df_clean = df[FEATURES].dropna()
        if df_clean.empty:
            console.print("[yellow]No complete rows found after dropping missing values. Nothing to predict.[/]")
            return 0
        results: list[dict[str, Any]] = []
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            transient=True,
        )
        with progress:
            task = progress.add_task("Predicting", total=len(df_clean))
            for _, row in df_clean.iterrows():
                values = {name: row[name] for name in FEATURES}
                pred, prob = model_app.predict_diabetes(fitted_model, values)
                results.append({**row.to_dict(), "prediction": pred, "probability": prob})
                progress.update(task, advance=1)
        # Export results to project root directory
        out_path = Path.cwd() / f"batch_predictions_{model_name.replace(' ', '_').lower()}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        console.print(f"[green]Batch predictions saved to {out_path}[/]")
        return 0

    # ---- single‑patient interactive flow ------------------------------------
    values = collect_features()
    values = review_features(values)
    if values is None:
        console.print("[yellow]Prediction cancelled. No result was generated.[/]")
        return 0
    try:
        prediction, probability = model_app.predict_diabetes(fitted_model, values)
    except Exception as error:
        console.print(f"[bold red]Prediction failed:[/] {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    class_label = "DIABETES" if prediction == 1 else "NON-DIABETES"
    result_style = "bold red" if prediction == 1 else "bold light_green"
    result = Table(box=box.ROUNDED, show_header=False, border_style="bright_magenta")
    result.add_column("Item", style="dim")
    result.add_column("Result")
    result.add_row("Model", model_name)
    result.add_row("Prediction", f"[{result_style}]{class_label}[/]")
    result.add_row("Diabetes probability", f"[bold]{probability:.1%}[/]")
    top_factors_list: list[str] = []
    try:
        top_factors = model_app.explain_prediction(fitted_model, values, top_k=3)
        if top_factors:
            top_factors_list = [model_app.humanize_factor(name, score) for name, score in top_factors]
            factors_str = "\n".join(f"• {factor}" for factor in top_factors_list)
            result.add_row("Top contributing factors", factors_str)
    except Exception:
        pass
    console.print(Panel(result, title="[bold bright_magenta]Prediction[/]", border_style="bright_magenta"))
    speak_text = f"Prediction complete. Result: {class_label}, with a diabetes probability of {probability:.0%}."
    if top_factors_list:
        speak_text += f" Main factors: {', '.join(top_factors_list)}."
    speak(speak_text)
    return 0



if __name__ == "__main__":
    if "--list-voices" in sys.argv:
        list_voices()
        raise SystemExit(0)
    try:
        raise SystemExit(run())
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled. Byee!")
        raise SystemExit(130)
