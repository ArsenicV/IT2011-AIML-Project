from __future__ import annotations

import curses
import random
import sys
import time
from pathlib import Path
from typing import Callable

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app as model_app

console = Console(force_terminal=True, color_system="truecolor")

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


class SnakeGame:
    def __init__(self, width: int = 24, height: int = 12) -> None:
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


def choose_model(
    input_fn: Callable[[str], str] = input,
    snake_fn: Callable[[], None] = play_snake,
) -> str:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold bright_magenta")
    table.add_column("Key", style="bold bright_cyan", width=5)
    table.add_column("Model", style="white")
    table.add_row("T", "FT-Transformer checkpoint")
    table.add_row("L", "Logistic Regression")
    table.add_row("X", "XGBoost")
    console.print(table)

    while True:
        console.print("[bold bright_magenta]Model key[/] [dim](or Q to quit):[/] ", end="")
        command = input_fn("")
        if command.strip().lower() in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if command.strip().lower() == "snake":
            snake_fn()
            console.print("\n[bold bright_magenta]Model selection[/]")
            console.print(table)
            continue
        model_name = MODEL_CHOICES.get(command.strip().lower())
        if model_name is not None:
            console.print(f"[green]Selected:[/] [bold]{model_name}[/]\n")
            return model_name
        console.print("[bold red]Enter T, L, or X.[/]")


def read_numeric(
    name: str,
    label: str,
    input_fn: Callable[[str], str] = input,
) -> float:
    minimum, maximum = NUMERIC_LIMITS[name]
    while True:
        console.print(f"[bold bright_cyan]{label}[/] [dim]({minimum:g}-{maximum:g})[/] ", end="")
        raw_value = input_fn("").strip()
        try:
            value = float(raw_value)
        except ValueError:
            console.print("[red]Enter a number.[/]")
            continue
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
    while True:
        hint = "yes/no or 0/1" if name in BINARY_FEATURES else "/".join(map(str, sorted(valid_values)))
        console.print(f"[bold bright_magenta]Choose[/] [dim]({hint})[/] ", end="")
        raw_value = input_fn("").strip().lower()
        if name in BINARY_FEATURES:
            binary_aliases = {"no": 0, "n": 0, "yes": 1, "y": 1}
            if raw_value in binary_aliases:
                return binary_aliases[raw_value]
        try:
            selected = int(raw_value)
        except ValueError:
            console.print(f"[red]Enter one of: {', '.join(str(value) for value in sorted(valid_values))}.[/]")
            continue
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
    model_name = choose_model()
    try:
        with console.status(f"[bold bright_cyan]Loading saved {model_name} model...[/]", spinner="dots"):
            fitted_model = model_app.load_model(model_name)
    except Exception as error:
        console.print(f"[bold red]Could not load model:[/] {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    values = collect_features()
    try:
        prediction, probability = model_app.predict_diabetes(fitted_model, values)
    except Exception as error:
        console.print(f"[bold red]Prediction failed:[/] {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    class_label = "higher-risk class" if prediction == 1 else "lower-risk class"
    result_style = "bold red" if prediction == 1 else "bold green"
    result = Table(box=box.ROUNDED, show_header=False, border_style="bright_magenta")
    result.add_column("Item", style="dim")
    result.add_column("Result")
    result.add_row("Model", model_name)
    result.add_row("Output", f"[{result_style}]{prediction} ({class_label})[/]")
    result.add_row("Class 1 probability", f"[bold]{probability:.1%}[/]")
    console.print(Panel(result, title="[bold bright_magenta]Prediction[/]", border_style="bright_magenta"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        raise SystemExit(130)