from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Grid, VerticalScroll
from textual.widgets import Button, Footer, Input, Label, Select, Static


DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "raw"
    / "diabetes_binary_health_indicators_BRFSS2015.csv"
)
MODEL_PATH = Path(__file__).resolve().parent / "weights" / "ft_transformer_best.pt"
LOGREG_MODEL_PATH = Path(__file__).resolve().parent / "weights" / "logreg (1).pt"
XGBOOST_MODEL_PATH = Path(__file__).resolve().parent / "weights" / "xgboost_diabetes_model.joblib"
XGBOOST_RUNNER_PATH = Path(__file__).resolve().parent / "xgboost_inference.py"

FEATURES = [
    "HighBP",
    "HighChol",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "HvyAlcoholConsump",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Age",
    "Education",
    "Income",
]
NUMERICAL_FEATURES = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Income"]
CATEGORICAL_FEATURES = [
    "HighBP",
    "HighChol",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "HvyAlcoholConsump",
    "DiffWalk",
    "Education",
]
NUMERIC_INPUTS = {"BMI", "MentHlth", "PhysHlth"}
FT_MODEL_NAME = "FT-Transformer (checkpoint)"
LOGREG_MODEL_NAME = "Logistic Regression"
XGBOOST_MODEL_NAME = "XGBoost"
MODEL_COMMANDS = {
    "t": FT_MODEL_NAME,
    "transformer": FT_MODEL_NAME,
    "ft-transformer": FT_MODEL_NAME,
    "l": LOGREG_MODEL_NAME,
    "logreg": LOGREG_MODEL_NAME,
    "logistic regression": LOGREG_MODEL_NAME,
    "logistic_regression": LOGREG_MODEL_NAME,
    "x": XGBOOST_MODEL_NAME,
    "xgboost": XGBOOST_MODEL_NAME,
}

BINARY_OPTIONS = [("No", 0), ("Yes", 1)]
AGE_OPTIONS = [
    ("18-24", 1), ("25-29", 2), ("30-34", 3), ("35-39", 4),
    ("40-44", 5), ("45-49", 6), ("50-54", 7), ("55-59", 8),
    ("60-64", 9), ("65-69", 10), ("70-74", 11), ("75-79", 12),
    ("80+", 13),
]
EDUCATION_OPTIONS = [
    ("Never attended / kindergarten", 1),
    ("Grades 1-8", 2),
    ("Grades 9-11", 3),
    ("Grade 12 / GED", 4),
    ("College 1-3 years", 5),
    ("College 4+ years", 6),
]
INCOME_OPTIONS = [
    ("Under $10,000", 1), ("$10,000-$15,000", 2),
    ("$15,000-$20,000", 3), ("$20,000-$25,000", 4),
    ("$25,000-$35,000", 5), ("$35,000-$50,000", 6),
    ("$50,000-$75,000", 7), ("$75,000 or more", 8),
]
HEALTH_OPTIONS = [
    ("Excellent", 1), ("Very good", 2), ("Good", 3), ("Fair", 4), ("Poor", 5)
]

FIELD_DEFINITIONS: list[tuple[str, str]] = [
    ("HighBP", "High blood pressure"),
    ("HighChol", "High cholesterol"),
    ("BMI", "Body mass index"),
    ("Smoker", "Smoked 100+ cigarettes"),
    ("Stroke", "History of stroke"),
    ("HeartDiseaseorAttack", "Heart disease / attack"),
    ("PhysActivity", "Physical activity"),
    ("HvyAlcoholConsump", "Heavy alcohol consumption"),
    ("GenHlth", "General health"),
    ("MentHlth", "Poor mental health days (0-30)"),
    ("PhysHlth", "Poor physical health days (0-30)"),
    ("DiffWalk", "Difficulty walking"),
    ("Age", "Age group"),
    ("Education", "Education level"),
    ("Income", "Income group"),
]


class FTTransformer(torch.nn.Module):
    def __init__(self, feature_count: int = 28) -> None:
        super().__init__()
        self.feature_weight = torch.nn.Parameter(torch.empty(feature_count, 64))
        self.feature_bias = torch.nn.Parameter(torch.empty(feature_count, 64))
        self.cls_token = torch.nn.Parameter(torch.empty(1, 1, 64))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=6)
        self.norm = torch.nn.LayerNorm(64)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = features.unsqueeze(-1) * self.feature_weight + self.feature_bias
        cls_token = self.cls_token.expand(features.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((cls_token, tokens), dim=1))
        return self.head(self.norm(encoded[:, 0])).squeeze(-1)


class XGBoostArtifact:
    def __init__(self, feature_names: list[str], feature_count: int) -> None:
        self.feature_names = feature_names
        self.n_features_in_ = feature_count


def run_xgboost(request: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(XGBOOST_RUNNER_PATH)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "XGBoost helper failed")
    return json.loads(result.stdout)


def make_preprocessor(*, drop_first: bool = False) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first" if drop_first else None),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def load_model(model_name: str) -> tuple[ColumnTransformer, Any]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    dataset = pd.read_csv(DATA_PATH)
    required_columns = set(FEATURES + ["Diabetes_binary"])
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing_columns))}")

    features = dataset[FEATURES]
    target = dataset["Diabetes_binary"].astype(int)
    x_train, _, y_train, _ = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )

    if model_name not in (FT_MODEL_NAME, LOGREG_MODEL_NAME, XGBOOST_MODEL_NAME):
        raise ValueError(f"Unknown model: {model_name}")

    preprocessor = make_preprocessor(drop_first=model_name == LOGREG_MODEL_NAME)
    train_features = preprocessor.fit_transform(x_train)

    encoder = preprocessor.named_transformers_["cat"]
    feature_names = NUMERICAL_FEATURES + list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    if model_name == FT_MODEL_NAME:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"FT-Transformer checkpoint not found: {MODEL_PATH}")
        if train_features.shape[1] != 28:
            raise ValueError(f"Transformer expects 28 preprocessed features; got {train_features.shape[1]}")
        model: Any = FTTransformer(feature_count=train_features.shape[1])
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
    else:
        import joblib

        model_path = LOGREG_MODEL_PATH if model_name == LOGREG_MODEL_NAME else XGBOOST_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"{model_name} model not found: {model_path}")
        if model_name == XGBOOST_MODEL_NAME:
            metadata = run_xgboost({"action": "inspect"})
            expected_features = metadata["feature_names"]
            model = XGBoostArtifact(expected_features, metadata["feature_count"])
        else:
            model = joblib.load(model_path)
            expected_features = list(getattr(model, "feature_names_in_", []))
        if expected_features and expected_features != feature_names:
            raise ValueError(
                f"{model_name} feature order does not match this app's preprocessing. "
                "Check the artifact's training preprocessing and feature order."
            )
        if model.n_features_in_ != train_features.shape[1]:
            raise ValueError(
                f"{model_name} expects {model.n_features_in_} features; "
                f"preprocessing produced {train_features.shape[1]}"
            )
    return preprocessor, model


def predict_diabetes(
    fitted: tuple[ColumnTransformer, Any], values: dict[str, int | float]
) -> tuple[int, float]:
    preprocessor, model = fitted
    row = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
    transformed_row = preprocessor.transform(row)
    if isinstance(model, XGBoostArtifact):
        result = run_xgboost(
            {
                "action": "predict",
                "feature_names": model.feature_names,
                "features": transformed_row[0].tolist(),
            }
        )
        return int(result["prediction"]), float(result["probability"])
    if isinstance(model, FTTransformer):
        tensor = torch.as_tensor(transformed_row, dtype=torch.float32)
        with torch.inference_mode():
            probability = float(torch.sigmoid(model(tensor))[0].item())
        return int(probability >= 0.5), probability

    if hasattr(model, "feature_names_in_"):
        encoder = preprocessor.named_transformers_["cat"]
        feature_names = NUMERICAL_FEATURES + list(
            encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        )
        transformed_row = pd.DataFrame(transformed_row, columns=feature_names)
    prediction = int(model.predict(transformed_row)[0])
    probability = float(model.predict_proba(transformed_row)[0][list(model.classes_).index(1)])
    return prediction, probability


class DiabetesApp(App[None]):
    TITLE = "DRiST"
    CSS = """
    Screen { align: left top; }
    #content { width: 100%; max-width: 112; height: 1fr; padding: 1 2; }
    #wordmark { height: 6; margin: 0 0 2 0; color: #a855f7; text-style: bold; }
    #selection { width: 100%; height: auto; padding: 1 0; }
    #model-prompt { height: 2; color: $text; }
    #model-command { width: 1fr; max-width: 52; margin: 1 0; }
    #selection-error { height: 2; color: $error; }
    #selection-actions { height: 3; margin: 0 0 2 0; }
    #continue { width: 20; }
    #details { display: none; width: 100%; height: 1fr; }
    #model-row { height: 5; align: left middle; margin: 0 0 1 0; }
    #selected-model { width: 1fr; color: #a855f7; text-style: bold; }
    #change-model { width: 20; }
    #fields { grid-size: 2; grid-columns: 1fr 1fr; grid-gutter: 0 2; height: auto; }
    .field { height: 5; }
    .field Label { height: 1; color: $text-muted; }
    .field Input, .field Select { width: 1fr; }
    #predict { width: 100%; margin: 1 0; }
    #status { height: auto; min-height: 2; padding: 0 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: tuple[ColumnTransformer, Any] | None = None
        self._training_generation = 0
        self._selected_model_name: str | None = None

    def compose(self) -> ComposeResult:
        yield Footer()
        with VerticalScroll(id="content"):
            yield Static(
                " ____   ____    _   ____   _____\n"
                "|  _ \\ |  _ \\  (_) / ___| |_   _|\n"
                "| | | || |_) | | | \\___ \\   | |\n"
                "| |_| ||  _ <  | |  ___) |  | |\n"
                "|____/ |_| \\_\\ |_| |____/   |_|\n"
                "Diabetes Risk Prediction Tool",
                id="wordmark",
                markup=False,
            )
            with Container(id="selection"):
                yield Static(
                    "Select [T]ransformer, [L]ogistic_Regression, [X]GBoost",
                    id="model-prompt",
                    markup=False,
                )
                yield Input(placeholder="Type T, L, or X and press Enter", id="model-command")
                yield Static("", id="selection-error")
                with Container(id="selection-actions"):
                    yield Button("Continue", variant="primary", id="continue")
            with VerticalScroll(id="details"):
                with Container(id="model-row"):
                    yield Static("", id="selected-model")
                    yield Button("Change model", id="change-model")
                with Grid(id="fields"):
                    for name, label in FIELD_DEFINITIONS:
                        with Container(classes="field"):
                            yield Label(label)
                            yield self._make_field(name)
                yield Button("Predict", variant="primary", id="predict")
                yield Static("Choose a model to begin.", id="status")

    @staticmethod
    def _make_field(name: str) -> Input | Select:
        if name in NUMERIC_INPUTS:
            defaults = {"BMI": "28", "MentHlth": "0", "PhysHlth": "0"}
            return Input(value=defaults[name], type="number", id=name)

        options = {
            "HighBP": BINARY_OPTIONS,
            "HighChol": BINARY_OPTIONS,
            "Smoker": BINARY_OPTIONS,
            "Stroke": BINARY_OPTIONS,
            "HeartDiseaseorAttack": BINARY_OPTIONS,
            "PhysActivity": BINARY_OPTIONS,
            "HvyAlcoholConsump": BINARY_OPTIONS,
            "DiffWalk": BINARY_OPTIONS,
            "GenHlth": HEALTH_OPTIONS,
            "Age": AGE_OPTIONS,
            "Education": EDUCATION_OPTIONS,
            "Income": INCOME_OPTIONS,
        }[name]
        return Select(options, value=options[0][1], allow_blank=False, id=name)

    def on_mount(self) -> None:
        self.query_one("#model-command", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "model-command":
            self._choose_model(event.value)

    def _choose_model(self, command: str) -> None:
        model_name = MODEL_COMMANDS.get(command.strip().lower())
        if model_name is None:
            self.query_one("#selection-error", Static).update("Enter T, L, or X to choose a model.")
            return
        self._selected_model_name = model_name
        self.query_one("#selection-error", Static).update("")
        self.query_one("#selected-model", Static).update(f"Selected model: {model_name}")
        self.query_one("#selection").display = False
        self.query_one("#details").display = True
        self._start_model_load(model_name)

    def _start_model_load(self, model_name: str) -> None:
        self._training_generation += 1
        self._model = None
        self.query_one("#status", Static).update(f"Loading saved {model_name} model...")
        self._load_model_in_background(model_name, self._training_generation)

    @work(thread=True, group="model-training", exclusive=True)
    def _load_model_in_background(self, model_name: str, generation: int) -> None:
        try:
            fitted = load_model(model_name)
            error_message = None
        except Exception as error:
            fitted = None
            error_message = f"{type(error).__name__}: {error}"
        self.call_from_thread(
            self._training_finished,
            generation,
            model_name,
            fitted,
            error_message,
        )

    def _training_finished(
        self,
        generation: int,
        model_name: str,
        fitted: tuple[ColumnTransformer, Any] | None,
        error_message: str | None,
    ) -> None:
        if generation != self._training_generation:
            return
        if error_message:
            self._model = None
            self.query_one("#status", Static).update(f"Model loading failed: {error_message}")
            return
        self._model = fitted
        self.query_one("#status", Static).update(f"{model_name} is ready.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "continue":
            self._choose_model(self.query_one("#model-command", Input).value)
            return
        if event.button.id == "change-model":
            self._training_generation += 1
            self._model = None
            self.query_one("#details").display = False
            self.query_one("#selection").display = True
            self.query_one("#model-command", Input).focus()
            return
        if event.button.id != "predict":
            return
        if self._model is None:
            self.query_one("#status", Static).update("Wait for the model to finish loading before predicting.")
            return

        try:
            values = self._read_form()
            prediction, probability = predict_diabetes(self._model, values)
        except ValueError as error:
            self.query_one("#status", Static).update(f"Check the inputs: {error}")
            return

        class_label = "higher-risk class" if prediction == 1 else "lower-risk class"
        self.query_one("#status", Static).update(
            f"Model output: {prediction} ({class_label}) | Class 1 probability: {probability:.1%}"
        )

    def _read_form(self) -> dict[str, int | float]:
        values: dict[str, int | float] = {}
        for name, _ in FIELD_DEFINITIONS:
            if name in NUMERIC_INPUTS:
                raw_value = self.query_one(f"#{name}", Input).value.strip()
                if not raw_value:
                    raise ValueError(f"{name} is required")
                value = float(raw_value)
                limits = {"BMI": (12, 98), "MentHlth": (0, 30), "PhysHlth": (0, 30)}
                minimum, maximum = limits[name]
                if not minimum <= value <= maximum:
                    raise ValueError(f"{name} must be between {minimum} and {maximum}")
                values[name] = value
            else:
                selected = self.query_one(f"#{name}", Select).value
                if not isinstance(selected, (int, float)):
                    raise ValueError(f"{name} is required")
                values[name] = int(selected)
        return values


if __name__ == "__main__":
    DiabetesApp().run()