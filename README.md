# DRiST TUI

## Install the base app

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS, install the OpenMP runtime once if you plan to use XGBoost:

```sh
brew install libomp
```

## Run the Textual app

```sh
python app.py
```

## Run the terminal CLI

```sh
python cli/app.py
```

The CLI displays the purple ASCII DRiST title, prompts for `T`, `L`, or `X`, loads that local model artifact, then guides you through the patient features. It validates entries and shows the prediction in a colored summary.

At the model prompt, enter `snake` for the hidden Snake game. Move with WASD or the arrow keys; press `Q` to return to model selection.

## Install offline voice input

From the project root, run these commands in order. The voice setup script installs the optional audio/speech packages and the local model; `requirements-voice.txt` also includes `requirements.txt`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/setup-voice.sh
```

The setup script installs the optional microphone package (`sounddevice`), text-to-speech package (`pyttsx3`), installs whisper.cpp with Homebrew on macOS if needed, and downloads the Whisper `base.en` model (about 142 MB). If a download is interrupted, rerun the script to resume it. It uses `.venv` if it exists; otherwise it uses the available system Python. Once setup completes, start the CLI with `python cli/app.py`.

At any feature prompt, type `voice` and press Enter for a single voice entry, or type `auto` to enable continuous hands-free voice mode. The CLI records from the operating system's default microphone, stops after about 0.45 seconds of silence (or 5 seconds maximum), then transcribes locally with whisper.cpp. It accepts the transcript without per-field confirmation, while numeric ranges and categorical options are still validated. If you are in auto mode and want to return to typing, just say "manual" or "stop". Before prediction, review the full input table: press Enter to continue, `E` to edit a feature, or `Q` to cancel. The temporary WAV is deleted after transcription. On macOS, allow microphone access for the terminal app running Python.

To use another microphone, list devices with `python -c "import sounddevice as sd; print(sd.query_devices())"`, then set its device index or name before launching:

```sh
export DRIST_AUDIO_DEVICE=2
python cli/app.py
```

Speech-to-text does not change or synthesize the speaker's voice; it recognizes whoever speaks into the selected microphone. The default model is Whisper `base.en`. To use another whisper.cpp model, set `DRIST_WHISPER_MODEL` to its local model file; to use a differently named executable, set `DRIST_WHISPER_CLI`. Without the optional executable or model, normal typed input continues to work. Downloaded model binaries are ignored by Git.

## Text-to-speech (TTS)

The CLI reads back transcribed voice input and speaks the final prediction result aloud using `pyttsx3`, which uses the operating system's native speech engine (macOS `say`) — fully offline, no network needed.

List all voices available on your system:

```sh
python cli/app.py --list-voices
```

Pick a voice by setting its name (or any substring of it) before launching:

```sh
export DRIST_TTS_VOICE=Daniel     # British male
export DRIST_TTS_VOICE=Samantha   # US female (default macOS voice)
export DRIST_TTS_VOICE=Rishi      # Indian English male
export DRIST_TTS_VOICE=Zarvox     # robotic novelty voice
python cli/app.py
```

Matching is case-insensitive and substring-based, so `daniel` and `Dan` both select **Daniel**. Other TTS controls:

| Variable | Default | Effect |
|---|---|---|
| `DRIST_TTS` | `1` | Set to `0` / `false` / `off` to disable TTS entirely |
| `DRIST_TTS_VOICE` | _(system default)_ | Voice name or substring, e.g. `Samantha`, `Zarvox` |
| `DRIST_TTS_RATE` | `175` | Speaking speed in words per minute |

Example — slow Zarvox robot voice:

```sh
DRIST_TTS_VOICE=Zarvox DRIST_TTS_RATE=130 python cli/app.py
```


## Host in a browser

```sh
python serve.py --port 8080
```

Open `http://localhost:8080` and leave the server terminal running. Stop it with Ctrl+C. Use `python serve.py --port 8000` to host on port 8000 instead. The app loads `ft_transformer_best.pt`, `logreg (1).pt`, or `xgboost_diabetes_model.joblib` from the repository root. It fits the matching notebook scaler/encoder on the training split from `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`, then applies it to the entered patient row. The Logistic Regression artifact expects 19 features (first category dropped); Transformer and XGBoost expect 28. XGBoost loading and inference run in a helper process to avoid a macOS OpenMP conflict with PyTorch.

The Transformer checkpoint stores weights without model configuration. Its tensor shapes identify a six-layer, width-64 transformer with a 256-wide feed-forward layer and a 32-unit head. The attention head count is not stored; this app assumes four heads. The Logistic Regression artifact was saved under scikit-learn 1.6.1; the app reports the version mismatch if run with a different scikit-learn version.

This is a project scaffold, not a validated clinical tool. The models are loaded from local artifacts supplied with the project.

# 

<br>

#### Project Pipeline
![project pipeline](Drist-Pipeline.png)

----

#### Assigned Dataset
