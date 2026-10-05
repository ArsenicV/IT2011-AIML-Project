         ____   ____    _   ____   _____
        |  _ \\ |  _ \\  (_) / ___| |_   _|
        | | | || |_) | | | \\___ \\   | |
        | |_| ||  _ <  | |  ___) |  | |
        |____/ |_| \\_\\ |_| |____/   |_|
        Diabetes Risk Prediction Tool




<br>

## Project overview
DRiST is a diabetes risk screening tool built on the idea that early warning shouldn't require a scheduled checkup. Using common, self-reportable health indicators - blood pressure, BMI, activity levels, general health - it predicts whether someone is likely diabetic and estimates their probability of risk. It's tuned to flag generously rather than miss someone who's actually at risk: a screening aid meant to prompt a conversation with a doctor, not replace one. 

<br>

## Repo structure 

```
|- cli ----> contains the command line version of the application
|
|- data
|   |- raw ----> raw dataset 
|
|- notebooks ----> data preprocessing pipeline
|
|- results
|   |- eda visualizations ----> plots generated from data preprocessing pipeline
|   |- outputs ----> preprocessed dataset
|
|- scripts ----> scripts needed to install in order to get STT/TTS functionality
|
|- weights ----> trained models by group members
|
|- plots ----> model evaluation plots


```

## Project Pipeline
![project pipeline](Drist-Pipeline.png)

----
<br>

## How to run 

<br>

### Run Locally

Clone the project repo first. then run,
```sh
cd IT2011-AIML-Project

```
Then activate the virtual environment using:

```sh
python3 -m venv .venv
source .venv/bin/activate

```

Then run

```sh
python -m pip install -r requirements.txt

```

Run the Textual app by using:

```sh
python app.py
```

> Note: If you're uisng MacOS, make sure to install the OpenMP runtime once if you plan to use XGBoost:
```sh
brew install libomp
```

<br>

### Host in a browser

```sh
python serve.py --port 8080
```

<br>

### Run the fully CLI version

```sh
python cli/app.py
```
Open localhost:8080 from your browser, and keep terminal running.

<br>

### Install offline voice input

To enable STT/TTS features, run these commands in order *from the project root*. The voice setup script installs the optional audio/speech packages and the local model; `requirements-voice.txt` also includes `requirements.txt`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-voice.txt
bash scripts/setup-voice.sh
```
<br>

### How to use the voice feature with cli

The setup script installs the optional microphone package , text-to-speech package, installs whisper.cpp with Homebrew on macOS if needed, and downloads the Whisper `base.en` model. Once setup completes, start the CLI with `python cli/app.py`.

```
Text commands related to voice feature:
`voice` - for a single voice entry
`auto` - to enable continuous hands-free voice mode

Voice commands related to voice feature:
`manual` & `stop` - return to typing (use this before prediction)
```

To change voice, change microphone, and other features, see [Additional information]

<br>

## Assigned Dataset


## Model selection (justification w charts)


## Plots (performance matrices & eval)


## Team members & roles


## Additional information

Additional information related to voice:
- To use another microphone, list devices with `python -c "import sounddevice as sd; print(sd.query_devices())"`, then set its device index or name before launching:

```sh
export DRIST_AUDIO_DEVICE=2
python cli/app.py
```

- To use another whisper.cpp model, set `DRIST_WHISPER_MODEL` to its local model file; to use a differently named executable, set `DRIST_WHISPER_CLI`. Without the optional executable or model, normal typed input continues to work.

- To change the voice (Text To Speech)
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

**Note!**
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


Used packages for this project:
sounddevice, pyttsx3, whisper.cpp, base.en







































