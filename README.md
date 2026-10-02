         ____   ____    _   ____   _____
        |  _ \\ |  _ \\  (_) / ___| |_   _|
        | | | || |_) | | | \\___ \\   | |
        | |_| ||  _ <  | |  ___) |  | |
        |____/ |_| \\_\\ |_| |____/   |_|
        Diabetes Risk Prediction Tool

A minimal Textual app for diabetes screening. Consist of FT-Tansformer mode, XGBoost, and Baseline Logistic Regression Model. At the model prompt, enter `T` for the FT-Transformer, `L` for Logistic Regression, or `X` for XGBoost, then enter patient details. 

## How to run?
#### Run Locally

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

#### Run the fully CLI version

```sh
python cli/app.py
```

It displays the purple ASCII DRiST title, then prompts for `T`, `L`, or `X`, loads the corresponding local model artifact, and guides you through each patient feature with colored choices and input validation. The prediction is shown in a colored summary.

At the model prompt, enter `snake` for the hidden Snake game. Move with WASD or the arrow keys; press `Q` to return to model selection.

On macOS, install XGBoost's OpenMP runtime once:

```sh
brew install libomp
```

## Host in a browser

```sh
python serve.py --port 8080
```

Open `http://localhost:8080` and leave the server terminal running. Stop it with Ctrl+C. Use `python serve.py --port 8000` to host on port 8000 instead. The app loads `ft_transformer_best.pt`, `logreg (1).pt`, or `xgboost_diabetes_model.joblib` from the repository root. It fits the matching notebook scaler/encoder on the training split from `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`, then applies it to the entered patient row. The Logistic Regression artifact expects 19 features (first category dropped); Transformer and XGBoost expect 28. XGBoost loading and inference run in a helper process to avoid a macOS OpenMP conflict with PyTorch.

The Transformer checkpoint stores weights without model configuration. Its tensor shapes identify a six-layer, width-64 transformer with a 256-wide feed-forward layer and a 32-unit head. The attention head count is not stored; this app assumes four heads. The Logistic Regression artifact was saved under scikit-learn 1.6.1; the app reports the version mismatch if run with a different scikit-learn version.

This is a project scaffold, not a validated clinical tool. The models are loaded from local artifacts supplied with the project.

# DRiST - Diabetics Risk Screening Tool

<br>

#### Project overview


#### Repo structure 


#### How to run 


#### Project Pipeline
![project pipeline](Drist-Pipeline.png)

----

#### Assigned Dataset


#### Model selection (justification w charts)


#### plots (performance matrices & eval)


#### Team members & roles


