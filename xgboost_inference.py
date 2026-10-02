from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "weights" / "xgboost_diabetes_model.joblib"


def main() -> None:
    request: dict[str, Any] = json.load(sys.stdin)
    model = joblib.load(MODEL_PATH)
    feature_names = list(model.feature_names_in_)

    if request["action"] == "inspect":
        response = {"feature_names": feature_names, "feature_count": model.n_features_in_}
    elif request["action"] == "predict":
        values = request["features"]
        frame = pd.DataFrame([values], columns=request["feature_names"])
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0][list(model.classes_).index(1)])
        response = {"prediction": prediction, "probability": probability}
    else:
        raise ValueError(f"Unknown XGBoost action: {request['action']}")

    print(json.dumps(response))


if __name__ == "__main__":
    main()
