import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Climate Early Warning", layout="wide")

st.title("Climate Risk Early Warning System")
st.write("Hybrid LSTM + Random Forest deployment using saved model files.")

@st.cache_resource
def load_artifacts():
    best_rf = joblib.load("best_rf.pkl")
    scaler_lstm = joblib.load("scaler_lstm.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    series_features = joblib.load("series_features.pkl")
    lookback = joblib.load("lookback.pkl")
    rain_high_threshold = joblib.load("rain_high_threshold.pkl")
    lstm_model = load_model("lstm_model.h5", compile=False)

    return {
        "best_rf": best_rf,
        "scaler_lstm": scaler_lstm,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
        "series_features": series_features,
        "lookback": lookback,
        "rain_high_threshold": rain_high_threshold,
        "lstm_model": lstm_model,
    }

artifacts = load_artifacts()

best_rf = artifacts["best_rf"]
scaler_lstm = artifacts["scaler_lstm"]
label_encoder = artifacts["label_encoder"]

feature_cols = [str(c) for c in list(artifacts["feature_cols"])]
series_features = [str(c) for c in list(artifacts["series_features"])]

lookback = int(artifacts["lookback"])
rain_high_threshold = float(artifacts["rain_high_threshold"])
lstm_model = artifacts["lstm_model"]

st.subheader("Upload prepared feature data")
uploaded_file = st.file_uploader("Upload a CSV with engineered features", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    st.write("Preview")
    st.dataframe(data.head())

    missing_series = [c for c in series_features if c not in data.columns]
    missing_rf = [c for c in feature_cols if c not in data.columns and c != "pred_rainfall_next"]

    if missing_series:
        st.error(f"Missing LSTM input columns: {missing_series}")
    elif missing_rf:
        st.error(f"Missing RF input columns: {missing_rf}")
    else:
        # LSTM needs the sequence features scaled
        series_df = data.loc[:, series_features].copy()
        series_scaled = scaler_lstm.transform(series_df)

        if len(series_scaled) < lookback:
            st.error(f"You need at least {lookback} rows for sequence prediction.")
        else:
            X_seq = []
            for i in range(lookback, len(series_scaled) + 1):
                X_seq.append(series_scaled[i - lookback:i, :])

            X_seq = np.array(X_seq)

            y_pred_lstm = lstm_model.predict(X_seq, verbose=0)

            target_col_index = series_features.index("rainfall")
            dummy_pred = np.zeros((len(y_pred_lstm), len(series_features)))
            dummy_pred[:, target_col_index] = y_pred_lstm.flatten()
            pred_rainfall_inv = scaler_lstm.inverse_transform(dummy_pred)[:, target_col_index]

            # align LSTM predictions back to rows
            data["pred_rainfall_next"] = np.nan
            data.loc[lookback - 1:, "pred_rainfall_next"] = pred_rainfall_inv

            pred_rows = data.dropna(subset=["pred_rainfall_next"]).copy()

            X_rf = pred_rows[feature_cols].copy()
            y_pred_rf = best_rf.predict(X_rf)

            if pd.api.types.is_numeric_dtype(y_pred_rf):
                y_pred_labels = label_encoder.inverse_transform(y_pred_rf.astype(int))
            else:
                y_pred_labels = y_pred_rf

            pred_rows["predicted_future_risk"] = y_pred_labels

            def warning_flag(row):
                if row["pred_rainfall_next"] >= rain_high_threshold:
                    return "Flood Watch"
                elif row["predicted_future_risk"] == "drought":
                    return "Drought Watch"
                else:
                    return "Normal Watch"

            pred_rows["warning_flag"] = pred_rows.apply(warning_flag, axis=1)

            st.subheader("Predictions")
            display_cols = ["pred_rainfall_next", "predicted_future_risk", "warning_flag"]
            if "date" in pred_rows.columns:
                display_cols = ["date"] + display_cols

            st.dataframe(pred_rows[display_cols].tail(20))

            st.subheader("Latest warning")
            latest = pred_rows.iloc[-1]
            if "date" in pred_rows.columns:
                st.write(f"Date: {latest['date']}")
            st.write(f"Predicted rainfall next step: {latest['pred_rainfall_next']:.2f}")
            st.write(f"Predicted future risk: {latest['predicted_future_risk']}")
            st.write(f"Warning: {latest['warning_flag']}")