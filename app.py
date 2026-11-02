from flask import Flask, request, render_template

import numpy as np
import pandas as pd
import pickle


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("yield_df.csv")

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)


# =========================================================
# LOAD ML FILES
# =========================================================

model = pickle.load(
    open("rf_model.pkl", "rb")
)

preprocessor = pickle.load(
    open("preprocessor.pkl", "rb")
)

feature_importance = pickle.load(
    open("feature_importance.pkl", "rb")
)

model_info = pickle.load(
    open("model_info.pkl", "rb")
)


# =========================================================
# DATA FOR DROPDOWN
# =========================================================

areas = sorted(
    df["Area"].dropna().unique().tolist()
)

crops = sorted(
    df["Item"].dropna().unique().tolist()
)


# =========================================================
# HELPER FUNCTION
# =========================================================

def make_prediction(
    year,
    rainfall,
    pesticides,
    temperature,
    area,
    crop
):

    input_data = pd.DataFrame(
        [[
            year,
            rainfall,
            pesticides,
            temperature,
            area,
            crop
        ]],
        columns=[
            "Year",
            "average_rain_fall_mm_per_year",
            "pesticides_tonnes",
            "avg_temp",
            "Area",
            "Item"
        ]
    )

    transformed = preprocessor.transform(
        input_data
    )

    prediction = model.predict(
        transformed
    )[0]

    return float(prediction)


# =========================================================
# YIELD LEVEL
# =========================================================

def get_yield_level(prediction):

    low = model_info["low_threshold"]

    high = model_info["high_threshold"]

    if prediction < low:

        return "Low", "yield-low"

    elif prediction < high:

        return "Medium", "yield-medium"

    else:

        return "High", "yield-high"


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        areas=areas,
        crops=crops
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        year = int(
            request.form["Year"]
        )

        rainfall = float(
            request.form[
                "average_rain_fall_mm_per_year"
            ]
        )

        pesticides = float(
            request.form[
                "pesticides_tonnes"
            ]
        )

        temperature = float(
            request.form["avg_temp"]
        )

        area = request.form["Area"]

        crop = request.form["Item"]


        # -----------------------------------------
        # Main Prediction
        # -----------------------------------------

        prediction = make_prediction(
            year,
            rainfall,
            pesticides,
            temperature,
            area,
            crop
        )


        # -----------------------------------------
        # Yield Level
        # -----------------------------------------

        yield_level, yield_class = get_yield_level(
            prediction
        )


        # =================================================
        # FEATURE 2 — TOP 3 CROP RECOMMENDATION
        # =================================================

        crop_predictions = []

        for selected_crop in crops:

            crop_yield = make_prediction(
                year,
                rainfall,
                pesticides,
                temperature,
                area,
                selected_crop
            )

            crop_predictions.append(
                {
                    "crop": selected_crop,
                    "yield": round(crop_yield, 2)
                }
            )


        crop_predictions = sorted(
            crop_predictions,
            key=lambda x: x["yield"],
            reverse=True
        )


        top_crops = crop_predictions[:3]


        # =================================================
        # FEATURE 3 — CROP BENCHMARK
        # =================================================

        crop_data = df[
            df["Item"] == crop
        ]["hg/ha_yield"]

        crop_average = (
            float(crop_data.mean())
            if len(crop_data) > 0
            else 0
        )


        if crop_average != 0:

            benchmark_difference = (
                (prediction - crop_average)
                / crop_average
            ) * 100

        else:

            benchmark_difference = 0


        benchmark_difference = round(
            benchmark_difference,
            2
        )


        # =================================================
        # FEATURE 4 — WHAT IF ANALYSIS
        # =================================================

        rainfall_plus = make_prediction(
            year,
            rainfall * 1.10,
            pesticides,
            temperature,
            area,
            crop
        )

        rainfall_minus = make_prediction(
            year,
            rainfall * 0.90,
            pesticides,
            temperature,
            area,
            crop
        )

        temperature_plus = make_prediction(
            year,
            rainfall,
            pesticides,
            temperature + 2,
            area,
            crop
        )


        scenarios = {

            "base": round(
                prediction,
                2
            ),

            "rainfall_plus_10": round(
                rainfall_plus,
                2
            ),

            "rainfall_minus_10": round(
                rainfall_minus,
                2
            ),

            "temperature_plus_2": round(
                temperature_plus,
                2
            )
        }


        # =================================================
        # FEATURE 5 — INSIGHT
        # =================================================

        sorted_importance = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_feature = sorted_importance[0]


        insight = (
            f"{top_feature[0]} is currently the "
            f"most influential feature in the "
            f"trained Random Forest model, with "
            f"approximately {top_feature[1]}% "
            f"of aggregated feature importance."
        )


        return render_template(

            "index.html",

            areas=areas,

            crops=crops,

            prediction=round(
                prediction,
                2
            ),

            yield_level=yield_level,

            yield_class=yield_class,

            selected_area=area,

            selected_crop=crop,

            selected_year=year,

            selected_rainfall=rainfall,

            selected_pesticides=pesticides,

            selected_temperature=temperature,

            top_crops=top_crops,

            feature_importance=sorted_importance,

            scenarios=scenarios,

            crop_average=round(
                crop_average,
                2
            ),

            benchmark_difference=benchmark_difference,

            insight=insight,

            model_r2=round(
                model_info["r2"],
                4
            ),

            model_mae=round(
                model_info["mae"],
                4
            )

        )


    except Exception as e:

        return render_template(

            "index.html",

            areas=areas,

            crops=crops,

            error=str(e)

        )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )