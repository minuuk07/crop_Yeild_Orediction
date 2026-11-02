import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("Loading dataset...")

df = pd.read_csv("yield_df.csv")

# Remove unnecessary index column if present
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print("Dataset shape:", df.shape)


# =========================================================
# 2. DEFINE FEATURES AND TARGET
# =========================================================

features = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "Area",
    "Item"
]

target = "hg/ha_yield"

X = df[features].copy()
y = df[target].copy()


# =========================================================
# 3. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# =========================================================
# 4. PREPROCESSING
# =========================================================

numeric_features = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp"
]

categorical_features = [
    "Area",
    "Item"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        )
    ]
)


# IMPORTANT:
# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Only transform test data
X_test_processed = preprocessor.transform(X_test)


# =========================================================
# 5. RANDOM FOREST
# =========================================================

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)


# =========================================================
# 6. RANDOM SEARCH HYPERPARAMETER TUNING
# =========================================================

param_grid = {

    "n_estimators": [
        100,
        150,
        200,
        250
    ],

    "max_depth": [
        None,
        10,
        15,
        20,
        25
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "max_features": [
        "sqrt",
        "log2",
        1.0
    ]
}


print("\nStarting Random Search...")
print("Please wait... This may take some time.")


random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=1
)


random_search.fit(
    X_train_processed,
    y_train
)


# =========================================================
# 7. BEST MODEL
# =========================================================

best_model = random_search.best_estimator_

print("\n================================")
print("BEST PARAMETERS")
print("================================")

print(random_search.best_params_)


# =========================================================
# 8. TEST PREDICTION
# =========================================================

y_pred = best_model.predict(X_test_processed)


# =========================================================
# 9. EVALUATION
# =========================================================

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)


print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print("R2 Score :", r2)
print("MAE      :", mae)
print("MSE      :", mse)
print("RMSE     :", rmse)


# =========================================================
# 10. FEATURE IMPORTANCE
# =========================================================

feature_names = preprocessor.get_feature_names_out()

importances = best_model.feature_importances_


# We aggregate one-hot encoded features
# back to their original feature groups.

importance_dict = {
    "Year": 0,
    "Rainfall": 0,
    "Pesticides": 0,
    "Temperature": 0,
    "Area": 0,
    "Crop": 0
}


for name, importance in zip(
    feature_names,
    importances
):

    if "Year" in name:
        importance_dict["Year"] += importance

    elif "average_rain_fall" in name:
        importance_dict["Rainfall"] += importance

    elif "pesticides" in name:
        importance_dict["Pesticides"] += importance

    elif "avg_temp" in name:
        importance_dict["Temperature"] += importance

    elif "Area" in name:
        importance_dict["Area"] += importance

    elif "Item" in name:
        importance_dict["Crop"] += importance


# Convert to percentage
total_importance = sum(importance_dict.values())

for key in importance_dict:
    importance_dict[key] = round(
        (importance_dict[key] / total_importance) * 100,
        2
    )


print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

for key, value in importance_dict.items():
    print(key, ":", value, "%")


# =========================================================
# 11. MODEL INFORMATION
# =========================================================

model_info = {

    "r2": float(r2),

    "mae": float(mae),

    "mse": float(mse),

    "rmse": float(rmse),

    "best_params": random_search.best_params_,

    "importance": importance_dict,

    "low_threshold": float(y.quantile(0.33)),

    "high_threshold": float(y.quantile(0.66))
}


# =========================================================
# 12. SAVE MODELS
# =========================================================

pickle.dump(
    best_model,
    open("rf_model.pkl", "wb")
)

pickle.dump(
    preprocessor,
    open("preprocessor.pkl", "wb")
)

pickle.dump(
    importance_dict,
    open("feature_importance.pkl", "wb")
)

pickle.dump(
    model_info,
    open("model_info.pkl", "wb")
)


print("\n================================")
print("TRAINING COMPLETED")
print("================================")

print("Created files:")
print("rf_model.pkl")
print("preprocessor.pkl")
print("feature_importance.pkl")
print("model_info.pkl") 