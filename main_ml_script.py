import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("XGBoost not installed. Install with: pip install xgboost to include it.")


# =========================
# 1. LOAD & CLEAN DATA
# =========================

def load_data(path: str = "megaGymDataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop useless index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Basic cleaning: strip spaces, standardize category text
    for col in ["Title", "Type", "BodyPart", "Equipment", "Level"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Keep only rows where Level is known (needed for classification)
    df = df[~df["Level"].isna()]

    df.reset_index(drop=True, inplace=True)
    return df


# =========================
# 2. CLASSIFICATION – MULTI-MODEL COMPARISON
# =========================
def train_level_models(df: pd.DataFrame):
    """
    Train multiple classifiers to predict Level from Type, BodyPart, Equipment.
    Compare accuracies, save plots/metrics.

    Returns (for UI):
        pipelines (dict): name -> fitted Pipeline
        accuracies (dict): name -> accuracy
        best_model_name (str)
        label_enc (LabelEncoder)
    """

    feature_cols = ["Type", "BodyPart", "Equipment"]
    target_col = "Level"

    X = df[feature_cols]
    y = df[target_col]

    # Encode labels (Beginner/Intermediate/Expert -> 0/1/2)
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Preprocess categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols)
        ]
    )

    models = {}

    # 1. Random Forest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # 2. SVM (RBF kernel)
    models["SVM"] = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=False,
        random_state=42
    )

    # 3. ANN / MLP (still used for comparison, but NO curves now)
    models["ANN_MLP"] = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=42
    )

    # 4. XGBoost (optional, main model + curves)
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,               # main model for comparison
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            random_state=42,
            eval_metric="mlogloss"
        )

    accuracies = {}
    pipelines = {}
    y_test_dict = {}
    y_pred_dict = {}

    # collect metrics per model
    metrics_rows = []

    for name, model in models.items():
        clf_pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        clf_pipeline.fit(X_train, y_train)
        y_pred = clf_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # store for later use
        accuracies[name] = acc
        pipelines[name] = clf_pipeline
        y_test_dict[name] = y_test
        y_pred_dict[name] = y_pred

        # metrics
        prec_macro = precision_score(y_test, y_pred, average="macro")
        recall_macro = recall_score(y_test, y_pred, average="macro")
        f1_macro = f1_score(y_test, y_pred, average="macro")

        prec_weighted = precision_score(y_test, y_pred, average="weighted")
        recall_weighted = recall_score(y_test, y_pred, average="weighted")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        metrics_rows.append({
            "Model": name,
            "Accuracy": acc,
            "Precision_macro": prec_macro,
            "Recall_macro": recall_macro,
            "F1_macro": f1_macro,
            "Precision_weighted": prec_weighted,
            "Recall_weighted": recall_weighted,
            "F1_weighted": f1_weighted,
        })

        # Optional console logs
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=label_enc.classes_))

    # Save metrics table
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("classification_metrics_summary.csv", index=False)

    # Accuracy vs model bar plot
    plt.figure()
    model_names = list(accuracies.keys())
    acc_values = [accuracies[m] for m in model_names]

    plt.bar(model_names, acc_values)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.title("Model Accuracy Comparison (Level Prediction)")
    plt.ylim(0, 1)
    for i, v in enumerate(acc_values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig("classification_accuracy_comparison.png")
    plt.close()

    # ----------------------------------------
    # Best model: FORCE XGBoost if available
    # ----------------------------------------
    if "XGBoost" in accuracies:
        best_model_name = "XGBoost"
    else:
        best_model_name = max(accuracies, key=accuracies.get)

    # Confusion Matrix Heatmap for Best Model
    best_y_test = y_test_dict[best_model_name]
    best_y_pred = y_pred_dict[best_model_name]

    cm = confusion_matrix(best_y_test, best_y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_enc.classes_,
        yticklabels=label_enc.classes_
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix Heatmap - {best_model_name}")
    plt.tight_layout()

    if best_model_name == "XGBoost":
        plt.savefig("confusion_matrix_xgboost_heatmap.png")
    else:
        plt.savefig(f"confusion_matrix_{best_model_name}_heatmap.png")

    plt.close()

    # =====================================================
    # NEW: XGBoost Training Curves (up to 100 epochs)
    # =====================================================
    if XGBClassifier is not None:
        print("\nTraining separate XGBoost (100 trees) for loss/accuracy curves...")

        # Separate preprocessor ONLY for this curve model
        xgb_curve_preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols)
            ]
        )

        # One-hot encode train / validation
        X_train_enc = xgb_curve_preprocessor.fit_transform(X_train)
        X_val_enc = xgb_curve_preprocessor.transform(X_test)

        # XGBoost model for curves (100 estimators = 100 epochs)
        xgb_curve = XGBClassifier(
            n_estimators=100,             # <= exactly 100 boosting rounds
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            random_state=42,
            eval_metric=["mlogloss", "merror"],  # loss + classification error
            use_label_encoder=False
        )

        xgb_curve.fit(
            X_train_enc,
            y_train,
            eval_set=[(X_train_enc, y_train), (X_val_enc, y_test)],
            verbose=False
        )

        results = xgb_curve.evals_result()

        # Training loss (mlogloss)
        train_loss = results["validation_0"]["mlogloss"]
        # Validation classification error -> accuracy
        val_error = results["validation_1"]["merror"]
        val_accuracy = [1.0 - e for e in val_error]

        epochs = range(1, len(train_loss) + 1)  # should be 1..100

    # --- Plot 1: Training Loss vs Epoch (XGBoost) ---
        plt.figure()
        plt.plot(epochs, train_loss, marker="o")
        plt.xlabel("Epoch (Boosting Round)")
        plt.ylabel("Training Loss (mlogloss)")
        plt.title("XGBoost Training Loss vs Epoch (100 trees)")
        plt.xticks(np.arange(0, 101, 10))   # spacing 10 units
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("xgb_training_loss_curve.png")
        plt.close()

        # --- Plot 2: Validation Accuracy vs Epoch (XGBoost) ---
        plt.figure()
        plt.plot(epochs, val_accuracy, marker="o")
        plt.xlabel("Epoch (Boosting Round)")
        plt.ylabel("Validation Accuracy")
        plt.title("XGBoost Validation Accuracy vs Epoch (100 trees)")
        plt.xticks(np.arange(0, 101, 10))   # spacing 10 units
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("xgb_validation_accuracy_curve.png")
        plt.close()

        print("Saved: xgb_training_loss_curve.png, xgb_validation_accuracy_curve.png")
       
    # *** IMPORTANT: match UI expectations: return ONLY 4 things ***
    return pipelines, accuracies, best_model_name, label_enc


# =========================
# 3. REGRESSION – RATING PREDICTION
# =========================

def train_rating_regressor(df: pd.DataFrame):
    """
    Predict Rating (0–10) from Type, BodyPart, Equipment, Level.
    Only uses rows where Rating is not NaN.

    Returns:
        reg_pipeline (Pipeline): fitted RandomForestRegressor pipeline
    """

    if "Rating" not in df.columns:
        print("\nNo 'Rating' column found in dataset, skipping regression.")
        return None

    df_rating = df[~df["Rating"].isna()].copy()

    if df_rating.empty:
        print("\nNo non-null Ratings in dataset, skipping regression.")
        return None

    feature_cols = ["Type", "BodyPart", "Equipment", "Level"]
    target_col = "Rating"

    X = df_rating[feature_cols]
    y = df_rating[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols)
        ]
    )

    reg = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    reg_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", reg),
        ]
    )

    reg_pipeline.fit(X_train, y_train)
    y_pred = reg_pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Regression: Rating Prediction ===")
    print("RMSE:", rmse)
    print("R^2:", r2)

    reg_metrics_df = pd.DataFrame([{
        "Model": "RandomForestRegressor",
        "RMSE": rmse,
        "R2": r2
    }])

    reg_metrics_df.to_csv("regression_metrics_summary.csv", index=False)

    return reg_pipeline


# =========================
# 4. RECOMMENDATION FUNCTION
# =========================

def recommend_exercises(
    df: pd.DataFrame,
    reg_model,
    body_part: str,
    equipment: str,
    level: str,
    top_n: int = 5,
):
    """
    Simple recommendation logic:

    1. Filter exercises by BodyPart, Equipment, Level.
    2. If real Rating is available, use it.
    3. If not, predict Rating using regression model.
    4. Sort by (real or predicted) Rating and return top-N.
    """

    # Normalize input
    body_part = body_part.strip()
    equipment = equipment.strip()
    level = level.strip()

    candidates = df.copy()

    if body_part.lower() != "any" and "BodyPart" in candidates.columns:
        candidates = candidates[candidates["BodyPart"].str.lower() == body_part.lower()]

    if equipment.lower() != "any" and "Equipment" in candidates.columns:
        candidates = candidates[candidates["Equipment"].str.lower() == equipment.lower()]

    if level.lower() != "any" and "Level" in candidates.columns:
        candidates = candidates[candidates["Level"].str.lower() == level.lower()]

    if candidates.empty:
        print("\nNo exercises match your filters. Try relaxing conditions.")
        return candidates

    if "Rating" not in candidates.columns:
        print("\nNo 'Rating' column available for recommendation ranking.")
        return candidates

    # For recommendation score: use Rating if present, else predict
    if reg_model is not None:
        feature_cols = ["Type", "BodyPart", "Equipment", "Level"]

        temp = candidates.copy()
        temp["PredRating"] = temp["Rating"]

        # Indices where Rating is NaN -> predict
        mask_nan = temp["Rating"].isna()
        if mask_nan.any():
            X_to_predict = temp.loc[mask_nan, feature_cols]
            pred_values = reg_model.predict(X_to_predict)
            temp.loc[mask_nan, "PredRating"] = pred_values

        # If still some are NaN (just in case), fill with mean
        temp["PredRating"] = temp["PredRating"].fillna(temp["PredRating"].mean())

        # Sort by PredRating
        temp = temp.sort_values(by="PredRating", ascending=False)
    else:
        # Fallback: just sort by existing Rating
        temp = candidates.sort_values(by="Rating", ascending=False)

    print(f"\n=== Top {top_n} Recommended Exercises ===")
    cols_to_show = ["Title", "Type", "BodyPart", "Equipment", "Level", "Rating"]
    if "PredRating" in temp.columns:
        cols_to_show.append("PredRating")

    print(temp[cols_to_show].head(top_n).to_string(index=False))


    return temp.head(top_n)
