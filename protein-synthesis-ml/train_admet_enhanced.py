"""
train_admet_enhanced.py

Enhanced ADMET model training with model selection, cross-validation, and calibration.
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
import xgboost as xgb

from admet_loader import load_admet_dataset, ADMET_TASKS
from featurization import smiles_to_matrix


def train_and_select_best_model(
    X_train, y_train, task_type: str, cv_folds: int = 5
) -> tuple:
    """
    Train multiple models and select the best based on cross-validation.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    task_type : str
        "classification" or "regression"
    cv_folds : int
        Number of cross-validation folds
        
    Returns
    -------
    tuple
        (best_model, best_score, model_name)
    """
    models = {}
    
    if task_type == "classification":
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            eval_metric="logloss"
        )
        
        # Cross-validation with ROC-AUC
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = "roc_auc"
    else:  # regression
        models["RandomForest"] = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
        )
        
        # Cross-validation with R2
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = "r2"
    
    # Evaluate each model
    best_model = None
    best_score = -np.inf
    best_name = None
    
    print(f"\nEvaluating {len(models)} models with {cv_folds}-fold cross-validation...")
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            mean_score = scores.mean()
            std_score = scores.std()
            print(f"  {name}: {mean_score:.4f} (+/- {std_score:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name
        except Exception as e:
            print(f"  {name}: Failed ({e})")
            continue
    
    if best_model is None:
        raise ValueError("No model could be trained successfully")
    
    print(f"\n✅ Best model: {best_name} (CV score: {best_score:.4f})")
    
    # Train best model on full training set
    best_model.fit(X_train, y_train)
    
    # Calibrate if classification
    if task_type == "classification":
        print("Calibrating probabilities...")
        calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)
        return calibrated, best_score, best_name
    else:
        return best_model, best_score, best_name


def train_admet_model_enhanced(task_key: str, save_model: bool = True, cv_folds: int = 5):
    """
    Train an ADMET model with enhanced features (model selection, calibration).
    
    Parameters
    ----------
    task_key : str
        One of the keys in ADMET_TASKS
    save_model : bool
        Whether to save the trained model to disk
    cv_folds : int
        Number of cross-validation folds for model selection
    """
    if task_key not in ADMET_TASKS:
        raise ValueError(
            f"Unknown task: {task_key}. Available tasks: {list(ADMET_TASKS.keys())}"
        )
    
    task_config = ADMET_TASKS[task_key]
    task_type = task_config["type"]
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading {task_key} dataset...")
    print(f"{'='*60}")
    
    df, detected_type = load_admet_dataset(task_key)
    
    if detected_type != task_type:
        print(f"Warning: Task type mismatch. Expected {task_type}, got {detected_type}")
    
    print(f"\nDataset loaded: {len(df)} compounds")
    
    # Handle label column name
    if task_type == "classification":
        label_col = "label"
        if label_col not in df.columns:
            raise ValueError(f"Classification task {task_key} missing 'label' column")
        print(f"Label distribution:")
        print(df[label_col].value_counts())
    else:  # regression
        label_col = "value"
        if label_col not in df.columns:
            raise ValueError(f"Regression task {task_key} missing 'value' column")
        print(f"Value statistics:")
        print(df[label_col].describe())
    
    # Featurize SMILES
    print(f"\n{'='*60}")
    print("Converting SMILES to Morgan fingerprints...")
    print(f"{'='*60}")
    
    X, idx = smiles_to_matrix(df["smiles"], radius=2, n_bits=2048)
    y = df[label_col].values[idx]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Valid SMILES: {len(idx)} / {len(df)}")
    
    # Train/test split
    if task_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:  # regression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Train and select best model
    print(f"\n{'='*60}")
    print("Training and selecting best model...")
    print(f"{'='*60}")
    
    model, cv_score, model_name = train_and_select_best_model(
        X_train, y_train, task_type, cv_folds=cv_folds
    )
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")
    
    if task_type == "classification":
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        test_auc = roc_auc_score(y_test, y_proba)
        print(f"Test ROC-AUC: {test_auc:.4f}")
        print(f"Test Accuracy: {(y_pred == y_test).mean():.4f}")
    else:  # regression
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
    
    # Save model
    if save_model:
        os.makedirs("models/admet", exist_ok=True)
        
        if task_type == "classification":
            model_path = os.path.join("models", "admet", f"{task_key}_clf.pkl")
        else:  # regression
            model_path = os.path.join("models", "admet", f"{task_key}_reg.pkl")
        
        joblib.dump(model, model_path)
        print(f"\n{'='*60}")
        print(f"Model saved to: {model_path}")
        print(f"Model type: {model_name}")
        print(f"CV score: {cv_score:.4f}")
        print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print(f"Training complete for {task_key}!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Train enhanced ADMET prediction models with model selection and calibration"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(ADMET_TASKS.keys()),
        help=f"ADMET task to train on. Available: {', '.join(ADMET_TASKS.keys())}",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model to disk",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    
    args = parser.parse_args()
    train_admet_model_enhanced(args.task, save_model=not args.no_save, cv_folds=args.cv_folds)


if __name__ == "__main__":
    main()

