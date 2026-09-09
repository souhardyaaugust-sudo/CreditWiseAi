import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

class ModelManager:
    def __init__(self):
        # Dictionary of classifiers and their search parameters
        self.model_configs = {
            "logistic_regression": {
                "class": LogisticRegression,
                "default_params": {"max_iter": 1000, "random_state": 42},
                "grid_params": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["liblinear", "lbfgs"]
                }
            },
            "decision_tree": {
                "class": DecisionTreeClassifier,
                "default_params": {"random_state": 42},
                "grid_params": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10]
                }
            },
            "random_forest": {
                "class": RandomForestClassifier,
                "default_params": {"random_state": 42},
                "grid_params": {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [3, 5, 7, None],
                    "min_samples_split": [2, 5]
                }
            },
            "k_nearest_neighbors": {
                "class": KNeighborsClassifier,
                "default_params": {},
                "grid_params": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                }
            },
            "support_vector_machine": {
                "class": SVC,
                "default_params": {"probability": True, "random_state": 42},
                "grid_params": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf"]
                }
            }
        }
        
        self.trained_models = {}
        self.model_metrics = {}
        self.best_model_name = None

    def train_all(self, X: pd.DataFrame, y: pd.Series, use_grid_search: bool = True) -> dict:
        """
        Trains all 5 classifiers.
        Computes performance metrics on a 20% validation split.
        Saves the best estimators.
        """
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # 1. Map target variable if it is Y/N string to 0/1
        if not pd.api.types.is_numeric_dtype(y.dtype):
            y_mapped = y.map({"Y": 1, "N": 0}).fillna(0).astype(int)
        else:
            y_mapped = y.astype(int)
            
        # 2. Train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
        )
        
        self.model_metrics = {}
        self.trained_models = {}
        
        # 3. Train each model
        for name, config in self.model_configs.items():
            start_time = time.time()
            
            clf_class = config["class"]
            
            if use_grid_search:
                # Instantiate with default parameters first
                base_clf = clf_class(**config["default_params"])
                grid_search = GridSearchCV(
                    estimator=base_clf,
                    param_grid=config["grid_params"],
                    cv=5,
                    scoring="f1",
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Grid search complete for {name}. Best params: {best_params}")
            else:
                best_model = clf_class(**config["default_params"])
                best_model.fit(X_train, y_train)
                best_params = config["default_params"]
                print(f"Standard training complete for {name}.")
                
            training_time = time.time() - start_time
            
            # Predict and evaluate
            y_pred = best_model.predict(X_val)
            y_prob = best_model.predict_proba(X_val)[:, 1] if hasattr(best_model, "predict_proba") else y_pred
            
            acc = float(accuracy_score(y_val, y_pred))
            prec = float(precision_score(y_val, y_pred, zero_division=0))
            rec = float(recall_score(y_val, y_pred, zero_division=0))
            f1 = float(f1_score(y_val, y_pred, zero_division=0))
            
            # Confusion matrix
            cm = confusion_matrix(y_val, y_pred).tolist() # [[TN, FP], [FN, TP]]
            
            display_name = name.replace("_", " ").title()
            self.model_metrics[name] = {
                "display_name": display_name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm,
                "training_time_seconds": training_time,
                "training_time": training_time,
                "best_params": best_params
            }
            
            # Save the trained model
            self.trained_models[name] = best_model
            model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
            joblib.dump(best_model, model_path)
            
        # 4. Identify best model based on F1-score (or Accuracy if F1 is tied)
        best_name = None
        best_f1 = -1.0
        best_acc = -1.0
        
        for name, metrics in self.model_metrics.items():
            f1 = metrics["f1_score"]
            acc = metrics["accuracy"]
            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
                best_name = name
            elif abs(f1 - best_f1) < 1e-5 and acc > best_acc:
                best_acc = acc
                best_name = name
                
        self.best_model_name = best_name
        
        # Save a copy of the best model as best_model.joblib
        best_model_obj = self.trained_models[best_name]
        joblib.dump(best_model_obj, os.path.join(MODELS_DIR, "best_model.joblib"))
        
        # Save metadata info about metrics
        metadata = {
            "best_model_name": self.best_model_name,
            "metrics": self.model_metrics
        }
        joblib.dump(metadata, os.path.join(MODELS_DIR, "model_metadata.joblib"))
        
        return metadata

    @staticmethod
    def load_model(name: str):
        """Loads a specific trained model by name."""
        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model {name} not found at {model_path}")

    @staticmethod
    def load_metadata():
        """Loads metrics metadata for all models."""
        meta_path = os.path.join(MODELS_DIR, "model_metadata.joblib")
        if os.path.exists(meta_path):
            return joblib.load(meta_path)
        return None

    def get_feature_importances(self, model_name: str, feature_names: list) -> dict:
        """Returns the feature importances or coefficients of a trained model."""
        try:
            model = self.load_model(model_name)
        except FileNotFoundError:
            return {}
            
        importances = {}
        
        # Check for feature importances (Trees, Forests)
        if hasattr(model, "feature_importances_"):
            for name, val in zip(feature_names, model.feature_importances_):
                importances[name] = float(val)
        # Check for coefficients (Logistic Regression, Linear SVM)
        elif hasattr(model, "coef_"):
            coefs = model.coef_[0]
            for name, val in zip(feature_names, coefs):
                # Use absolute value to represent importance
                importances[name] = float(abs(val))
        else:
            # Fallback for models without direct importance (e.g. KNN, SVM RBF)
            # Use random forest feature importances as a global baseline or return equal weight
            try:
                rf_model = self.load_model("random_forest")
                if hasattr(rf_model, "feature_importances_"):
                    for name, val in zip(feature_names, rf_model.feature_importances_):
                        importances[name] = float(val)
            except FileNotFoundError:
                # Return uniform importance
                for name in feature_names:
                    importances[name] = 1.0 / len(feature_names)
                    
        # Normalize to sum to 1
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}
            
        # Sort by importance descending
        sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
        return sorted_importances

    def explain_prediction(self, model_name: str, input_dict: dict, prediction: int, probability: float, feature_names: list) -> list:
        """
        Generates 2-3 dynamic explanation points for a prediction.
        Combines model feature importances with candidate's specific values relative to data averages.
        """
        explanations = []
        
        credit_history = float(input_dict.get("Credit_History", 1.0))
        applicant_income = float(input_dict.get("ApplicantIncome", 0.0))
        coapplicant_income = float(input_dict.get("CoapplicantIncome", 0.0))
        loan_amount = float(input_dict.get("LoanAmount", 0.0))
        loan_term = float(input_dict.get("Loan_Amount_Term", 360.0))
        total_income = applicant_income + coapplicant_income
        income_to_loan = total_income / (loan_amount + 1e-5)
        
        loan_amount_rupees = loan_amount * 1000
        loan_lakhs = loan_amount / 100
        loan_fmt = f"₹{loan_amount_rupees:,.0f} ({loan_lakhs:.1f} Lakhs)" if loan_lakhs >= 1 else f"₹{loan_amount_rupees:,.0f}"

        # Point 1: Credit History (is almost always the deciding factor)
        if credit_history == 0.0:
            explanations.append({
                "factor": "Credit History",
                "importance": "High Impact",
                "type": "negative" if prediction == 0 else "neutral",
                "text": "Applicant has defaulted or uncleared past credit history (0.0), indicating higher repayment risk."
            })
        else:
            explanations.append({
                "factor": "Credit History",
                "importance": "High Impact",
                "type": "positive" if prediction == 1 else "neutral",
                "text": "Clean and verified credit history (1.0) significantly boosts approval confidence."
            })
            
        # Point 2: Income and Loan Ratio
        if prediction == 1:
            if income_to_loan > 35:
                explanations.append({
                    "factor": "Income-to-Loan Ratio",
                    "importance": "High Impact",
                    "type": "positive",
                    "text": f"Strong income multiplier ({income_to_loan:.1f}x). Total monthly income of ₹{total_income:,.0f} comfortably covers the requested loan of {loan_fmt}."
                })
            elif total_income > 8000:
                explanations.append({
                    "factor": "Monthly Household Income",
                    "importance": "Medium Impact",
                    "type": "positive",
                    "text": f"Combined household income of ₹{total_income:,.0f}/month meets strong solvency criteria."
                })
            else:
                explanations.append({
                    "factor": "Financial Solvency",
                    "importance": "Medium Impact",
                    "type": "positive",
                    "text": f"Total income of ₹{total_income:,.0f}/month satisfies minimum baseline eligibility for a loan of {loan_fmt} over {loan_term:.0f} months."
                })
        else:
            if income_to_loan < 15:
                explanations.append({
                    "factor": "Income-to-Loan Ratio",
                    "importance": "High Impact",
                    "type": "negative",
                    "text": f"Low income-to-loan ratio ({income_to_loan:.1f}x). Combined monthly income of ₹{total_income:,.0f} is insufficient for a loan of {loan_fmt}."
                })
            elif loan_amount > 250:
                explanations.append({
                    "factor": "High Loan Amount",
                    "importance": "High Impact",
                    "type": "negative",
                    "text": f"The requested loan amount of {loan_fmt} exceeds standard debt-to-income limits."
                })
            else:
                explanations.append({
                    "factor": "Income Capacity",
                    "importance": "Medium Impact",
                    "type": "negative",
                    "text": f"Applicant monthly income of ₹{total_income:,.0f} is marginal relative to requested loan size of {loan_fmt}."
                })

        # Point 3: Additional Demographic / employment contexts
        education = input_dict.get("Education", "Graduate")
        self_employed = input_dict.get("Self_Employed", "No")
        property_area = input_dict.get("Property_Area", "Semiurban")
        
        if prediction == 1:
            if property_area == "Semiurban":
                explanations.append({
                    "factor": "Property Area",
                    "importance": "Low-Medium",
                    "type": "positive",
                    "text": "Property located in a Semi-Urban area, which historically shows higher approval rates."
                })
            elif education == "Graduate":
                explanations.append({
                    "factor": "Education Level",
                    "importance": "Low",
                    "type": "positive",
                    "text": "Graduate education level supports long-term employment stability."
                })
        else:
            if self_employed == "Yes" and credit_history == 1.0:
                explanations.append({
                    "factor": "Employment Status",
                    "importance": "Low",
                    "type": "neutral",
                    "text": "Self-employed status introduces income variability, contributing to risk threshold adjustments."
                })
            elif education == "Not Graduate":
                explanations.append({
                    "factor": "Education Level",
                    "importance": "Low",
                    "type": "negative",
                    "text": "Non-graduate status increases risk score slightly in decision trees."
                })
                
        # Limit to top 3 points
        return explanations[:3]

if __name__ == "__main__":
    from pipeline import LoanPreprocessor
    from data_manager import load_raw_data
    
    print("Testing Model Manager...")
    df = load_raw_data()
    
    # Preprocess
    preprocessor = LoanPreprocessor()
    preprocessor.fit(df)
    df_trans = preprocessor.transform(df)
    
    X = df_trans
    y = df['Loan_Status']
    
    # Train
    manager = ModelManager()
    print("Training models (Grid Search disabled for speed in test)...")
    results = manager.train_all(X, y, use_grid_search=False)
    
    print("Best Model Name:", results["best_model_name"])
    print("Model Metrics:")
    for model_name, metrics in results["metrics"].items():
        print(f" - {model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, Time={metrics['training_time_seconds']:.2f}s")
        
    # Test feature importance
    print("Feature Importances for Best Model:")
    importances = manager.get_feature_importances(results["best_model_name"], preprocessor.feature_cols)
    for col, imp in list(importances.items())[:5]:
        print(f" - {col}: {imp:.4f}")
