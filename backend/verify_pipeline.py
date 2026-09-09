import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_manager import load_raw_data, get_dataset_stats
from pipeline import LoanPreprocessor
from model_manager import ModelManager
from main import app
from fastapi.testclient import TestClient

class TestCreditAnalyzerPipeline(unittest.TestCase):
    
    def test_01_data_manager(self):
        """Test dataset loading and statistics generation."""
        print("\n--- Testing Data Manager ---")
        df = load_raw_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        print(f"Dataset loaded. Total rows: {len(df)}, Columns: {list(df.columns)}")
        
        stats = get_dataset_stats()
        self.assertIn("total_records", stats)
        self.assertEqual(stats["total_records"], len(df))
        self.assertIn("loan_status_distribution", stats)
        print("Dataset stats successfully verified.")

    def test_02_preprocessor(self):
        """Test preprocessor fit and transform operations."""
        print("\n--- Testing Preprocessor ---")
        df = load_raw_data()
        X = df.drop(columns=['Loan_ID', 'Loan_Status'])
        
        preprocessor = LoanPreprocessor()
        preprocessor.fit(X)
        
        # Test full dataset transform
        X_trans = preprocessor.transform(X)
        self.assertEqual(X_trans.shape[0], X.shape[0])
        self.assertEqual(X_trans.shape[1], 14) # 7 categorical + 4 numerical + 3 engineered
        
        # Verify engineered features exist
        self.assertIn("Total_Income", X_trans.columns)
        self.assertIn("Income_to_Loan_Ratio", X_trans.columns)
        self.assertIn("Loan_Amount_per_Term", X_trans.columns)
        
        # Test single prediction transform
        test_sample = {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "1",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 4500,
            "CoapplicantIncome": 1500,
            "LoanAmount": 100,
            "Loan_Amount_Term": 360,
            "Credit_History": 1.0,
            "Property_Area": "Semiurban"
        }
        
        single_trans = preprocessor.transform_single(test_sample)
        self.assertEqual(single_trans.shape, (1, 14))
        print("Preprocessor transformations verified.")

    def test_03_model_manager(self):
        """Test model training, benchmarking, and predictions."""
        print("\n--- Testing Model Manager ---")
        df = load_raw_data()
        preprocessor = LoanPreprocessor()
        preprocessor.fit(df.drop(columns=['Loan_ID', 'Loan_Status']))
        X_trans = preprocessor.transform(df.drop(columns=['Loan_ID', 'Loan_Status']))
        y = df['Loan_Status']
        
        manager = ModelManager()
        # Train baseline (standard) to keep test fast
        results = manager.train_all(X_trans, y, use_grid_search=False)
        
        self.assertIn("best_model_name", results)
        self.assertIn("metrics", results)
        self.assertIn(results["best_model_name"], results["metrics"])
        
        # Check active feature importance
        importances = manager.get_feature_importances(results["best_model_name"], preprocessor.feature_cols)
        self.assertGreater(len(importances), 0)
        self.assertIn("Credit_History", importances)
        
        # Check dynamic explanation
        test_sample = {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "1",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 4500,
            "CoapplicantIncome": 1500,
            "LoanAmount": 100,
            "Loan_Amount_Term": 360,
            "Credit_History": 1.0,
            "Property_Area": "Semiurban"
        }
        
        expls = manager.explain_prediction(
            results["best_model_name"], test_sample, 1, 0.85, preprocessor.feature_cols
        )
        self.assertGreater(len(expls), 0)
        print("Model Manager training and explanations verified.")

    def test_04_fastapi_endpoints(self):
        """Test FastAPI endpoints using TestClient."""
        print("\n--- Testing FastAPI Endpoints ---")
        client = TestClient(app)
        
        # Test stats
        res_stats = client.get("/api/stats")
        self.assertEqual(res_stats.status_code, 200)
        stats = res_stats.json()
        self.assertIn("total_records", stats)
        self.assertIn("correlation_matrix", stats)
        
        # Test dataset
        res_data = client.get("/api/dataset?page=1&page_size=5")
        self.assertEqual(res_data.status_code, 200)
        grid = res_data.json()
        self.assertEqual(len(grid["records"]), 5)
        self.assertIn("total_records", grid)
        
        # Test predict
        test_payload = {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5000,
            "CoapplicantIncome": 2000,
            "LoanAmount": 150,
            "Loan_Amount_Term": 360,
            "Credit_History": 1.0,
            "Property_Area": "Urban",
            "model_name": "best_model"
        }
        res_pred = client.post("/api/predict", json=test_payload)
        self.assertEqual(res_pred.status_code, 200)
        pred_res = res_pred.json()
        self.assertIn("prediction_status", pred_res)
        self.assertIn("approval_probability", pred_res)
        self.assertIn("explanations", pred_res)
        
        # Test history
        res_hist = client.get("/api/history")
        self.assertEqual(res_hist.status_code, 200)
        history = res_hist.json()
        self.assertGreater(len(history), 0) # Should have at least the one we just predicted!
        
        print("All API endpoints responded successfully.")

if __name__ == "__main__":
    unittest.main()
