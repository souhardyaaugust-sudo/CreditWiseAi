import os
import sqlite3
from datetime import datetime

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "history.db")

def ensure_db_dir():
    """Ensure database directory exists."""
    os.makedirs(DB_DIR, exist_ok=True)

class HistoryManager:
    def __init__(self):
        ensure_db_dir()
        self.initialize_db()

    def get_connection(self):
        """Returns a connection to the SQLite database."""
        return sqlite3.connect(DB_PATH)

    def initialize_db(self):
        """Creates the prediction history table if it doesn't exist."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gender TEXT,
                married TEXT,
                dependents TEXT,
                education TEXT,
                self_employed TEXT,
                applicant_income REAL,
                coapplicant_income REAL,
                loan_amount REAL,
                loan_amount_term REAL,
                credit_history REAL,
                property_area TEXT,
                predicted_status TEXT,
                probability REAL,
                model_used TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

    def add_prediction(self, input_dict: dict, predicted_status: str, probability: float, model_used: str) -> bool:
        """Saves a prediction record to the database."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute("""
                INSERT INTO prediction_history (
                    gender, married, dependents, education, self_employed,
                    applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                    credit_history, property_area, predicted_status, probability, model_used, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(input_dict.get("Gender", "Male")),
                str(input_dict.get("Married", "No")),
                str(input_dict.get("Dependents", "0")),
                str(input_dict.get("Education", "Graduate")),
                str(input_dict.get("Self_Employed", "No")),
                float(input_dict.get("ApplicantIncome", 0)),
                float(input_dict.get("CoapplicantIncome", 0)),
                float(input_dict.get("LoanAmount", 0)),
                float(input_dict.get("Loan_Amount_Term", 360)),
                float(input_dict.get("Credit_History", 1.0)),
                str(input_dict.get("Property_Area", "Semiurban")),
                predicted_status,
                float(probability),
                model_used,
                timestamp
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving prediction history: {e}")
            return False

    def get_history(self, limit: int = 100) -> list:
        """Retrieves prediction history records from the database."""
        try:
            conn = self.get_connection()
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM prediction_history 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            history = []
            
            for row in rows:
                history.append({
                    "id": row["id"],
                    "inputs": {
                        "Gender": row["gender"],
                        "Married": row["married"],
                        "Dependents": row["dependents"],
                        "Education": row["education"],
                        "Self_Employed": row["self_employed"],
                        "ApplicantIncome": row["applicant_income"],
                        "CoapplicantIncome": row["coapplicant_income"],
                        "LoanAmount": row["loan_amount"],
                        "Loan_Amount_Term": row["loan_amount_term"],
                        "Credit_History": row["credit_history"],
                        "Property_Area": row["property_area"]
                    },
                    "predicted_status": row["predicted_status"],
                    "probability": row["probability"],
                    "model_used": row["model_used"],
                    "timestamp": row["timestamp"]
                })
                
            conn.close()
            return history
        except Exception as e:
            print(f"Error fetching prediction history: {e}")
            return []

if __name__ == "__main__":
    print("Testing History Manager...")
    manager = HistoryManager()
    
    # Test insertion
    test_input = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban"
    }
    
    if manager.add_prediction(test_input, "Approved", 0.88, "random_forest"):
        print("Test prediction saved successfully.")
        
    history = manager.get_history(limit=5)
    print(f"Retrieved {len(history)} historical records.")
    if history:
        print("Latest prediction record:", history[0])
