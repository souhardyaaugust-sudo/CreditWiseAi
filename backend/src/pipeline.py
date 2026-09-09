import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")

class LoanPreprocessor:
    def __init__(self, num_impute_strategy="median"):
        self.num_impute_strategy = num_impute_strategy
        self.scaler = StandardScaler()
        
        # Imputers placeholders
        self.num_impute_values = {}
        self.cat_impute_values = {}
        
        # Mappings
        self.mappings = {
            "Gender": {"Male": 1, "Female": 0},
            "Married": {"Yes": 1, "No": 0},
            "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
            "Education": {"Graduate": 1, "Not Graduate": 0},
            "Self_Employed": {"Yes": 1, "No": 0},
            "Credit_History": {1.0: 1, 0.0: 0, 1: 1, 0: 0},
            "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}
        }
        
        # Columns definitions
        self.num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        self.cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
        self.engineered_cols = ['Total_Income', 'Income_to_Loan_Ratio', 'Loan_Amount_per_Term']
        self.feature_cols = self.cat_cols + self.num_cols + self.engineered_cols
        
    def fit(self, df: pd.DataFrame):
        """Fits the imputer values and scaling parameters on a dataframe."""
        df_copy = df.copy()
        
        # 1. Fit numerical imputers
        for col in self.num_cols:
            if col in df_copy.columns:
                if self.num_impute_strategy == "mean":
                    val = float(df_copy[col].mean())
                elif self.num_impute_strategy == "mode":
                    val = float(df_copy[col].mode()[0])
                else:  # default to median
                    val = float(df_copy[col].median())
                self.num_impute_values[col] = val
                df_copy[col] = df_copy[col].fillna(val)
                
        # 2. Fit categorical imputers (always use mode)
        for col in self.cat_cols:
            if col in df_copy.columns:
                val = df_copy[col].mode()[0]
                # Convert numeric-like categorical mode (e.g. Credit_History) to float if needed
                if col == "Credit_History":
                    val = float(val)
                self.cat_impute_values[col] = val
                df_copy[col] = df_copy[col].fillna(val)
                
        # 3. Perform Feature Engineering
        df_copy = self._engineer_features(df_copy)
        
        # 4. Apply categorical mappings
        for col in self.cat_cols:
            mapping = self.mappings[col]
            # Convert values to correct types before mapping
            df_copy[col] = df_copy[col].map(mapping).fillna(0) # Fallback to 0 if unknown category
            
        # 5. Fit Scaler on all features (both numerical and engineered numericals)
        scale_cols = self.num_cols + self.engineered_cols
        self.scaler.fit(df_copy[scale_cols])
        
        return self

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to create engineered features."""
        df_copy = df.copy()
        
        # Total Household Income
        df_copy['Total_Income'] = df_copy['ApplicantIncome'] + df_copy['CoapplicantIncome']
        
        # Income to Loan Ratio (Total Income / Loan Amount in thousands)
        # Add a tiny epsilon to avoid division by zero
        df_copy['Income_to_Loan_Ratio'] = df_copy['Total_Income'] / (df_copy['LoanAmount'] + 1e-5)
        
        # Loan Amount per Term month
        df_copy['Loan_Amount_per_Term'] = df_copy['LoanAmount'] / (df_copy['Loan_Amount_Term'] + 1e-5)
        
        return df_copy

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a dataframe using the fitted imputer and scaler parameters."""
        df_copy = df.copy()
        
        # 1. Apply numerical imputations
        for col in self.num_cols:
            if col in df_copy.columns:
                val = self.num_impute_values.get(col, 0.0)
                df_copy[col] = df_copy[col].fillna(val)
                
        # 2. Apply categorical imputations
        for col in self.cat_cols:
            if col in df_copy.columns:
                val = self.cat_impute_values.get(col, "Missing")
                df_copy[col] = df_copy[col].fillna(val)
                
        # 3. Perform Feature Engineering
        df_copy = self._engineer_features(df_copy)
        
        # 4. Map categorical columns
        for col in self.cat_cols:
            mapping = self.mappings[col]
            # Map values, fill na for any unseen categories
            df_copy[col] = df_copy[col].map(mapping).fillna(0)
            
        # 5. Scale features
        scale_cols = self.num_cols + self.engineered_cols
        scaled_features = self.scaler.transform(df_copy[scale_cols])
        
        # Replace numerical columns in the copy with scaled values
        for i, col in enumerate(scale_cols):
            df_copy[col] = scaled_features[:, i]
            
        # Ensure we return only the feature columns in standard order
        return df_copy[self.feature_cols]

    def transform_single(self, input_dict: dict) -> pd.DataFrame:
        """Transforms a single prediction input dictionary to a model-ready DataFrame."""
        # Convert single input dict to dataframe
        df = pd.DataFrame([input_dict])
        
        # Clean target and ID columns if they somehow got in
        if 'Loan_ID' in df.columns:
            df = df.drop(columns=['Loan_ID'])
        if 'Loan_Status' in df.columns:
            df = df.drop(columns=['Loan_Status'])
            
        transformed_df = self.transform(df)
        return transformed_df

    def save(self):
        """Saves the preprocessor to backend/models/preprocessor.joblib."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self, PREPROCESSOR_PATH)
        print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

    @staticmethod
    def load():
        """Loads a saved preprocessor from file."""
        if os.path.exists(PREPROCESSOR_PATH):
            return joblib.load(PREPROCESSOR_PATH)
        else:
            raise FileNotFoundError(f"No preprocessor file found at {PREPROCESSOR_PATH}. Fit one first.")

if __name__ == "__main__":
    from data_manager import load_raw_data
    print("Testing preprocessor...")
    df = load_raw_data()
    
    # Split features and target
    X = df.drop(columns=['Loan_ID', 'Loan_Status'])
    y = df['Loan_Status']
    
    # Fit and transform
    preprocessor = LoanPreprocessor()
    preprocessor.fit(X)
    X_trans = preprocessor.transform(X)
    
    print("Preprocessed shape:", X_trans.shape)
    print("First preprocessed sample:\n", X_trans.iloc[0])
    
    # Test single prediction pipeline
    test_sample = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5849,
        "CoapplicantIncome": 0,
        "LoanAmount": 128,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban"
    }
    
    single_trans = preprocessor.transform_single(test_sample)
    print("Transformed single sample shape:", single_trans.shape)
    print("Transformed single values:", single_trans)
    
    # Save
    preprocessor.save()
    loaded = LoanPreprocessor.load()
    print("Loader successfully verified.")
