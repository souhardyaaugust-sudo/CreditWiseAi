import os
import urllib.request
import pandas as pd

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE_PATH = os.path.join(DATA_DIR, "loan_data.csv")

# Standard fallback URL (Kaggle dataset equivalent on GitHub)
FALLBACK_URL = "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv"

def ensure_data_dir():
    """Ensure data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)

def download_dataset():
    """
    Downloads the Loan Prediction dataset.
    Tries using Kaggle API first. If it fails or is unconfigured,
    falls back to downloading from the GitHub raw URL.
    """
    ensure_data_dir()
    
    if os.path.exists(DATA_FILE_PATH):
        print(f"Dataset already exists at: {DATA_FILE_PATH}")
        return True

    # Check for Kaggle credentials before importing/authenticating
    home_dir = os.path.expanduser("~")
    kaggle_json = os.path.join(home_dir, ".kaggle", "kaggle.json")
    kaggle_token = os.path.join(home_dir, ".kaggle", "access_token")
    has_creds = (
        os.path.exists(kaggle_json) or 
        os.path.exists(kaggle_token) or 
        "KAGGLE_USERNAME" in os.environ or 
        "KAGGLE_API_TOKEN" in os.environ
    )

    if has_creds:
        try:
            print("Attempting to download dataset from Kaggle...")
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "altruistdelhite04/loan-prediction-problem-dataset", 
                path=DATA_DIR, 
                unzip=True
            )
            
            # Check what was downloaded and rename/move if necessary
            files = os.listdir(DATA_DIR)
            train_file = None
            for f in files:
                if f.startswith("train") and f.endswith(".csv"):
                    train_file = f
                    break
                    
            if train_file:
                os.rename(os.path.join(DATA_DIR, train_file), DATA_FILE_PATH)
                print("Successfully downloaded and processed dataset from Kaggle.")
                return True
        except Exception as e:
            print(f"Kaggle download failed. Reason: {e}")
            print("Falling back to downloading from public GitHub repository...")
    else:
        print("Kaggle credentials not found in environment or ~/.kaggle/. Using GitHub fallback.")

    # Fallback to direct raw download
    try:
        print("Downloading from fallback raw GitHub link...")
        urllib.request.urlretrieve(FALLBACK_URL, DATA_FILE_PATH)
        print(f"Successfully downloaded fallback dataset from GitHub to: {DATA_FILE_PATH}")
        return True
    except Exception as e:
        print(f"Fallback download failed: {e}")
        return False

def load_raw_data() -> pd.DataFrame:
    """Loads the raw dataset into a pandas DataFrame."""
    if not os.path.exists(DATA_FILE_PATH):
        success = download_dataset()
        if not success:
            raise FileNotFoundError(f"Dataset could not be downloaded/found at {DATA_FILE_PATH}")
    
    # Read the data
    df = pd.read_csv(DATA_FILE_PATH)
    
    # Standardize column names (Kaggle uses Loan_ID, Gender, Married, etc.)
    # We will keep them as-is or make sure they match expectations.
    return df

def get_dataset_stats():
    """Generates summary statistics of the raw dataset."""
    df = load_raw_data()
    
    # Missing values
    missing = df.isnull().sum().to_dict()
    
    # Target variable distribution (Loan_Status: Y/N)
    loan_status_dist = {}
    if 'Loan_Status' in df.columns:
        loan_status_dist = df['Loan_Status'].value_counts().to_dict()
        # Convert keys to strings
        loan_status_dist = {str(k): int(v) for k, v in loan_status_dist.items()}
        
    # Categorical distributions
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    categorical_dists = {}
    for col in cat_cols:
        if col in df.columns:
            # Handle nulls by filling with 'Missing' for statistics display
            filled_col = df[col].fillna('Missing').astype(str)
            dist = filled_col.value_counts().to_dict()
            categorical_dists[col] = {k: int(v) for k, v in dist.items()}
            
    # Numerical summaries
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    numerical_stats = {}
    for col in num_cols:
        if col in df.columns:
            desc = df[col].describe().to_dict()
            # Clean NaN values if any
            desc = {k: (v if not pd.isna(v) else 0) for k, v in desc.items()}
            numerical_stats[col] = desc

    return {
        "total_records": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "missing_values": missing,
        "loan_status_distribution": loan_status_dist,
        "categorical_distributions": categorical_dists,
        "numerical_stats": numerical_stats,
        "columns": list(df.columns)
    }

if __name__ == "__main__":
    # Test download and statistics
    print("Testing data manager...")
    if download_dataset():
        stats = get_dataset_stats()
        print(f"Loaded successfully. Total records: {stats['total_records']}")
        print(f"Loan status distribution: {stats['loan_status_distribution']}")
    else:
        print("Data download failed.")
