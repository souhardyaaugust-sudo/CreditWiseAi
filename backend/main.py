import os
import sys
import warnings
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager

# Silence deprecation and future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_manager import download_dataset, load_raw_data, get_dataset_stats
from pipeline import LoanPreprocessor
from model_manager import ModelManager
from history_manager import HistoryManager

# Setup managers
history_manager = HistoryManager()
model_manager = ModelManager()

# Ensure data and models are initialized on startup using lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting CreditAnalyzer AI application...")
    
    # 1. Download data if needed
    download_dataset()
    
    # 2. Train baseline models if none exist
    models_exist = os.path.exists(os.path.join(os.path.dirname(__file__), "models", "model_metadata.joblib"))
    preprocessor_exists = os.path.exists(os.path.join(os.path.dirname(__file__), "models", "preprocessor.joblib"))
    
    if not preprocessor_exists or not models_exist:
        print("No trained models found. Training baseline models on startup...")
        try:
            df = load_raw_data()
            # Split features and target
            X = df.drop(columns=['Loan_ID', 'Loan_Status'])
            y = df['Loan_Status']
            
            # Fit and save preprocessor
            preprocessor = LoanPreprocessor()
            preprocessor.fit(X)
            preprocessor.save()
            
            # Transform
            X_trans = preprocessor.transform(X)
            
            # Train baseline models (no grid search for fast startup)
            model_manager.train_all(X_trans, y, use_grid_search=False)
            print("Baseline models trained successfully on startup.")
        except Exception as e:
            print(f"Error training models on startup: {e}")
    yield

from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app with lifespan
app = FastAPI(title="CreditAnalyzer AI API", version="1.0.0", lifespan=lifespan)

# Configure CORS for decoupled frontend hosting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define API request/response models
class PreprocessConfig(BaseModel):
    impute_strategy: str = Field(default="median", description="Strategy for numeric missing values: mean, median, mode")

class TrainConfig(BaseModel):
    use_grid_search: bool = Field(default=True, description="Enable GridSearchCV hyperparameter tuning")

class PredictRequest(BaseModel):
    Gender: str = Field(..., examples=["Male"])
    Married: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["0"])
    Education: str = Field(..., examples=["Graduate"])
    Self_Employed: str = Field(..., examples=["No"])
    ApplicantIncome: float = Field(..., examples=[5000])
    CoapplicantIncome: float = Field(..., examples=[0])
    LoanAmount: float = Field(..., examples=[120])
    Loan_Amount_Term: float = Field(..., examples=[360])
    Credit_History: float = Field(..., examples=[1.0])
    Property_Area: str = Field(..., examples=["Semiurban"])
    model_name: Optional[str] = Field(default=None, description="Model to use for prediction (default: best model)")

@app.get("/api/stats")
def get_stats():
    """Returns dataset overview statistics and correlation matrix."""
    try:
        stats = get_dataset_stats()
        
        # Add correlation matrix for numeric columns
        df = load_raw_data()
        num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        
        # Simple imputation for correlation calculation
        df_num = df[num_cols].copy()
        for col in num_cols:
            df_num[col] = df_num[col].fillna(df_num[col].median())
            
        corr_matrix = df_num.corr().to_dict()
        stats["correlation_matrix"] = corr_matrix
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset")
def get_dataset(
    page: int = Query(1, ge=1),
    page_size: int = Query(15, ge=1, le=100),
    search: Optional[str] = None,
    status_filter: Optional[str] = None,
    education_filter: Optional[str] = None,
    gender_filter: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_order: str = Query("asc", regex="^(asc|desc)$")
):
    """Returns paginated, filtered, and sorted records from the dataset."""
    try:
        df = load_raw_data()
        
        # Apply Search (Loan_ID)
        if search:
            df = df[df['Loan_ID'].str.contains(search, case=False, na=False)]
            
        # Apply Filters
        if status_filter:
            df = df[df['Loan_Status'] == status_filter]
        if education_filter:
            df = df[df['Education'] == education_filter]
        if gender_filter:
            df = df[df['Gender'] == gender_filter]
            
        # Apply Sorting
        if sort_by and sort_by in df.columns:
            ascending = (sort_order == "asc")
            df = df.sort_values(by=sort_by, ascending=ascending)
            
        total_records = len(df)
        
        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_df = df.iloc[start_idx:end_idx].copy()
        
        # Convert NaN values to None for clean JSON response
        paginated_df = paginated_df.astype(object).where(paginated_df.notna(), None)
        records = paginated_df.to_dict(orient="records")
        
        return {
            "records": records,
            "total_records": total_records,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_records + page_size - 1) // page_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/preprocess")
def configure_preprocessing(config: PreprocessConfig):
    """Fits and saves a new data preprocessor configuration."""
    try:
        df = load_raw_data()
        X = df.drop(columns=['Loan_ID', 'Loan_Status'])
        
        preprocessor = LoanPreprocessor(num_impute_strategy=config.impute_strategy)
        preprocessor.fit(X)
        preprocessor.save()
        
        return {"status": "success", "message": f"Preprocessor re-fit using '{config.impute_strategy}' strategy."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
def train_models(config: TrainConfig):
    """Trains all classifiers, optionally running GridSearchCV hyperparameter optimization."""
    try:
        df = load_raw_data()
        X = df.drop(columns=['Loan_ID', 'Loan_Status'])
        y = df['Loan_Status']
        
        # Load or fit preprocessor
        try:
            preprocessor = LoanPreprocessor.load()
        except FileNotFoundError:
            preprocessor = LoanPreprocessor()
            preprocessor.fit(X)
            preprocessor.save()
            
        X_trans = preprocessor.transform(X)
        
        # Train and return metrics metadata
        metadata = model_manager.train_all(X_trans, y, use_grid_search=config.use_grid_search)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
def get_models_metadata():
    """Retrieves trained models comparison metrics and best model indicator."""
    metadata = model_manager.load_metadata()
    if not metadata:
        raise HTTPException(status_code=404, detail="No trained model metadata found. Please train models first.")
    return metadata

@app.post("/api/predict")
def predict_loan(request: PredictRequest):
    """Executes a loan approval prediction on the applicant profile."""
    try:
        # 1. Load Preprocessor
        try:
            preprocessor = LoanPreprocessor.load()
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Preprocessor is not trained. Please fit preprocessor or train models.")
            
        # 2. Get active model name
        metadata = model_manager.load_metadata()
        active_model_name = request.model_name
        if not active_model_name or active_model_name == "best_model":
            if metadata:
                active_model_name = metadata["best_model_name"]
            else:
                active_model_name = "logistic_regression" # default fallback
                
        # 3. Load Model
        try:
            model = model_manager.load_model(active_model_name)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=f"Model '{active_model_name}' has not been trained. Please train models first.")
            
        # 4. Format and Preprocess Input
        input_data = request.model_dump(exclude={"model_name"})
        X_single = preprocessor.transform_single(input_data)
        
        # 5. Predict
        pred = int(model.predict(X_single)[0])
        status = "Approved" if pred == 1 else "Rejected"
        
        # Probability estimation
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_single)[0][1])
        else:
            # Fallback for models without predict_proba (e.g. some SVM configurations)
            prob = 1.0 if pred == 1 else 0.0
            
        # If predicted Status is Rejected, the probability shown should represent approval probability
        # approval probability is 'prob' (likelihood of class 1)
        
        # 6. Generate explanations
        explanations = model_manager.explain_prediction(
            active_model_name, input_data, pred, prob, preprocessor.feature_cols
        )
        
        # 7. Save to database history
        history_manager.add_prediction(input_data, status, prob, active_model_name)
        
        return {
            "prediction_status": status,
            "approval_probability": prob,
            "model_used": active_model_name,
            "explanations": explanations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
def get_prediction_history(limit: int = Query(50, ge=1, le=200)):
    """Returns the prediction log history from the database."""
    return history_manager.get_history(limit=limit)

# Serve Frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    
    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
        
    @app.get("/{filename}")
    def serve_frontend_root_file(filename: str):
        file_path = os.path.join(FRONTEND_DIR, filename)
        if os.path.exists(file_path):
            return FileResponse(file_path)
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    # Start server
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
