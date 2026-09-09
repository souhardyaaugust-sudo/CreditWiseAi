# CreditAnalyzer AI

### Intelligent Credit Risk & Loan Approval Prediction Platform

CreditAnalyzer AI is a machine learning-powered platform designed to predict whether a loan application should be approved based on applicant information and financial history. The platform assists financial institutions in evaluating loan applications by analyzing applicant data, estimating credit risk, comparing multiple classification models, and providing explainable decision factors.

---

## 🚀 Key Features

*   **Overview Dashboard**: Visualizes loan approval distributions, property demographic distributions, monthly income trends, and credit history correlations.
*   **Dataset Explorer**: View historical credit records with pagination, multi-column sorting, and flexible filtering (by gender, education, and approval status).
*   **Data Pipeline & Feature Engineering**: Configure numerical missing value imputations (mean, median, mode) and inspect real-time correlation heatmaps of numeric and engineered features (`Total_Income`, `Income_to_Loan_Ratio`, `Loan_Amount_per_Term`).
*   **Model Benchmarking Suite**: Train and optimize 5 different machine learning classifiers:
    *   *Logistic Regression*
    *   *Decision Tree Classifier*
    *   *Random Forest Classifier*
    *   *K-Nearest Neighbors (KNN)*
    *   *Support Vector Machine (SVM)*
    Includes hyperparameter tuning via 5-fold cross-validation (`GridSearchCV`) and confusion matrix visualization.
*   **Loan Predictor**: Enter new applicant demographic and financial figures to evaluate loan suitability in real-time. Features an interactive approval probability ring and dynamic decision factor explanations.
*   **Prediction History Log**: An audit trail of evaluated loan applicants stored in a local database.

---

## 🛠️ Technology Stack

*   **Backend**: Python, FastAPI, Uvicorn, SQLite
*   **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
*   **Frontend**: Vanilla HTML5, CSS3 (Glassmorphic dark-theme design), JavaScript ES6, Chart.js, FontAwesome

---

## 📁 Directory Structure

```text
CreditAnalyzer AI/
│
├── backend/
│   ├── main.py                 # FastAPI application and route entry point
│   ├── requirements.txt        # Backend dependencies
│   ├── verify_pipeline.py      # Programmatic pipeline test suite
│   │
│   ├── src/                    # Backend source modules
│   │   ├── data_manager.py     # Dataset download (Kaggle/GitHub) & stats
│   │   ├── pipeline.py         # Data preprocessing & feature engineering
│   │   ├── model_manager.py    # ML models training, tuning & explanations
│   │   └── history_manager.py  # Prediction history SQLite manager
│   │
│   ├── data/                   # Directory containing loan_data.csv
│   ├── database/               # Directory containing history.db (SQLite)
│   └── models/                 # Preprocessor & model serializations (.joblib)
│
├── frontend/
│   ├── index.html              # Main dashboard layout
│   ├── style.css               # Glassmorphism dark mode stylesheet
│   └── app.js                  # Frontend API client and Chart.js scripts
│
├── AGENT.md                    # Core project specifications
└── README.md                   # This README file
```

---

## ⚙️ Installation & Setup

Ensure you have **Python 3.10+** and the **`uv`** package manager installed.

### 1. Set Up the Virtual Environment
Navigate to the root directory and create a virtual environment using `uv`:
```bash
uv venv
```

### 2. Activate the Environment
*   **Windows (PowerShell)**:
    ```powershell
    .venv\Scripts\activate
    ```
*   **macOS / Linux**:
    ```bash
    source .venv/bin/activate
    ```

### 3. Install Dependencies
Install the required packages from `backend/requirements.txt`:
```bash
uv pip install -r backend/requirements.txt
```

---

## 🎮 How to Run

### 1. Launch the Application
Start the FastAPI server by running:
```bash
uv run python backend/main.py
```
*Note: On startup, the server will check for the dataset. If it isn't found locally, it will download it. It will also compile a set of baseline models so the application is ready for predictions immediately.*

### 2. Access the Dashboard
Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🧪 Running Tests
You can run the programmatic pipeline tests to verify that the dataset download, data preprocessors, model trainers, and FastAPI endpoints are all running successfully:
```bash
uv run python backend/verify_pipeline.py
```
*The test runner uses FastAPI's `TestClient` and will output `OK` when all assertions successfully pass.*
