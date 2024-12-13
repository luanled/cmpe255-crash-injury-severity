
# Crash Injury Severity Prediction
This project aims to predict the severity of injuries resulting from crashes based on various factors such as driver age, sobriety, party type, and time of the crash. The goal is to build predictive models using machine learning techniques and evaluate their performance.

---

## Project Structure

```bash
cmpe255-crash-injury-severity/
│
├── data/
│   ├── raw/                # Raw data files before any processing
│   ├── processed/          # Cleaned and processed data files
│
├── notebooks/              # Jupyter notebooks used for data analysis
│   ├── Brian_visualize_preprocessed_data.ipynb   # Initial data exploration and analysis
│   ├── random_forest.ipynb                       # Random Forest model implementation
│   ├── gradient_boosting_classifier.ipynb        # Gradient Boosting model implementation
│   ├── logistic_regression.ipynb                 # Logistic Regression implementation
│   ├── SMOTE_method.ipynb                        # SMOTE oversampling method applied
│   ├── balanced_random_forest.ipynb              # Balanced Random Forest classifier
│   ├── class_weight.ipynb                        # Class-weighted Random Forest implementation
│   ├── random_forest_feature_importance_analysis.ipynb  # Random Forest feature importance analysis
│   ├── random_forest_vs_logistic_regression.ipynb       # Random Forest vs Logistic Regression comparison
│   ├── severity_correlation_patterns.ipynb       # Analysis of severity correlation patterns
│   ├── visualize_preprocessed_data.ipynb         # Visualizing preprocessed data
│
├── models/                # Saved trained models
│   ├── balanced_random_forest_pipeline.pkl
│   ├── class_weight_random_forest_pipeline.pkl
│   ├── gbm_pipeline.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── smote_random_forest_pipeline.pkl
│
├── parameters/            # Model parameters for each model
│   ├── balanced_random_forest_parameters.json
│   ├── class_weight_random_forest_parameters.json
│   ├── gbm_parameters.json
│   ├── logistic_regression_parameters.json
│   ├── random_forest_parameters.json
│
├── scripts/               # Python scripts for various tasks
│   ├── preprocess_data.py            # Data cleaning and preprocessing
│   ├── merge_dataset.py              # Merge 2 preprocessed datasets into 1 for training and testing
│
├── requirements.txt       # Python dependencies
└── README.md              # Project description and instructions
```

---

## Requirements
- Python >= 3.8
- Jupyter Notebook
- Required libraries listed in `requirements.txt`

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/cmpe255-crash-injury-severity.git
   cd cmpe255-crash-injury-severity
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the raw crash and vehicle datasets are placed in the `data/raw` directory:
   - Place the raw data files in `data/raw`.
   - Ensure the `data/processed` folder exists for saving output files after preprocessing.

---

## Usage

### 1. Data Preprocessing
Run the preprocessing script to clean and prepare the data for analysis:
```bash
python scripts/preprocess_data.py
python scripts/merge_dataset.py
```

### 2. Model Training
Explore the Jupyter notebooks for training machine learning models:
- `random_forest.ipynb`: Implements the Random Forest model.
- `gradient_boosting_classifier.ipynb`: Implements the Gradient Boosting Classifier.
- `logistic_regression.ipynb`: Implements Logistic Regression.
- `SMOTE_method.ipynb`: Applies SMOTE to handle class imbalance.
- `class_weight.ipynb`: Implements class-weighted Random Forest.

### 3. Evaluate Models
Run the relevant notebooks above for evaluating models on the test dataset.

### 4. View Confusion Matrix
Confusion matrix generation is included in each notebook above as part of the evaluation.

---

## Outputs
- **Trained Models**: Saved in the `models/` directory as `.pkl` files.
- **Model Parameters**: Saved in the `parameters/` directory as `.json` files.
- **Processed Data**: Saved in the `data/processed` directory.

---