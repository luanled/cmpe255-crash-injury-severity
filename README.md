# Crash Injury Severity Prediction
This project aims to predict the severity of injuries resulting from crashes based on various factors such as driver age, sobriety, party type, speeding, and time of the crash.
## Project Structure
```bash
cmpe255-crash-injury-severity/
│
├──  data/
│  ├──  raw/  # Raw data files before any processing
│  ├──  processed/  # Cleaned and processed data files
│
├──  notebooks/  # Jupyter notebooks used for data analysis
│  ├──  visualize_preprocessed_data.ipynb  # Initial exploration and analysis
│
├──  scripts/
│  ├──  preprocess_data.py  # Python script for data cleaning and preprocessing
│
├──  requirements.txt  # List of Python dependencies
└──  README.md
```
## Install requirements
```bash
pip  install  -r  requirements.txt
```
Make sure the crash and vehicle datasets are in data/raw, and data/processed is existed to contain the output files.