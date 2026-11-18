# Sleep Quality Classifier

A machine learning model and Streamlit dashboard that predict and explain sleep quality from survey data collected from military personnel.

## What it does

- Turns survey answers (bedtimes, frequencies, categorical answers) into numeric features.
- Trains a Random Forest classifier for four sleep quality levels, from *Very Bad* to *Very Good*.
- Handles class imbalance with SMOTE and tunes hyperparameters with randomized search and stratified 5-fold cross-validation.
- Shows the most important factors and lets you try predictions in an interactive dashboard.
- Generates an analysis report in Markdown and HTML.

## Evaluation

The training pipeline is set up so that the reported scores reflect real, unseen answers:

1. A stratified test set (20%) is split off **before** any fitted preprocessing.
2. Missing-value means and scaling are fitted on the training set only.
3. SMOTE runs only on training data, inside each cross-validation fold, so no synthetic samples reach validation or test data.
4. The model is compared against a majority-class baseline, on both accuracy and macro F1.

Results are saved to `models/metrics.json`, and the dashboard and report read them from there.

> **Note:** an earlier version applied SMOTE to the whole dataset before splitting, which let synthetic copies of training samples into the test set and inflated the reported accuracy (89%). The pipeline was fixed as described above. Retraining needs the original survey dataset, which is not included in this repository, so the saved model and plots are still from the earlier version.

## Running it

```bash
cd sleep
pip install -r requirements.txt

# place the survey data at sleep/data/P2.4_Military_Responses.csv
cd src
python optimized_sleep_model.py   # train, evaluate, save model, metrics and plots
python generate_report.py         # optional: Markdown/HTML report
streamlit run sleep_dashboard.py  # dashboard on http://localhost:8501
```

## Project structure

```text
sleep/
  src/
    optimized_sleep_model.py   training, tuning and evaluation
    sleep_dashboard.py         Streamlit dashboard
    dashboard_utils.py         data loading and charts
    data_preparation.py        data preprocessing helpers
    generate_report.py         report generation
  models/                      saved model and metrics
  results/                     confusion matrix and feature importance plots
```

## Tech stack

Python · pandas · scikit-learn · imbalanced-learn · Streamlit · Plotly · Matplotlib / Seaborn

## Disclaimer

For educational and research purposes only. The model is not clinically validated and should not be used for medical decisions.
