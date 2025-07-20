Sleep Quality Dashboard
A machine learning-based dashboard to analyze, predict, and visualize sleep quality for military personnel based on survey data.

📂 Project Structure
sleep_dashboard/
├── data/                     # Dataset CSV files (input and processed)
├── models/                   # Saved optimized models (pkl)
├── results/                  # Generated plots (feature importance, confusion matrix)
├── reports/                  # Markdown and HTML analysis reports
├── dashboard_config.py       # Configuration for dashboard (colors, mappings, etc.)
├── dashboard_utils.py        # Utility functions for data loading and visualization
├── data_preparation.py       # Data preprocessing and feature engineering
├── generate_report.py        # Auto-generate analysis reports in Markdown and HTML
├── optimized_sleep_model.py  # Model training, tuning, evaluation pipeline
├── sleep_dashboard.py        # Streamlit app for the dashboard
├── requirements.txt          # Python dependencies
└── README.md                 # This file

🚀 How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Data Preparation
Ensure your dataset is placed inside the data/ directory. Expected input:

data/P2.4_Military_Responses.csv
3️⃣ Train the Model
python optimized_sleep_model.py
Generates:

Trained model models/optimized_sleep_model.pkl

Evaluation plots results/

4️⃣ Generate Analysis Report (Optional)
python generate_report.py
Output:
Markdown report reports/
HTML version for easy sharing

5️⃣ Launch the Dashboard
streamlit run sleep_dashboard.py
Access via:
http://localhost:8501
🛠️ Technologies Used
Python 3.10+
Streamlit
Scikit-learn
Pandas / NumPy
Seaborn / Matplotlib / Plotly
Imbalanced-learn (SMOTE)

📊 Key Features
Analyze sleep quality survey data
Predict sleep quality using Random Forest (optimized)
Visualize key contributing factors
Interactive user interface (Streamlit)
Auto-generated reports in Markdown/HTML
Supports both simple and advanced prediction modes

⚠️ Disclaimer
This project is for educational and research purposes only. It is not clinically validated for medical decision-making.

