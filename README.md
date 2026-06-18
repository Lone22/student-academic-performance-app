# Student Performance Predictor

This project is an AI/ML-based web application that predicts a student's final academic grade using features such as study hours, attendance, assignments completed, previous semester marks, and class participation.

## Project Overview
- The application uses a synthetic dataset with 500 records.
- The workflow includes data cleaning, missing value handling, duplicate removal, visualization, model comparison, and prediction.
- The final web app is built using Flask and a simple HTML/CSS frontend.

## Features
- Data preprocessing and exploratory visualization
- Comparison of multiple regression models
- Prediction API for academic performance
- Simple web interface for users

## Project Files
- `app.py` - Flask web application and prediction endpoint
- `train_model.py` - Data preprocessing, model comparison, and model saving
- `data/generate_dataset.py` - Script to generate the synthetic dataset
- `data/student_performance.csv` - Generated dataset used for training
- `templates/index.html` - Frontend form for user inputs
- `static/style.css` - Styling for the web page
- `requirements.txt` - Required Python packages
- `model.pkl` - Saved trained model
- `student_performance_app.ipynb` - Notebook for ML workflow demonstration

## Data Preprocessing & Modeling
- Missing values are handled using mean/mode-based filling.
- Duplicate records are removed.
- Feature correlation and grade distribution are visualized.
- Two regression models are compared:
  - Linear Regression
  - Random Forest Regressor
- The best-performing model is saved for use in the web app.

## Setup Instructions
1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate the dataset:
   ```bash
   python data/generate_dataset.py
   ```
4. Train and compare models:
   ```bash
   python train_model.py
   ```
5. Run the web application:
   ```bash
   python app.py
   ```
6. Open the browser at:
   ```text
   http://127.0.0.1:5000
   ```

## How to Use
- Enter the required student details in the form.
- Click **Predict Performance**.
- The app will display the predicted final grade and performance category.

## Submission Notes
- ZIP the full project folder (excluding unnecessary virtual environment folders if needed).
- Include the source code, dataset, notebook, and README.
- Make sure the `model.pkl` file is present if you want the trained model included in the submission.

## Notes
- The dataset is synthetic and designed for academic demonstration purposes.
- The model output is a predicted grade score and a category such as Good, Average, or Needs Improvement.
