# Student Performance Predictor

This project is a simple AI/ML-based web application that predicts a student’s final academic grade based on features such as study hours, attendance, assignments completed, previous semester marks, and class participation.

## Features
- Synthetic dataset with 500+ records
- Data preprocessing and regression model training using scikit-learn
- Flask backend serving a prediction API
- Web frontend for user input and prediction display

## Files
- `app.py` - Flask application and prediction endpoint
- `train_model.py` - Model training and evaluation script
- `data/generate_dataset.py` - Generates a synthetic student performance dataset
- `data/student_performance.csv` - Generated dataset
- `templates/index.html` - Web frontend
- `static/style.css` - Frontend styling
- `requirements.txt` - Python dependencies

## Setup
1. Create a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate the dataset:
   ```bash
   python data/generate_dataset.py
   ```
4. Train the model:
   ```bash
   python train_model.py
   ```
5. Run the web application:
   ```bash
   python app.py
   ```
6. Open `http://127.0.0.1:5000` in your browser.

## Usage
Enter the student features on the web form and click `Predict Performance`. The app returns a predicted final grade and a performance category.

## Notes
- The dataset is synthetic and designed to demonstrate model training, evaluation, and frontend integration.
- The model predicts the final grade as a numeric score, with categories derived from grade ranges.
