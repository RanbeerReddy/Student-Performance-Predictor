# Student Exam Performance Prediction

A machine learning web application that predicts student mathematics exam scores based on various demographic and academic factors. Built using Python, Flask, and scikit-learn with a focus on end-to-end machine learning pipeline implementation.

## 🚀 Features

- **Web-based Prediction Interface**: User-friendly web application for real-time predictions
- **Comprehensive ML Pipeline**: Complete machine learning workflow from data ingestion to model deployment
- **Multiple Model Evaluation**: Comparison of various regression algorithms including Linear Regression, Random Forest, XGBoost, and CatBoost
- **Data Preprocessing**: Automated feature engineering and scaling
- **Model Persistence**: Trained models saved for production use

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, XGBoost, CatBoost
- **Data Processing**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML, CSS (Bootstrap)
- **Model Serialization**: pickle, dill

## 📊 Dataset

The project uses the "Students Performance in Exams" dataset containing:
- Student demographics (gender, race/ethnicity, parental education)
- Academic factors (lunch type, test preparation course)
- Exam scores (reading, writing) as input features
- Mathematics score as the target variable

## 🏗️ Project Structure

```
├── app.py                    # Main Flask application
├── application.py            # Alternative Flask app entry point
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup configuration
├── artifacts/               # Trained models and preprocessors
│   ├── model_trainer.pkl
│   ├── preprocessor.pkl
│   └── *.csv
├── src/                     # Source code
│   ├── __init__.py
│   ├── exception.py         # Custom exception handling
│   ├── logger.py           # Logging configuration
│   ├── utils.py            # Utility functions
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   └── pipeline/
│       ├── predict_pipeline.py
│       └── train_pipeline.py
├── templates/               # HTML templates
│   ├── index.html
│   └── home.html
├── notebook/                # Jupyter notebooks for EDA and training
└── catboost_info/           # CatBoost training logs
```

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ML-1-project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web app**
   Open your browser and navigate to `http://localhost:5000`

## 📈 Model Performance

The application uses Linear Regression as the final model, achieving:
- **R² Score**: ~88% on test data
- **Mean Absolute Error**: Low prediction error
- **Mean Squared Error**: Minimal variance in predictions

## 🎯 Usage

1. Navigate to the prediction page
2. Fill in the student information:
   - Gender
   - Race/Ethnicity
   - Parental Level of Education
   - Lunch Type
   - Test Preparation Course
   - Reading Score (0-100)
   - Writing Score (0-100)
3. Click "Predict" to get the mathematics score prediction

## 🤖 Machine Learning Pipeline

### Data Ingestion
- Loads and validates the dataset
- Performs train-test split

### Data Transformation
- Handles categorical features with One-Hot Encoding
- Scales numerical features using StandardScaler
- Manages missing values with appropriate imputation

### Model Training
- Evaluates multiple regression algorithms
- Selects the best performing model based on R² score
- Saves the trained model and preprocessor for inference

### Prediction Pipeline
- Loads saved model and preprocessor
- Transforms input data
- Generates predictions in real-time

## 📝 Key Learnings

- End-to-end ML project implementation
- Flask web application development
- Model serialization and deployment
- Feature engineering and preprocessing
- Model evaluation and selection
- Production-ready code structure

## 🔍 Future Enhancements

- [ ] Add more advanced models (Neural Networks)
- [ ] Implement model monitoring and retraining
- [ ] Add API endpoints for programmatic access
- [ ] Improve UI/UX with modern frontend framework
- [ ] Add model explainability features
- [ ] Deploy to cloud platform (AWS/Heroku)


## 👤 Author

Rabeer Reddy
- LinkedIn: https://www.linkedin.com/in/ranbeerreddy-mukpogle-528b48290/
- Email: reddyranbeer@gmail.com

---

*This project demonstrates practical application of machine learning concepts in a real-world scenario, showcasing the complete lifecycle from data to deployment.*