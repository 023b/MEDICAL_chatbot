# Disease Prediction System

A machine learning-based web application that predicts diseases based on symptoms provided by users.

## Overview

This application uses multiple machine learning algorithms (Decision Tree, Random Forest, and Naive Bayes) to predict potential diseases based on symptoms. The system provides confidence scores, severity levels, and recommendations for each prediction.

## Features

- User-friendly web interface for symptom selection
- Multiple ML models for more reliable predictions
- Comparative analysis of prediction results
- Downloadable PDF report with visualization
- Patient information tracking

## Technologies Used

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **PDF Generation**: FPDF
- **Frontend**: HTML, CSS (assumed)

## Dataset

The system uses two datasets:
- `Training.csv`: Used to train the machine learning models
- `Testing.csv`: Used to evaluate the models' performance

The datasets contain symptom information mapped to various diseases.

## Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/disease-prediction.git
   cd disease-prediction
   ```

2. Create a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Run the application
   ```
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000/`

## Requirements

Create a `requirements.txt` file with the following dependencies:
```
flask==2.0.1
numpy==1.21.2
pandas==1.3.3
scikit-learn==1.0
matplotlib==3.4.3
fpdf==1.7.2
```

## Usage

1. Enter patient details (name, age, blood group)
2. Select symptoms from the provided checklist
3. Submit the form to get disease predictions
4. View results with confidence scores and recommendations
5. Download a PDF report of the predictions

## Project Structure

```
disease-prediction/
│
├── app.py                  # Main Flask application
├── Training.csv            # Training dataset
├── Testing.csv             # Testing dataset
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
│
├── static/                 # Static files
│   ├── css/                # CSS stylesheets
│   └── js/                 # JavaScript files
│
└── templates/              # HTML templates
    ├── index.html          # Home page with symptom selection
    └── result.html         # Results page showing predictions
```

## How It Works

1. **Data Preparation**: The system loads and processes the disease datasets.
2. **Model Training**: Three ML algorithms are trained on the dataset.
3. **User Input**: The user selects symptoms through the web interface.
4. **Prediction**: The application runs the input through all three models.
5. **Results**: Predictions from all models are displayed with confidence scores.
6. **Report Generation**: A PDF report with visualizations can be downloaded.

## Machine Learning Models

1. **Decision Tree Classifier**: A tree-based model that makes decisions based on symptom patterns.
2. **Random Forest Classifier**: An ensemble of decision trees for more robust predictions.
3. **Naive Bayes Classifier**: A probabilistic model based on Bayes' theorem.

## Future Improvements

- Add more machine learning models for higher accuracy
- Implement user accounts for tracking medical history
- Add more detailed recommendations and information about diseases
- Create a mobile application for easier access
- Improve the visualization of results
- Add multilingual support

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- [Your Name](https://github.com/yourusername)

## Acknowledgments

- The dataset providers
- The scikit-learn development team
- Flask and Python community
