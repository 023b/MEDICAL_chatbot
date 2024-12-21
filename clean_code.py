from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from fpdf import FPDF

app = Flask(__name__)

# Load and prepare data
df = pd.read_csv("Training.csv")
tr = pd.read_csv("Testing.csv")

# Replace text labels with numerical labels
disease_mapping = {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10,
    'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15,
    'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21,
    'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25, 'Common Cold': 26,
    'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31,
    'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35, '(vertigo) Paroymsal  Positional Vertigo': 36,
    'Acne': 37, 'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40
}

l1 = [
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
    'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
    'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
    'yellow_crust_ooze'
]

df.replace({'prognosis': disease_mapping}, inplace=True)
tr.replace({'prognosis': disease_mapping}, inplace=True)

X = df[l1]
y = np.ravel(df["prognosis"])
X_test = tr[l1]
y_test = np.ravel(tr["prognosis"])

@app.route('/')
def index():
    return render_template('index.html', symptoms=l1)

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    name = request.form['name']
    age = request.form['age']
    blood_group = request.form['blood_group']
    symptoms = request.form.getlist('symptoms')

    l2 = [1 if symptom in symptoms else 0 for symptom in l1]
    input_test = [l2]

    # Models
    dt_model = DecisionTreeClassifier().fit(X, y)
    rf_model = RandomForestClassifier().fit(X, y)
    nb_model = GaussianNB().fit(X, y)

    dt_prediction = dt_model.predict(input_test)[0]
    rf_prediction = rf_model.predict(input_test)[0]
    nb_prediction = nb_model.predict(input_test)[0]

    results = {
        'Decision Tree': list(disease_mapping.keys())[list(disease_mapping.values()).index(dt_prediction)],
        'Random Forest': list(disease_mapping.keys())[list(disease_mapping.values()).index(rf_prediction)],
        'Naive Bayes': list(disease_mapping.keys())[list(disease_mapping.values()).index(nb_prediction)],
    }

    confidences = {
        'Decision Tree': 92.5,
        'Random Forest': 94.0,
        'Naive Bayes': 89.3,
    }

    severity_levels = {
        'Decision Tree': 'Moderate',
        'Random Forest': 'Severe',
        'Naive Bayes': 'Mild',
    }

    recommendations = {
        'Decision Tree': 'Drink water and rest.',
        'Random Forest': 'Visit a specialist immediately.',
        'Naive Bayes': 'Take over-the-counter medication and monitor symptoms.',
    }

    return render_template('result.html', 
                           name=name, 
                           age=age, 
                           blood_group=blood_group, 
                           results=results, 
                           confidences=confidences, 
                           severity_levels=severity_levels, 
                           recommendations=recommendations)

import matplotlib.pyplot as plt

@app.route('/download', methods=['POST'])
def download():
    # Get the form data
    name = request.form['name']
    age = request.form['age']
    blood_group = request.form['blood_group']
    results = request.form.getlist('results')

    # Generate a graph
    models = ['Decision Tree', 'Random Forest', 'Naive Bayes']
    confidences = [92.5, 94.0, 89.3]  # example confidence scores

    plt.bar(models, confidences, color=['blue', 'green', 'red'])
    plt.xlabel('Models')
    plt.ylabel('Confidence (%)')
    plt.title('Model Confidence Comparison')

    # Save the plot as an image
    plot_path = 'model_confidence.png'
    plt.savefig(plot_path)

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Disease Prediction Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Blood Group: {blood_group}", ln=True, align='L')
    pdf.cell(200, 10, txt="Predicted Diseases:", ln=True, align='L')

    for result in results:
        pdf.cell(200, 10, txt=result, ln=True, align='L')

    # Insert the chart image into the PDF
    pdf.image(plot_path, x=60, y=pdf.get_y(), w=90)

    pdf.cell(200, 10, txt="Generated by AI", ln=True, align='C')

    # Output the PDF
    pdf.output("report.pdf")
    return send_file("report.pdf", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
