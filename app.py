from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('lung_cancer_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get form data
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        smoking = int(request.form['smoking'])
        yellow_fingers = int(request.form['yellow_fingers'])
        anxiety = int(request.form['anxiety'])
        peer_pressure = int(request.form['peer_pressure'])
        chronic_disease = int(request.form['chronic_disease'])
        fatigue = int(request.form['fatigue'])
        allergy = int(request.form['allergy'])
        wheezing = int(request.form['wheezing'])
        alcohol_consuming = int(request.form['alcohol_consuming'])
        coughing = int(request.form['coughing'])
        shortness_of_breath = int(request.form['shortness_of_breath'])
        swallowing_difficulty = int(request.form['swallowing_difficulty'])
        chest_pain = int(request.form['chest_pain'])

        # Prepare input for model
        features = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                              chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                              coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])

        # Predict
        pred = model.predict(features)[0]
        if pred == 1:
            prediction = "The person is suffering from lung cancer. Thank you for visiting our website"
        else:
            prediction = "The person is not suffering from lung cancer. Thank you for visiting our website."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
