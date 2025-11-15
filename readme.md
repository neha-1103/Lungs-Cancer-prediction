<b><h1>Lung Cancer Prediction using Machine Learning<h1><b>

<p>This project predicts whether a person is likely to have lung cancer based on health and lifestyle indicators such as age, smoking habits, and respiratory symptoms.
It uses a Random Forest Classifier trained on the Survey Lung Cancer Dataset and provides a simple Flask web interface for real-time predictions.<p>

<b>Project Structure <b>
├── app.py                     # Flask web application
├── lung_cancer_classifier.py  # Model training and evaluation script
├── lung_cancer_model.pkl      # Saved trained model
├── survey lung cancer.csv     # Dataset used for training
├── templates/
│   └── index.html             # Frontend HTML form

<b><h2>Model Training (lung_cancer_classifier.py)<h2><b>

Algorithm: Random Forest Classifier

Dataset: survey lung cancer.csv

Target Variable: LUNG_CANCER

Features: Gender, Age, Smoking, Yellow Fingers, Anxiety, Peer Pressure, Chronic Disease, Fatigue, Allergy, Wheezing, Alcohol Consumption, Coughing, Shortness of Breath, Swallowing Difficulty, Chest Pain

<h3>Steps Performed:<h3>

1.Load and preprocess dataset (GENDER and LUNG_CANCER converted to numeric values).

2.Split dataset into training and testing sets (80/20 split).

3.Train a RandomForestClassifier.

4.Save the trained model as lung_cancer_model.pkl.

Display model performance (accuracy and classification report).

<h3>Web Application (app.py)<h3>

Framework: Flask

Model Used: lung_cancer_model.pkl

Input Method: Web form (index.html)

Output: Displays whether the person is likely suffering from lung cancer or not.

<h3>Workflow:<h3>

User submits the form with health details.

Flask collects inputs and converts them into a numerical array.

The model predicts the result (YES or NO).

The prediction message is displayed on the webpage.

<h3>Dependencies<h3>

Install all necessary packages before running the project:

flask 
numpy 
pandas 
scikit-learn 
joblib

<b>Run the training script to generate the model file.<b>

<b>python lung_cancer_classifier.py<b>

This will create a file named lung_cancer_model.pkl.

<h3>Launch the Web App<h3>
Start the Flask server:
python app.py


Open your browser and go to:

http://127.0.0.1:5000/

<h3>Use the Application<h3>

Fill out the form with patient details and submit to get the prediction.

