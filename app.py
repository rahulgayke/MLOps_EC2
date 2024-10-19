from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
# from wtforms.validators import NumberRange
# import numpy as np 
import joblib

def return_prediction_rf(rf_classifier, scaler_x, scaler_y, sample_json):
    # Fields for input json
    Fixed_Acidity = sample_json['Fixed_Acidity']
    Volatile_Acidity = sample_json['Volatile_Acidity']
    Citric_Acid = sample_json['Citric_Acid']
    Residual_Sugar = sample_json['Residual_Sugar']
    Total_Sulfur_Dioxide = sample_json['Total_Sulfur_Dioxide']
    pH = sample_json['pH']
    Alcohol = sample_json['Alcohol']
    
    sample = [[Fixed_Acidity, Volatile_Acidity, Citric_Acid, Residual_Sugar, Total_Sulfur_Dioxide, pH, Alcohol]]
    sample = scaler_x.transform(sample)
    
    wine_class = rf_classifier.predict(sample)
    wine_class = scaler_y.inverse_transform(wine_class.reshape(-1, 1))
    
    return wine_class

app = Flask(__name__)

# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'mysecretkey'

# LOAD THE MODEL AND THE SCALERS
rf_classifier = joblib.load("./model/random_forest_model.joblib")
scaler_x = joblib.load("./model/scaler_x.joblib")
scaler_y = joblib.load("./model/scaler_y.joblib")
# C:\Rahul Data\IMP Data\BITS\Semister 3\MLOPS\Assignment 2\MLOPS_Assignment-2_Grp_69
class FlowerForm(FlaskForm):

    Fixed_Acidity = StringField('Fixed Acidity')
    Volatile_Acidity = StringField('Volatile Acidity')
    Citric_Acid = StringField('Citric Acid')
    Residual_Sugar = StringField('Residual Sugar')
    Total_Sulfur_Dioxide = StringField('Total Sulfur Dioxide')
    pH = StringField('pH')
    Alcohol = StringField('Alcohol')

    submit = SubmitField('Predicted Wine Class')


@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = FlowerForm()
    # If the form is valid on submission
    if form.validate_on_submit():
        # Grab the data from the form

        session['Fixed_Acidity'] = form.Fixed_Acidity.data
        session['Volatile_Acidity'] = form.Volatile_Acidity.data
        session['Citric_Acid'] = form.Citric_Acid.data
        session['Residual_Sugar'] = form.Residual_Sugar.data
        session['Total_Sulfur_Dioxide'] = form.Total_Sulfur_Dioxide.data
        session['pH'] = form.pH.data
        session['Alcohol'] = form.Alcohol.data

        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['Fixed_Acidity'] = float(session['Fixed_Acidity'])
    content['Volatile_Acidity'] = float(session['Volatile_Acidity'])
    content['Citric_Acid'] = float(session['Citric_Acid'])
    content['Residual_Sugar'] = float(session['Residual_Sugar'])
    content['Total_Sulfur_Dioxide'] = float(session['Total_Sulfur_Dioxide'])
    content['pH'] = float(session['pH'])
    content['Alcohol'] = float(session['Alcohol'])

    ### Use only one model at a time
    results = return_prediction_rf(rf_classifier=rf_classifier, scaler_x=scaler_x, scaler_y=scaler_y, sample_json=content)

    if int(results[0][0]) == 0:
        result = "LOW"
    else:
        result = "HIGH"
    print(f"results : {result}")

    return render_template('prediction.html', results=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)