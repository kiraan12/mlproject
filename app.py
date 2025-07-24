from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction form
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # Ensure home.html exists
    else:
        try:

            reading_score = float(request.form.get('reading_score'))
            writing_score = float(request.form.get('writing_score'))

            # Create CustomData object from form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),  # ✅ Use same name as HTML
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=reading_score,
                writing_score=writing_score
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            print("Input DataFrame:")
            print(pred_df)

            # Prediction pipeline
            pipeline = PredictPipeline()
            prediction = pipeline.predict(pred_df)

            print("Prediction result:", prediction[0])

            return render_template('home.html', results=round(prediction[0], 2))

        except Exception as e:
            return f"Prediction failed: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)  # Add debug=True for better error messages
