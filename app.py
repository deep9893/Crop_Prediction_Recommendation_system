from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model =  pickle.load(open('rfc.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input data from the form
    input_data = [float(request.form['N']), float(request.form['P']), float(request.form['K']),
                  float(request.form['temperature']), float(request.form['humidity']),
                  float(request.form['ph']), float(request.form['rainfall'])]

    # Convert the input data to a NumPy array
    input_array = np.array(input_data).reshape(1, -1)

    # Make predictions using the loaded model
    prediction = model.predict(input_array)

    # Process the prediction or return it to the user
    return f"The predicted output is: {prediction[0]}"

if __name__ == '__main__':
    app.run(debug=True)
