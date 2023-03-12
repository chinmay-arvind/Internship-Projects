# Source code adapted from: https://towardsdatascience.com/deploy-a-machine-learning-model-using-flask-da580f84e60c
# Importing necessary libraries
from flask import Flask, request, jsonify
import pickle
import tensorflow as tf

# Creating app
app = Flask(__name__)

# Loading the model from the pickle file that it was saved in
with open('mnist_predictive_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Routes to predict when called and returns response to the client
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json(force=True)

    # Convert the data into a numpy array
    predict_input = tf.convert_to_tensor([data['input']])

    # Make a prediction
    prediction = model.predict(predict_input)

    # Return the prediction as a JSON response
    response = {'prediction': int(prediction[0][0])}
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=8080)