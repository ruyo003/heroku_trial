from flask import Flask, request, jsonify
import pickle
import numpy as np
import os


multimodel_path = os.path.join(os.path.dirname(__file__), 'multi_model.pkl')


# deserilization
with open(multimodel_path, 'rb') as multimodel_file:
    multi_model = pickle.load(multimodel_file)

# creating the flask application
in_app = Flask(__name__)

# defining the prediction endpoints
@in_app.route('/predict', methods = ['POST'])

def predictin():
    data = request.get_json(force = True)
    prediction = multi_model.predict(np.array(data['features']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    in_app.run(debug = True)

