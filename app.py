import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the model
regmodel= pickle.load(open('regmodel.pkl', 'rb'))
scaler= pickle.load(open('scaler.pkl','rb'))

#this will be the home page of the web-app
@app.route('/')
def home():
    return render_template('home.html')

#following function will be called when the user hits the predict button
# @app.route('/predict', methods=['POST'])
# def predict_api():
#    data= request.get_json()['data']
#    print(data)
#    print(np.array(list(data.values())).reshape(1,-1))
#    new_data= scaler.transform(np.array(list(data.values())).reshape(1,-1))
#    output= regmodel.predict(new_data)
#    print(output[0])
#    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
   data= [float(x) for x in request.form.values()]
   new_data= np.array(data).reshape(1,-1)
   print(new_data)
   final_input = scaler.transform(new_data)

   output= regmodel.predict(final_input)[0]
   return render_template('home.html', prediction_text='Predicted House Price: {}'.format(output))
   

if __name__ == '__main__':
   app.run(debug=True)