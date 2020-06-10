# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:04:55 2020

@author: VINOTHKUMAR
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:56:37 2020

@author: CHANDRUSAEKAR
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import pickle

# =============================================================================
# import numpy
# =============================================================================


app=Flask(__name__)

file_Name = "Cancer.pkl"
fileObject = open(file_Name,'rb')  
# load the object from the file into var b
model = pickle.load(fileObject)  


@app.route('/')
def welcome():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])

def predict():
       
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    if str(output)== '0':
        prediction_text = 'No Cancer'
    else :
        prediction_text = 'Go for Trearment - Cancer =YES'
    
    return render_template('index1.html', prediction_text=prediction_text)
    
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

#port = int(os.getenv("PORT"))
if __name__ == "__main__":
	app.run()

