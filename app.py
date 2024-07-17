import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Retrieve data from the request
        data = request.json.get('data')
        
        # Log received data
        print("Received data:", data)
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Convert data to numpy array and reshape
        new_data = np.array(list(data.values())).reshape(1, -1)
        
        # Log the shape and content of new_data
        print("New data shape:", new_data.shape)
        print("New data content:", new_data)
        
        # Perform prediction
        output = regmodel.predict(new_data)
        
        # Log the prediction output
        print("Prediction output:", output)
        
        # Return prediction result as JSON
        return jsonify({"prediction": output.tolist()})
    
    except Exception as e:
        # Log the exception
        print("Exception occurred:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    print(final_input)
    output=regmodel.predict(final_input)
    return render_template("home.html",prediction_text="The House Price Prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)