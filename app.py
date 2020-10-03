import numpy as np
import os
from flask import Flask, request, jsonify, render_template
from model import * 


app = Flask(__name__)

#training the model when app starts 
#if not os.path.exists('artifacts/rf_model.sav'):
#    df=pd.read_csv('titanic.csv')
#    Training(df)

#training the model when app starts 
df=pd.read_csv('titanic.csv')
Training(df)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    age,pclass,sex = [x for x in request.form.values()]
    age=int(age)
    pclass=int(pclass)
    sex=str(sex)
    
    test_data=pd.DataFrame({"Age":[age],
                           "Pclass":[pclass],
                           "Sex":[sex]})
    
    prediction=Inference(test_data)
    output = prediction['predictions'][0]

    return render_template('index.html', prediction_text='Predicted Label: {}'.format(output))

## Create a directory in a known location to save files to.
#uploads_dir = os.path.join(app.instance_path, 'artifacts')  
#@app.route('/train', methods=['GET', 'POST'])
#def train():
#    if request.method == 'POST':
#        # save the single "profile" file
#        profile = request.files['train_file']
#        profile.save(os.path.join(uploads_dir, secure_filename(profile.filename)))
#        output="Successfully Uploaded"
#        return render_template('train.html', status='Predicted Label: {}'.format(output))
#    return render_template('train.html')

if __name__ == "__main__":
    app.run(debug=True)