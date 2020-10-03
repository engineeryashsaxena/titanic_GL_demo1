import pandas as pd
import numpy as np 
import pickle
from sklearn.preprocessing import LabelEncoder
import json
from sklearn.ensemble import RandomForestClassifier
import datetime 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os 

def Save_Encoding_Gender(df):
    le_gender=LabelEncoder()
    le_gender.fit(df['Sex'])
    le_gender_filename = model_artifacts_dir+'Column_Sex_LabelEncoder.sav'
    pickle.dump(le_gender, open(le_gender_filename, 'wb'))
    print( "Encoding object saved for column : Sex as : "+le_gender_filename)

def Save_NA_imputations(data):
    with open(model_artifacts_dir+'NA_imputations.json', 'w') as fp:
        json.dump(data, fp)
    print( "Saved NA imputations as JSON :"+'NA_imputations.json')

def Save_cols_to_select(cols_to_select):
    data={"cols_to_select":cols_to_select}
    with open(model_artifacts_dir+'Cols_For_Model.json', 'w') as fp:
        json.dump(data, fp)
    print ("Saved Cols to Select as JSON  : "+'Cols_For_Model.json')

def Load_Encoding_Gender():
    le_gender_filename = 'Column_Sex_LabelEncoder.sav'
    le_gender = pickle.load(open(model_artifacts_dir+le_gender_filename, 'rb'))
    return le_gender 

def Load_NA_imputations():
    with open(model_artifacts_dir+'NA_imputations.json', 'r') as fp:
        data = json.load(fp)
    return data

def Load_cols_to_select():
    with open(model_artifacts_dir+'Cols_For_Model.json', 'r') as fp:
        data = json.load(fp)
    return data

def Transformations(df):
    #LE Encoding Sex column
    le_gender=Load_Encoding_Gender()
    df['Sex']=le_gender.transform(df['Sex'])
    
    #NA imputations
    na_imputations=Load_NA_imputations()
    for col_name in na_imputations.keys():
        value=na_imputations[col_name]
        df[col_name]=df[col_name].fillna(value)
    
    #Cols to select 
    cols_to_select=Load_cols_to_select()['cols_to_select']
    print ("Transformations done ")
    return df[cols_to_select]



def FitModel(features,target):
    rcf=RandomForestClassifier(n_estimators=20,max_depth=3)
    rf_model=rcf.fit(features,target)
    model_filename = model_artifacts_dir+'rf_model.sav'
    pickle.dump(rf_model, open(model_filename, 'wb'))
    print( "Model saved as : "+model_filename)
    
def GetPrediction(features):
    model_filename = model_artifacts_dir+'rf_model.sav'
    model = pickle.load(open(model_filename, 'rb'))
    pred=model.predict(features)
    pred_json={'predictions':list(map(str,pred)),
              "prediction_time":str(datetime.datetime.now())}
    with open(model_artifacts_dir+'Predictions.json', 'w') as fp:
        json.dump(pred_json, fp)
    print( "Predictions saved at : "+'Predictions.json')

def LoadPredictions():
    with open(model_artifacts_dir+'Predictions.json', 'r') as fp:
        data = json.load(fp)
    return data

def Evaluate(pred,org):
    pred=list(map(int,LoadPredictions()['predictions']))
    score=accuracy_score(org,pred)
    print( "Accuracy Score:"+str(score))
    return score


def Training(train):
    
    #Saving transformation details 
    Save_Encoding_Gender(train)
    NA_imputations={"Age":train['Age'].median()}
    Save_NA_imputations(NA_imputations)
    cols_to_consider=['Pclass','Sex','Age']
    Save_cols_to_select(cols_to_consider)
    
    #Tranformation 
    features=Transformations(train)
    target=train['Survived']
    
    #Fitting model 
    FitModel(features,target)
    print ("Training done ")
    
def Inference(test,return_pred=False):
    
    #Tranformation 
    features=Transformations(test)
    GetPrediction(features)
    if return_pred:
        print (LoadPredictions())
    return LoadPredictions()


model_artifacts_dir="artifacts/"
if not os.path.exists(model_artifacts_dir):
    os.mkdir(model_artifacts_dir)

