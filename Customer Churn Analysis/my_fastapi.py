# Description: This file is used to test the model. It loads the model and the scaler and then uses them to predict the output.

#library imports
import os 
import uvicorn # for running the app
from fastapi import FastAPI # for creating the app
import tensorflow as tf # for loading the model
import numpy as np # for array manipulation
import pickle #for loading the scaler pickle file
import warnings  #filter warnings
warnings.filterwarnings('ignore') 




#encoding for credit score
credict_score_dict = {'A':720,'B':690,'C':660,'D':630,'E':600,'F':570,'G':540}

#ecoding for payment history
payment = {'N':0,'Y':1}

#encoding for home ownership and loan purposes
target_encoder = {'home_ownership': {'MORTGAGE': 0.12570663493008033,
  'OWN': 0.07469040247678019,
  'other_rent': 0.31565275176705127},
 'loan_purposes': {'DEBTCONSOLIDATION': 0.28587874136607827,
  'EDUCATION': 0.17216798388346505,
  'MED_HOME': 0.26477883422902027,
  'PERSONAL': 0.1988770150335084,
  'VENTURE': 0.1481028151774786}}


#loading the scaler pickle file (standart_scale _transform i used)
with open("scale.pkl", "rb") as f:
   scale_transform = pickle.load(f)

#loading the model
tf_model = tf.keras.models.load_model('model_weights.h5')

#creating the app
app = FastAPI()

@app.get('/')
def func(ad):
    return {'hell':'hell'}

class Transformations:
  def __init__(self,salary ,home_ownership,employment_time,loan_purposes,credit_score,Credit_Amount,loan_rate,loan_percentage ,Payment_History,Credit_History_Length) -> None:
      self.salary = salary
      self.home_ownership = home_ownership
      self.employment_time = employment_time
      self.loan_purposes = loan_purposes
      self.credit_score = credit_score
      self.Credit_Amount = Credit_Amount
      self.loan_rate = loan_rate
      self.loan_percentage = loan_percentage
      self.Payment_History = Payment_History
      self.Credit_History_Length = Credit_History_Length
  def transform(self):
    if self.home_ownership in ['RENT','OTHER']:
      self.home_ownership = 'other_rent'
    if self.loan_purposes in ['MEDICAL','HOMEIMPROVEMENT']:
      self.loan_purposes =  'MED_HOME'
    self.salary = np.log(self.salary)
    self.credit_score = credict_score_dict[self.credit_score] 
    self.Payment_History = payment[self.Payment_History]
    self.home_ownership = target_encoder['home_ownership'][self.home_ownership]
    self.loan_purposes = target_encoder['loan_purposes'][self.loan_purposes]
    data =  scale_transform.transform( np.array([
    self.salary ,self.home_ownership,self.employment_time,self.loan_purposes,
    self.credit_score,self.Credit_Amount,self.loan_rate,self.loan_percentage ,
    self.Payment_History , self.Credit_History_Length]).reshape(1,-1))
    return data




@app.post("/predict")
async def predict(data:dict):
    # preditions = np.array( ts)
    ts = Transformations( float( data['salary']),
                         data['home_ownership'],float(data['employment_time']) , 
                         data['loan_purposes'],data['credit_score'],float(data['Credit_Amount']),float(data['loan_rate'] ),
                         float(data['loan_percentage']),data['Payment_History'],float(data['Credit_History_Length'])).transform()
    preditions = np.array(ts)
    pred = np.where(tf_model.predict(preditions,verbose=0)[0][0] >0.4 ,1,0 )
    return   int(pred)


# it is for running the app and in local host , change host and port for your own use
uvicorn.run(app, host= '127.0.0.1', port= 8000)


#for testing  
# test_1 = {"salary": 67000,
#  "home_ownership": "MORTGAGE",
#  "employment_time": 2.0,
#  "loan_purposes": "HOMEIMPROVEMENT",
#  "credit_score": "B",
#  "Credit_Amount": 12150,
#  "loan_rate": 10.37,
#  "loan_percentage": 0.18,
#  "Payment_History": "N",
#  "Credit_History_Length": 3}
