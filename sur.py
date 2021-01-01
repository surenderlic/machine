import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask,render_template,request
import pickle

df= pd.read_csv("cancer.csv")
#print(df.info())
df1 = df.drop(['id','diagnosis'],axis=1)
#print(df1.head())
y = df.diagnosis
y1=y[:].str.replace('B','BENIGN,NOT CANCER')
y2=y1[:].str.replace('M','MALIGNANT IS CANCER')
#print(y.head())
x = df1[['radius_mean','texture_mean','perimeter_mean','area_mean']]
#print(x.head())
lr = LogisticRegression()
lr.fit(x,y2)
y_predict=lr.predict(x)
acc=accuracy_score(y_predict,y2)
#print(round(acc,2))
#lr_predict=lr.predict([[7.76,24.54,47.92,181.0]])[0]
#print(lr_predict)
with open('cancer.pkl','wb') as f :
   pickle.dump(lr,f)

lr_model=pickle.load(open('cancer.pkl','rb'))
#a=lr_model.predict([[6.7,3.3,5.7,2.1]])[0]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        rm = request.form['rm']
        tm = request.form['tm']
        pm = request.form['pm']
        am = request.form['am']
        data = [[float(rm), float(tm), float(pm), float(am)]]
        lr = pickle.load(open('cancer.pkl','rb'))
        prediction = lr.predict(data)[0]
        
    return render_template('home.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

