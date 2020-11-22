import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        spl = request.form['sepallength']
        spw = request.form['sepalwidth']
        ptl = request.form['petallength']
        ptw = request.form['petalwidth']

        data = [[float(spl), float(spw), float(ptl), float(ptw)]]

        lr = pickle.load(open('iris.pkl', 'rb'))
        prediction = lr.predict(data)[0]

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)

df = pd.read_csv('Iris_csv.csv')


print(df.head())
#print(df.isna().sum())

x = df.drop(['class'], axis=1)
y = df['class']

lr = LogisticRegression(class_weight='balanced')
lr.fit(x, y)
y_pred = lr.predict(x)
print(y_pred)
acc = accuracy_score(y, y_pred)
print('Accuracy:', round(acc, 2) * 100)

lr.predict([[5.1,3.5,1.4,0.2]])[0]
print(x)

with open('iris.pkl', 'wb') as f:
    pickle.dump(lr, f)

lr_model = pickle.load(open('iris.pkl', 'rb'))

x=lr_model.predict([[5.1, 3.5, 1.4, 0.2]])[0]
print('result:',x)
