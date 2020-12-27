import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from flask import Flask,render_template,request

df = pd.read_csv("iris.csv")
df.head()
#print(df.keys())
x = df.drop(['variety'],axis=1)
y = df['variety']

lr = LogisticRegression(class_weight='balanced')
lr.fit(x,y)

y_predict=lr.predict(x)
#a=lr.predict([[6.7,3.3,5.7,2.1]])[0]
#print(a)

acc = accuracy_score(y,y_predict)
#print(round(acc,2))

with open('iris.pkl','wb') as f :
   pickle.dump(lr,f)

lr_model=pickle.load(open('iris.pkl','rb'))
a=lr_model.predict([[6.7,3.3,5.7,2.1]])[0]
#print(a)



app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   if request.method == 'POST':
      spl = request.form['spl']
      spw = request.form['spw']
      ptl = request.form['ptl']
      ptw = request.form['ptw']

      data =  [[float(spl),float(spw),float(ptl),float(ptw)]]

      lr = pickle.load(open('iris.pkl','rb'))
      prediction=lr.predict(data)[0]
   return render_template('index.html',prediction=prediction)

if __name__ == '__main__':
   app.run(debug=True)




