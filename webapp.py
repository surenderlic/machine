import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('Iris.csv')

# df.head()

# df.isna().sum()

x = df.drop(['variety'], axis=1)
y = df['variety']

lr = LogisticRegression(class_weight='balanced')
lr.fit(x, y)
y_pred = lr.predict(x)

acc = accuracy_score(y, y_pred)
print('Accuracy:', round(acc, 2) * 100)

lr.predict([[5.1, 3.5, 1.4, 0.2]])[0]

with open('iris.pkl', 'wb') as f:
   pickle.dump(lr, f)

lr_model = pickle.load(open('iris.pkl', 'rb'))

lr_model.predict([[5.1, 3.5, 1.4, 0.2]])[0]

