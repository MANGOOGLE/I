import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([[30], [45], [70], [18], [50]])
y = np.array([0, 0, 1, 0, 0])

model = LogisticRegression()
model.fit(x,y)

prediction = model.predict([[65]])
probability = model.predict_proba([[65]])

print("prediction ", prediction)
print("probability ", probability)
