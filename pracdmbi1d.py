import numpy as np
from sklearn.neighbors import KNeighborsClassifier

x = np.array([[1, 4], [2, 5], [3, 7], [5, 8], [4, 3], [6, 2]])
y = np.array([1, 1, 1, 1, 0, 0])

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)

prediction = model.predict([[6, 4]])
probability = model.predict_proba([[6, 4]])

print("prediction ", prediction)
print("probability ", probability)
