import libs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

x = np.array([[25,0], [35, 1], [40, 2], [50, 2], [20, 0]])
y = np.array([0, 1, 1, 1, 0])

model = DecisionTreeClassifier(max_depth = 3, random_state = 42)
model.fit(x,y)

prediction = model.predict([[65, 2]])
print("Will buyer make purchase? yes: 1 no: 0 ", prediction)

plt.figure(figsize=(10,6))
tree.plot_tree(model, feature_names=["age", "income"], class_names=["no", "yes"],
               filled=True)
plt.show()


