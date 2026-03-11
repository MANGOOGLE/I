import libs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([2000, 2200, 2400, 2600, 2800, 3000]).reshape(-1, 1)
y = np.array([100000, 120000, 140000, 160000, 180000, 200000])

model = LinearRegression()
model.fit(x,y)

print("intercept", model.intercept_)
print("slope", model.coef_[0])

predicted_price = model.predict([[2500]])
print("predicted price: ", predicted_price)

plt.scatter(x, y, color="blue", label="actual data")
plt.plot(x, model.predict(x), color="red", label="regression")
plt.scatter(2500, predicted_price, color="yellow", label="predicted price")

plt.xlabel("square footage")
plt.ylabel("price")

plt.legend()
plt.show()
                  
