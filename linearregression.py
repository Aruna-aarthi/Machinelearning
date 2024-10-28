import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


data=np.array([[1500,300000],[2400,450000],[1800,350000],[2200,40000],[3000,500000],
               [3500,550000],[1200,250000],[2000,380000],[2500,470000],[2700,490000]])

x=data[:,0].reshape(-1,1)
y=data[:,1]

model=linear_model.LinearRegression()
model.fit(x,y)

predicted_prices=model.predict(x)

plt.figure(figsize=(10,6))
plt.scatter(data[:,0],data[:,1],color="red",marker="*")
plt.plot(data[:,0], predicted_prices,color="blue")
plt.xlabel('area')
plt.ylabel('price')
plt.show()



