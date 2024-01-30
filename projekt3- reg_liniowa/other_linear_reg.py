import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


file = np.loadtxt('dane/dane1.txt')
X = file[:,[0]]
y = file[:,[1]]

plt.plot(X, y, 'r*')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

plt.plot(X_train, y_train, 'g*')
plt.plot(X_test, y_test, 'b*')
plt.show()

# Simple linear regression
F1 = np.hstack([X_train, np.ones(X_train.shape)])
V1 = np.linalg.inv(F1.T @ F1) @ F1.T @ y_train

E1_train = y_train - (V1[0] * X_train + V1[1])
MSE1_train = np.mean(E1_train**2)
print(MSE1_train)

E1_test = y_test - (V1[0] * X_test + V1[1])
MSE1_test = np.mean(E1_test**2)
print(MSE1_test)

plt.plot(X_train, y_train, 'g*')
plt.plot(X_test, y_test, 'b*')
plt.plot(X, V1[0] * X + V1[1])
plt.show

# quadratic reg
F2 = np.hstack([X_train**2, X_train, np.ones(X_train.shape)])
V2 = np.linalg.pinv(F2) @ y_train

E2_train = y_train - (V2[0] * np.square(X_train) + V2[1] * X_train + V2[2])
MSE2_train = np.mean(E2_train**2)
print(MSE2_train)

E2_test = y_test - (V2[0] * np.square(X_test) + V2[1] * X_test + V2[2])
MSE2_test = np.mean(E2_test**2)
print(MSE2_test)

plt.plot(X_train, y_train, 'g*')
plt.plot(X_test, y_test, 'b*')
plt.plot(X, V1[0] * X + V1[1])
plt.plot(X, V2[0] * X * X + V2[1] * X + V2[2])
plt.show

# inverse linear
F3 = np.hstack([1/X_train, np.ones(X_train.shape)])
V3 = np.linalg.pinv(F3) @ y_train

E3_train = y_train - (V3[0]/X_train + V3[1])
MSE3_train = np.mean(E3_train**2)
print(MSE2_train)

E3_test = y_test - (V3[0]/X_test + V3[1])
MSE3_test = np.mean(E3_test**2)
print(MSE3_test)

plt.plot(X_train, y_train, 'g*')
plt.plot(X_test, y_test, 'b*')
plt.plot(X, V1[0] * X + V1[1])
plt.plot(X, V2[0] * X * X + V2[1] * X + V2[2])
plt.plot(X, V3[0]/X + V3[1])
plt.show

# polynomial model: fifth degree (y = ax^5 + bx^4 + cx^3 + dx^2 + ex + f)
F4 = np.hstack([X_train**5,X_train**4,X_train**3,X_train**2,X_train, np.ones(X_train.shape)])
V4 = np.linalg.pinv(F4) @ y_train

# TRAIN
E4_train = y_train - (V4[0]*np.power(X_train, 5) + V4[1]*np.power(X_train, 4) + V4[2]*np.power(X_train, 3) +V4[3]*np.power(X_train, 2) + V4[4]*X_train + V4[5])
MSE4_train = np.mean(E4_train**2)
print(MSE4_train)

# TEST
E4_test = y_test - (V4[0]*np.power(X_test, 5) + V4[1]*np.power(X_test, 4) + V4[2]*np.power(X_test, 3) +V4[3]*np.power(X_test, 2) + V4[4]*X_test + V4[5])
MSE4_test = np.mean(E4_test**2)
print(MSE4_test)

plt.plot(X_train, y_train, 'g*')
plt.plot(X_test, y_test, 'b*')
plt.plot(X, V1[0]*X + V1[1])
plt.plot(X, V2[0]*X*X + V2[1]*X, V2[2])
plt.plot(X, V3[0]/X + V3[1])
plt.plot(X, V4[0]*X**5 + V4[1]*X**4 + V4[2]*X**3 +V4[3]*X*X + V4[4]*X, V4[5])
plt.show