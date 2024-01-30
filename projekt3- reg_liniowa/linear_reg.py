import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Załadowanie danych z pliku
file_path = 'dane/dane1.txt'  # Zmień na ścieżkę do jednego z plików
data = np.loadtxt(file_path)

# Podział danych na wektory X i y
X = data[:, 0].reshape(-1, 1)  # Wektor cech
y = data[:, 1]  # Wektor wartości docelowych

# Podział danych na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Liniowy
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Weryfikacja Modelu 1
y_pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)

# Model 2: Wielomianowy stopnia 2
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)

# Weryfikacja Modelu 2
y_pred_poly = poly_reg.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)

# Wyświetlenie wyników
print("Błąd średniokwadratowy Modelu 1 (liniowy):", mse_lin)
print("Błąd średniokwadratowy Modelu 2 (wielomianowy):", mse_poly)

# Porównanie modeli
plt.scatter(X_test, y_test, color='black', label='Dane rzeczywiste')
plt.plot(X_test, y_pred_lin, color='blue', label='Model 1 (liniowy)')
plt.plot(np.sort(X_test, axis=0), poly_reg.predict(poly_features.transform(np.sort(X_test, axis=0))), color='red', label='Model 2 (wielomianowy)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
