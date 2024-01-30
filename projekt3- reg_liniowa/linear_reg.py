import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Załadowanie danych z pliku
file_path = 'dane/dane1.txt'  # Ścieżka do pliku z danymi
data = np.loadtxt(file_path)  # Wczytanie danych z pliku

# Podział danych na wektory X i y
X = data[:, 0].reshape(-1, 1)  # Przekształcenie pierwszej kolumny na wektor cech
y = data[:, 1]  # Druga kolumna to wektor wartości docelowych

# Podział danych na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Podział danych na treningowe i testowe

# Model 1: Liniowy
lin_reg = LinearRegression()  # Utworzenie modelu regresji liniowej
lin_reg.fit(X_train, y_train)  # Trenowanie modelu na danych treningowych

# Weryfikacja Modelu 1 na zbiorze testowym
y_pred_lin_test = lin_reg.predict(X_test)  # Przewidywanie wartości na danych testowych
mse_lin_test = mean_squared_error(y_test, y_pred_lin_test)  # Obliczenie MSE dla danych testowych

# Weryfikacja Modelu 1 na zbiorze treningowym
y_pred_lin_train = lin_reg.predict(X_train)  # Przewidywanie wartości na danych treningowych
mse_lin_train = mean_squared_error(y_train, y_pred_lin_train)  # Obliczenie MSE dla danych treningowych

print("Błąd średniokwadratowy Modelu 1 (liniowy) na zbiorze testowym:", mse_lin_test)
print("Błąd średniokwadratowy Modelu 1 (liniowy) na zbiorze treningowym:", mse_lin_train)

# Model 2: Wielomianowy stopnia 2
poly_features = PolynomialFeatures(degree=2, include_bias=False)  # Utworzenie cech wielomianowych stopnia 2
X_poly_train = poly_features.fit_transform(X_train)  # Transformacja danych treningowych
X_poly_test = poly_features.transform(X_test)  # Transformacja danych testowych

poly_reg = LinearRegression()  # Utworzenie modelu regresji liniowej
poly_reg.fit(X_poly_train, y_train)  # Trenowanie modelu na przekształconych danych treningowych

# Weryfikacja Modelu 2 na zbiorze testowym
y_pred_poly_test = poly_reg.predict(X_poly_test)  # Przewidywanie wartości na danych testowych
mse_poly_test = mean_squared_error(y_test, y_pred_poly_test)  # Obliczenie MSE dla danych testowych

# Weryfikacja Modelu 2 na zbiorze treningowym
y_pred_poly_train = poly_reg.predict(X_poly_train)  # Przewidywanie wartości na danych treningowych
mse_poly_train = mean_squared_error(y_train, y_pred_poly_train)  # Obliczenie MSE dla danych treningowych

print("Błąd średniokwadratowy Modelu 2 (wielomianowy) na zbiorze testowym:", mse_poly_test)
print("Błąd średniokwadratowy Modelu 2 (wielomianowy) na zbiorze treningowym:", mse_poly_train)

# Porównanie modeli
plt.scatter(X_test, y_test, color='black', label='Dane rzeczywiste')
plt.plot(X_test, y_pred_lin_test, color='blue', label='Model 1 (liniowy)')
plt.plot(np.sort(X_test, axis=0), poly_reg.predict(poly_features.transform(np.sort(X_test, axis=0))), color='red', label='Model 2 (wielomianowy)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

"""
Na podstawie przedstawionych błędów średniokwadratowych (MSE) dla obu modeli, można ocenić, 
który z nich lepiej dopasowuje się do danych. Niższy MSE wskazuje na lepsze dopasowanie modelu do danych. 
Analiza:

Model 1 (liniowy):

MSE na zbiorze testowym: 2.3588362348108873
MSE na zbiorze treningowym: 1.3694551998943036
Wyniki te wskazują, że model liniowy ma umiarkowane dopasowanie do danych, 
z nieco lepszymi wynikami na zbiorze treningowym.

Model 2 (wielomianowy):

MSE na zbiorze testowym: 0.01698043752924276
MSE na zbiorze treningowym: 0.008945713096845807
Te wyniki są znacznie niższe niż dla modelu liniowego, 
co wskazuje na znacznie lepsze dopasowanie modelu wielomianowego do danych, 
zarówno na zbiorze treningowym, jak i testowym.
"""
