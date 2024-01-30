import numpy as np  # Importowanie biblioteki NumPy do operacji na macierzach i wektorach
from sklearn import datasets  # Importowanie zestawów danych z biblioteki sklearn
from sklearn.model_selection import train_test_split  # Importowanie funkcji do podziału danych na zbiory treningowe i testowe

class Perceptron(object):
    # Klasa Perceptron, implementacja perceptronu pojedynczej warstwy
    def __init__(self, eta=0.01, n_iter=10):  # Konstruktor z parametrami: współczynnik uczenia eta i liczba iteracji n_iter
        self.eta = eta  # Przypisanie współczynnika uczenia
        self.n_iter = n_iter  # Przypisanie liczby iteracji

    def fit(self, X, y):  # Metoda do trenowania perceptronu
        self.w_ = np.zeros(1 + X.shape[1])  # Inicjalizacja wag z zerami
        for _ in range(self.n_iter):  # Pętla przez określoną liczbę iteracji
            for xi, target in zip(X, y):  # Iteracja przez próbki i ich etykiety
                update = self.eta * (target - self.predict(xi))  # Obliczenie aktualizacji wagi
                self.w_[1:] += update * xi  # Aktualizacja wag
                self.w_[0] += update  # Aktualizacja biasu (wagi zerowej)
        return self

    def net_input(self, X):  # Metoda do obliczania całkowitego bodźca dla próbki
        return np.dot(X, self.w_[1:]) + self.w_[0]  # Iloczyn skalarny wektora wag i wejść plus bias

    def predict(self, X):  # Metoda do przewidywania etykiety klasy
        return np.where(self.net_input(X) >= 0.0, 1, -1)  # Zwraca 1 lub -1 w zależności od bodźca

class MultiClassPerceptron:
    # Klasa do obsługi perceptronu wieloklasowego
    def __init__(self):
        self.perceptrons = []  # Lista do przechowywania perceptronów dla każdej klasy

    def train(self, X, y, eta=0.01, n_iter=10000):
        self.classes = np.unique(y)  # Ustalenie unikalnych etykiet klas
        for c in self.classes:  # Dla każdej klasy
            ppn = Perceptron(eta, n_iter)  # Tworzenie nowego perceptronu
            y_binary = np.where(y == c, 1, -1)  # Tworzenie binarnej etykiety dla aktualnej klasy
            ppn.fit(X, y_binary)  # Trenowanie perceptronu dla binarnej klasyfikacji
            self.perceptrons.append(ppn)  # Dodanie trenowanego perceptronu do listy

    def predict(self, X):
        predictions = np.array([ppn.predict(X) for ppn in self.perceptrons]).T  # Przewidywanie przez wszystkie perceptrony
        return np.argmax(predictions, axis=1)  # Wybór klasy z najwyższym wynikiem

def main():
    iris = datasets.load_iris()  # Wczytanie zbioru danych Iris
    X = iris.data[:, [2, 3]]  # Wybór dwóch cech z danych,
    # jesli zakomentujemy po 'data' to perceptrony beda trenowane na wszystkich cechach(4.)
    y = iris.target  # Przypisanie etykiet

    # Podział na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    mcp = MultiClassPerceptron()  # Utworzenie instancji klasyfikatora wieloklasowego
    mcp.train(X_train, y_train)  # Trenowanie klasyfikatora

    y_pred = mcp.predict(X_test)  # Przewidywanie etykiet na zbiorze testowym
    accuracy = np.mean(y_pred == y_test)  # Obliczenie dokładności
    print("Dokładność: ", accuracy)  # Wyświetlenie dokładności

if __name__ == '__main__':
    main()  # Wywołanie głównej funkcji, jeśli skrypt jest uruchamiany bezpośrednio

"""
ficzery lub ich brak:
ta sledzi liczbe bledow w kazdej iteracji ('self.errors_')

brak bezposredniej oceny dokladnosci na zbiorze testowym poprzez porownanie etykiet prawdziwych i przewidywanych

tamta implementacja ma logike do tworzenia binarnych etykiet dla kazdej klasy, 
co jest niezbedne w kontekscie klasyfikacji wieloklasowej
"""