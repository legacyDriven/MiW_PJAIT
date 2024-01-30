import numpy as np  # Importowanie biblioteki NumPy do obliczeń numerycznych
from sklearn import datasets  # Importowanie funkcjonalności do wczytywania popularnych zestawów danych
from sklearn.model_selection import train_test_split  # Funkcja do dzielenia danych na zbiory treningowe i testowe

# Klasa implementująca regresję logistyczną metodą spadku gradientu
class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta  # Współczynnik uczenia
        self.n_iter = n_iter  # Liczba iteracji w procesie uczenia
        self.random_state = random_state  # Ziarno dla generatora liczb losowych

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)  # Generator liczb losowych dla inicjalizacji wag
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # Inicjalizacja wag

        for i in range(self.n_iter):  # Pętla iteracyjnego uczenia
            net_input = self.net_input(X)  # Obliczenie bodźca
            output = self.activation(net_input)  # Aktywacja (funkcja sigmoidalna)
            errors = (y - output)  # Obliczenie błędu predykcji
            self.w_[1:] += self.eta * X.T.dot(errors)  # Aktualizacja wag
            self.w_[0] += self.eta * errors.sum()  # Aktualizacja biasu (wagi zerowej)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]  # Obliczenie całkowitego bodźca

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))  # Funkcja aktywacji (sigmoid)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)  # Przewidywanie klasy

    def predict_proba(self, X):
        # Obliczanie prawdopodobieństwa przynależności próbki do klasy
        net_input = self.net_input(X)
        proba = self.activation(net_input)
        return proba

# Klasa implementująca wieloklasową regresję logistyczną metodą "one-vs-rest"
class MultiClassLogisticRegression:
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta  # Współczynnik uczenia
        self.n_iter = n_iter  # Liczba iteracji w procesie uczenia
        self.random_state = random_state  # Ziarno dla generatora liczb losowych
        self.classifiers = []  # Lista klasyfikatorów dla każdej klasy

    def fit(self, X, y):
        self.classes = np.unique(y)  # Znalezienie unikalnych klas
        for c in self.classes:  # Dla każdej klasy
            y_binary = np.where(y == c, 1, 0)  # Tworzenie binarnego wektora celu
            lrgd = LogisticRegressionGD(self.eta, self.n_iter, self.random_state)
            lrgd.fit(X, y_binary)  # Trenowanie klasyfikatora dla danej klasy
            self.classifiers.append(lrgd)  # Dodanie wytrenowanego klasyfikatora do listy

    def predict(self, X):
        probabilities = np.array([classifier.activation(classifier.net_input(X)) for classifier in self.classifiers])
        class_indices = np.argmax(probabilities, axis=0)  # Wybór klasy z najwyższym prawdopodobieństwem
        return self.classes[class_indices]

def main():
    iris = datasets.load_iris()  # Wczytanie danych zbioru Iris
    X = iris.data[:, [2, 3]]  # Wybór dwóch cech z danych
    y = iris.target  # Przypisanie etykiet
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)  # Podział danych na treningowe i testowe

    mlr = MultiClassLogisticRegression(eta=0.05, n_iter=1000, random_state=1)  # Utworzenie instancji wieloklasowego klasyfikatora
    mlr.fit(X_train, y_train)  # Trenowanie klasyfikatora na danych treningowych

    # Obliczanie prawdopodobieństw dla każdej klasy
    for idx, classifier in enumerate(mlr.classifiers):
        proba = classifier.predict_proba(X_test)  # Obliczanie prawdopodobieństwa dla każdej klasy
        print(f"Prawdopodobieństwa przynależności do klasy {idx}:\n", proba)

    y_pred = mlr.predict(X_test)  # Przewidywanie klas na danych testowych
    accuracy = np.mean(y_pred == y_test)  # Obliczenie dokładności
    print("Dokładność: ", accuracy)  # Wyświetlenie dokładności

if __name__ == '__main__':
    main()  # Uruchomienie głównej funkcji
