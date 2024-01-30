# Importowanie potrzebnych bibliotek
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions

# Klasa implementująca regresję logistyczną z gradientowym spadkiem
class LogisticRegressionGD1(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta  # Współczynnik uczenia
        self.n_iter = n_iter  # Liczba iteracji
        self.random_state = random_state  # Ziarno losowości

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)  # Generator liczb losowych
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # Inicjalizacja wag
        self.cost_ = []  # Lista do przechowywania kosztu w każdej iteracji

        for i in range(self.n_iter):  # Pętla iteracyjnego uczenia
            net_input = self.net_input(X)  # Obliczanie całkowitego bodźca
            output = self.activation(net_input)  # Obliczanie wartości funkcji aktywacji
            errors = (y - output)  # Obliczanie błędu
            self.w_[1:] += self.eta * X.T.dot(errors)  # Aktualizacja wag
            self.w_[0] += self.eta * errors.sum()  # Aktualizacja biasu
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))  # Obliczenie kosztu
            self.cost_.append(cost)  # Dodanie kosztu do listy
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]  # Obliczanie całkowitego bodźca

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))  # Funkcja aktywacji sigmoidalna

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)  # Przewidywanie klasy

    def predict_proba(self, X):
        output = self.predict(X)  # Przewidywanie klasy
        return np.vstack((1 - output, output)).T  # Zwracanie prawdopodobieństwa przynależności do klas

# Klasa służąca do klasyfikacji wieloklasowej
class Classifier1:
    def __init__(self, lrgd1, lrgd2, lrgd3):
        self.lrgd1 = lrgd1  # Pierwszy klasyfikator
        self.lrgd2 = lrgd2  # Drugi klasyfikator
        self.lrgd3 = lrgd3  # Trzeci klasyfikator

    def predict(self, x):
        # Przypisywanie klasy na podstawie predykcji trzech klasyfikatorów
        return np.where(self.lrgd1.predict(x) == 1, 0,
                np.where(self.lrgd2.predict(x) == 1, 1,
                np.where(self.lrgd3.predict(x) == 1, 2 ,-1)))

# Główna funkcja
def main():
    iris = datasets.load_iris()  # Wczytanie danych zbioru Iris
    X = iris.data[:, [2, 3]]  # Wybór cech
    y = iris.target  # Wybór etykiet
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)  # Podział na zbiory treningowe i testowe

    # Przygotowanie danych dla każdej klasy
    y_train_01_subset = y_train.copy()
    y_train_02_subset = y_train.copy()
    y_train_03_subset = y_train.copy()
    X_train_01_subset = X_train.copy()

    # Przypisanie etykiet dla każdej klasy
    y_train_01_subset[(y_train == 1) | (y_train == 2)] = -1
    y_train_01_subset[(y_train_01_subset == 0)] = 1
    y_train_01_subset[(y_train_01_subset == -1)] = 0

    y_train_02_subset[(y_train == 2) | (y_train == 0)] = 0
    y_train_02_subset[(y_train_02_subset == 1)] = 1

    y_train_03_subset[(y_train == 1) | (y_train == 0)] = 0
    y_train_03_subset[(y_train_03_subset == 2)] = 1

    # Wytrenowanie klasyfikatorów
    lrgd1 = LogisticRegressionGD1(eta=0.05, n_iter=1000, random_state=1)
    lrgd1.fit(X_train_01_subset, y_train_01_subset)

    lrgd2 = LogisticRegressionGD1(eta=0.05, n_iter=1000, random_state=1)
    lrgd2.fit(X_train_01_subset, y_train_02_subset)
    lrgd3 = LogisticRegressionGD1(eta=0.05, n_iter=1000, random_state=1)
    lrgd3.fit(X_train_01_subset, y_train_03_subset)

    # Predykcja i ocena skuteczności
    y1_predict = lrgd1.predict(X_train_01_subset)
    y2_predict = lrgd2.predict(X_train_01_subset)
    y3_predict = lrgd3.predict(X_train_01_subset)

    print("Result 1: ", y1_predict)
    print("Result 2: ", y2_predict)
    print("Result 3: ", y3_predict)

    accuracy_1 = accuracy_score(y1_predict, y_train_01_subset)
    accuracy_2 = accuracy_score(y2_predict, y_train_02_subset)
    accuracy_3 = accuracy_score(y3_predict, y_train_03_subset)

    print("Accuracy 1: ", accuracy_1)
    print("Accuracy 2: ", accuracy_2)
    print("Accuracy 3: ", accuracy_3)

    # Obliczenie ogólnej skuteczności
    total_accuracy = (accuracy_1 + accuracy_2 + accuracy_3) / 3
    print("Averall accuracy: ", total_accuracy)

    # Klasyfikacja i wizualizacja wyników
    _classifier = Classifier1(lrgd1, lrgd2, lrgd3)
    plot_decision_regions(X=X_train, y=y_train, classifier=_classifier)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()


    # Averall
    # accuracy: 0.8285714285714285
'''
Podejście z Binarnymi Predykcjami: 
W tym podejściu, każdy klasyfikator decyduje, czy próbka należy do swojej klasy, czy nie. 
Następnie, na podstawie tych decyzji, przypisujesz klasę próbce. To podejście jest prostsze, 
ale może nie być tak precyzyjne jak podejście z prawdopodobieństwami, 
szczególnie w przypadkach, gdy kilka klas ma podobne prawdopodobieństwa.
'''
