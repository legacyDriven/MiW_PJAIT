from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score

# Funkcja do rysowania wykresów powierzchni decyzyjnych dla klasyfikatora
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
        # Konfiguracja znaczników i mapy kolorów
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # Rysowanie wykresu powierzchni decyzyjnej
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # Rysowanie próbek
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl, edgecolor='k')

# Definicja klasy perceptronu
class Perceptron(object):
        def __init__(self, eta=0.01, n_iter=10):
                self.eta = eta  # Współczynnik uczenia
                self.n_iter = n_iter  # Liczba iteracji

        def fit(self, X, y):
                self.w_ = np.zeros(1 + X.shape[1])  # Inicjalizacja wag
                self.errors_ = []  # Lista błędów

                for _ in range(self.n_iter):
                        errors = 0
                        for xi, target in zip(X, y):
                                update = self.eta * (target - self.predict(xi))
                                self.w_[1:] += update * xi
                                self.w_[0] += update
                                errors += int(update != 0.0)
                        self.errors_.append(errors)
                return self

        def net_input(self, X):
                return np.dot(X, self.w_[1:]) + self.w_[0]  # Obliczenie sumy ważonej

        def predict(self, X):
                return np.where(self.net_input(X) >= 0.0, 1, -1)  # Prognozowanie klasy

# Dodatkowa klasa klasyfikatora wykorzystująca dwa perceptrony
class Classifier:
        def __init__(self, ppn1, ppn2):
                self.ppn1 = ppn1  # Pierwszy perceptron
                self.ppn2 = ppn2  # Drugi perceptron

        def predict(self, x):
                # Klasyfikacja przy użyciu obu perceptronów
                return np.where(self.ppn1.predict(x) == 1, 0, np.where(self.ppn2.predict(x) == 1, 2, 1))

# Główna funkcja programu
def main():
        # Wczytywanie zestawu danych Iris
        iris = datasets.load_iris()
        X = iris.data[:, [2, 3]]  # Wybór dwóch cech
        y = iris.target  # Klasy

        # Dzielenie danych na zestawy treningowe i testowe
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

        # Przygotowanie danych do klasyfikacji binarnej dla obu perceptronów
        y_train_01_subset = y_train.copy()
        y_train_02_subset = y_train.copy()

        X_train_01_subset = X_train.copy()

        # Modyfikacja etykiet dla perceptronu 1
        y_train_01_subset[(y_train == 1) | (y_train == 2)] = -1
        y_train_01_subset[(y_train_01_subset == 0)] = 1

        # Modyfikacja etykiet dla perceptronu 2
        y_train_02_subset[(y_train == 1) | (y_train == 0)] = -1
        y_train_02_subset[(y_train_02_subset == 2)] = 1

        # Trenowanie perceptronów
        ppn1 = Perceptron(eta=0.1, n_iter=300)
        ppn1.fit(X_train_01_subset, y_train_01_subset)

        ppn2 = Perceptron(eta=0.1, n_iter=300)
        ppn2.fit(X_train_01_subset, y_train_02_subset)

        # Ocena dokładności perceptronów
        accuracy_1 = accuracy(ppn1.predict(X_train), y_train_01_subset)
        accuracy_2 = accuracy(ppn2.predict(X_train), y_train_02_subset)
        print("Perceptron 1 accuracy: ", accuracy_1)
        print("Perceptron 2 accuracy: ", accuracy_2)

        total_accuracy(accuracy_1, accuracy_2)

        # Utworzenie klasyfikatora wykorzystującego oba perceptrony
        classifier = Classifier(ppn1, ppn2)

        # Rysowanie wykresów
        plot_decision_regions(X=X_train, y=y_train, classifier=classifier)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.legend(loc='upper left')
        plt.show()

# Funkcja do obliczania dokładności
def accuracy(y_results, y_train):
        return accuracy_score(y_results, y_train)

# Funkcja do obliczania średniej dokładności
def total_accuracy(accuracy_1, accuracy_2):
        total_accuracy = (accuracy_1 + accuracy_2) / 2
        print("Overall accuracy: ", total_accuracy)

if __name__ == '__main__':
        main()

        '''
Ładowanie i przygotowanie danych: wykorzystany jest zbiór danych Iris, wybieramy dwie cechy (kolumny 2 i 3) 
i dzielimy dane na zestawy treningowe i testowe.

Modyfikacja etykiet klas: Dla każdego perceptronu modyfikujemy etykiety klas tak,
 aby każdy perceptron mógł nauczyć się rozróżniać jedną klasę od pozostałych (przyjmując wartości 1 i -1).

Trenowanie perceptronów: Dwa perceptrony są trenowane na zmodyfikowanych danych.

Klasyfikacja: Używamy obu perceptronów do klasyfikacji próbek, a następnie łączymy ich wyniki, aby uzyskać końcową klasyfikację wieloklasową.

Ocena dokładności: Obliczamy dokładność każdego perceptronu oraz średnią dokładność.

Wizualizacja: Użycie plot_decision_regions do wizualizacji obszarów decyzyjnych klasyfikatora.
        
        '''
