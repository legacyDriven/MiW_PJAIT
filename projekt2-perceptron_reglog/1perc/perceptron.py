import matplotlib.pyplot as plt  # Importowanie biblioteki matplotlib do tworzenia wykresów
import numpy as np  # Importowanie biblioteki NumPy do obliczeń numerycznych
from sklearn.model_selection import train_test_split  # Funkcja do podziału danych na zbiory treningowe i testowe
from sklearn import datasets  # Importowanie zbiorów danych z biblioteki sklearn
from plotka import plot_decision_regions  # Importowanie funkcji do rysowania obszarów decyzyjnych


class Perceptron(object):  # Definicja klasy Perceptron

    def __init__(self, eta=0.01,
                 n_iter=10):  # Konstruktor klasy z parametrami: współczynnik uczenia (eta) i liczba iteracji (n_iter)
        self.eta = eta  # Przypisanie współczynnika uczenia do zmiennej instancji
        self.n_iter = n_iter  # Przypisanie liczby iteracji do zmiennej instancji

    def fit(self, X, y):  # Metoda do trenowania perceptronu
        self.w_ = np.zeros(1 + X.shape[1])  # Inicjalizacja wag z zerami

        for _ in range(self.n_iter):  # Pętla przez określoną liczbę iteracji
            errors = 0  # Zmienna do śledzenia liczby błędów
            for xi, target in zip(X, y):  # Iteracja przez próbki i ich odpowiednie etykiety
                update = self.eta * (target - self.predict(xi))  # Obliczenie aktualizacji wagi
                self.w_[1:] += update * xi  # Aktualizacja wag oprócz biasu
                self.w_[0] += update  # Aktualizacja biasu
        return self  # Zwrócenie instancji klasy

    def net_input(self, X):  # Metoda do obliczenia całkowitego bodźca
        return np.dot(X, self.w_[1:]) + self.w_[0]  # Obliczenie iloczynu skalarnego plus bias

    def predict(self, X):  # Metoda do przewidywania etykiety klasy
        return np.where(self.net_input(X) >= 0.0, 1, -1)  # Zwraca 1 jeśli bodziec jest nieujemny, w przeciwnym razie -1


def main():  # Główna funkcja

    iris = datasets.load_iris()  # Wczytanie zbioru danych Iris
    X = iris.data[:, [2, 3]]  # Wybór dwóch cech
    y = iris.target  # Przypisanie etykiet
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1,
                                                        stratify=y)  # Podział na zbiory treningowe i testowe

    # Przygotowanie danych do perceptronu (perceptron przyjmuje wyjście 1 lub -1)
    X_train_01_subset = X_train  # Używamy wszystkich danych treningowych
    y_train_01_subset = y_train  # Kopiujemy etykiety
    y_train_01_subset[(y_train != 2)] = -1  # Etykiety różne od 2 są ustawiane na -1
    y_train_01_subset[(y_train == 2)] = 1  # Etykiety równe 2 są ustawiane na 1
    print(y_train_01_subset)  # Wyświetlenie przetworzonych etykiet
    ppn = Perceptron(eta=0.2, n_iter=200)  # Utworzenie instancji perceptronu
    ppn.fit(X_train_01_subset, y_train_01_subset)  # Trenowanie perceptronu

    plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=ppn)  # Wyświetlenie obszarów decyzyjnych
    plt.xlabel(r'$x_1$')  # Etykieta osi X
    plt.ylabel(r'$x_2$')  # Etykieta osi Y
    plt.legend(loc='upper left')  # Umieszczenie legendy
    plt.show()  # Wyświetlenie wykresu


if __name__ == '__main__':  # Sprawdzenie, czy skrypt jest uruchamiany bezpośrednio
    main()  # Uruchomienie głównej funkcji
