import numpy as np  # Do macierzy
import matplotlib.pyplot as plt  # Do wykresów

# Inicjalizacja macierzy przejścia
# Stany: 0 - Papier, 1 - Kamień, 2 - Nożyce
transition_matrix = np.array([[1/3, 1/3, 1/3],  # Prawdopodobieństwa przejścia z Papieru
                              [1/3, 1/3, 1/3],  # Prawdopodobieństwa przejścia z Kamienia
                              [1/3, 1/3, 1/3]])  # Prawdopodobieństwa przejścia z Nożyc

def update_transition_matrix(matrix, current_state, win_lose_draw):  # Zmiana macierzy przejścia
    """
    Aktualizuje macierz przejścia na podstawie wyniku gry.
    """
    lose_prob_increase = 0.1  # Zwiększenie prawdopodobieństwa przejścia przy przegranej
    if win_lose_draw == 1:  # Wygrana
        pass  # Brak zmian w macierzy przejścia
    elif win_lose_draw == -1:  # Przegrana
        matrix[current_state] = np.clip(matrix[current_state] + lose_prob_increase, 0, 1) # Zwiększenie prawdopodobieństwa przejścia
        matrix[current_state] /= matrix[current_state].sum()  # Normalizacja
    # Remis nie zmienia macierzy

def play_rps(transition_matrix, opponent_move):
    """
    Wybiera ruch w grze "papier, kamień, nożyce" i zwraca wynik.
    """
    # 0 - Papier, 1 - Kamień, 2 - Nożyce
    current_state = np.random.choice([0, 1, 2], p=transition_matrix[0])  # Wybór ruchu
    if current_state == opponent_move:  # Remis
        return 0, current_state  # Remis
    elif (current_state == 0 and opponent_move == 1) or \
         (current_state == 1 and opponent_move == 2) or \
         (current_state == 2 and opponent_move == 0):
        return 1, current_state  # Wygrana
    else:
        return -1, current_state  # Przegrana

# Symulacja gier i aktualizacja macierzy przejścia
n_games = 50  # Liczba gier
cash = 0  # Stan kasy
cash_history = []  # Historia stanu kasy

for _ in range(n_games):
    opponent_move = np.random.choice([0, 1, 2])  # Losowy ruch przeciwnika
    result, current_state = play_rps(transition_matrix, opponent_move)  # Wybór ruchu i wynik gry
    cash += result  # Aktualizacja stanu kasy
    cash_history.append(cash)  # Dodanie stanu kasy do historii
    update_transition_matrix(transition_matrix, current_state, result)  # Aktualizacja macierzy przejścia

# Wykres zmian stanu kasy
plt.plot(cash_history)  # Wykres
plt.xlabel('Numer gry')  # Podpis osi x
plt.ylabel('Stan kasy')  # Podpis osi y
plt.title('Zmiana stanu kasy w grze "papier, kamień, nożyce"')
plt.show()

"""
Ten program symuluje serię gier w "papier, kamień, nożyce", wykorzystując macierz przejścia do decydowania o ruchach. 
Macierz przejścia jest aktualizowana w zależności od wyniku każdej gry. Kluczowe elementy programu to:

1 Inicjalizacja Macierzy Przejścia: Na początku, program tworzy macierz przejścia o równych prawdopodobieństwach, sugerując, że początkowo każdy ruch (papier, kamień, nożyce) jest równie prawdopodobny.

1 Funkcja update_transition_matrix: Ta funkcja aktualizuje macierz przejścia po każdej grze. Jeśli gracz przegra, prawdopodobieństwo wybranego ruchu wzrasta o 0.1, zachowując jednak całą macierz jako prawidłowo znormalizowaną.

3 Funkcja play_rps: Funkcja symuluje grę, wybierając ruch na podstawie aktualnej macierzy przejścia i porównując go z ruchem przeciwnika. Zwraca wynik gry (wygrana, przegrana, remis) oraz wykonany ruch.

4 Symulacja Gry i Aktualizacja Macierzy: Program symuluje serię gier (50), losowo wybierając ruch przeciwnika. Po każdej grze aktualizuje stan kasy i macierz przejścia.

5 Wykres Stanu Kasy: Po symulacji, program rysuje wykres, który pokazuje, jak zmienia się stan kasy po każdej grze.

Podsumowując, program symuluje dynamiczną grę w "papier, kamień, nożyce", 
gdzie strategia gracza ewoluuje w czasie w oparciu o wyniki poprzednich gier.
"""