import numpy as np  # Importowanie biblioteki NumPy dla operacji na macierzach
import matplotlib.pyplot as plt  # Importowanie biblioteki Matplotlib do tworzenia wykresów
import enum  # Importowanie modułu enum do tworzenia wyliczeń

# Enum do reprezentacji opcji w grze: kamień, papier, nożyce
class RPS(enum.Enum):
    Rock = 1
    Paper = 2
    Scissors = 3

# Funkcja do pobierania wejścia od użytkownika z walidacją
def get_user_input():
    choices = { "r": RPS.Rock, "p": RPS.Paper, "s": RPS.Scissors }  # Słownik mapujący wejście użytkownika na enumy
    while True:  # Pętla nieskończona do uzyskania poprawnego wejścia
        user_input = input("Choose Rock (r), Paper (p), Scissors (s): ").lower()  # Pobieranie wejścia od użytkownika
        if user_input in choices:  # Sprawdzanie, czy wejście jest poprawne
            return choices[user_input]  # Zwracanie wyboru użytkownika
        else:
            print("Invalid input. Please choose 'r', 'p', or 's'.")  # Komunikat o błędzie, jeśli wejście jest niepoprawne

# Funkcja aktualizująca macierz przejścia na podstawie wyniku gry
def update_transition_matrix(matrix, current_state, result):
    learning_rate = 0.05  # Współczynnik nauki
    if result == -1:  # Jeśli gracz przegrał
        winning_move = (current_state.value + 1) % 3  # Obliczenie ruchu, który wygrał
        matrix[current_state.value - 1][winning_move] += learning_rate  # Aktualizacja macierzy
        matrix[current_state.value - 1] /= np.sum(matrix[current_state.value - 1])  # Normalizacja macierzy

# Funkcja do rozgrywki i określenia wyniku
def play_rps(user_choice, transition_matrix):
    # Wybór ruchu AI na podstawie macierzy przejścia
    ai_choice = np.random.choice([RPS.Rock, RPS.Paper, RPS.Scissors], p=transition_matrix[user_choice.value - 1])
    if ai_choice == user_choice:  # Sprawdzanie czy jest remis
        return 0, ai_choice
    elif (ai_choice.value - user_choice.value) % 3 == 1:  # Sprawdzanie czy AI wygrało
        return -1, ai_choice
    else:  # W przypadku wygranej gracza
        return 1, ai_choice

# Główna funkcja zarządzająca grą
def main():
    # Inicjalizacja macierzy przejścia
    transition_matrix = np.full((3, 3), 1/3)  # Macierz przejścia wypełniona wartościami 1/3
    num_games = int(input("How many games to play? "))  # Pobieranie liczby gier od użytkownika
    total_score = 0  # Inicjalizacja całkowitego wyniku
    scores = []  # Lista do przechowywania wyników

    for _ in range(num_games):  # Pętla dla każdej gry
        user_choice = get_user_input()  # Pobieranie wyboru użytkownika
        result, ai_choice = play_rps(user_choice, transition_matrix)  # Rozgrywka i otrzymywanie wyniku
        update_transition_matrix(transition_matrix, user_choice, result)  # Aktualizacja macierzy przejścia
        total_score += result  # Aktualizacja wyniku
        scores.append(total_score)  # Dodawanie wyniku do listy
        print(f"AI chose {ai_choice.name}. Current score: {total_score}")  # Wyświetlanie wyboru AI i wyniku

    # Tworzenie wykresu
    plt.plot(scores)
    plt.xlabel('Game Number')  # Etykieta osi X
    plt.ylabel('Total Score')  # Etykieta osi Y
    plt.title('Rock Paper Scissors Game Progress')  # Tytuł wykresu
    plt.show()  # Wyświetlanie wykresu

# Uruchomienie programu
if __name__ == "__main__":
    main()

    '''
    Jeśli macierz przejścia dla wyboru kamienia (Rock) przez gracza wskazuje na prawdopodobieństwa
     50% dla papieru (Paper), 25% dla kamienia (Rock) i 25% dla nożyc (Scissors), 
     AI będzie wybierało papier w przybliżeniu w połowie przypadków, 
     a kamień i nożyce w przybliżeniu w jednej czwartej przypadków każde.

    To oznacza, że wybór AI nie jest deterministyczny, lecz probabilistyczny. 
    AI "losuje" swój ruch każdorazowo, kiedy ma dokonać wyboru, 
    ale prawdopodobieństwa tych wyborów są ważone zgodnie z wartościami w macierzy przejścia. 
    Takie podejście sprawia, że zachowanie AI jest bardziej dynamiczne i trudniejsze do przewidzenia, 
    co jest kluczowym aspektem w grach takich jak kamień, papier, nożyce.
    '''
