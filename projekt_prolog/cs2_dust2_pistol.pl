% Definicje dynamicznych predykatów
:- dynamic player/2.  % player(lokalizacja, bron) - przechowuje informacje o graczu
:- dynamic enemy/3.  % enemy(lokalizacja, bron, status) - przechowuje informacje o wrogach
:- dynamic bomb/1.  % bomb(lokalizacja) - przechowuje informacje o bombie
:- dynamic bombPlanted/2. % bombPlanted(Site, Position) - przechowuje informacje o podłożonej bombie

% Definicje bombsite'ów
bombsite(siteA).
bombsite(siteB).

% Warunki dozwolonego podłożenia bomby na danej lokalizacji:
isBombPlantable(a_default).
isBombPlantable(a_plat).

% Predykat do zaplanowania bomby na danej lokalizacji:
plantBomb(Site, Position) :-
    isBombPlantable(Site), % Sprawdzamy, czy lokalizacja jest dozwolona
    \+ bombPlanted(_, _), % Sprawdzamy, czy bomba nie została już podłożona
    assert(bombPlanted(Site, Position)). % Podłóż bombę na danej lokalizacji

% Definicje lokalizacji
locations([t_spawn, suicide, top_mid, mid, catwalk, xbox, right_side_mid, outside_long, long_doors, blue, long_corner, side_pit,
pit, pit_plat, a_long, a_car, a_cross, elevator, short_boost, a_ramp, a_barrels, goose, a_default, a_plat,
ninja, a_short, ct_spawn, ct_mid]).

% Definicje połączeń
paths([(t_spawn, suicide), (suicide, top_mid), (suicide, right_side_mid), (t_spawn, outside_long), (top_mid, right_side_mid),
(outside_long, long_doors), (outside_long, top_mid), (long_doors, blue), (blue, long_doors), (blue, long_corner), (blue, side_pit),
(long_corner, side_pit), (long_corner, pit), (long_corner, pit_plat), (side_pit, pit), (side_pit, long_corner), (pit, long_corner),
(pit_plat, a_long), (pit_plat, pit), (a_long, a_car), (a_long, a_cross), (a_long, a_ramp), (a_long, long_corner), (a_car, a_cross),
(a_car, a_long), (a_car, a_ramp), (a_cross, elevator), (elevator, a_cross), (elevator, short_boost), (elevator, ct_spawn),
(short_boost, ct_spawn), (short_boost, elevator), (a_ramp, a_barrels), (a_barrels, a_ramp), (a_barrels, goose), (goose, barrels),
(goose, a_default), (a_ramp, a_default), (a_default, goose), (a_default, barrels), (a_default, elevator), (a_default, a_plat),
(a_plat, a_default), (a_plat, ninja), (a_plat, elevator), (a_plat, short_boost), (a_plat, a_short), (a_short, ninja),
(a_short, a_plat), (a_short, short_boost), (a_short, ct_spawn)]).

% Funkcja sprawdzająca, czy istnieje połączenie między dwoma lokalizacjami
is_connected(Location1, Location2) :-
    paths(Paths),
    (member((Location1, Location2), Paths) ; member((Location2, Location1), Paths)).
    % Sprawdza, czy para lokalizacji jest częścią zdefiniowanych ścieżek
    % member sprawdza, czy dany element jest elementem listy

% Definicje broni, nazwa, cena, avgDamage
bron(glock18, 200, 15).
bron(usp-s, 300, 20).
bron(five-seven, 500, 18).
bron(p250, 300, 22).
bron(deagle, 700, 40).
bron(tec9, 500, 26).
bron(dual-elites, 500, 17).

% Definicje utility
utility(smoke, 300).
utility(flashbang, 200).
utility(hegrenade, 300).
utility(molotov, 400).
utility(decoy, 50).
utility(ctmolotov, 600).

% Definicje rzutów utility
% utility_throw(Start, Target, Affected, Type, SuccessRate) - definiuje rzuty utility
utility_throw(t_spawn, x_box, [x_box, top_mid], smoke, 80).
utility_throw(outside_long, side_pit, [side_pit, long_corner, pit], flashbang, 80).
utility_throw(long_doors, blue, [blue, long_corner], molotov, 70).
% ... (inne rzuty utility)

% Stosowanie efektów utility na listę pozycji
apply_effects(AffectedPositions, UtilityType) :-
    maplist(apply_effect(UtilityType), AffectedPositions).

% Stosowanie efektu utility na jedną pozycję
apply_effect(UtilityType, AffectedPosition) :-
    UtilityType == flashbang,
    findall(Enemy, enemy(AffectedPosition, Enemy, _), Enemies), % Znajdź wszystkich przeciwników na tej pozycji
    maplist(apply_flash_effect, Enemies), % Stosuj efekt flashbang na wszystkich przeciwnikach
    !.

% Predykat rzucania utility
throw_utility(StartPosition, UtilityType) :-
    player(StartPosition, _), % Sprawdza, czy gracz jest na odpowiedniej pozycji startowej
    utility_throw(StartPosition, TargetPosition, AffectedPositions, UtilityType, SuccessRate),
    random(0, 100, Roll), % Losowanie szansy powodzenia
    (Roll =< SuccessRate -> apply_effects(AffectedPositions, UtilityType), % Jeśli losowanie jest udane, stosuje efekty
                             format('Utility thrown from ~w to ~w affecting ~w~n', [StartPosition, TargetPosition, AffectedPositions])
                           ; format('Utility throw failed at ~w. Chance was ~d but rolled ~d.~n', [StartPosition, SuccessRate, Roll])), !.
    % Informuje gracza o powodzeniu lub niepowodzeniu rzutu

% Stosowanie efektów utility na listę pozycji
apply_effects(AffectedPositions, UtilityType) :-
    maplist(apply_effect(UtilityType), AffectedPositions).
    % Stosuje efekt utility na każdej pozycji z listy

% Stosowanie efektu utility na jedną pozycję
apply_effect(flashbang, AffectedPosition) :-
    findall(enemy(AffectedPosition, Weapon, _), enemy(AffectedPosition, Weapon, _), Enemies),
    maplist(apply_flash_effect, Enemies),
    !.
    % Stosuje efekt flashbang na wszystkich wrogach na danej pozycji
apply_effect(smoke, AffectedPosition) :-
    % Tu można dodać logikę dla efektu smoke...
    !.
apply_effect(_, _). % Dla innych utility, które nie mają dodatkowych efektów

% Zastosuj efekt flashbang na przeciwniku
apply_flash_effect(enemy(Position, Weapon, _)) :-
    retract(enemy(Position, Weapon, _)),
    assert(enemy(Position, Weapon, flashed)),
    format('Enemy at ~w is flashed~n', [Position]).
    % Zmienia status wroga na "flashed" (oszołomiony)


% Wykonanie ruchu w turze
perform_move(Position) :-
    player(CurrentPosition, Weapon),
    is_connected(CurrentPosition, Position),
    retract(player(CurrentPosition, Weapon)),
    assert(player(Position, Weapon)),
    format('Moved to ~w from ~w.~n', [Position, CurrentPosition]).
    % Pozwala graczowi przenieść się do nowej lokalizacji, jeśli jest ona połączona z aktualną

% Rekurencyjne szukanie ścieżki od punktu startowego do celu
find_path(Start, Goal, Path) :- find_path(Start, Goal, [Start], Path).
find_path(Goal, Goal, _, [Goal]).
find_path(Start, Goal, Visited, [Start|Path]) :-
    is_connected(Start, Next),
    \+ member(Next, Visited),  % Zapobieganie cyklom
    find_path(Next, Goal, [Next|Visited], Path).

% Przykładowe użycie: find_path(t_spawn, a_plat, Path).







%initializeGame :-
%    retractall(player(_,_)),
%    retractall(enemy(_,_,_)),
%    retractall(bomb(_)),
%    retractall(bombPlanted),
%    retractall(inventory(_,_)),
%    assert(player(terroristSpawn, hasPistol)),
%    assert(inventory(terroristSpawn, [])),
%    assert(enemy(siteA, hasPistol, alive)),
%    assert(enemy(long, hasPistol, alive)),
%    assert(enemy(short, hasPistol, alive)),
%    assert(bomb(terroristSpawn)).
%
%/* INVENTORY MANAGEMENT */
%addItem(Item) :-
%    player(CurrentPos, _),
%    inventory(CurrentPos, CurrentInventory),
%    \+ member(Item, CurrentInventory),
%    retract(inventory(CurrentPos, CurrentInventory)),
%    assert(inventory(CurrentPos, [Item|CurrentInventory])),
%    write(Item), write(' added to inventory.'), nl.
%
%removeItem(Item) :-
%    player(CurrentPos, _),
%    inventory(CurrentPos, CurrentInventory),
%    member(Item, CurrentInventory),
%    delete(CurrentInventory, Item, NewInventory),
%    retract(inventory(CurrentPos, CurrentInventory)),
%    assert(inventory(CurrentPos, NewInventory)),
%    write(Item), write(' removed from inventory.'), nl.
%
%showInventory :-
%    player(CurrentPos, _),
%    inventory(CurrentPos, Inventory),
%    write('Your inventory: '), write(Inventory), nl.
%
%/* ... [Reszta kodu, jak wcześniej] ... */
%
%take(Item) :-
%    player(CurrentPos, _),
%    Item == bomb,
%    bomb(CurrentPos),
%    addItem(bomb),
%    retract(bomb(CurrentPos)),
%    !.
%take(_) :-
%    write('Nothing to take here.'), nl.
%
%/* MAIN */
%startGame :-
%    initializeGame,
%    write('Game started. You are a terrorist on Dust2. Your goal is to plant the bomb at site A.'), nl,
%    showInventory.
%
%gameLoop :-
%    write('Enter your next move (move/2, take/1, plantBomb, showInventory): '), nl,
%    read(Command),
%    call(Command),
%    (bombPlanted -> defendBomb ; gameLoop),
%    !.
%gameLoop :-
%    write('Game over.'), nl.
%
%:- startGame,
%   gameLoop.