:- dynamic player/2.
:- dynamic enemy/3.
:- dynamic bomb/1.
:- dynamic bombPlanted/0.
:- dynamic inventory/2.

initializeGame :-
    retractall(player(_,_)),
    retractall(enemy(_,_,_)),
    retractall(bomb(_)),
    retractall(bombPlanted),
    retractall(inventory(_,_)),
    assert(player(terroristSpawn, hasPistol)),
    assert(inventory(terroristSpawn, [])),
    assert(enemy(siteA, hasPistol, alive)),
    assert(enemy(long, hasPistol, alive)),
    assert(enemy(short, hasPistol, alive)),
    assert(bomb(terroristSpawn)).

/* INVENTORY MANAGEMENT */
addItem(Item) :-
    player(CurrentPos, _),
    inventory(CurrentPos, CurrentInventory),
    \+ member(Item, CurrentInventory),
    retract(inventory(CurrentPos, CurrentInventory)),
    assert(inventory(CurrentPos, [Item|CurrentInventory])),
    write(Item), write(' added to inventory.'), nl.

removeItem(Item) :-
    player(CurrentPos, _),
    inventory(CurrentPos, CurrentInventory),
    member(Item, CurrentInventory),
    delete(CurrentInventory, Item, NewInventory),
    retract(inventory(CurrentPos, CurrentInventory)),
    assert(inventory(CurrentPos, NewInventory)),
    write(Item), write(' removed from inventory.'), nl.

showInventory :-
    player(CurrentPos, _),
    inventory(CurrentPos, Inventory),
    write('Your inventory: '), write(Inventory), nl.

/* ... [Reszta kodu, jak wczeœniej] ... */

take(Item) :-
    player(CurrentPos, _),
    Item == bomb,
    bomb(CurrentPos),
    addItem(bomb),
    retract(bomb(CurrentPos)),
    !.
take(_) :-
    write('Nothing to take here.'), nl.

/* MAIN */
startGame :-
    initializeGame,
    write('Game started. You are a terrorist on Dust2. Your goal is to plant the bomb at site A.'), nl,
    showInventory.

gameLoop :-
    write('Enter your next move (move/2, take/1, plantBomb, showInventory): '), nl,
    read(Command),
    call(Command),
    (bombPlanted -> defendBomb ; gameLoop),
    !.
gameLoop :-
    write('Game over.'), nl.

:- startGame,
   gameLoop.
