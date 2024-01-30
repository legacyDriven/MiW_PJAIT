import numpy as np
import time

#macierz przejscia
t1 = ['R','Su']
p_t1=[[0.4,0.6],[0.3,0.7]]

#macierz emisji
t2 = ['Walk','Shop','Clean']
p_t2=[[0.1,0.4,0.5],[0.6,0.3,0.1]]

print(sum(p_t2[0]))

state = np.random.choice(t1, p= [0.5, 0.5])
n = 20 
for i in range(n):
    if state == 'R':
        activity = np.random.choice(t2, p=p_t2[0])
        print(state)
        print(activity)
        state = np.random.choice(t1, p=p_t1[0])
    elif state == 'Su':
        activity = np.random.choice(t2, p=p_t2[1])
        print(state)
        print(activity)        
        state = np.random.choice(t1,p=p_t1[1])
    print("\n")
    time.sleep(0.5)
  
