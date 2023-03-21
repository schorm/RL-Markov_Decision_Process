from multiprocessing import current_process
from operator import index
import numpy as np
Actions=[np.array([0,-1]),np.array([-1,0]),np.array([0,1]),np.array([1,0])]#west,north,east,south
Prob=0.25
#5x5 grid
grid1_size=5
A1_POS=[0,1]
A1_Goal_Pos=[4,1]
B1_Pos=[0,3]
B1_Goal_Pos=[2,3]
discount1=0.85

#7x7 grid
grid2_size=7
A2_POS=[2,1]
A2_Goal_Pos=[6,1]
B2_Pos=[0,5]
B2_Goal_Pos=[3,5]
discount2=0.75

def step1(curr_state,curr_action):
    if curr_state==A1_POS:
        return A1_Goal_Pos,10
    if curr_state==B1_Pos:
        return B1_Goal_Pos,5
    
    next_state=(np.array(curr_state)+curr_action).tolist()
    pos_x,pos_y=next_state
    reward=0.0
    if pos_x<0 or pos_x>=grid1_size or pos_y < 0 or pos_y>=grid1_size:
        reward=-1.0 
        next_state=curr_state
    return next_state,reward

def step2(curr_state,curr_action):
    if curr_state==A2_POS:
        return A2_Goal_Pos,10
    if curr_state==B2_Pos:
        return B2_Goal_Pos,5
    
    next_state=(np.array(curr_state)+curr_action).tolist()
    pos_x,pos_y=next_state
    reward=0.0
    if pos_x<0 or pos_x>=grid2_size or pos_y < 0 or pos_y>=grid2_size:
        reward=-1.0 
        next_state=curr_state
    return next_state,reward

def linear_5():
    A=np.zeros((grid1_size*grid1_size,grid1_size*grid1_size))# A is[25x25]
    for i in range(grid1_size*grid1_size):
        for j in range(grid1_size*grid1_size):
            if(i==j):
                A[i,j]-=1#

    B=np.zeros(grid1_size*grid1_size)# b is [1x25]
    for xi in range (grid1_size):
        for yi in range (grid1_size):
            currpos=[xi,yi]
            index=xi*grid1_size+yi
            for action in Actions:
                next_pos,reward=step1(currpos,action)
                next_x,next_y=next_pos
                next_index=next_x*grid1_size+next_y
                A[index,next_index]+=Prob*discount1
                B[index]+=Prob*reward

    x=np.linalg.solve(A,B)*(-1)
    print("5x5 v(s):")
    print(x.reshape(grid1_size,grid1_size))

def linear_7():
    A=np.zeros((grid2_size*grid2_size,grid2_size*grid2_size))# A is[25x25]
    for i in range(grid2_size*grid2_size):
        for j in range(grid2_size*grid2_size):
            if(i==j):
                A[i,j]-=1#

    B=np.zeros(grid2_size*grid2_size)# b is [1x25]
    for xi in range (grid2_size):
        for yi in range (grid2_size):
            currpos=[xi,yi]
            index=xi*grid2_size+yi
            for action in Actions:
                next_pos,reward=step2(currpos,action)
                next_x,next_y=next_pos
                next_index=next_x*grid2_size+next_y
                A[index,next_index]+=Prob*discount2
                B[index]+=Prob*reward

    x=np.linalg.solve(A,B)*(-1)
    print("7x7 v(s):")
    print(x.reshape(grid2_size,grid2_size))


linear_5()
print()
linear_7()

