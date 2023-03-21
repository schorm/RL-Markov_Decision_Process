
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

def value2():
    stateValue=np.zeros((grid2_size,grid2_size))
    
    while(True):
        newstatevalue=np.zeros((grid2_size,grid2_size))
        for i in range(grid2_size):
            for j in range(grid2_size):
                v_2=[]
                for action in Actions:
                    currpos=[i,j]
                    next_pos,reward=step2(currpos,action)
                    next_x,next_y=next_pos
                    v_2.append(Prob*(reward+discount2*stateValue[next_x,next_y]))
                newstatevalue[i][j]=np.sum(v_2)
        if np.sum(np.abs(stateValue-newstatevalue))==0:# calcuate completed
            print("7x7 v(s):")
            print(newstatevalue)
            break
        stateValue=newstatevalue


def value1():
    stateValue=np.zeros((grid1_size,grid1_size))
    
    while(True):
        newstatevalue=np.zeros((grid1_size,grid1_size))
        for i in range(grid1_size):
            for j in range(grid1_size):
                v_1=[]
                for action in Actions:
                    currpos=[i,j]
                    next_pos,reward=step1(currpos,action)
                    next_x,next_y=next_pos
                    v_1.append(Prob*(reward+discount1*stateValue[next_x,next_y]))
                newstatevalue[i][j]=np.sum(v_1)
        if np.sum(np.abs(stateValue-newstatevalue))==0:# calcuate completed
            print("5x5 v(s):")
            print(newstatevalue)
            break
        stateValue=newstatevalue

value1()
print()
value2()


