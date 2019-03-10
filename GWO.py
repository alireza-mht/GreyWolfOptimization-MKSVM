# -*- coding: utf-8 -*-


import random
import numpy
from solution import solution
import math
import time
from MultiKernelSVM import *

    

def GWO(lb,ub,dim,SearchAgents_no,Max_iter):
    

    # initialize alpha, beta, and delta_pos
    Alpha_pos=numpy.zeros(dim)
    Alpha_score=float("-inf")
    
    Beta_pos=numpy.zeros(dim)
    Beta_score=float("-inf")
    
    Delta_pos=numpy.zeros(dim)
    Delta_score=float("-inf")
    
    #Initialize the positions of search agents
    Positions=numpy.random.uniform(0,1,(SearchAgents_no,dim)) *(ub-lb)+lb
    
    Convergence_curve=numpy.zeros(Max_iter)
    s=solution()

     # Loop counter
    #print("GWO is optimizing  \""+objf.__name__+"\"")
    
    timerStart=time.time()
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")

    z = MultiKernelSVM()
    z.preprocessing()
    pos1=0;
    # Main loop
    a_max = 2
    a_min = 2-(Max_iter-1)*((2)/(Max_iter))
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):

            Positions = binerization(Positions)
            # Return back the search agents that go beyond the boundaries of the search space
            # Positions[i,:]=numpy.clip(Positions[i,:], lb, ub)

            # Calculate objective function for each search agent
            fitness=z.fit(Positions[i,:])

            # Update Alpha, Beta, and Delta

            if fitness>Alpha_score :
                Alpha_score=fitness; # Update alpha
                Alpha_pos = assignValue(Positions[i,:])

            
            if (fitness<Alpha_score and fitness>Beta_score ):
                Beta_score=fitness  # Update beta
                Beta_pos=assignValue(Positions[i,:])
            
            
            if (fitness<Alpha_score and fitness<Beta_score and fitness>Delta_score):
                Delta_score=fitness # Update delta
                Delta_pos=assignValue(Positions[i,:])
            
        
        
        
        # a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        ln = math.log((a_min/a_max),2)
        Alpha_a = a_max* math.exp(pow(((l+1)/(Max_iter)),2)*ln)
        Beta_a =a_max* math.exp(pow(((l+1)/(Max_iter)),3)*ln)
        Delta_a = (Alpha_a + Beta_a)/2
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]
                # r1 = 0.2*Alpha_score
                # r2= 0.2*Alpha_score

                A1=2*Alpha_a*r1-Alpha_a # Equation (3.3)
                C1=2*r2 # Equation (3.4)
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]) # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha # Equation (3.6)-part 1
                           
                r1=random.random()
                r2=random.random()

                # r1 = random.random()  # r1 is a random number in [0,1]
                # r2 = random.random()  # r2 is a random number in [0,1]


                A2=2*Beta_a*r1-Beta_a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]) # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta # Equation (3.6)-part 2
                #
                r1=random.random()
                r2=random.random()


                # r1 = random.random()  # r1 is a random number in [0,1]
                # r2 = random.random()  # r2 is a random number in [0,1]


                A3=2*Delta_a*r1-Delta_a # Equation (3.3)
                C3=2*r2 # Equation (3.4)
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta # Equation (3.5)-part 3
                
                Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)
                
            
        
        
        Convergence_curve[l]=Alpha_score;


        if (l%1==0):
               print(['At iteration '+ str(l)+ ' the best fitness is '+ str(Alpha_score)]);
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=Convergence_curve
    s.optimizer= str(Alpha_pos)
    s.objfname="IGWO (0.2fitness)-1000data- 10 pop - 5itr "
    
    
    
    
    return s

def assignValue(pos):
    wolfPos = []
    for i in pos:
        wolfPos.append(i)
    return np.array(wolfPos)

def binerization(pos):
    k = 0
    h = 0
    for i in pos:
        h = 0
        for j in i:
            if (j <= 0.5):
                pos[k, h] = 0
            else:
                pos[k, h] = 1
            h+=1
        k+=1
    return pos