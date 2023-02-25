#CSE 691: Topics in Reinforcement Learning Spring 2023, Homework Assignment 2 Implementation
#Jamison Weber

import numpy as np
import random
import pandas as pd

N=10
p_1=0.25
p_2=0.25
x_0=2
x_bar=10
beta=1.4

#Utilities
def reset_mem():
    for i in range(x_0-N-1,x_0+N+1):
        mem[i]={}
        for j in range(0,N): 
            mem[i][j]=' '

    for i in range(x_0-N-1,x_0+N+1):
        policy_mem[i]={}
        for j in range(0,N+1): 
            policy_mem[i][j]=' '

def display_results(): 
    policy_val_arr=[]
    policy_arr=[]
    for i in range(x_0-N-1,x_0+N+1):
        policy_row=[]
        policy_val_row=[]
        for j in range(0,N): 
            policy_row.append(policy_mem[i][j])
            if mem[i][j]!=' ':
                policy_val_row.append(np.round(mem[i][j],3))
            else:
                policy_val_row.append(mem[i][j])
        policy_arr.append(policy_row)
        policy_val_arr.append(policy_val_row)
    policy_arr=np.flip(np.array(policy_arr),axis=0)
    policy_val_arr=np.flip(np.array(policy_val_arr),axis=0)
    #You may need to adjust the window size below if you change the parameters
    policyPlot = pd.DataFrame(policy_arr[N-x_bar:N+3]).rename(index = lambda s: abs(s-x_bar-2))
    print(policyPlot)
    valuePlot = pd.DataFrame(policy_val_arr[N-x_bar:N+3]).rename(index = lambda s: abs(s-x_bar-2))
    print(valuePlot)


# Part a: (Exact Dynamic Programming)
def J(x,k):
    if k==N:
        mem[x][k]=x
    else:
        if x==0:
            if mem[x][k]==' ':
                expected_future=p_1*J(x+1,k+1)+(1-p_1)*J(x,k+1)
                if x>=expected_future:
                    mem[x][k]=x
                    policy_mem[x][k]='S'
                else:
                    mem[x][k]=expected_future
                    policy_mem[x][k]='D'
        elif x==x_bar:
            if mem[x][k]==' ':
                expected_future=p_2*J(x-1,k+1)+(1-p_2)*J(x,k+1)
                if x>=expected_future:
                    mem[x][k]=x
                    policy_mem[x][k]='S'
                else:
                    mem[x][k]=expected_future
                    policy_mem[x][k]='D'
        else:
            if mem[x][k]==' ':
                expected_future=p_1*J(x+1,k+1)+(1-(p_1+p_2))*J(x,k+1)+p_2*J(x-1,k+1)
                if x>=expected_future:
                    mem[x][k]=x
                    policy_mem[x][k]='S'
                else:
                    mem[x][k]=expected_future
                    policy_mem[x][k]='D'     
    return mem[x][k]

mem={}
policy_mem={}
reset_mem()
opt=J(x_0,0,)
print('Optimal reward is '+str(opt))
display_results()

#Part b: (Heuristic Base Policy and Monte Carlo Simulation)
mem={}
policy_mem={}
reset_mem()

def H(x,k,x_init,write):
    if x>=beta*x_init or k==N:
        if x>=beta*x_init:
            if write:
                policy_mem[x][k]='S'
        if write:   
            mem[x][k]=x
        return x
    else:
        if write:
            policy_mem[x][k]='D'
        if x==0:
            expected_future=p_1*H(x+1,k+1,x_init,write)+(1-(p_1))*H(x,k+1,x_init,write)
            if write:
                mem[x][k]=expected_future
            return expected_future
        elif x==x_bar:
            expected_future=(1-(p_2))*H(x,k+1,x_init,write)+p_2*H(x-1,k+1,x_init,write)
            if write:
                mem[x][k]=expected_future
            return expected_future
        else:
            expected_future=p_1*H(x+1,k+1,x_init,write)+(1-(p_1+p_2))*H(x,k+1,x_init,write)+p_2*H(x-1,k+1,x_init,write)
            if write:
                mem[x][k]=expected_future
            return expected_future

print('******************************************************************')
print('Heuristic Reward Starting From x_0 (Exact DP)')
print(H(x_0,0,x_0,True))
display_results()

def monte_carlo_H(x,k,x_init,numSamples):
    big_summation=0
    for i in range(numSamples):
        summation=x
        for j in range(N-k):
            if summation>=x_init*beta:
                break
            r=random.random()
            if summation==x_bar:
                if r<=p_2:
                    summation-=1
            elif summation==0:
                if r<=p_1:
                    summation+=1
            else:
                if r<=p_1:
                    summation+=1
                elif p_1<r<=p_1+p_2:
                    summation-=1     
        big_summation+=summation
    return big_summation/numSamples
            
#Part c: (Rollout)
mem={}
policy_mem={}
reset_mem()

def rollout(x,k):
    if k==N:
        mem[x][k]=x
    else:
        if x==0:
            expected_future=p_1*H(x+1,k+1,x+1,False)+(1-p_1)*H(x,k+1,x,False)
            if x>=expected_future:
                mem[x][k]=x
                policy_mem[x][k]='S'
            else:
                mem[x][k]=expected_future
                policy_mem[x][k]='D'
        elif x==x_bar:
            expected_future=p_2*H(x-1,k+1,x-1,False)+(1-p_2)*H(x,k+1,x,False)
            if x>=expected_future:
                mem[x][k]=x
                policy_mem[x][k]='S'
            else:
                mem[x][k]=expected_future
                policy_mem[x][k]='D'
        else:
            expected_future=p_1*H(x+1,k+1,x+1,False)+(1-(p_1+p_2))*H(x,k+1,x,False)+p_2*H(x-1,k+1,x-1,False)
            if x>=expected_future:
                mem[x][k]=x
                policy_mem[x][k]='S'
            else:
                mem[x][k]=expected_future
                policy_mem[x][k]='D'
        #Fill out the rest of the table
        if k != N:
            if x+1 <= x_bar:
                rollout(x+1,k+1)
            rollout(x,k+1)
            if x-1 >= 0: 
                rollout(x-1,k+1)
    return mem[x][k]

sol=rollout(x_0,0)
print('******************************************************************')
print('Rollout reward at x_0  with exact DP Heuristic is '+str(sol))
display_results()


mem={}
policy_mem={}
reset_mem()
#Rollout with Monte Carlo Simulation
def monte_carlo_rollout(x,k,samples):
        if k==N:
            mem[x][k]=x
        else:
            if x==0:
                if mem[x][k]==' ':
                    expected_future=p_1*monte_carlo_H(x+1,k+1,x+1,samples)+(1-p_1)*monte_carlo_H(x,k+1,x,samples)
                    if x>=expected_future:
                        mem[x][k]=x
                        policy_mem[x][k]='S'
                    else:
                        mem[x][k]=expected_future
                        policy_mem[x][k]='D'
            elif x==x_bar:
                if mem[x][k]==' ':
                    expected_future=p_2*monte_carlo_H(x-1,k+1,x-1,samples)+(1-p_2)*monte_carlo_H(x,k+1,x,samples)
                    if x>=expected_future:
                        mem[x][k]=x
                        policy_mem[x][k]='S'
                    else:
                        mem[x][k]=expected_future
                        policy_mem[x][k]='D'
            else:
                if mem[x][k]==' ':
                    expected_future=p_1*monte_carlo_H(x+1,k+1,x+1,samples)+(1-(p_1+p_2))*monte_carlo_H(x,k+1,x,samples)+p_2*monte_carlo_H(x-1,k+1,x-1,samples)
                    if x>=expected_future:
                        mem[x][k]=x
                        policy_mem[x][k]='S'
                    else:
                        mem[x][k]=expected_future
                        policy_mem[x][k]='D'
            #Fill out the rest of the table
            if k != N:
                if x+1 <= x_bar:
                    monte_carlo_rollout(x+1,k+1,samples)
                monte_carlo_rollout(x,k+1,samples)
                if x-1 >= 0: 
                    monte_carlo_rollout(x-1,k+1,samples)    
        return mem[x][k]
display_monte_carlo = False
if display_monte_carlo:
    sol=monte_carlo_rollout(x_0,0,20)
    print('******************************************************************')
    print('Monte Carlo Rollout reward at x_0  with 20 samples is '+str(sol))
    display_results()

    mem={}
    policy_mem={}
    reset_mem()
    sol=monte_carlo_rollout(x_0,0,200)
    print('******************************************************************')
    print('Monte Carlo Rollout reward at x_0  with 200 samples is '+str(sol))
    display_results()

print('******************************************************************')
total_sol=0
computed_values=[]
for a in range(400):
    mem={}
    policy_mem={}
    reset_mem()
    sol=monte_carlo_rollout(x_0,0,20)
    computed_values.append(sol)
    total_sol+=sol
total_sol/=400
print("Average Rollout Reward at x_0 with Monte Carlo Heuristic (20 samples) is "+str(total_sol) )
variance=0
for v in computed_values:
    variance+=(v-total_sol)**2
variance=variance/399
print("Rollout Reward Sample Variance at x_0 with Monte Carlo Heuristic (20 samples) is "+str(variance) )

total_sol=0
computed_values=[]
for a in range(400): 
    mem={}
    policy_mem={}
    reset_mem()
    sol=monte_carlo_rollout(x_0,0,200)
    computed_values.append(sol)
    total_sol+=sol

total_sol/=400
print("Average Rollout Reward at x_0 with Monte Carlo Heuristic (200 samples) is "+str(total_sol) )
variance=0
for v in computed_values:
    variance+=(v-total_sol)**2
variance=variance/399
print("Rollout Reward Sample Variance at x_0 with Monte Carlo Heuristic (200 samples) is "+str(variance) )

# Part d: (multistep lookahead)

#d is the number of steps to look ahead
def multistep_rollout(x,k,d,current_level=0,start_k=0):
    #Fill out the rest of the table
    if k+d < N:
        if x+1 <= x_bar:
            multistep_rollout(x+1,k+1,d,0,k+1)
        multistep_rollout(x,k+1,d,0,k+1)
        if x-1 >= 0: 
            multistep_rollout(x-1,k+1,d,0,k+1)   
    if current_level==d:
        return H(x,d+start_k,x,False)
    else:
        if x==0:
            expected_future=p_1*multistep_rollout(x+1,k+1,d,current_level+1,start_k)+(1-p_1)*multistep_rollout(x,k+1,d,current_level+1,start_k)
            if x>=expected_future:
                if current_level == 0:
                    mem[x][k]=x
                    policy_mem[x][k]='S'
                return x
            else:
                if current_level == 0:
                    mem[x][k]=expected_future
                    policy_mem[x][k]='D'
                return expected_future
        elif x==x_bar:
            expected_future=p_2*multistep_rollout(x-1,k+1,d,current_level+1,start_k)+(1-p_2)*multistep_rollout(x,k+1,d,current_level+1,start_k)
            if x>=expected_future:
                if current_level == 0:
                    mem[x][k]=x
                    policy_mem[x][k]='S'
                return x
            else:
                if current_level == 0:
                    mem[x][k]=expected_future
                    policy_mem[x][k]='D'
                return expected_future
        else:
            expected_future=p_1*multistep_rollout(x+1,k+1,d,current_level+1,start_k)+(1-(p_1+p_2))*multistep_rollout(x,k+1,d,current_level+1, start_k)+p_2*multistep_rollout(x-1,k+1,d,current_level+1, start_k)
            if x>=expected_future:
                if current_level == 0:
                    mem[x][k]=x
                    policy_mem[x][k]='S'
                return x
            else:
                if current_level == 0:
                    mem[x][k]=expected_future
                    policy_mem[x][k]='D' 
                return expected_future   


mem={}
policy_mem={}
reset_mem()
sol=multistep_rollout(x_0,0,2,0,0)
print('******************************************************************')
print("2 step lookahead rollout solution is", sol)
display_results()



   