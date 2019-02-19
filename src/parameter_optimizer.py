import numpy as np
from copy import copy
import itertools
import os
import random

def cartesian_product(List_of_Lists):
    '''
    Cartesian Product, each dimension is a variable whose name is the key to
    a dictionart with possible values for that variable as the value.
    Used to form a dictionary of full state viarables where the key
    is the state variable in string form and the values are a two element list:
    the first element is the state variable itself, the second is initially 
    np.nan, spot used for representing the value of the state if visited
    '''
    return_combos={}
    poss_combos=list(itertools.product(*List_of_Lists))
    for x in poss_combos:
        y=tuple(x)
        return_combos[y]=[0,-np.inf,0]
    return return_combos

class parameter_optimizer():

    def __init__(self, GLB_VARS, InputString):
        self.GLB_VARS=GLB_VARS
        self.Pols=GLB_VARS.get_global_variable('Policies')
        self.r_function=GLB_VARS.get_global_variable('Reward_Function')
        self.InputString=InputString
        self.nodes=GLB_VARS.get_global_variable('Nodes')   
        
    def optimize_system_parameters(self,params,file_name):
        
        #Not exactly ready
        
        os.chdir("..")
        os.chdir("data")
        f=open(file_name,'w')
        f.write('Optimize System Parameters'+os.linesep)
        f.write(self.InputString)
        
        time=self.GLB_VARS.get_global_variable('Horizon')
        
        state={}
        for node in self.nodes:
            state[node]=self.nodes[node].get_initial_preds()
        state['T']=0
     
        #2nd param is name of the policy
        pol_name=params[1]
        
        #3rd param is name of the component
        Name=params[1]
        
        #4th param is a list giving [min, max] for each policy parameter
        Ranges=params[3]
        
        #5th param is a list of the same size as the 4th parmameter where each
        #element represents how many evenly spaced values of the parameter we want to consider
        Spacing=params[3]
        
        #1st param is the number of noisy observations of the function we get to see for each parameter set
        trials=params[0]
        
        Values=[]
        for i in xrange(len(Ranges)):
            V=np.linspace(Ranges[i][0],Ranges[i][1],Spacing[i])
            Values.append(list(V))
        Param_Values=cartesian_product(Values)
        
        Rewards={}
        Cumulative_Rewards={}
        
        for pol in Param_Values:
            Rewards[pol]=np.zeros((time,trials))
            Cumulative_Rewards[pol]=np.zeros((time,trials))
            
        for i in xrange(trials):
            state={}
            for node in self.nodes:
                state[node]=self.nodes[node].get_initial_preds()
            state['T']=0
            States={}
            for pol in Param_Values:
                States[pol]=copy(state)
            for j in xrange(time):
                #Same randomn outcomes for each node in a trial for a 'fair' comparison
                SharedRandomSeeds={}
                for node in self.nodes:
                    SharedRandomSeeds[node]=random.randint(1,1000000)
                for pol in Param_Values:
                    
                    self.nodes[Name]=pol
    
                    Dec=self.Pols[pol_name].decision(States[pol])
                            
                    ###RECIEVE REWARD AND ACCUMULATE
                    Rewards[pol][j,i]=self.r_function(States[pol],Dec,self.nodes)
                    
                    if j==0:
                        Cumulative_Rewards[pol][j,i]=Rewards[pol][j,i]
                    else:
                        Cumulative_Rewards[pol][j,i]=Cumulative_Rewards[pol][j-1,i]+Rewards[pol][j,i]
                    
                    ###Make pre-post decision transition
                    PostDecSt={}
                    for node in self.nodes:
                        PostDecSt[node]=self.nodes[node].pre_to_post_ds_transition(States[pol],Dec)
                    States[pol]=PostDecSt
                    States[pol]['T']=j
                    ###Make post-pre decision transition
                    PreDecSt={}
                    for node in self.nodes:
                        self.nodes[node].set_random_seed(SharedRandomSeeds[node])
                        PreDecSt[node]=self.nodes[node].post_to_pre_ds_transition(States[pol])
                    States[pol]=PreDecSt
                    States[pol]['T']=j+1
            
            for pol in Param_Values:    
                Param_Val=Param_Values[pol]
                Param_Val[0]+=1
                ##1st element is number of trials for specific combo
                ##2nd is estimate of mean
                ##3rd is estimate of variance
                ##recursive updating, (frequentist)
                n=float(Param_Val[0])
                R=Cumulative_Rewards[pol][time-1,i]
                if n==1:
                    Param_Val[1]=R
                elif n==2:
                    Param_Val[2]=(1/n)*np.power(R-Param_Val[1],2)
                    Param_Val[1]=(1-1/n)*Param_Val[1]+(1/n)*R
                else:
                    Param_Val[2]=((n-2)/(n-1))*Param_Val[2]+(1/n)+(1/n)*np.power(R-Param_Val[1],2)
                    Param_Val[1]=(1-1/n)*Param_Val[1]+(1/n)*R
                Param_Values[pol]=Param_Val
            
        Max=-np.inf
        MP=None
        for p in Param_Values:
            if Param_Values[p][1]>Max:
                MP=p
                Max=Param_Values[p][1]
                
        print MP
        print Max   
        os.chdir('..')
        os.chdir('data')
        f=open(file_name,'w')
        f.write('Optimize System Parameters'+ os.linesep)
        f.write(self.InputString)
        Str="Best Parameters (in terms of Mean Estimate): "+str(MP)
        Str=Str+" Number of Trails: "+str(Param_Values[MP][0])
        Str=Str+" Mean Estimate: "+str(Param_Values[MP][1])
        Str=Str+" Variance Estimate: "+str(Param_Values[MP][2])+"\n\n"
        f.write(Str)
        for p in Param_Values:
            Str="Parameters: "+str(p)
            Str=Str+" Number of Trails: "+str(Param_Values[p][0])
            Str=Str+" Mean Estimate: "+str(Param_Values[p][1])
            Str=Str+" Variance Estimate: "+str(Param_Values[p][2])+"\n"
            f.write(Str)
        f.close()
            
            
        
    def optimize_policy_parameters(self,params,file_name):
        
        os.chdir("..")
        os.chdir("data")
        f=open(file_name,'w')
        f.write('Optimize System Parameters'+os.linesep)
        f.write(self.InputString)
        
        time=self.GLB_VARS.get_global_variable('Horizon')
        
        #2nd param is name of the policy
        pol_name=params[1]
        
        #3rd param is a list giving [min, max] for each policy parameter
        Ranges=params[2]
        
        #4th param is a list of the same size as the 3rd parmameter where each
        #element represents how many evenly spaced values of the parameter we want to consider
        Spacing=params[3]
        
        #1st param is the number of noisy observations of the function we get to see for each parameter set
        trials=params[0]
        
        Values=[]
        for i in xrange(len(Ranges)):
            V=np.linspace(Ranges[i][0],Ranges[i][1],Spacing[i])
            Values.append(list(V))
        Param_Values=cartesian_product(Values)
        
        Rewards={}
        Cumulative_Rewards={}
        
        for pol in Param_Values:
            Rewards[pol]=np.zeros((time,trials))
            Cumulative_Rewards[pol]=np.zeros((time,trials))
            
        for i in xrange(trials):
            state={}
            for node in self.nodes:
                state[node]=self.nodes[node].get_initial_preds()
            state['T']=0
            States={}
            for pol in Param_Values:
                States[pol]=copy(state)
            for j in xrange(time):
                #Same randomn outcomes for each node in a trial for a 'fair' comparison
                SharedRandomSeeds={}
                for node in self.nodes:
                    SharedRandomSeeds[node]=random.randint(1,1000000)
                for pol in Param_Values:
                    
                    self.Pols[pol_name].set_new_params(self.GLB_VARS,pol)
    
                    Dec=self.Pols[pol_name].decision(States[pol])
                            
                    ###RECIEVE REWARD AND ACCUMULATE
                    Rewards[pol][j,i]=self.r_function(States[pol],Dec,self.nodes)
                    
                    if j==0:
                        Cumulative_Rewards[pol][j,i]=Rewards[pol][j,i]
                    else:
                        Cumulative_Rewards[pol][j,i]=Cumulative_Rewards[pol][j-1,i]+Rewards[pol][j,i]
                    
                    ###Make pre-post decision transition
                    PostDecSt={}
                    for node in self.nodes:
                        PostDecSt[node]=self.nodes[node].pre_to_post_ds_transition(States[pol],Dec)
                    States[pol]=PostDecSt
                    States[pol]['T']=j
                    ###Make post-pre decision transition
                    PreDecSt={}
                    for node in self.nodes:
                        self.nodes[node].set_random_seed(SharedRandomSeeds[node])
                        PreDecSt[node]=self.nodes[node].post_to_pre_ds_transition(States[pol])
                    States[pol]=PreDecSt
                    States[pol]['T']=j+1
            
            for pol in Param_Values:    
                Param_Val=Param_Values[pol]
                Param_Val[0]+=1
                ##1st element is number of trials for specific combo
                ##2nd is estimate of mean
                ##3rd is estimate of variance
                ##recursive updating, (frequentist)
                n=float(Param_Val[0])
                R=Cumulative_Rewards[pol][time-1,i]
                if n==1:
                    Param_Val[1]=R
                elif n==2:
                    Param_Val[2]=(1/n)*np.power(R-Param_Val[1],2)
                    Param_Val[1]=(1-1/n)*Param_Val[1]+(1/n)*R
                else:
                    Param_Val[2]=((n-2)/(n-1))*Param_Val[2]+(1/n)+(1/n)*np.power(R-Param_Val[1],2)
                    Param_Val[1]=(1-1/n)*Param_Val[1]+(1/n)*R
                Param_Values[pol]=Param_Val
            
        Max=-np.inf
        MP=None
        for p in Param_Values:
            if Param_Values[p][1]>Max:
                MP=p
                Max=Param_Values[p][1]
                
        print MP
        print Max
        os.chdir('..')
        os.chdir('data')
        f=open(file_name,'w')
        f.write('Optimize Policy Parameters'+ os.linesep)
        f.write(self.InputString)
        Str="Best Parameters (in terms of Mean Estimate): "+str(MP)
        Str=Str+" Number of Trails: "+str(Param_Values[MP][0])
        Str=Str+" Mean Estimate: "+str(Param_Values[MP][1])
        Str=Str+" Variance Estimate: "+str(Param_Values[MP][2])+"\n\n"
        f.write(Str)
        for p in Param_Values:
            Str="Parameters: "+str(p)
            Str=Str+" Number of Trails: "+str(Param_Values[p][0])
            Str=Str+" Mean Estimate: "+str(Param_Values[p][1])
            Str=Str+" Variance Estimate: "+str(Param_Values[p][2])+"\n"
            f.write(Str)
        f.close()