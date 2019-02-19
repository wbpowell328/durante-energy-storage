from Node import Node
import numpy as np
import bisect
import random

def round_to_array(Vec,Val):
    Ind=bisect.bisect_right(Vec,Val)-1
    if Ind<0:
        Ret=Vec[0]
    elif Ind==len(Vec)-1:
        Ret=Vec[Ind]
    else:
        diff=abs(Val-Vec[Ind])-abs(Val-Vec[Ind+1])
        if diff>0:
            Ret=Vec[Ind+1]
        else:
            Ret=Vec[Ind]
    return Ret

def round_to_array_ind(Vec,Val):
    Ind=bisect.bisect_right(Vec,Val)-1
    if Ind<0:
        Ret=0
    elif Ind==len(Vec)-1:
        Ret=Ind
    else:
        diff=abs(Val-Vec[Ind])-abs(Val-Vec[Ind+1])
        if diff>0:
            Ret=Ind+1
        else:
            Ret=Ind
    return Ret


class basic_resource_node(Node):
    
    def __init__(self, Name, FilePath, headers,params):
        
        Plen=len(params)
        self.name=Name
        self.max_cap=30000
        self.d_int=float(self.max_cap)/30
        self.max_charge_rate=float(self.d_int)
        self.conv_rate=1.0
        self.loss_rate=1.0
        if Plen>=1:
            self.d_int=float(params[0])
        if Plen>=2:
            self.max_cap=float(params[1])
        if Plen>=3:
            self.max_charge_rate=float(params[2])
        if Plen>=4:
            self.conv_rate=float(params[3])
        if Plen>=5:
            self.loss_rate=float(params[4])
            

        self.all_states=list(np.arange(0.0,self.max_cap+self.d_int,self.d_int))
        self.all_states_indicies=range(len(self.all_states))

        
        self.maximum=max(self.all_states)
        
    def get_name(self):
        """
        Return name of node (string)
        """
        return self.name
    
    def get_discretization_interval(self):
        """
        Return the interval which this node/process is discretized by (float or int)
        """
        return self.d_int
    
    def get_time_step(self):
        """
        Return time step of node/process in integer multiple of quickest changing process (int)
        """
        return 1   

    def get_postds_value(self, postds):
        
        """
        Return value of node (the state which the index represents)
        """        
        return self.all_states[postds[self.name]]
    
    def get_preds_value(self, preds):
        """
        Return value of node (the state which the index represents)
        """
        return self.all_states[preds[self.name]]
    
    def get_postds(self,postds):
        """
        Return post decision state value of node (index, an int)
        """
        return postds[self.name]
    
    def get_preds(self,preds):
        """
        Return pre decision state value of node (index, an int)
        """
        return preds[self.name]
    
    def get_max(self):
        """
        Return maximum value of node (float/int)
        """
        return self.maximum
    
    def get_min(self):
        """
        Return minimum value of node (float/int)
        """
        return 0.0
    
    def get_forecast(self, t):
        """
        Return forecast of node at time t (int)
        """
        return 0
    
    def get_possible_postds(self, t):
        """
        Return list of possible post decision states of node at time t
        """
        return self.all_states_indicies
    
    def get_possible_preds(self, t):
        """
        Return list of possible pre decision states of node at time t
        """
        return self.all_states_indicies
    
    def get_postds_to_preds_probabilities(self, postds):
        """
        Return a 2 element list of the forward states (list comprising first
        element) and their probabilities (list comprising second element)
        
        Possible simplified Markov chain representation of process 
        """
        
        return [[postds[self.name]],[1.0]]
    
    def pre_to_post_ds_transition(self,preds,dec):
        """
        After a decision is made, return post decision state (return int index)
        """
        
        sumFlow=0;
        for d in dec:
            D=d.split(':')
            if D[0]==self.name:
                if dec[d]<0.0:
                    sumFlow-=dec[d]*self.conv_rate
                else:
                    sumFlow-=dec[d]
            if D[1]==self.name:
                if dec[d]<0.0:
                    sumFlow+=dec[d]*self.conv_rate
                else:
                    sumFlow+=dec[d]
        
        endval=self.all_states[preds[self.name]]+sumFlow
        
        return round_to_array_ind(self.all_states,endval)
    
    def post_to_pre_ds_transition(self, postds):
        """
        Make transition from post to pre decision state (return int index)
        """
        return round_to_array_ind(self.all_states,self.all_states[postds[self.name]]*self.loss_rate)
    
    def get_initial_preds(self):
        """
        Return an initial pre decision state at time t
        """
        return len(self.all_states)/2
    
    def set_random_seed(self, rint):
        """
        set specific random seed of random number generator
        """
        random.seed(rint)
        
#    def set_new_params(self, params):
#        """
#        Set new parameters for battery
#        """
#        self.max_cap=params[0]
#        self.max_charge_rate=float(params[1])
#        self.conv_rate=float(params[2])
#        self.d_int=params[4]
#        self.all_states=np.arange(0.0,self.max_cap+self.d_int,self.d_int)
#        self.leakage_rate=float(params[3])
#        self.maximum=max(self.all_states)
#        
#    def get_all_params(self):
#        """
#        Get parameters for battery
#        """
#        params=[]
#        params.append(self.max_cap)
#        params.append(self.max_charge_rate)
#        params.append(self.conv_rate)
#        params.append(self.leakage_rate)
#        params.append(self.d_int)
#        return params


        
       
    

    

    

        
        
        