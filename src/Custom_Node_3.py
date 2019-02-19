from Node import Node
import random

class Custom_Node_3(Node):
    
    def __init__(self, Name, FilePath, params):
        self.name=Name
        self.fp=FilePath
        self.pars=params
    
    def get_name(self):
        """
        Return name of Node (string)
        """
        return self.name
    
    def get_discretization_interval(self):
        """
        Return the interval which this node/process is discretized by (float)
        """
        return        
    
    def get_time_step(self):
        """
        Return time step of node in integer multiple of quickest changing process (int)
        """
        return   
    
    def get_postds_value(self, postds):
        """
        Return value of node (the state which the index represents)
        """
        return
    
    def get_preds_value(self, preds):
        """
        Return value of node (the state which the index represents)
        """
        return
    
    def get_postds(self,postds):
        """
        Return post decision state value of node (index, an int)
        """
        return
    
    def get_preds(self,preds):
        """
        Return pre decision state value of node (index, an int)
        """
        return
        
    def get_max(self):
        """
        Return maximum value of node (float)
        """
        return
    
    def get_min(self):
        """
        Return minimum value of node (float)
        """
        return  
      
    def get_forecast(self, t):
        """
        Return forecast of node at time t (int)
        """
        return
    
    def get_possible_postds(self, t):
        """
        Return list of possible post decision states of node at time t
        """
        return
        
    def get_possible_preds(self, t):
        """
        Return list of possible pre decision states of node at time t
        """
        return
        
    def get_postds_to_preds_probabilities(self, postds):
        """
        Return a 2 element list of the forward states (list comprising first
        element) and their probabilities (list comprising second element)
        
        Possible simplified Markov chain representation of process 
        """
        return        
    
    def pre_to_post_ds_transition(self,preds,dec):
        """
        After a decision is made, return post decision state (return int index)
        """
        return
    
    def post_to_pre_ds_transition(self, postds):
        """
        Make transition from post to pre decision state (return int index)
        """
        return
    
    def get_initial_preds(self):
        """
        Return an initial pre decision state at time t
        """
        return
    
    def set_random_seed(self, rint):
        """
        set specific random seed of random number generator
        """
        random.seed(rint)
        return
    
    
    