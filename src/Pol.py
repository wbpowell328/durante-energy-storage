import abc

class Pol():
    __metaclass__ = abc.ABCMeta   
    
    @abc.abstractmethod
    def get_name(self):
        """
        Return name of policy
        """
        return
        
    @abc.abstractmethod
    def get_all_params(self):
        """
        Return params of policy, in list, in order
        """
        return
    
    @abc.abstractmethod
    def set_new_params(self, params):
        """
        Update policy parameters
        """
    
    @abc.abstractmethod
    def offline_stage(self):
        """
        Perform any computations necessary prior to online decisions,
        if not necessary, just return from function
        """
        return
    
    @abc.abstractmethod
    def learn_after_decision(self, state, decision, reward):
        """
        Learn, after returning decision, from observing reward and state
        """
        return
    
    @abc.abstractmethod
    def get_learn_after_each_decision(self):
        """
        return boolean whether or not we learn after each decision
        """
        return 
    
    @abc.abstractmethod
    def learn_after_trial(self, cumulative_reward):
        """
        Learn after entire trial
        """
        return 
    
    @abc.abstractmethod
    def get_learn_after_each_trial(self):
        """
        return boolean whether or not we learn after each trial (policy evaluation)
        """
        return 
    
    
    @abc.abstractmethod
    def decision(self, state):
        """
        Make decision based on current state of system
        """
        return