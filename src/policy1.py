from Pol import Pol

class policy1(Pol):
    
    def __init__(self, name, FilePath, GLB_VARS, params):
        self.name=name
        self.glb_vars=GLB_VARS
        self.params=params
        self.nodes=GLB_VARS.get_global_variable('Nodes')        
    
    def get_name(self):
        return self.name
        
    def offline_stage(self):
        """
        can leave blank
        """
        return
        
    def get_all_params(self):
        return self.params
    
    def learn_after_decision(self, state, decision, reward):
        """
        Learn, after returning decision, from observing reward and state
        """
        return
    
    def get_learn_after_each_decision(self):
        """
        return boolean whether or not we learn after each decision
        """
        return 0

    def learn_after_trial(self, cumulative_reward):
        """
        Learn after entire trial
        """
        return 
    
    def get_learn_after_each_trial(self):
        """
        return boolean whether or not we learn after each trial (policy evaluation)
        """
        return 0
        
    def set_new_params(self, GLB_VARS, params):
        self.params=params
        
    def decision(self, state):
        """
        from state, return decision according to policy
        """

        return