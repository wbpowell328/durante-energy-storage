from Pol import Pol
import numpy as np

class policy_CFA(Pol):
    
    def __init__(self, name, FilePath, GLB_VARS, params):
        self.name=name
        self.glb_vars=GLB_VARS
        self.params=params
        self.nodes=GLB_VARS.get_global_variable('Nodes')
        self.feasible_decisions=GLB_VARS.get_global_variable('Decision_Space')
        self.r_fnct=GLB_VARS.get_global_variable('Reward_Function')
    
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
        Dec_Space=self.feasible_decisions.allowed_actions(state)
        Max=-1*np.inf
        Ret=None
        for D in Dec_Space:
            One_Step_Rew=self.r_fnct(state, D, self.nodes)
            EC_term=self.params[0]*(D['G:R_1']+D['E:R_1']+D['R_1:L'])
            V=One_Step_Rew-EC_term
            if V>Max:
                Max=V
                Ret=D
        return Ret