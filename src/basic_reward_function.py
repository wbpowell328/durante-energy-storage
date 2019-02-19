from reward_function import reward_function

class basic_reward_function(reward_function):
    
    def __init__(self,name,rew_function,GLB_VARS):
        self.name=name
        self.reward_fnct=rew_function
        self.glb_vars=GLB_VARS
        self.price_vars=GLB_VARS.get_global_variable('Price_Vars')
        self.energy_vars=GLB_VARS.get_global_variable('Energy_Vars')
        self.resource_vars=GLB_VARS.get_global_variable('Resource_Vars')
        
    def get_name(self):
        """
        Return name of reward function
        """
        return self.name
        
    def get_reward(self, state, action):
        """
        Return reward of taking an action in a certain state
        """
        return self.reward_fnct(state, action,self.price_vars, self.energy_vars, self.resource_vars)

    def get_reward_2(self, p,action):
        """
        Return reward of taking an action in a certain state
        """
        return self.reward_fnct(p, action,self.price_vars, self.energy_vars, self.resource_vars)