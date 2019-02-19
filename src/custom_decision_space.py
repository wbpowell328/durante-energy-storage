from decision_space import decision_space

class custom_decision_space(decision_space):
    
    def __init__(self,GLB_VARS):
        
        self.GLB_VARS=GLB_VARS
    
    def allowed_actions(self, current_state):
        '''
        From the current state, return a list of feasible actions.
        An action is a dictionaries defining the decision
        '''
        return