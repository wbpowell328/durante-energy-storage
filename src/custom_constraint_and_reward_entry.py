from Constraint_and_Reward_Function_Entry import Constraint_and_Reward_Function_Entry

class custom_constraint_and_reward_entry(Constraint_and_Reward_Function_Entry):
  
    def equality_constraints(self):
        """
        Return a single function for finding slack variable(s) in each equality constraint
        
        Inputs to functions are, in order: state, decision, nodes
        
        {slack_decision_variable_1:equality constraint solving for it,
        slack_decision_variable_2:equality constraint solving for it}
        """
        def Equality_Constraints(state,decision,nodes):
            return
        
        return Equality_Constraints
        
    def equality_constraint_variables(self):
        """
        Return list of all decision_variables involved in an equality constraint
        """
        return []
        
    def remaining_constraints(self):
        """
        Return list of functions which return boolean True/False values, True if
        the constraint is satisfied, False otherwise. Inputs to functions are, 
        in order: state, decision, nodes
        """
        def c1(state, decision, nodes):
            return
        def c2(state, decision, nodes):
            return
        
        return [c1,c2]
        
    def reward_function(self):
        """
        Return a function, which given the current state and an action, will
        return a reward. Inputs to function are, in order: state, decision, nodes
        """
        def R(state,decision,nodes):
            return 0.0
        
        return R