import abc

class Constraint_and_Reward_Function_Entry():
    __metaclass__ = abc.ABCMeta   
    
    @abc.abstractmethod
    def equality_constraints(self):
        """
        Return a single function for finding slack variable(s) in each equality constraint
        (typically of the G:L type) that returns a dicitonary as follows:
        
        Inputs to functions are, in order: decision, energy_vars, resource_vars, state
        
        {slack_decision_variable_1:equality constraint solving for it,
        slack_decision_variable_2:equality constraint solving for it}
        """
        return
        
    @abc.abstractmethod
    def equality_constraint_variables(self):
        """
        Return list of all decision_variables involved in an equality constraint
        """
        return
        
    @abc.abstractmethod
    def remaining_constraints(self):
        """
        Return list of functions which return boolean True/False values, True if
        the constraint is satisfied, False otherwise. Inputs to functions are, 
        in order: decision, energy_vars, resource_vars, state
        """
        return
        
    @abc.abstractmethod
    def reward_function(self):
        """
        Return a function, which given the current state and an action, will
        return a reward. Inputs to function are, in order: state, decision,
        price_vars,energy_vars, resource_vars
        """
        return