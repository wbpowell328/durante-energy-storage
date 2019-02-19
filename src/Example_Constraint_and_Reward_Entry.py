from Constraint_and_Reward_Function_Entry import Constraint_and_Reward_Function_Entry

class Example_Constraint_and_Reward_Entry(Constraint_and_Reward_Function_Entry):
    
    def equality_constraints(self):
        """
        Return a single function for finding slack variable(s) in each equality constraint
        (typically of the G:L type) that returns a dicitonary as follows:
        
        Inputs to functions are, in order: state, decision, nodes
        
        {slack_decision_variable_1:equality constraint solving for it,
        slack_decision_variable_2:equality constraint solving for it}
        
        """
        
        def Equality_Constraints(state,decision,nodes):
            return {'G:L':nodes['L'].get_preds_value(state)-decision['E:L']-(nodes['R_1'].conv_loss*decision['R_1:L'])}
        
        return Equality_Constraints
        
    def equality_constraint_variables(self):
        """
        Return list of all decision variables involved in an equality constraint
        """
        return ['E:L','G:L','R_1:L']
        
    def remaining_constraints(self):
        """
        Return list of functions which return boolean True/False values, True if
        the constraint is satisfied, False otherwise. Inputs to functions are, 
        in order: state, decision, nodes
        """
        
        def iec1(state,decision,nodes):
            return decision['E:L']+decision['E:R_1']<=nodes['E'].get_preds_value(state)
        def iec2(state,decision,nodes):
            return decision['R_1:L']<=nodes['R_1'].get_preds_value(state)
        def iec3(state,decision,nodes):
            return decision['G:R_1']>=-(nodes['R_1'].get_preds_value(state))   
        def iec4(state,decision,nodes):
            return decision['G:L']>=0.0
        def iec5(state,decision,nodes):
            return decision['E:L']>=0.0
        def iec6(state,decision,nodes):
            return decision['E:R_1']>=0.0
        def iec7(state,decision,nodes):
            return decision['R_1:L']>=0.0

        Inequality_Constraints=[iec1,iec2,iec3,iec4,iec5,iec6,iec7]
        
        return Inequality_Constraints
        
    def reward_function(self):
        """
        Return a function, which given the current state and an action, will
        return a reward. Inputs to function are, in order: state, decision, nodes
        """
        def R(state, decision, nodes):
            return -1.0/1000*nodes['G'].get_preds_value(state)*(decision['G:R_1']+decision['G:L'])
        
        return R