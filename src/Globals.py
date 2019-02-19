class Globals:

    '''
    Class with dicitionary, used for passing around variables thoughout the
    package
    '''

    def __init__(self):
        
        self.Global_Vars={}
        
    def set_global_variable(self,var_name,var_val):
        
        self.Global_Vars[var_name]=var_val
        
    def get_global_variable(self, var_name):
        
        return self.Global_Vars[var_name]
        
    def get_globals(self):
        
        return self.Global_Vars