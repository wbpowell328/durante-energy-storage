import abc

class decision_space():
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def allowed_actions(self, current_state):
        '''
        From the current state, return a two element list.
        The first element is a list of dictionaries defining allowable
        forward actions, and the second is a list of post decision states which
        result from the corresponding action
        '''
        return
        