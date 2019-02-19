import abc

class reward_function():
    __metaclass__ = abc.ABCMeta   
    
    @abc.abstractmethod
    def get_name(self):
        """
        Return name of reward function
        """
        return
    
    @abc.abstractmethod
    def get_reward(self, state, action):
        """
        Return reward of taking an action in a certain state
        """
        return