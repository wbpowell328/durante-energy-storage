from Pol import Pol
import os
import numpy as np

class policy_PostDS_LU_table(Pol):

    
    def __init__(self, Name, FilePath, GLB_VARS, params):
        
        self.name=Name
        self.glb_vars=GLB_VARS
        self.params=params
        self.Nodes=GLB_VARS.get_global_variable('Nodes')
        self.SortedNodeList=sorted(self.Nodes.keys())
        self.dec_space=GLB_VARS.get_global_variable('Decision_Space')
        self.Reward=GLB_VARS.get_global_variable('Reward_Function')
        self.Discount_Factor=GLB_VARS.get_global_variable('Discount_Factor')
        ###
        os.chdir("..")
        os.chdir("data")
        self.fn=FilePath
        
        ###
        f=open(self.fn)
        fd=f.readlines()
        for lnnum in xrange(len(fd)):
            if fd[lnnum][0:20]=='End Input Parameters':
                strt=lnnum+1
                break
        file_data=fd[strt:len(fd)]
        f.close()
        
        ind=(len(file_data)-1)
        T=0
        self.FileDict={}
        self.FileDict[T]={}
        while ind>=0:
            Line=file_data[ind]
            if Line[0:4]=='Post':
                SplLin1=Line.split(':')[0]
                SplLin2=SplLin1.split(' ')
                T=int(SplLin2[len(SplLin2)-1])+1
                self.FileDict[T]={}
            else:
                SplLin=Line.split(' Value:')
                self.FileDict[T][SplLin[0]]=float(SplLin[1].rstrip())
            ind-=1
        
        os.chdir("..")
        os.chdir("input")
        

    def get_name(self):
        
        return self.name
        
    def offline_stage(self):
        
        return
        
    def get_all_params(self):
        
        return
    
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
        
        return
    
    def decision(self, state):
        """
        Make decision based on current state of system
        """
        T=state['T']
        Feasible_Decisions=self.dec_space.allowed_actions(state)
        BestDec={}
        BestExpectedVal=-np.inf
        for i in xrange(len(Feasible_Decisions)):
            TempPostDS={}
            for node in self.SortedNodeList:
                TempPostDS[node]=self.Nodes[node].pre_to_post_ds_transition(state,Feasible_Decisions[i])
            TempPostDS['T']=T
            DecVal=self.Reward(state,Feasible_Decisions[i],self.Nodes)+self.Discount_Factor*self.FileDict[T][str(TempPostDS)]
            if DecVal>BestExpectedVal:
                BestDec=Feasible_Decisions[i]
                BestExpectedVal=DecVal
        
        return BestDec