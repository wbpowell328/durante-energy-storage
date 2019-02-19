from Pol import Pol
import os
import numpy as np
import ast
import itertools



def apply_basis_functions(state,BFs):
    Output=[]
    for fun in BFs:
        Output.append(fun(state))
    return Output

def combine_independent_processes(Dict):
    '''
    Input a dicitionary with each exogenous process as the keys, and a
    2 element list consiting of first: list of possible process states in the
    next time step, second: the corresponding probability of reaching that state
    
    Outputs a two element list, the first is a list of combined partial state
    variabes consisting of only the entered exogenous processes, and the
    corresponding probability of reaching the combined partial state
    '''
    return_states=[]
    probabilities=[]
    big_list=[]
    key_list=sorted(Dict.keys())
    for key in key_list:
        small_list=[]
        for j in xrange(len(Dict[key][0])):
            small_list.append([Dict[key][0][j],Dict[key][1][j]])
        big_list.append(small_list)
    poss_states=list(itertools.product(*big_list))
    for i in xrange(len(poss_states)):
        Prob=1.0
        tuplist=[]
        for j in xrange(len(poss_states[i])):
            tuplist.append(poss_states[i][j][0])
            Prob=Prob*poss_states[i][j][1]
        ret_tup=tuple(tuplist)
        return_states.append(ret_tup)
        probabilities.append(Prob)
    return [return_states, probabilities]

class policy_PreDS_Lin_VFA(Pol):

    
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

        ###DEFINE BASIS FUNCTIONS HERE
        def f_1(state):
            return self.Nodes['R_1'].get_preds_value(state)
        def f_2(state):
            return self.Nodes['E'].get_preds_value(state)
        def f_3(state):
            return self.Nodes['G'].get_preds_value(state)

        
        #Need to put these in a list
        self.Basis_Functions=[f_1,f_2,f_3]
        
        ###END BASIS FUNCTION DEFINITION        
        

        
        
        f=open(self.fn)
        fd=f.readlines()
        for lnnum in xrange(len(fd)):
            if fd[lnnum][0:20]=='End Input Parameters':
                strt=lnnum+1
                break
        file_data=fd[strt:len(fd)]
        f.close()
        
        ind=(len(file_data)-1)
        T=1
        self.FileDict={}
        self.FileDict[T]={}
        while ind>=0:
            Line=file_data[ind]
            if Line[0:3]=='Par':
                SplLin1=Line.split(':')[0]
                SplLin2=SplLin1.split(' ')
                T=int(SplLin2[len(SplLin2)-1])+1
                self.FileDict[T]={}
            else:
                self.FileDict[T]=ast.literal_eval(Line.rstrip())
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
        beta_hat=self.FileDict[T+1]
        Feasible_Decisions=self.dec_space.allowed_actions(state)
        BestDec={}
        BestExpectedVal=-np.inf
        for i in xrange(len(Feasible_Decisions)):
            TempPostDS={}
            for node in self.SortedNodeList:
                TempPostDS[node]=self.Nodes[node].pre_to_post_ds_transition(state,Feasible_Decisions[i])
            TempPostDS['T']=T
            Possible_Sts={}
            for node in self.SortedNodeList:
                if (T%self.Nodes[node].get_time_step())==0:
                    Full_Outcome_Space=self.Nodes[node].get_postds_to_preds_probabilities(TempPostDS)
                    Possible_Sts[node]=[Full_Outcome_Space[0],Full_Outcome_Space[1]]
                else:
                    Possible_Sts[node]=[[state[node]],[1.0]]
            Possible_Sts['T']=[[T+1],[1.0]]
            Next_Pre_Dec_States=combine_independent_processes(Possible_Sts)
            PostDSVal=0.0
            for j in xrange(len(Next_Pre_Dec_States[0])):
                st=Next_Pre_Dec_States[0][j]
                TempPreDS={}
                for node in self.SortedNodeList:
                    TempPreDS[node]=st[self.SortedNodeList.index(node)]
                TempPreDS['T']=T+1
                BFs=apply_basis_functions(TempPreDS,self.Basis_Functions)
                Val=beta_hat[len(beta_hat)-1]
                for bf in xrange(len(BFs)):
                    Val+=(beta_hat[bf]*BFs[bf])
                PostDSVal+=(Val*Next_Pre_Dec_States[1][j])
            DecVal=self.Reward(state,Feasible_Decisions[i],self.Nodes)+self.Discount_Factor*PostDSVal
            if DecVal>BestExpectedVal:
                BestDec=Feasible_Decisions[i]
                BestExpectedVal=DecVal
        
        return BestDec