import itertools
from copy import copy
import timeit
import collections
import numpy as np
import os

def state_space(Dict):
    '''
    Cartesian Product, each dimension is a variable whose name is the key to
    a dictionart with possible values for that variable as the value.
    Used to form a list of full state viarables
    '''
    return_states=[]
    big_list=[]
    key_list=Dict.keys()
    L=len(key_list)
    for key in Dict:
        big_list.append(Dict[key])
    poss_states=list(itertools.product(*big_list))
    for x in xrange(len(poss_states)):
        New_dict={}
        for y in xrange(L):
            New_dict[key_list[y]]=poss_states[x][y]
        return_states.append(New_dict)
    return return_states
    
def state_space_form_dictionary(Dict,fixed_val_to_set):
    '''
    Cartesian Product, each dimension is a variable whose name is the key to
    a dictionart with possible values for that variable as the value.
    Used to form a dictionary of full state viarables where the key
    is the state variable in string form and the values are a two element list:
    the first element is the state variable itself, the second is fixed_val_to_set, used
    for representing the value of the state
    '''
    return_states={}
    big_list=[]
    key_list=sorted(Dict.keys())
    for key in key_list:
        big_list.append(Dict[key])
    poss_states=list(itertools.product(*big_list))
    for st in poss_states:
        return_states[st]=fixed_val_to_set
    return return_states

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

class full_backward_MDP_solver_new():
    
    def __init__(self, GLB_VARS,InputString):
        self.glb_vars=GLB_VARS   
        self.InputString=InputString
    
    def backward_MDP(self, file_name):
        '''
        Solve full disretized MDP and output values for all post-decision 
        states in a text file with file name specified by file_name
        '''
        
        start = timeit.default_timer()
        os.chdir('..')
        os.chdir('data')
        Stop_Time=self.glb_vars.get_global_variable('Horizon')
        f=open(file_name,'w')
        f.write(self.InputString)
        Discount_Factor=self.glb_vars.get_global_variable('Discount_Factor')
        Nodes=self.glb_vars.get_global_variable('Nodes')
        SortedNodeList=sorted(Nodes.keys())
        Decision_Space=self.glb_vars.get_global_variable('Decision_Space')
        Reward_Function=self.glb_vars.get_global_variable('Reward_Function')
        
        ###GET ALL FINAL STATES FOR EXOGENOUS PROCESSES AND RESOURCES
        ###AT TIME T, FORM ALL COMBINED FULL STATES WITH A CARTESIAN PRODUCT
        Possible_States={}
        for node in SortedNodeList:
            last_time=Stop_Time
            while (not (last_time%Nodes[node].get_time_step()==0)):
                last_time-=1
            Possible_States[node]=Nodes[node].get_possible_preds(last_time)
        Possible_States['T']=[Stop_Time]
        Curr_states=state_space_form_dictionary(Possible_States,0.0)
        Iter=0
        for T in range(Stop_Time,0,-1):
            step_start=timeit.default_timer()
            
            ###GET ALL POST DECISION STATES FOR EXOGENOUS PROCESSES AND RESOURCES
            ###AT TIME t-1, FORM ALL COMBINED POST DECISION STATES 
            ###WITH A CARTESIAN PRODUCT
            PDS_States={}
            for node in SortedNodeList:
                if (T%Nodes[node].get_time_step())==0:
                    PDS_States[node]=Nodes[node].get_possible_postds(T-1)
                else:
                    PDS_States[node]=Possible_States[node]
            PDS_States['T']=[T-1]
            Post_Decision_States=state_space_form_dictionary(PDS_States,0.0)
            T1=timeit.default_timer()
            
            ###FOR EACH POST DECISION STATE:
            f.write('Post-decision state values at time '+str(T-1) + ':'+os.linesep)
            for pds in Post_Decision_States:
                TempPostDS={}
                for node in SortedNodeList:
                    TempPostDS[node]=pds[SortedNodeList.index(node)]
                TempPostDS['T']=T-1
                ###FIND PROBABILITY OF TRANSFERING TO EACH NEXT PRE DECISION STATE
                ###AND FIND PDS VALUE AS SUM OF 
                ###Prob(NEXT PRE DEC STATE|PDS)*Value(NEXT PRE DEC STATE)
                ###FOR ALL NEXT PRE DECISION STATES
                Possible_Sts={}
                for node in SortedNodeList:
                    if (T%Nodes[node].get_time_step())==0:
                        Possible_Sts[node]=Nodes[node].get_postds_to_preds_probabilities(TempPostDS)
                    else:
                        Possible_Sts[node]=[[pds[SortedNodeList.index(node)]],[1.0]]
                Possible_Sts['T']=[[T],[1.0]]
                Next_Pre_Dec_States=combine_independent_processes(Possible_Sts)
                Val=0.0
                for i in xrange(len(Next_Pre_Dec_States[0])):
                    st=Next_Pre_Dec_States[0][i]
                    Val+=Next_Pre_Dec_States[1][i]*Curr_states[st]     
                Post_Decision_States[pds]=Val
                f.write(str(TempPostDS))
                f.write(' Value:'+str(Val)+'\n')
            #T2=timeit.default_timer()
            T2=timeit.default_timer()
            print 'Found time '+str(T-1)+' post decision state values in '+str(T2-T1)+ ' sec.'    
            print 'Number of post decision states: ', str(len(Post_Decision_States))
            
            ###GET ALL PREVIOUS STATES FOR EXOGENOUS PROCESSES AND RESOURCES
            ###AT TIME t-1, FORM ALL COMBINED FULL STATES WITH A CARTESIAN PRODUCT
            for node in SortedNodeList:
                if (T%Nodes[node].get_time_step())==0:
                    Possible_States[node]=Nodes[node].get_possible_preds(T-1)
            Possible_States['T']=[T-1]
            Curr_states=state_space_form_dictionary(Possible_States,0.0)
            
            ###FOR EACH PRE DECISION STATE
            for s in Curr_states:
                ###FIRST FIND FEASIBLE ACTIONS ACCORDING TO DECISION SPACE
                TempPreDS={}
                for node in SortedNodeList:
                    TempPreDS[node]=s[SortedNodeList.index(node)]
                TempPreDS['T']=T-1
                Actions_to_PDS=Decision_Space.allowed_actions(TempPreDS)
                MaxVal=-1*np.inf
                #OptAct=None
                ###OPTIMAL ACTION IS THAT THAT MAXIMIZES SUM OF IMMEDIATE
                ###REWARD PLUS DISCOUNTED VALUE OF POST DECISION STATE (SUM
                ###OF EXPECTED FUTURE REWARDS)
                for i in range(len(Actions_to_PDS)):
                    A=Actions_to_PDS[i]
                    pds_list=[]
                    for node in SortedNodeList:
                        pds_list.append(Nodes[node].pre_to_post_ds_transition(TempPreDS,A))
                    pds_list.append(T-1)
                    pds_tuple=tuple(pds_list)

                    V=Reward_Function(TempPreDS,A,Nodes)+Discount_Factor*Post_Decision_States[pds_tuple]
                    if V>MaxVal:
                        MaxVal=V
                        #OptAct=A
                #T3=timeit.default_timer()
                Curr_states[s]=MaxVal
#            for s in Curr_states:
#                f.write(str(Curr_states[s][0]))
#                f.write(' Value:'+str(Curr_states[s][1]))
#                f.write(' Opt Action:'+str(Curr_states[s][2])+'\n')
            Iter+=1
            step_end=timeit.default_timer()
            Str=str(Iter)+' steps backward out of ' +str(Stop_Time)+ ' completed '
            Str=Str+'in '+str(step_end-start)+' seconds, the most recent step took '+str(step_end-step_start)+' seconds.'
            print Str
        f.close()