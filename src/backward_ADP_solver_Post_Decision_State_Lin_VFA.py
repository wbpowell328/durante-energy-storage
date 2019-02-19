import itertools
import timeit
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
    a dictionary with possible values for that variable as the value.
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

def state_space_form_list(Dict):
    '''
    Cartesian Product, each dimension is a variable whose name is the key to
    a dictionary with possible values for that variable as the value.
    Used to form a dictionary of full state viarables where the key
    is the state variable in string form and the values are a two element list:
    the first element is the state variable itself, the second is fixed_val_to_set, used
    for representing the value of the state
    '''
    big_list=[]
    key_list=sorted(Dict.keys())
    for key in key_list:
        big_list.append(Dict[key])
    poss_states=list(itertools.product(*big_list))
    return poss_states

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

class backward_ADP_solver_Post_Decision_State_Lin_VFA():
    
    def __init__(self, GLB_VARS,InputString):
        self.glb_vars=GLB_VARS   
        self.InputString=InputString
    
    def backward_MDP(self, Monte_Carlo_samples,file_name):
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

        ###DEFINE BASIS FUNCTIONS HERE
        def f_1(state):
            return Nodes['R_1'].get_postds_value(state)
        def f_2(state):
            return Nodes['E'].get_postds_value(state)
        def f_3(state):
            return Nodes['G'].get_postds_value(state)

        
        #Need to put these in a list
        Basis_Functions=[f_1,f_2,f_3]
        
        ###END BASIS FUNCTION DEFINITION        
        
        def apply_basis_functions(state):
            Output=[]
            for fun in Basis_Functions:
                Output.append(fun(state))
            return Output
        
        def PostDS_VFA(pds, beta_hat):
            BFs=apply_basis_functions(pds)
            Val=beta_hat[len(beta_hat)-1]
            for i in range(len(BFs)):
                Val+=BFs[i]*beta_hat[i]
            return Val
        
        
        Iter=0
        TimeTPlus1PostDS_ParamVec=[]
        for i in range(len(Basis_Functions)+1):
            TimeTPlus1PostDS_ParamVec.append(0.0)
            
        for T in range(Stop_Time,0,-1):
            
            Basis_function_evals=[[] for i in range(len(Basis_Functions))]
            Values_of_states=[]
            
            step_start=timeit.default_timer()
            Post_Decision_States={}
            for node in SortedNodeList:
                if (T%Nodes[node].get_time_step()==0):
                    Post_Decision_States[node]=Nodes[node].get_possible_postds(T-1)
                else:
                    Post_Decision_States[node]=Nodes[node].get_possible_preds(T-1)
            Post_Decision_States['T']=[T-1]
            PostDSFull=state_space_form_list(Post_Decision_States)
            for MCSample in xrange(Monte_Carlo_samples):
                Ind=np.random.choice(len(PostDSFull),1)[0]
                pds=PostDSFull[Ind]
                TempPostDS={}
                for node in SortedNodeList:
                    TempPostDS[node]=pds[SortedNodeList.index(node)]
                TempPostDS['T']=T-1
                BFs=apply_basis_functions(TempPostDS)
                ###FIND PROBABILITY OF TRANSFERING TO EACH NEXT PRE DECISION STATE
                ###AND FIND PDS VALUE AS SUM OF 
                ###Prob(NEXT PRE DEC STATE|PDS)*Value(NEXT PRE DEC STATE)
                ###FOR ALL NEXT PRE DECISION STATES
                if T==Stop_Time:
                    for bf in xrange(len(BFs)):
                            Basis_function_evals[bf].append(BFs[bf])
                    Values_of_states.append(0.0)  
                else:                    
                    TempPreDS={}
                    for node in SortedNodeList:
                        TempPreDS[node]=Nodes[node].post_to_pre_ds_transition(TempPostDS)
                    TempPreDS['T']=T
                    
                    Actions_to_PDS=Decision_Space.allowed_actions(TempPreDS)
                    MaxVal=-1*np.inf
                    for i in range(len(Actions_to_PDS)):
                        A=Actions_to_PDS[i]
                        TempNextPostDS={}
                        for node in SortedNodeList:
                            TempNextPostDS[node]=Nodes[node].pre_to_post_ds_transition(TempPreDS,A)
                        TempNextPostDS['T']=T
                        V=Reward_Function(TempPreDS,A,Nodes)+Discount_Factor*PostDS_VFA(TempNextPostDS,TimeTPlus1PostDS_ParamVec)
                        if V>MaxVal:
                            MaxVal=V
                            
                    for bf in xrange(len(BFs)):
                        Basis_function_evals[bf].append(BFs[bf])
                    Values_of_states.append(MaxVal)       
                        

            f.write('Parameter Vector for time '+str(T-1) + ':'+os.linesep)
            x=Basis_function_evals
            X = np.column_stack(x+[[1]*len(x[0])])
            beta_hat = np.linalg.lstsq(X,Values_of_states)[0]
            Write_String='['
            for b in xrange(len(beta_hat)):
                if b<(len(beta_hat)-1):
                    Write_String=Write_String+str(beta_hat[b])+','
                else:
                    Write_String=Write_String+str(beta_hat[b])
            Write_String=Write_String+']'
            f.write(Write_String+os.linesep)
            TimeTPlus1PostDS_ParamVec=beta_hat

            Iter+=1
            step_end=timeit.default_timer()
            Str=str(Iter)+' steps backward out of ' +str(Stop_Time)+ ' completed '
            Str=Str+'in '+str(step_end-start)+' seconds, the most recent step took '+str(step_end-step_start)+' seconds.'
            print(Str)
        f.close()