import itertools
import numpy as np
import os
import bisect

def round_to_array(Vec,Val):
    Ind=bisect.bisect_right(Vec,Val)-1
    if Ind<0:
        Ret=Vec[0]
    elif Ind==len(Vec)-1:
        Ret=Vec[Ind]
    else:
        diff=abs(Val-Vec[Ind])-abs(Val-Vec[Ind+1])
        if diff>0:
            Ret=Vec[Ind+1]
        else:
            Ret=Vec[Ind]
    return Ret

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


class ForwardADP_API():
    
    def __init__(self, GLB_VARS,InputString):
        self.glb_vars=GLB_VARS
        self.InputString=InputString


    def API(self, M, N, file_name):
        '''
        Solve sampled disretized MDP and output values for all post-decision 
        states in a text file with file name specified by file_name
        
        The probability of sampling a pre-decision state is given by sample_prob
        However, the number of sampled pre-decision states is lower bounded
        by the number of previous post decision states; the implementation
        ensures every post decision state will have a reachable next
        pre-decision state so that there exists an approximate value for
        each post decision state
        '''
        
        ###ALL COMENTS FROM FULL_MDP_SOLVER HOLD HERE
        ###WITH THE EXPETION OF ONE AREA AND THE SAMPLING MENTIONED ABOVE
        os.chdir('..')
        os.chdir('data')
        
        Stop_Time=self.glb_vars.get_global_variable('Horizon')
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
        
        Init_Beta_Hat=[]
        for i in range(len(Basis_Functions)+1):
            Init_Beta_Hat.append(0.0)
            
        Current_Beta_Hats=[Init_Beta_Hat for i in range(Stop_Time)]

        for n in xrange(N):
            print n
            CumRews=np.zeros(Stop_Time)
            StateValuePairs={}
            for t in range(Stop_Time):
                StateValuePairs[t]=[[[] for i in range(len(Basis_Functions))],[]]
            for m in xrange(M):
                state={}
                for node in SortedNodeList:
                    state[node]=Nodes[node].get_initial_preds()
                state['T']=0                
                Path=[]
                Rews=[]
                for t in range(Stop_Time):
                    Actions_to_PDS=Decision_Space.allowed_actions(state)
                    MaxVal=-1*np.inf
                    for i in range(len(Actions_to_PDS)):
                        A=Actions_to_PDS[i]
                        pds={}
                        for node in SortedNodeList:
                            pds[node]=Nodes[node].pre_to_post_ds_transition(state,A)
                        pds[node]=t
                        V=Reward_Function(state,A,Nodes)+Discount_Factor*PostDS_VFA(pds,Current_Beta_Hats[t])
                        if V>MaxVal:
                            MaxVal=V
                            Dec=A
                    Rews.append(Reward_Function(state,Dec,Nodes))
                    for node in SortedNodeList:
                        state[node]=Nodes[node].pre_to_post_ds_transition(state,Dec)
                    state['T']=t
                    Path.append(state)
                    for node in SortedNodeList:
                        state[node]=Nodes[node].post_to_pre_ds_transition(state)
                    state['T']=t+1
                    
                for t in range(Stop_Time-1,-1,-1):
                    st=Path[t]
                    if t==(Stop_Time-1):
                        CumRews[t]=0.0
                    else:
                        CumRews[t]=Rews[t]+CumRews[t+1]
                    for i in range(len(Basis_Functions)):
                        StateValuePairs[t][0][i].append(Basis_Functions[i](st))
                    StateValuePairs[t][1].append(CumRews[t])
            
            print ('Policy Update: '+str(n))
            f=open(file_name+'_'+str(n),'w')
            f.write("Approximate Policy Iteration with Linear Regression for Post Decision States"+os.linesep)
            f.write(self.InputString)     
            for t in range(Stop_Time-1,-1,-1):
                f.write('Parameter Vector for time '+str(t) + ':'+os.linesep)
                x=StateValuePairs[t][0]
                X = np.column_stack(x+[[1]*len(x[0])])
                beta_hat = np.linalg.lstsq(X,StateValuePairs[t][1])[0]
                Current_Beta_Hats[t]=beta_hat
                Write_String='['
                for b in xrange(len(beta_hat)):
                    if b<(len(beta_hat)-1):
                        Write_String=Write_String+str(beta_hat[b])+','
                    else:
                        Write_String=Write_String+str(beta_hat[b])
                Write_String=Write_String+']'
                f.write(Write_String+os.linesep)
#                svr_rbf = SVR(kernel='rbf')
#                svr_rbf.fit(X,StateValuePairs[t][1])

            f.close()