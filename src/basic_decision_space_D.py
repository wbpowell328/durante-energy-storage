from copy import copy
import itertools
from decision_space import decision_space 
import numpy as np 
import bisect 

def possible_decisions_1(decision_ranges):
    '''
    Cartesian Product, each flow variable entered in the dictionary is a dimension
    with a set of trimmed down possible actions
    '''
    all_decision_states=[]
    big_list=[]
    key_list=decision_ranges.keys()
    L=len(key_list)
    for key in decision_ranges:
        big_list.append(decision_ranges[key])
    poss_states=list(itertools.product(*big_list))
    for x in xrange(len(poss_states)):
        New_dict={}
        for y in xrange(L):
            New_dict[key_list[y]]=poss_states[x][y]
        all_decision_states.append(New_dict)
    return all_decision_states
    
def possible_decisions_2(partial_decisions, decision_ranges):
    '''
    Special Cartesian Product where some partial decisions have been made
    The partial decisions comprise one dimension, while the remaining
    flow variables each comprise their own dimension
    '''
    all_decision_states=[]
    big_list=[]
    big_list.append(partial_decisions)
    key_list=decision_ranges.keys()
    L=len(key_list)
    for key in decision_ranges:
        big_list.append(decision_ranges[key])
    poss_states=list(itertools.product(*big_list))
    for s in poss_states:
        New_dict=copy(s[0])
        for i in xrange(L):
            New_dict[key_list[i]]=s[1+i]
        all_decision_states.append(New_dict)
    return all_decision_states

class basic_decision_space_D(decision_space):
    
    def __init__(self, GLOBAL_VARS):
        
        self.gl_vars=GLOBAL_VARS
        self.inequality_constraints=GLOBAL_VARS.get_global_variable('Inequality_Constraints')
        self.equality_constraints=GLOBAL_VARS.get_global_variable('Equality_Constraints')
        self.decision_variables=GLOBAL_VARS.get_global_variable('Decision_Vars')
        self.decision_vars=[]
        for D in self.decision_variables:
            D2=D.split(':')
            self.decision_vars.append(D2)
        self.energy_vars=GLOBAL_VARS.get_global_variable('Energy_Vars')
        self.resource_vars=GLOBAL_VARS.get_global_variable('Resource_Vars')
        for res_name in self.resource_vars:
            self.res_name=res_name
        self.equality_decision_vars=GLOBAL_VARS.get_global_variable('Equality_Decision_Variables')
        
        
    def allowed_actions(self, current_state):
        
        '''
        From the current state, return a two element list.
        The first element is a list of dictionaries defining allowable
        forward actions, and the second is a list of post decision states which
        result from the corresponding action
        '''
        
        ###ALL COMMENTS IN THE BASIC_DECSION_SPACE_SHORTCUTS_2.PY FILE
        ###APPLY HERE AS WELL

        ###THE DIFFERENCE IN THIS ACTION SPACE IS THAT THE NUMBER OF POSSIBLE
        ###ACTIONS PER FLOW VARIABLE IS LIMITED TO THE MINIMUM OF A FIXED NUMBER
        ###WHICH IS ENTERED AS A PARAMETER AND THE NUMBER OF ALLOWED ACTIONS
        ###FOR THE FLOW VARIABLE BETWEEN THE VARIABLES' DISCRETIZED STATES        
        t=current_state['T']
        E=current_state['E']
        try:
            E_curr=current_state['E']['E']
        except:
            E_curr=current_state['E']
        L_curr=current_state['L']
        R_curr=current_state['R_1']
        Decision_Possibilities={}
        Res1=self.energy_vars['R_1']
        d_int=Res1.d_int
        R_states=Res1.get_all_states()
        E_to_L=min(E_curr,L_curr)
        L_left=L_curr-E_to_L
        E_left=E_curr-E_to_L
        R_max=Res1.get_max()
        CR=Res1.get_max_change(t)
        Eff=Res1.conv_rate
        
        RL=[0.0,min(CR,R_curr,L_left)]
        Decision_Possibilities['R_1:L']=RL
        ER=min((R_max-R_curr), E_left, CR)
        Unfilled_R=R_max-(R_curr+ER)
        Decisions_1=[]
        for rl in RL:
            D={}
            D['R_1:L']=rl
            D['E:L']=E_to_L
            D['E:R_1']=ER
            Decisions_1.append(D)
        #T5=timeit.default_timer()
        #print T5-T4
        IEC_decisions_forward=[]
        for D in Decisions_1:
            D['G:L']=L_left-Eff*D['R_1:L']
            GRlist=set(np.arange(max(-(CR-D['R_1:L']),-(R_curr-D['R_1:L'])),min(CR-ER,Unfilled_R)+d_int,d_int))
            GRlist.add(0.0)
            for GR1 in set(GRlist):
                DD=copy(D)
                DD['G:R_1']=GR1
                IEC_decisions_forward.append(DD)
        #T6=timeit.default_timer()
        #print T6-T5
#        Decision_Possibilities={}
#        GR1=[max(-CR,-R_curr,-(CR-RL[0])),0,min(CR-ER,Unfilled_R)]
#        Decision_Possibilities['G:R_1']=list(set(GR1))
#        possible_decisions_2(EC_decisions_forward, Decision_Possibilities)

        new_states=[]
        #print 'g'
        for D in IEC_decisions_forward:
            ret_state=copy(current_state)
            if D['G:R_1']>=0:
                R1=R_curr+Eff*D['G:R_1']+Eff*D['E:R_1']-D['R_1:L']
            else:
                R1=R_curr+D['G:R_1']+Eff*D['E:R_1']-D['R_1:L']
            
            Ind=bisect.bisect_right(R_states,R1)-1
            if Ind<0:
                ret_state['R_1']=R_states[0]
            elif Ind==(len(R_states)-1):
                ret_state['R_1']=R_states[len(R_states)-1]
            else:
                V1=abs(R1-R_states[Ind])
                V2=abs(R1-R_states[Ind+1])
                if V1<V2:
                    ret_state['R_1']=R_states[Ind]
                else:
                    ret_state['R_1']=R_states[Ind+1]
            
            new_states.append(ret_state)
            
            
        #T8=timeit.default_timer()
        #print T8-T7
        #print 'e'
        return [IEC_decisions_forward, new_states]
        
    def allowed_actions_2(self, t,R_curr,E_curr,L_curr):
        
        '''
        From the current state, return a two element list.
        The first element is a list of dictionaries defining allowable
        forward actions, and the second is a list of post decision states which
        result from the corresponding action
        '''
        
        ###ALL COMMENTS IN THE BASIC_DECSION_SPACE_SHORTCUTS_2.PY FILE
        ###APPLY HERE AS WELL

        ###THE DIFFERENCE IN THIS ACTION SPACE IS THAT THE NUMBER OF POSSIBLE
        ###ACTIONS PER FLOW VARIABLE IS LIMITED TO THE MINIMUM OF A FIXED NUMBER
        ###WHICH IS ENTERED AS A PARAMETER AND THE NUMBER OF ALLOWED ACTIONS
        ###FOR THE FLOW VARIABLE BETWEEN THE VARIABLES' DISCRETIZED STATES        
        
        
        #T3=timeit.default_timer()
        Decision_Possibilities={}
        Res1=self.energy_vars['R_1']
        d_int=Res1.d_int
        R_states=Res1.get_all_states()
        E_to_L=min(E_curr,L_curr)
        L_left=L_curr-E_to_L
        E_left=E_curr-E_to_L
        R_max=Res1.get_max()
        CR=Res1.get_max_change(t)
        Eff=Res1.conv_rate
        
        RL=[0.0,min(CR,R_curr,L_left)]
        Decision_Possibilities['R_1:L']=RL
        ER=min((R_max-R_curr), E_left, CR)
        Unfilled_R=R_max-(R_curr+ER)
        Decisions_1=[]
        for rl in RL:
            D={}
            D['R_1:L']=rl
            D['E:L']=E_to_L
            D['E:R_1']=ER
            Decisions_1.append(D)
        #T5=timeit.default_timer()
        #print T5-T4
        IEC_decisions_forward=[]
        for D in Decisions_1:
            D['G:L']=L_left-Eff*D['R_1:L']
            GRlist=set(np.arange(max(-(CR-D['R_1:L']),-(R_curr-D['R_1:L'])),min(CR-ER,Unfilled_R)+d_int,d_int))
            GRlist.add(0.0)
            for GR1 in set(GRlist):
                DD=copy(D)
                DD['G:R_1']=GR1
                IEC_decisions_forward.append(DD)
        #T6=timeit.default_timer()
        #print T6-T5
#        Decision_Possibilities={}
#        GR1=[max(-CR,-R_curr,-(CR-RL[0])),0,min(CR-ER,Unfilled_R)]
#        Decision_Possibilities['G:R_1']=list(set(GR1))
#        possible_decisions_2(EC_decisions_forward, Decision_Possibilities)

        new_states=[]
        #print 'g'
        for D in IEC_decisions_forward:
            if D['G:R_1']>=0:
                R1=R_curr+Eff*D['G:R_1']+Eff*D['E:R_1']-D['R_1:L']
            else:
                R1=R_curr+D['G:R_1']+Eff*D['E:R_1']-D['R_1:L']
            
            Ind=bisect.bisect_right(R_states,R1)-1
            if Ind<0:
                R=R_states[0]
            elif Ind==(len(R_states)-1):
                R=R_states[len(R_states)-1]
            else:
                V1=abs(R1-R_states[Ind])
                V2=abs(R1-R_states[Ind+1])
                if V1<V2:
                    R=R_states[Ind]
                else:
                    R=R_states[Ind+1]
            
            new_states.append(R)
            
            
        #T8=timeit.default_timer()
        #print T8-T7
        #print 'e'
        return [IEC_decisions_forward, new_states]