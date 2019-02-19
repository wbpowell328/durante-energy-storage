from copy import copy
import itertools
from decision_space import decision_space 
import numpy as np
import timeit 
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

class basic_decision_space_A(decision_space):
    
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
        
        
        #T3=timeit.default_timer()
        t=current_state['T']
        Decision_Possibilities={}
        for dv in self.decision_vars:
            if dv[0]+':'+dv[1] in self.equality_decision_vars:
                ###START WITH EQUALITY CONSTRAINTS###
            
                ###USE KNOWLEDGE OF STANDARD ENERGY STORAGE PROBLEMS TO QUICKLY
                ###NARROW DOWN DECISION CANDIDIATES
                if (dv[0][0]=='R' or dv[1][0]=='R'):
                    if (dv[0][0]=='R') and (dv[1][0]=='R'):
                        Res1=self.energy_vars[dv[0]]
                        Res2=self.energy_vars[dv[1]]
                        d_int_1=Res1.get_discretization_interval()
                        d_int_2=Res2.get_discretization_interval()
                        d_int=max(d_int_1,d_int_2)
                        lf_1=Res1.get_conversion_loss(current_state)
                        lf_2=Res2.get_conversion_loss(current_state)
                        lf=lf_1*lf_2
                        V_1=Res1.get_value(current_state)
                        M_1=Res1.get_max()
                        V_2=Res2.get_value(current_state)
                        M_2=Res2.get_max()
                        CR_1=Res1.get_max_change(t)
                        CR_2=Res2.get_max_change(t)
                        change_rate_pos=min(CR_1,float(CR_2)/lf,V_1,float(M_2-V_2)/lf)
                        change_rate_neg=max(-1*float(CR_1)/lf,-1*CR_2,-1*(M_1-V_1)/lf,-1*V_2)
                        DP1=np.arange(change_rate_neg,change_rate_pos+d_int, d_int)        
                        if 0.0 not in DP1:
                            bisect.insort(DP1, 0.0)
                        Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                    elif (dv[0][0]=='R'):
                        if (dv[1][0]=='G'):
                            Res=self.energy_vars[dv[0]]
                            d_int=Res.get_discretization_interval()
                            loss_factor=Res.get_conversion_loss(current_state)
                            V=Res.get_value(current_state)
                            M=Res.get_max()
                            CR=Res.get_max_change(t)
                            change_rate_pos=min(CR,V)
                            change_rate_neg=max(-1*CR,-1*(M-V))
                            if change_rate_neg==0.0:
                                DP1=list(np.arange(0.0,change_rate_pos+d_int,d_int))
                            elif(change_rate_pos==0.0):
                                DP1=list(np.arange(change_rate_neg, 0.0+d_int, d_int)/loss_factor)
                            else:
                                DP1pos=list(np.arange(0.0,change_rate_pos+d_int,d_int))
                                DP1=list(np.arange(change_rate_neg, 0.0+d_int, d_int)/loss_factor)
                                DP1.extend(DP1pos[1:len(DP1pos)])
                            if 0.0 not in DP1:
                                bisect.insort(DP1, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                        elif (dv[1][0]=='E'):
                            Res=self.energy_vars[dv[0]]
                            E=self.energy_vars[dv[1]]
                            d_int=Res.get_discretization_interval()
                            loss_factor=Res.get_conversion_loss(current_state)
                            V=Res.get_value(current_state)
                            M=Res.get_max()
                            CR=Res.get_max_change(t)
                            E_val=E.get_value(current_state)
                            change_rate_pos=0.0
                            change_rate_neg=max(-1*CR,-1*(M-V),-1*E_val)
                            DP1=np.arange(change_rate_neg,change_rate_pos+d_int, d_int)/loss_factor        
                            if 0.0 not in DP1:
                                bisect.insort(DP1, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP1 if (x>=-1*E_val and x<=0.0)]
                        elif(dv[1][0]=='L'):
                            Res=self.energy_vars[dv[0]]
                            d_int=Res.get_discretization_interval()
                            V=Res.get_value(current_state)
                            loss_factor=Res.get_conversion_loss(current_state)
                            L=float(self.energy_vars[dv[1]].get_value(current_state))
                            CR=Res.get_max_change(t)
                            change_rate_pos=min(CR,V,L/loss_factor)
                            change_rate_neg=0.0
                            DP1=np.arange(change_rate_neg,change_rate_pos+d_int, d_int)        
                            if 0.0 not in DP1:
                                bisect.insort(DP1, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP1 if (x>=0 and x<=(L/loss_factor))]
                    elif(dv[1][0]=='R'):   
                        if (dv[0][0]=='G'):
                            Res=self.energy_vars[dv[1]]
                            d_int=Res.get_discretization_interval()
                            loss_factor=Res.get_conversion_loss(current_state)
                            V=Res.get_value(current_state)
                            M=Res.get_max()
                            CR=Res.get_max_change(t)
                            change_rate_pos=min(CR,M-V)
                            change_rate_neg=max(-1*CR,-1*V)
                            if change_rate_neg==0.0:
                                DP1=list(np.arange(0.0,change_rate_pos+d_int,d_int)/loss_factor)
                            elif change_rate_pos==0.0:
                                DP1=list(np.arange(change_rate_neg, 0.0+d_int, d_int))
                            else:
                                DP1pos=list(np.arange(0.0,change_rate_pos+d_int,d_int)/loss_factor)
                                DP1=list(np.arange(change_rate_neg, 0.0+d_int, d_int))
                                DP1.extend(DP1pos[1:len(DP1pos)])
                            if 0.0 not in DP1:
                                bisect.insort(DP1, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                        elif (dv[0][0]=='E'):
                            Res=self.energy_vars[dv[1]]
                            E=self.energy_vars[dv[0]]
                            d_int=Res.get_discretization_interval()
                            loss_factor=Res.get_conversion_loss(current_state)
                            V=Res.get_value(current_state)
                            M=Res.get_max()
                            CR=Res.get_max_change(t)
                            E_val=E.get_value(current_state)
                            change_rate_pos=min(CR,M-V,E_val)
                            change_rate_neg=0.0
                            DP1=np.arange(change_rate_neg,change_rate_pos+d_int, d_int)/loss_factor        
                            if 0.0 not in DP1:
                                bisect.insort(DP1, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP1 if (x>=0.0 and x<=E_val)]
                        elif(dv[0][0]=='L'):
                            Res=self.energy_vars[dv[1]]
                            d_int=Res.get_discretization_interval()
                            V=Res.get_value(current_state)
                            loss_factor=Res.get_conversion_loss(current_state)
                            L=float(self.energy_vars[dv[0]].get_value(current_state))
                            CR=Res.get_max_change(t)
                            change_rate_pos=0.0
                            change_rate_neg=-1*min(CR,V,L/loss_factor)
                            DP1=np.arange(change_rate_neg,change_rate_pos+d_int, d_int)        
                            if 0.0 not in DP1:
                                bisect.insort(DP1, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP1 if (x>=-1*(L/loss_factor) and x<=0.0)]
                else:
                    if (dv[0][0]=='G' and dv[1][0]=='L') or (dv[1][0]=='G' and dv[0][0]=='L'):
                        pass
                    elif (dv[1][0]=='L' or dv[0][0]=='L'):
                        if (dv[1][0]=='L'):
                            if (dv[0][0]=='E'):
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[0]].get_discretization_interval()
                                Decision_Possibilities[dv[0]+':'+dv[1]]=np.arange(0.0,max([Curr_val,Curr_val_2])+d_int, d_int)
                                
                            else:
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[0]].get_discretization_interval()
                                Decision_Possibilities[dv[0]+':'+dv[1]]=np.arange(0,max([Curr_val,Curr_val_2])+d_int, d_int)
                               
                        else:
                            if (dv[1][0]=='E'):
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[1]].get_discretization_interval()
                                Decision_Possibilities[dv[0]+':'+dv[1]]=np.arange(-1*max([Curr_val,Curr_val_2]),0.0+d_int, d_int)
                                
                            else:
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[1]].get_discretization_interval()
                                Decision_Possibilities[dv[0]+':'+dv[1]]=np.arange(-1*max([Curr_val,Curr_val_2]),0.0+d_int, d_int)
                                
                    else:
                        if (dv[0][0]=='E' or dv[1][0]=='E'):
                            if (dv[0][0]=='E'):
                                Proc=self.energy_vars[dv[0]]
                                Curr_val=Proc.get_value(current_state)
                                d_int=Proc.get_discretization_interval()
                                Decision_Possibilities[dv[0]+':'+dv[1]]=np.arange(0,Curr_val+d_int, d_int)
                            else:
                                Proc=self.energy_vars[dv[1]]
                                Curr_val=Proc.get_value(current_state)
                                d_int=Proc.get_discretization_interval()
                                Decision_Possibilities[dv[0]+':'+dv[1]]=np.arange(-1*Curr_val,0.0+d_int, d_int)
        
        ###CARTESIAN PRODUCT FOR EQUALITY VARS MINUS THE SLACK VARS (G:L TYPE)
        EC_decisions_forward=possible_decisions_1(Decision_Possibilities)
        
        for D in EC_decisions_forward:
            #FIND SLACK VARS FOR COMBINATIONS OF PARTIAL DECISIONS
            #SPEEDS UP FINDING DECSISIONS DRASTICALLY
            Slack_Vars=self.equality_constraints(D,self.energy_vars,self.resource_vars,current_state)
            D.update(Slack_Vars)
            
        Decision_Possibilities={}
        for dv in self.decision_vars:
            if dv[0]+':'+dv[1] not in self.equality_decision_vars:
            #NOW WE FIND FLOW VARIABLES NOT INVOLVED IN AN EQUALITY CONSTRAINT
                if (dv[0][0]=='R' or dv[1][0]=='R'):
                    if (dv[0][0]=='R') and (dv[1][0]=='R'):
                        Res1=self.energy_vars[dv[0]]
                        Res2=self.energy_vars[dv[1]]
                        d_int_1=Res1.get_discretization_interval()
                        d_int_2=Res2.get_discretization_interval()
                        d_int=max(d_int_1,d_int_2)
                        lf_1=Res1.get_conversion_loss(current_state)
                        lf_2=Res2.get_conversion_loss(current_state)
                        lf=lf_1*lf_2
                        V_1=Res1.get_value(current_state)
                        M_1=Res1.get_max()
                        V_2=Res2.get_value(current_state)
                        M_2=Res2.get_max()
                        CR_1=Res1.get_max_change(t)
                        CR_2=Res2.get_max_change(t)
                        change_rate_pos=min(CR_1,float(CR_2)/lf,V_1,float(M_2-V_2)/lf)
                        change_rate_neg=max(-1*float(CR_1)/lf,-1*CR_2,-1*(M_1-V_1)/lf,-1*V_2)
                        DP1=np.arange(change_rate_neg,change_rate_pos+d_int, d_int)        
                        if 0.0 not in DP1:
                            bisect.insort(DP1, 0.0)
                        Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                    elif (dv[0][0]=='R'):
                        if (dv[1][0]=='G'):
                            Res=self.energy_vars[dv[0]]
                            d_int=Res.get_discretization_interval()
                            loss_factor=Res.get_conversion_loss(current_state)
                            V=Res.get_value(current_state)
                            M=Res.get_max()
                            CR=Res.get_max_change(t)
                            change_rate_pos=min(CR,V)
                            change_rate_neg=max(-1*CR,-1*(M-V))
                            if change_rate_neg==0.0:
                                DP1=list(np.arange(0.0,change_rate_pos+d_int,d_int))
                            elif(change_rate_pos==0.0):
                                DP1=list(np.arange(change_rate_neg, 0.0+d_int, d_int)/loss_factor)
                            else:
                                DP1pos=list(np.arange(0.0,change_rate_pos+d_int,d_int))
                                DP1=list(np.arange(change_rate_neg, 0.0+d_int, d_int)/loss_factor)
                                DP1.extend(DP1pos[1:len(DP1pos)])
                            if 0.0 not in DP1:
                                bisect.insort(DP1, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                        elif (dv[1][0]=='E'):
                            Res=self.energy_vars[dv[0]]
                            E=self.energy_vars[dv[1]]
                            d_int=Res.get_discretization_interval()
                            loss_factor=Res.get_conversion_loss(current_state)
                            V=Res.get_value(current_state)
                            M=Res.get_max()
                            CR=Res.get_max_change(t)
                            E_val=E.get_value(current_state)
                            change_rate_pos=0.0
                            change_rate_neg=max(-1*CR,-1*(M-V),-1*E_val)
                            DP1=np.arange(change_rate_neg,change_rate_pos+d_int, d_int)/loss_factor        
                            if 0.0 not in DP1:
                                bisect.insort(DP1, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP1 if (x>=-1*E_val and x<=0.0)]
                        elif(dv[1][0]=='L'):
                            Res=self.energy_vars[dv[0]]
                            d_int=Res.get_discretization_interval()
                            V=Res.get_value(current_state)
                            loss_factor=Res.get_conversion_loss(current_state)
                            L=float(self.energy_vars[dv[1]].get_value(current_state))
                            CR=Res.get_max_change(t)
                            change_rate_pos=min(CR,V,L/loss_factor)
                            change_rate_neg=0.0
                            DP1=np.arange(change_rate_neg,change_rate_pos+d_int, d_int)        
                            if 0.0 not in DP1:
                                bisect.insort(DP1, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP1 if (x>=0 and x<=(L/loss_factor))]
                    elif(dv[1][0]=='R'):   
                        if (dv[0][0]=='G'):
                            Res=self.energy_vars[dv[1]]
                            d_int=Res.get_discretization_interval()
                            loss_factor=Res.get_conversion_loss(current_state)
                            V=Res.get_value(current_state)
                            M=Res.get_max()
                            CR=Res.get_max_change(t)
                            change_rate_pos=min(CR,M-V)
                            change_rate_neg=max(-1*CR,-1*V)
                            if change_rate_neg==0.0:
                                DP1=list(np.arange(0.0,change_rate_pos+d_int,d_int)/loss_factor)
                            elif change_rate_pos==0.0:
                                DP1=list(np.arange(change_rate_neg, 0.0+d_int, d_int))
                            else:
                                DP1pos=list(np.arange(0.0,change_rate_pos+d_int,d_int)/loss_factor)
                                DP1=list(np.arange(change_rate_neg, 0.0+d_int, d_int))
                                DP1.extend(DP1pos[1:len(DP1pos)])
                            if 0.0 not in DP1:
                                bisect.insort(DP1, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                        elif (dv[0][0]=='E'):
                            Res=self.energy_vars[dv[1]]
                            E=self.energy_vars[dv[0]]
                            d_int=Res.get_discretization_interval()
                            loss_factor=Res.get_conversion_loss(current_state)
                            V=Res.get_value(current_state)
                            M=Res.get_max()
                            CR=Res.get_max_change(t)
                            E_val=E.get_value(current_state)
                            change_rate_pos=min(CR,M-V,E_val)
                            change_rate_neg=0.0
                            DP1=np.arange(change_rate_neg,change_rate_pos+d_int, d_int)/loss_factor        
                            if 0.0 not in DP1:
                                bisect.insort(DP1, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP1 if (x>=0.0 and x<=E_val)]
                        elif(dv[0][0]=='L'):
                            Res=self.energy_vars[dv[1]]
                            d_int=Res.get_discretization_interval()
                            V=Res.get_value(current_state)
                            loss_factor=Res.get_conversion_loss(current_state)
                            L=float(self.energy_vars[dv[0]].get_value(current_state))
                            CR=Res.get_max_change(t)
                            change_rate_pos=0.0
                            change_rate_neg=-1*min(CR,V,L/loss_factor)
                            DP1=np.arange(change_rate_neg,change_rate_pos+d_int, d_int)        
                            if 0.0 not in DP1:
                                bisect.insort(DP1, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP1 if (x>=-1*(L/loss_factor) and x<=0.0)]
                else:
                    if (dv[0][0]=='G' and dv[1][0]=='L') or (dv[1][0]=='G' and dv[0][0]=='L'):
                        pass
                    elif (dv[1][0]=='L' or dv[0][0]=='L'):
                        if (dv[1][0]=='L'):
                            if (dv[0][0]=='E'):
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[0]].get_discretization_interval()
                                Decision_Possibilities[dv[0]+':'+dv[1]]=np.arange(0.0,max([Curr_val,Curr_val_2])+d_int, d_int)
                                
                            else:
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[0]].get_discretization_interval()
                                Decision_Possibilities[dv[0]+':'+dv[1]]=np.arange(0,max([Curr_val,Curr_val_2])+d_int, d_int)
                               
                        else:
                            if (dv[1][0]=='E'):
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[1]].get_discretization_interval()
                                Decision_Possibilities[dv[0]+':'+dv[1]]=np.arange(-1*max([Curr_val,Curr_val_2]),0.0+d_int, d_int)
                                
                            else:
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[1]].get_discretization_interval()
                                Decision_Possibilities[dv[0]+':'+dv[1]]=np.arange(-1*max([Curr_val,Curr_val_2]),0.0+d_int, d_int)
                                
                    else:
                        if (dv[0][0]=='E' or dv[1][0]=='E'):
                            if (dv[0][0]=='E'):
                                Proc=self.energy_vars[dv[0]]
                                Curr_val=Proc.get_value(current_state)
                                d_int=Proc.get_discretization_interval()
                                Decision_Possibilities[dv[0]+':'+dv[1]]=np.arange(0,Curr_val+d_int, d_int)
                            else:
                                Proc=self.energy_vars[dv[1]]
                                Curr_val=Proc.get_value(current_state)
                                d_int=Proc.get_discretization_interval()
                                Decision_Possibilities[dv[0]+':'+dv[1]]=np.arange(-1*Curr_val,0.0+d_int, d_int)
                                
        #ANOTHER CARTESIAN PRODUCT, THIS TIME ONE DIMENSION IS A PARTIAL DECISION FOUND EARLIER FROM EQUALITY CONSTRAINTS
        IEC_decisions_forward=possible_decisions_2(EC_decisions_forward, Decision_Possibilities)
        
        decisions_forward=[]
        new_states=[]
        for D in IEC_decisions_forward:
            Del=0
            Vars=copy(current_state)
            #REMOVE DECSION IF AN INEQUALITY CONSTRAINT VIOLATED
            for const in self.inequality_constraints:
                if not const(D,self.energy_vars, self.resource_vars,current_state):
                    Del=1
                    break
            if (Del==0):
                #REMOVE DECISION IF RESOURCE DROPS BELOW EMPTY OR RISES ABOVE MAX
                #DUE TO DECISION
                for dec in D:
                    spl=dec.split(':')
                    if spl[0][0]=='R' and spl[1][0]=='R':
                        if D[dec]>0.0:
                            Vars[spl[0]]-=D[dec]
                            if Vars[spl[0]]<0.0:
                                Del=1
                                break
                            Vars[spl[1]]+=(D[dec]*self.energy_vars[spl[0]].get_conversion_loss(current_state)*self.energy_vars[spl[1]].get_conversion_loss(current_state))
                            if Vars[spl[1]]>self.energy_vars[spl[1]].get_max():
                                Del=1
                                break
                        elif D[dec]<0.0:
                            Vars[spl[0]]-=(D[dec]*self.energy_vars[spl[0]].get_conversion_loss(current_state)*self.energy_vars[spl[1]].get_conversion_loss(current_state))
                            if Vars[spl[0]]>self.energy_vars[spl[0]].get_max():
                                Del=1
                                break
                            Vars[spl[1]]+=D[dec]
                            if Vars[spl[1]]<0.0:
                                Del=1
                                break
                    elif spl[0][0]=='R':
                        if D[dec]>0.0:
                            Vars[spl[0]]-=D[dec]
                            if Vars[spl[0]]<0:
                                Del=1
                                break
                        elif D[dec]<0.0:
                            Vars[spl[0]]-=(D[dec]*self.resource_vars[spl[0]].get_conversion_loss(current_state))
                            if Vars[spl[0]]>self.resource_vars[spl[0]].get_max():
                                Del=1
                                break
                    elif spl[1][0]=='R':
                        if D[dec]>0.0:
                            Vars[spl[1]]+=(D[dec]*self.resource_vars[spl[1]].get_conversion_loss(current_state))
                            if Vars[spl[1]]>self.resource_vars[spl[1]].get_max():
                                Del=1
                                break
                        elif D[dec]<0.0:
                            Vars[spl[1]]+=D[dec]
                            if Vars[spl[1]]<0:
                                Del=1
                                break
            if (Del==0):
                #IF ALL TESTS ARE PASSED, THIS IS A FEASIBLE ACTION AND
                #CAN BE ADDED TO LIST OF FEASIBLE ACTIONS
                for res in self.resource_vars:
                    Val=Vars[self.resource_vars[res].get_name()]
                    States=self.resource_vars[res].get_all_states()
                    if Val not in States:
                        Ind=bisect.bisect_right(States,Val)-1
                        if Ind<0:
                            Vars[self.resource_vars[res].get_name()]=States[0]
                        else:
                            Vars[self.resource_vars[res].get_name()]=States[Ind]
                decisions_forward.append(D)
                new_states.append(Vars)
            
        return [decisions_forward, new_states]
        