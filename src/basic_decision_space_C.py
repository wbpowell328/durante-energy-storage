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

class basic_decision_space_C(decision_space):
    
    def __init__(self, GLOBAL_VARS, num_poss_per_decision_var):
        
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
        self.num_poss_per_decision_var=num_poss_per_decision_var       
        
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
                        if len(DP1)>(self.num_poss_per_decision_var[dv[0]+':'+dv[1]]):
                            DP3=[]
                            DP2=list(np.linspace(change_rate_neg,change_rate_pos,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]]))
                            for d in DP2:
                                Ind=bisect.bisect_right(DP1,d)-1
                                if Ind<0:
                                    DP3.append(DP1[0])
                                elif Ind==len(DP1)-1:
                                    DP3.append(DP1[Ind])
                                else:
                                    diff=abs(d-DP1[Ind])-abs(d-DP1[Ind+1])
                                    if diff>0:
                                        DP3.append(DP1[Ind+1])
                                    else:
                                        DP3.append(DP1[Ind])
                        else:
                            DP3=DP1
                        if 0.0 not in DP3:
                            bisect.insort(DP3, 0.0)
                        Decision_Possibilities[dv[0]+':'+dv[1]]=DP3
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
                            if len(DP1)>(self.num_poss_per_decision_var[dv[0]+':'+dv[1]]):
                                DP3=[]
                                DP2=list(np.linspace(change_rate_neg,change_rate_pos,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])/loss_factor)
                                for d in DP2:
                                    Ind=bisect.bisect_right(DP1,d)-1
                                    if Ind<0:
                                        DP3.append(DP1[0])
                                    elif Ind==len(DP1)-1:
                                        DP3.append(DP1[Ind])
                                    else:
                                        diff=abs(d-DP1[Ind])-abs(d-DP1[Ind+1])
                                        if diff>0:
                                            DP3.append(DP1[Ind+1])
                                        else:
                                            DP3.append(DP1[Ind])
                            else:
                                DP3=DP1
                            if 0.0 not in DP3:
                                bisect.insort(DP3, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=DP3
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
                            if len(DP1)>(self.num_poss_per_decision_var[dv[0]+':'+dv[1]]):
                                DP3=[]
                                DP2=list(np.linspace(change_rate_neg,change_rate_pos,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])/loss_factor)
                                for d in DP2:
                                    Ind=bisect.bisect_right(DP1,d)-1
                                    if Ind<0:
                                        DP3.append(DP1[0])
                                    elif Ind==len(DP1)-1:
                                        DP3.append(DP1[Ind])
                                    else:
                                        diff=abs(d-DP1[Ind])-abs(d-DP1[Ind+1])
                                        if diff>0:
                                            DP3.append(DP1[Ind+1])
                                        else:
                                            DP3.append(DP1[Ind])
                            else:
                                DP3=DP1
                            if 0.0 not in DP3:
                                bisect.insort(DP3, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP3 if (x>=-1*E_val and x<=0.0)]
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
                            if len(DP1)>(self.num_poss_per_decision_var[dv[0]+':'+dv[1]]):
                                DP3=[]
                                DP2=list(np.linspace(change_rate_neg,change_rate_pos,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]]))
                                for d in DP2:
                                    Ind=bisect.bisect_right(DP1,d)-1
                                    if Ind<0:
                                        DP3.append(DP1[0])
                                    elif Ind==len(DP1)-1:
                                        DP3.append(DP1[Ind])
                                    else:
                                        diff=abs(d-DP1[Ind])-abs(d-DP1[Ind+1])
                                        if diff>0:
                                            DP3.append(DP1[Ind+1])
                                        else:
                                            DP3.append(DP1[Ind])
                            else:
                                DP3=DP1
                            if 0.0 not in DP3:
                                bisect.insort(DP3, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP3 if (x>=0 and x<=(L/loss_factor))]
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
                            if len(DP1)>(self.num_poss_per_decision_var[dv[0]+':'+dv[1]]):
                                DP3=[]
                                DP2=list(np.linspace(change_rate_neg,change_rate_pos,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])/loss_factor)
                                for d in DP2:
                                    Ind=bisect.bisect_right(DP1,d)-1
                                    if Ind<0:
                                        DP3.append(DP1[0])
                                    elif Ind==len(DP1)-1:
                                        DP3.append(DP1[Ind])
                                    else:
                                        diff=abs(d-DP1[Ind])-abs(d-DP1[Ind+1])
                                        if diff>0:
                                            DP3.append(DP1[Ind+1])
                                        else:
                                            DP3.append(DP1[Ind])
                            else:
                                DP3=DP1
                            if 0.0 not in DP3:
                                bisect.insort(DP3, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=DP3
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
                            if len(DP1)>(self.num_poss_per_decision_var[dv[0]+':'+dv[1]]):
                                DP3=[]
                                DP2=list(np.linspace(change_rate_neg,change_rate_pos,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])/loss_factor)
                                for d in DP2:
                                    Ind=bisect.bisect_right(DP1,d)-1
                                    if Ind<0:
                                        DP3.append(DP1[0])
                                    elif Ind==len(DP1)-1:
                                        DP3.append(DP1[Ind])
                                    else:
                                        diff=abs(d-DP1[Ind])-abs(d-DP1[Ind+1])
                                        if diff>0:
                                            DP3.append(DP1[Ind+1])
                                        else:
                                            DP3.append(DP1[Ind])
                            else:
                                DP3=DP1
                            if 0.0 not in DP3:
                                bisect.insort(DP3, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP3 if (x>=0.0 and x<=E_val)]
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
                            if len(DP1)>(self.num_poss_per_decision_var[dv[0]+':'+dv[1]]):
                                DP3=[]
                                DP2=list(np.linspace(change_rate_neg,change_rate_pos,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]]))
                                for d in DP2:
                                    Ind=bisect.bisect_right(DP1,d)-1
                                    if Ind<0:
                                        DP3.append(DP1[0])
                                    elif Ind==len(DP1)-1:
                                        DP3.append(DP1[Ind])
                                    else:
                                        diff=abs(d-DP1[Ind])-abs(d-DP1[Ind+1])
                                        if diff>0:
                                            DP3.append(DP1[Ind+1])
                                        else:
                                            DP3.append(DP1[Ind])
                            else:
                                DP3=DP1
                            if 0.0 not in DP3:
                                bisect.insort(DP3, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP3 if (x>=-1*(L/loss_factor) and x<=0.0)]
                else:
                    if (dv[0][0]=='G' and dv[1][0]=='L') or (dv[1][0]=='G' and dv[0][0]=='L'):
                        pass
                    elif (dv[1][0]=='L' or dv[0][0]=='L'):
                        if (dv[1][0]=='L'):
                            if (dv[0][0]=='E'):
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                Res=self.energy_vars[self.res_name]
                                loss_factor=float(Res.get_conversion_loss(current_state))
                                V=Res.get_value(current_state)
                                M=Res.get_max()
                                CR=Res.get_max_change(t)
                                Batt_First=min([Curr_val_2,(M-V)/loss_factor,CR/loss_factor])
                                Sts=Res.get_all_states()
                                Ind=bisect.bisect_right(Sts,(V+Batt_First))-1
                                if Ind<0:
                                    Act_change=(Sts[0]-V)/loss_factor
                                else:
                                    Act_change=(Sts[Ind]-V)/loss_factor
                                E_to_L=min(Curr_val,Curr_val_2-Act_change)
                                Decision_Possibilities[dv[0]+':'+dv[1]]=[min([Curr_val,Curr_val_2])]
                                Renew_Left=[Curr_val_2-min([Curr_val,Curr_val_2])]
                                if E_to_L not in Decision_Possibilities[dv[0]+':'+dv[1]]:
                                    Decision_Possibilities[dv[0]+':'+dv[1]].append(E_to_L)
                                    Renew_Left.append(Curr_val_2-E_to_L)
                            else:
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[0]].get_discretization_interval()
                                DP1=np.arange(0,max([Curr_val,Curr_val_2])+d_int, d_int)
                                DP2=np.linspace(0,max([Curr_val,Curr_val_2]),num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])
                                if len(DP2)>len(DP1):
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                                else:
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP2
                        else:
                            if (dv[1][0]=='E'):
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[1]].get_discretization_interval()
                                DP1=np.arange(-1*max([Curr_val,Curr_val_2]),0.0+d_int, d_int)
                                DP2=np.linspace(-1*max([Curr_val,Curr_val_2]),0.0,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])
                                if len(DP2)>len(DP1):
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                                else:
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP2
                            else:
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[1]].get_discretization_interval()
                                DP1=np.arange(-1*max([Curr_val,Curr_val_2]),0.0+d_int, d_int)
                                DP2=np.linspace(-1*max([Curr_val,Curr_val_2]),0.0,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])
                                if len(DP2)>len(DP1):
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                                else:
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP2
                    else:
                        if (dv[0][0]=='E' or dv[1][0]=='E'):
                            if (dv[0][0]=='E'):
                                Proc=self.energy_vars[dv[0]]
                                Curr_val=Proc.get_value(current_state)
                                d_int=Proc.get_discretization_interval()
                                DP1=np.arange(0,Curr_val+d_int, d_int)
                                DP2=np.linspace(0,Curr_val,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])
                                if len(DP2)>len(DP1):
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                                else:
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP2
                            else:
                                Proc=self.energy_vars[dv[1]]
                                Curr_val=Proc.get_value(current_state)
                                d_int=Proc.get_discretization_interval()
                                DP1=np.arange(-1*Curr_val,0.0+d_int, d_int)
                                DP2=np.linspace(-1*Curr_val,0.0,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])
                                if len(DP2)>len(DP1):
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                                else:
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP2
        #T4=timeit.default_timer()
        #print 's'
        #print T4-T3
        
        EC_decisions_forward=possible_decisions_1(Decision_Possibilities)
        #T5=timeit.default_timer()
        #print T5-T4
        for D in EC_decisions_forward:
            Del=0
            Slack_Vars=self.equality_constraints(D,self.energy_vars,self.resource_vars,current_state)
            D.update(Slack_Vars)
        #T6=timeit.default_timer()
        #print T6-T5
        Decision_Possibilities={}
        for dv in self.decision_vars:
            if dv[0]+':'+dv[1] not in self.equality_decision_vars:
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
                        if len(DP1)>(self.num_poss_per_decision_var[dv[0]+':'+dv[1]]):
                            DP3=[]
                            DP2=list(np.linspace(change_rate_neg,change_rate_pos,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]]))
                            for d in DP2:
                                Ind=bisect.bisect_right(DP1,d)-1
                                if Ind<0:
                                    DP3.append(DP1[0])
                                elif Ind==len(DP1)-1:
                                    DP3.append(DP1[Ind])
                                else:
                                    diff=abs(d-DP1[Ind])-abs(d-DP1[Ind+1])
                                    if diff>0:
                                        DP3.append(DP1[Ind+1])
                                    else:
                                        DP3.append(DP1[Ind])
                        else:
                            DP3=DP1
                        if 0.0 not in DP3:
                            bisect.insort(DP3, 0.0)
                        Decision_Possibilities[dv[0]+':'+dv[1]]=DP3
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
                            if len(DP1)>(self.num_poss_per_decision_var[dv[0]+':'+dv[1]]):
                                DP3=[]
                                DP2=list(np.linspace(change_rate_neg,change_rate_pos,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])/loss_factor)
                                for d in DP2:
                                    Ind=bisect.bisect_right(DP1,d)-1
                                    if Ind<0:
                                        DP3.append(DP1[0])
                                    elif Ind==len(DP1)-1:
                                        DP3.append(DP1[Ind])
                                    else:
                                        diff=abs(d-DP1[Ind])-abs(d-DP1[Ind+1])
                                        if diff>0:
                                            DP3.append(DP1[Ind+1])
                                        else:
                                            DP3.append(DP1[Ind])
                            else:
                                DP3=DP1
                            if 0.0 not in DP3:
                                bisect.insort(DP3, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=DP3
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
                            if len(DP1)>(self.num_poss_per_decision_var[dv[0]+':'+dv[1]]):
                                DP3=[]
                                DP2=list(np.linspace(change_rate_neg,change_rate_pos,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])/loss_factor)
                                for d in DP2:
                                    Ind=bisect.bisect_right(DP1,d)-1
                                    if Ind<0:
                                        DP3.append(DP1[0])
                                    elif Ind==len(DP1)-1:
                                        DP3.append(DP1[Ind])
                                    else:
                                        diff=abs(d-DP1[Ind])-abs(d-DP1[Ind+1])
                                        if diff>0:
                                            DP3.append(DP1[Ind+1])
                                        else:
                                            DP3.append(DP1[Ind])
                            else:
                                DP3=DP1
                            if 0.0 not in DP3:
                                bisect.insort(DP3, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP3 if (x>=-1*E_val and x<=0.0)]
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
                            if len(DP1)>(self.num_poss_per_decision_var[dv[0]+':'+dv[1]]):
                                DP3=[]
                                DP2=list(np.linspace(change_rate_neg,change_rate_pos,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]]))
                                for d in DP2:
                                    Ind=bisect.bisect_right(DP1,d)-1
                                    if Ind<0:
                                        DP3.append(DP1[0])
                                    elif Ind==len(DP1)-1:
                                        DP3.append(DP1[Ind])
                                    else:
                                        diff=abs(d-DP1[Ind])-abs(d-DP1[Ind+1])
                                        if diff>0:
                                            DP3.append(DP1[Ind+1])
                                        else:
                                            DP3.append(DP1[Ind])
                            else:
                                DP3=DP1
                            if 0.0 not in DP3:
                                bisect.insort(DP3, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP3 if (x>=0 and x<=(L/loss_factor))]
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
                            if len(DP1)>(self.num_poss_per_decision_var[dv[0]+':'+dv[1]]):
                                DP3=[]
                                DP2=list(np.linspace(change_rate_neg,change_rate_pos,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])/loss_factor)
                                for d in DP2:
                                    Ind=bisect.bisect_right(DP1,d)-1
                                    if Ind<0:
                                        DP3.append(DP1[0])
                                    elif Ind==len(DP1)-1:
                                        DP3.append(DP1[Ind])
                                    else:
                                        diff=abs(d-DP1[Ind])-abs(d-DP1[Ind+1])
                                        if diff>0:
                                            DP3.append(DP1[Ind+1])
                                        else:
                                            DP3.append(DP1[Ind])
                            else:
                                DP3=DP1
                            if 0.0 not in DP3:
                                bisect.insort(DP3, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=DP3
                        elif (dv[0][0]=='E'):
                            Res=self.energy_vars[dv[1]]
                            E=self.energy_vars[dv[0]]
                            d_int=Res.get_discretization_interval()
                            loss_factor=Res.get_conversion_loss(current_state)
                            V=Res.get_value(current_state)
                            M=Res.get_max()
                            CR=Res.get_max_change(t)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[]
                           
                            for r in Renew_Left:
                                change=min(CR/loss_factor,(M-V)/loss_factor,r)*loss_factor
                                Sts=Res.get_all_states()
                                Ind=bisect.bisect_right(Sts,(V+change))-1
                                if Ind<0:
                                    Act_change=(Sts[0]-V)/loss_factor
                                else:
                                    Act_change=(Sts[Ind]-V)/loss_factor
                                
                                Decision_Possibilities[dv[0]+':'+dv[1]].append(Act_change)
                            
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
                            if len(DP1)>(self.num_poss_per_decision_var[dv[0]+':'+dv[1]]):
                                DP3=[]
                                DP2=list(np.linspace(change_rate_neg,change_rate_pos,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]]))
                                for d in DP2:
                                    Ind=bisect.bisect_right(DP1,d)-1
                                    if Ind<0:
                                        DP3.append(DP1[0])
                                    elif Ind==len(DP1)-1:
                                        DP3.append(DP1[Ind])
                                    else:
                                        diff=abs(d-DP1[Ind])-abs(d-DP1[Ind+1])
                                        if diff>0:
                                            DP3.append(DP1[Ind+1])
                                        else:
                                            DP3.append(DP1[Ind])
                            else:
                                DP3=DP1
                            if 0.0 not in DP3:
                                bisect.insort(DP3, 0.0)
                            Decision_Possibilities[dv[0]+':'+dv[1]]=[x for x in DP3 if (x>=-1*(L/loss_factor) and x<=0.0)]
                else:
                    if (dv[0][0]=='G' and dv[1][0]=='L') or (dv[1][0]=='G' and dv[0][0]=='L'):
                        pass
                    elif (dv[1][0]=='L' or dv[0][0]=='L'):
                        if (dv[1][0]=='L'):
                            if (dv[0][0]=='E'):
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[0]].get_discretization_interval()
                                DP1=np.arange(0.0,max([Curr_val,Curr_val_2])+d_int, d_int)
                                DP2=np.linspace(0.0,max([Curr_val,Curr_val_2]),num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])
                                if len(DP2)>len(DP1):
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                                else:
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP2
                            else:
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[0]].get_discretization_interval()
                                DP1=np.arange(0,max([Curr_val,Curr_val_2])+d_int, d_int)
                                DP2=np.linspace(0,max([Curr_val,Curr_val_2]),num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])
                                if len(DP2)>len(DP1):
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                                else:
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP2
                        else:
                            if (dv[1][0]=='E'):
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[1]].get_discretization_interval()
                                DP1=np.arange(-1*min([Curr_val,Curr_val_2]),0.0+d_int, d_int)
                                DP2=np.linspace(-1*min([Curr_val,Curr_val_2]),0.0,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])
                                if len(DP2)>len(DP1):
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                                else:
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP2
                            else:
                                Curr_val=self.energy_vars[dv[1]].get_value(current_state)
                                Curr_val_2=self.energy_vars[dv[0]].get_value(current_state)
                                d_int=self.energy_vars[dv[1]].get_discretization_interval()
                                DP1=np.arange(-1*min([Curr_val,Curr_val_2]),0.0+d_int, d_int)
                                DP2=np.linspace(-1*min([Curr_val,Curr_val_2]),0.0,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])
                                if len(DP2)>len(DP1):
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                                else:
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP2
                    else:
                        if (dv[0][0]=='E' or dv[1][0]=='E'):
                            if (dv[0][0]=='E'):
                                Proc=self.energy_vars[dv[0]]
                                Curr_val=Proc.get_value(current_state)
                                d_int=Proc.get_discretization_interval()
                                DP1=np.arange(0,Curr_val+d_int, d_int)
                                DP2=np.linspace(0,Curr_val,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])
                                if len(DP2)>len(DP1):
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                                else:
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP2
                            else:
                                Proc=self.energy_vars[dv[1]]
                                Curr_val=Proc.get_value(current_state)
                                d_int=Proc.get_discretization_interval()
                                DP1=np.arange(-1*Curr_val,0.0+d_int, d_int)
                                DP2=np.linspace(-1*Curr_val,0.0,num=self.num_poss_per_decision_var[dv[0]+':'+dv[1]])
                                if len(DP2)>len(DP1):
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP1
                                else:
                                    Decision_Possibilities[dv[0]+':'+dv[1]]=DP2                  
        #T7=timeit.default_timer()
        IEC_decisions_forward=possible_decisions_2(EC_decisions_forward, Decision_Possibilities)
        #print len(IEC_decisions_forward)
        decisions_forward=[]
        new_states=[]
        for D in IEC_decisions_forward:
            Del=0
            Vars=copy(current_state)
            if self.inequality_constraints:
                for const in self.inequality_constraints:
                    if not const(D,self.energy_vars, self.resource_vars,current_state):
                        Del=1
                        break
            if (Del==0):
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
            
        #T8=timeit.default_timer()
        #print T8-T7
        #print 'e'
        return [decisions_forward, new_states]