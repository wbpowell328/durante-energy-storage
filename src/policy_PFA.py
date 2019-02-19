from Pol import Pol
import bisect

class policy_PFA(Pol):
    
    def __init__(self, name, FilePath, GLB_VARS, params):
        self.name=name
        self.glb_vars=GLB_VARS
        self.params=params
        self.nodes=GLB_VARS.get_global_variable('Nodes')        
    
    def get_name(self):
        return self.name
        
    def offline_stage(self):
        return
        
    def get_all_params(self):
        return self.params
    
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
        self.params=params
        
    def decision(self, state):
        L=self.nodes['L'].get_preds_value(state)
        #print L
        E=self.nodes['E'].get_preds_value(state)
        #print E
        R=self.nodes['R_1'].get_preds_value(state)
        #print R
        P=self.nodes['G'].get_preds_value(state)
        #print P
        loss_factor=self.nodes['R_1'].conv_rate
        Rmax=self.nodes['R_1'].maximum
        CR=self.nodes['R_1'].max_charge_rate
        R_states=self.nodes['R_1'].all_states
        EL=min(L,E)
        L_left=L-EL
        E_left=E-EL
        if P>self.params[1]:
            RL1=min(L_left/loss_factor,R,CR/loss_factor)
            Ind=bisect.bisect_right(R_states,R-RL1)-1
            if Ind<0:
                NextR=R_states[0]
            elif Ind==len(R_states)-1:
                NextR=R_states[Ind]
            else:
                diff=abs(R-RL1-R_states[Ind])-abs(R-RL1-R_states[Ind+1])
                if diff>0:
                    NextR=R_states[Ind+1]
                else:
                    NextR=R_states[Ind]
            RL=R-NextR
        else:
            RL=0.0
        
        GL=L_left-RL*loss_factor
        ER=min(E_left,CR,Rmax-R)/loss_factor
        R_space=Rmax-(R+ER)        
        R_1=R-RL
        if P<self.params[0]:
            GR1=min(CR-ER,R_space)/loss_factor
            Ind=bisect.bisect_right(R_states,R_1+GR1)-1
            if Ind<0:
                NextR=R_states[0]
            elif Ind==len(R_states)-1:
                NextR=R_states[Ind]
            else:
                diff=abs(R_1+GR1-R_states[Ind])-abs(R_1+GR1-R_states[Ind+1])
                if diff>0:
                    NextR=R_states[Ind+1]
                else:
                    NextR=R_states[Ind]
            GR=NextR-R_1
        elif P>self.params[1]:
            GR1=-1*min(CR-RL,R_1)
            Ind=bisect.bisect_right(R_states,R+GR1)-1
            if Ind<0:
                NextR=R_states[0]
            elif Ind==len(R_states)-1:
                NextR=R_states[Ind]
            else:
                diff=abs(R_1+GR1-R_states[Ind])-abs(R_1+GR1-R_states[Ind+1])
                if diff>0:
                    NextR=R_states[Ind+1]
                else:
                    NextR=R_states[Ind]
            GR=NextR-R
        else:
            GR=0.0
        
        D={}
        D['E:L']=EL
        D['R_1:L']=RL
        D['G:L']=GL
        D['E:R_1']=ER
        D['G:R_1']=GR

        return D