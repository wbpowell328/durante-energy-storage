from Pol import Pol
import numpy as np
import scipy as sp      

class policy_DLA(Pol):
    
    def __init__(self, name, FilePath, GLB_VARS, params):
        self.name=name
        self.glb_vars=GLB_VARS
        self.params=params
        self.horizon=params[0]
        self.nodes=GLB_VARS.get_global_variable('Nodes')
        self.r_fnct=GLB_VARS.get_global_variable('Reward_Function')
        self.eta=self.nodes['R_1'].conv_rate
        self.beta=self.nodes['R_1'].max_charge_rate
        self.R_max=self.nodes['R_1'].get_max()
        self.E_max=self.nodes['E'].get_max()
        Aeq=np.zeros((2*self.horizon,7*self.horizon)) 
        Beq=np.zeros(2*self.horizon)
        Aiq=np.zeros((11*self.horizon,7*self.horizon))
        Biq=np.zeros(11*self.horizon)
        for i in xrange(self.horizon):
            ind1=7*i
            ind2=2*i
            ind3=11*i
            Aeq[ind2,ind1+2]=1.0
            Aeq[ind2,ind1+3]=1.0
            Aeq[ind2,ind1+5]=self.eta
            if ind2==0:
                Aeq[ind2+1,ind1+6]=1.0
            else:
                Aeq[ind2+1,ind1+6]=1.0
                Aeq[ind2+1,ind1-1]=-1.0
                Aeq[ind2+1,ind1-2]=1.0                
                Aeq[ind2+1,ind1-7]=-self.eta
                Aeq[ind2+1,ind1-6]=1.0
                Aeq[ind2+1,ind1-3]=-self.eta
            Aiq[ind3,ind1+3]=1.0
            Aiq[ind3,ind1+4]=1.0
            Aiq[ind3+1,ind1+1]=1.0
            Aiq[ind3+1,ind1+5]=1.0
            Aiq[ind3+2,ind1+4]=1.0
            Aiq[ind3+2,ind1]=1.0
            Aiq[ind3+3,ind1]=-1.0
            Aiq[ind3+4,ind1+1]=-1.0
            Aiq[ind3+5,ind1+2]=-1.0
            Aiq[ind3+6,ind1+3]=-1.0
            Aiq[ind3+7,ind1+4]=-1.0
            Aiq[ind3+8,ind1+5]=-1.0
            Aiq[ind3+9,ind1+6]=-1.0
            Aiq[ind3+10,ind1+6]=1.0
            Biq[ind3+10]=self.R_max
            Biq[ind3+1]=self.beta
            Biq[ind3+2]=self.beta
                
        self.Aeq=Aeq
        self.Beq=Beq
        self.Aiq=Aiq
        self.Biq=Biq
#        self.corr_coeff_E=self.energy_vars['E'].ARcoeff
#        self.corr_coeff_P=self.price_vars['P'].ARcoeff
#        self.ErrorCorrMatrixWind=np.zeros((self.horizon-1,self.horizon-1))
#        self.ErrorCorrMatrixPrice=np.zeros((self.horizon-1,self.horizon-1))
#        price_std=self.price_vars['P'].stdev
#        wind_std=self.energy_vars['E'].stdev
#        wind_std_vec=[]
#        price_std_vec=[]
#        for i in xrange(self.horizon-1):
#            wind_std_vec.append(wind_std*float(i)*self.stdconst)
#            price_std_vec.append(wind_std*float(i)*self.stdconst)
#        #print wind_std_vec
#        #print price_std_vec
#        for i in xrange(self.horizon-1):
#            for j in xrange(self.horizon-1):
#                if i==j:
#                    self.ErrorCorrMatrixWind[i,i]=np.power(wind_std_vec[i],2)
#                    self.ErrorCorrMatrixPrice[i,i]=np.power(price_std_vec[i],2)
#                else:
#                    self.ErrorCorrMatrixWind[i,j]=wind_std_vec[i]*wind_std_vec[j]*np.power(self.corr_coeff_E,abs(i-j))
#                    self.ErrorCorrMatrixPrice[i,j]=price_std_vec[i]*price_std_vec[j]*np.power(self.corr_coeff_P,abs(i-j))
#        #print self.ErrorCorrMatrixPrice
#        #print self.ErrorCorrMatrixWind
        

        
    def get_name(self):
        return self.name
        
    def offline_stage(self):
        """
        can leave blank
        """
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
        
        '''
        x=[GR0,RG0,GL0,EL0,ER0,RL0,R0,GR1,...]^T
        '''
        t=state['T']
        t_left=self.glb_vars.get_global_variable('Horizon')-t
        T=min(self.horizon,t_left+1)
#        print t
#        print T
        R=self.nodes['R_1'].get_preds_value(state)
        E=np.zeros(T)
        P=np.zeros(T)
        L=np.zeros(T)     
        for j in xrange(T):
            if j==0:
                E[j]=np.round(self.nodes['E'].get_preds_value(state))
                P[j]=np.round(self.nodes['G'].get_preds_value(state))
                L[j]=np.round(self.nodes['L'].get_preds_value(state))
            else:
                E[j]=np.round(self.nodes['E'].get_forecast(t+j))
                P[j]=np.round(self.nodes['G'].get_forecast(t+j))
                L[j]=np.round(self.nodes['L'].get_forecast(t+j))

        self.Beq[1]=R
        c=np.zeros(7*T)
        for i in xrange(T):
            ind1=7*i
            ind2=2*i
            ind3=11*i
            c[ind1]=P[i]
            c[ind1+1]=-self.eta*P[i]
            c[ind1+2]=P[i]
            self.Beq[ind2]=L[i]
            self.Biq[ind3]=E[i]
        
        aeq=self.Aeq[0:(2*T),0:(7*T)]

        beq=self.Beq[0:(2*T)]

        aiq=self.Aiq[0:(11*T),0:(7*T)]
     
        biq=self.Biq[0:(11*T)]
        
        DecVec1=sp.optimize.linprog(c, A_ub=aiq, b_ub=biq, A_eq=aeq, b_eq=beq)
#        print DecVec1
        DecVec=DecVec1['x']
#        print R
        Dec={}
        Dec['G:R_1']=DecVec[0]-DecVec[1]
        Dec['G:L']=DecVec[2]
        Dec['E:L']=DecVec[3]
        Dec['E:R_1']=DecVec[4]
        Dec['R_1:L']=DecVec[5]
        
        return Dec