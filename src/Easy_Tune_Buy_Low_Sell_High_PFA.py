import numpy as np
import timeit
import copy
import matplotlib.pyplot as plt
import bisect
import scipy as sp
import os
import collections
from policy1 import policy1

def post_bound(GLB_VARS, state, sample_paths, horizon):      
    horizon=horizon
    resource_vars=GLB_VARS.get_global_variable('Resource_Vars')
    eta=resource_vars['R_1'].conv_rate
    beta=resource_vars['R_1'].max_charge_rate
    R_max=resource_vars['R_1'].maximum
    Aeq=np.zeros((2*horizon,7*horizon)) 
    Beq=np.zeros(2*horizon)
    Aiq=np.zeros((11*horizon,7*horizon))
    Biq=np.zeros(11*horizon)
    for i in xrange(horizon):
        ind1=7*i
        ind2=2*i
        ind3=11*i
        Aeq[ind2,ind1+2]=1.0
        Aeq[ind2,ind1+3]=1.0
        Aeq[ind2,ind1+5]=eta
        if ind2==0:
            Aeq[ind2+1,ind1+6]=1.0
        else:
            Aeq[ind2+1,ind1+6]=1.0
            Aeq[ind2+1,ind1-1]=-1.0
            Aeq[ind2+1,ind1-2]=1.0                
            Aeq[ind2+1,ind1-7]=-eta
            Aeq[ind2+1,ind1-6]=1.0
            Aeq[ind2+1,ind1-3]=-eta
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
        Biq[ind3+10]=R_max
        Biq[ind3+1]=beta
        Biq[ind3+2]=beta
    
    '''
    x=[GR0,RG0,GL0,EL0,ER0,RL0,R0,GR1,...]^T
    '''
    t=state['T']
    T=horizon
#        print t
#        print T
    E_pre=sample_paths['E'][t:(t+T)]
    P_pre=sample_paths['P'][t:(t+T)]
    L1=sample_paths['L'][t:(t+T)]
    R=state['R_1']
    E=np.zeros(T)
    P=np.zeros(T)
    L=np.zeros(T)
    for j in xrange(T):
        E[j]=np.round(E_pre[j]['E'])
        P[j]=np.round(P_pre[j]['P'])
        L[j]=np.round(L1[j])

    Beq[1]=R
    c=np.zeros(7*T)
    for i in xrange(T):
        ind1=7*i
        ind2=2*i
        ind3=11*i
        c[ind1]=P[i]*(1.0/100.0)
        c[ind1+1]=-eta*P[i]*(1.0/100.0)
        c[ind1+2]=P[i]*(1.0/100.0)
        Beq[ind2]=L[i]
        Biq[ind3]=E[i]
    Options={'maxiter':10000}
    DecVec1=sp.optimize.linprog(c, A_ub=Aiq, b_ub=Biq, A_eq=Aeq, b_eq=Beq,options=Options)
    print(DecVec1)
#        print DecVec1
    DecVec=DecVec1['x']
    objfun=DecVec1['fun']
#        print R
    
    return [DecVec,objfun]
    
def sample_paths(Init_State, Time, Processes_to_sample, Processes):
    
    '''
    For each exogenous process, given an initial state, produce a full length
    sample path that, per trial, does not vary accross policies
    '''
    state=Init_State
    init_time=Init_State['T']
    Sample_Paths={}
    for Proc in Processes:
        if Proc.get_name() in Processes_to_sample:
            Sample_Paths[Proc.get_name()]=[]
            Sample_Paths[Proc.get_name()].append(state[Proc.get_name()])
    for T in range(Time):
        for Proc in Processes:
            if Proc.get_name() in Processes_to_sample:
                if ((T+init_time+1)%Proc.get_time_step())==0:
                    state[Proc.get_name()]=Proc.forward_transition(state)
                Sample_Paths[Proc.get_name()].append(state[Proc.get_name()])
        state['T']=init_time+T+1
    return Sample_Paths

class Easy_Tune:

    def __init__(self, params,GLB_VARS,InputString):
        self.glb_vars=GLB_VARS
        self.evp=params[3]
        self.pvp=params[5]
        self.fvp=params[7]
        self.processes=GLB_VARS.get_global_variable('Processes')
        self.evp_avg_or_ind=params[4]
        self.pvp_avg_or_ind=params[6]
        self.fvp_avg_or_ind=params[8]
        self.r_fnct=GLB_VARS.get_global_variable('Reward_Function')
        self.resource_vars=GLB_VARS.get_global_variable('Resource_Vars')
        self.energy_vars=GLB_VARS.get_global_variable('Energy_Vars')
        self.price_vars=GLB_VARS.get_global_variable('Price_Vars')
        self.eta=self.resource_vars['R_1'].conv_rate
        self.InputString=InputString
        self.pb=params[9]
        self.buy_low=params[10]
        self.sell_high=params[11]
        self.Pols={}
        
    def test_policies(self, init_time, time, trials, file_name):
        
        '''
        Initial state held constant,
        Sample paths conducted at start time defined by initial state
        and carried out for T=time steps.
        Sequential decision process made for each policy on the same set
        of exogenous process sample paths for k trials
        Output results plotted according to inputs and written to file
        specified by file_name
        '''
        state={}
        for proc in self.processes:
            state[proc.get_name()]=proc.get_initial_state(init_time)
        for res in self.resource_vars:
            state[self.resource_vars[res].get_name()]=self.resource_vars[res].get_max()/2
            state['T']=init_time
        Pol_offline_times={}
        Pol_online_times={}
        Rewards={}
        Cumulative_Rewards={}
        Energy_Plots={}
        Price_Plots={}
        Flow_Plots={}
        os.chdir("..")
        os.chdir("data")
        f=open(file_name,'w')
        f.write('Policy Comparisons'+os.linesep)
        f.write(self.InputString)
        
        
        ###INITIALIZE DATA COLLECTION OBJECTS
        
        if self.pb:
            self.Pols['PB']='PB'
        for theta_L in np.linspace(self.buy_low[0],self.buy_low[1],self.buy_low[2]):
            for theta_U in np.linspace(self.sell_high[0],self.sell_high[1],self.sell_high[2]):
                if theta_L<=theta_U:
                    self.Pols[str((theta_L,theta_U))]=policy1((theta_L,theta_U),'Nada', self.glb_vars, [theta_L,theta_U])
        for pol_name in self.Pols:
            if self.evp:
                Energy_Plots[pol_name]={}
                for e_var in self.evp:
                    Energy_Plots[pol_name][e_var]=np.zeros((time, trials))
            if self.pvp:
                Price_Plots[pol_name]={}
                for p_var in self.pvp:
                    Price_Plots[pol_name][p_var]=np.zeros((time, trials))
            if self.fvp:
                Flow_Plots[pol_name]={}
                for f_var in self.fvp:
                    Flow_Plots[pol_name][f_var]=np.zeros((time, trials))
            Pol_online_times[pol_name]=np.zeros((time,trials))
            Rewards[pol_name]=np.zeros((time,trials))
            Cumulative_Rewards[pol_name]=np.zeros((time,trials))
            if not pol_name=='PB': 
                T0=timeit.default_timer()
                self.Pols[pol_name].offline_stage()
                T1=timeit.default_timer()
                Pol_offline_times[pol_name]=T1-T0
            else:
                Pol_offline_times[pol_name]=0.0
        ###TEMPORARY###
        processes_to_sample=[]
        for proc in self.processes:
            processes_to_sample.append(proc.get_name())
        
        E_forc=np.zeros(time)
        P_forc=np.zeros(time)
        
        for i in xrange(trials):
            print(i)
            init_state=copy.copy(state)
            States={}
            for pol in self.Pols:
                States[pol]=copy.copy(state)
            proc_sample_paths=sample_paths(init_state, time, processes_to_sample, self.processes)
            if self.pb:
                p_bound=post_bound(self.glb_vars,state,proc_sample_paths,time)
                dec_vec=p_bound[0]
                rew=p_bound[1]
            for j in xrange(time):
                print(j)
                for pol in self.Pols:
                    E_forc[j]=self.energy_vars['E'].get_forecast(States[pol]['T'])
                    if type(States[pol]['P']) is collections.OrderedDict:
                        P_forc[j]=self.price_vars['P'].get_forecast_plus_seas(States[pol]['T'])
                    else:
                        P_forc[j]=States[pol]['P']
                    ###COLLECT PLOTTING DATA
                    if self.evp:
                        for e_var in self.evp:
                            Energy_Plots[pol][e_var][j,i]=self.energy_vars[e_var].get_value(States[pol])
                    if self.pvp:
                        for p_var in self.pvp:
                            Price_Plots[pol][p_var][j,i]=self.price_vars[p_var].get_value(States[pol])
                    ###MAKE DECISION ACCORDING TO POLICY
                    T2=timeit.default_timer()
#                    print States[pol]
                    if pol[0:3]=='DLA':
                        Dec=self.Pols[pol].decision(States[pol],proc_sample_paths,time-j)
                    elif pol=='PB':
                        DecVec=dec_vec[(j*7):((j+1)*7)]
                        Dec1={}
                        Dec1['G:R_1']=DecVec[0]-DecVec[1]
                        Dec1['G:L']=DecVec[2]
                        Dec1['E:L']=DecVec[3]
                        Dec1['E:R_1']=DecVec[4]
                        Dec1['R_1:L']=DecVec[5]
                        Rnew={}
                        if Dec1['G:R_1']>0:
                            Rnew['R_1']=States[pol]['R_1']+self.eta*Dec1['G:R_1']+self.eta*Dec1['E:R_1']-Dec1['R_1:L']
                        else:
                            Rnew['R_1']=States[pol]['R_1']+Dec1['G:R_1']+self.eta*Dec1['E:R_1']-Dec1['R_1:L']
                        Dec=[Dec1,Rnew]
                    else:
                        Dec=self.Pols[pol].decision(States[pol])
                    D=Dec[0]
                    T3=timeit.default_timer()
                    ###TIME DECISION
                    Pol_online_times[pol][j,i]=T3-T2
                    ###COLLECT PLOTTING DATA ON FLOW VARIABLES
                    if self.fvp:
                        for f_var in self.fvp:
                            Flow_Plots[pol][f_var][j,i]=D[f_var]
                            
                    ###RECIEVE REWARD AND ACCUMULATE
                    Rewards[pol][j,i]=self.r_fnct.get_reward(States[pol],D)
                    if j==0:
                        Cumulative_Rewards[pol][j,i]=Rewards[pol][j,i]
                    else:
                        Cumulative_Rewards[pol][j,i]=Cumulative_Rewards[pol][j-1,i]+Rewards[pol][j,i]
                    if j==(time-1):
                        print(pol+':'+str(Cumulative_Rewards[pol][j,i]))
                        f.write(pol+':'+str(Cumulative_Rewards[pol][j,i])+os.linesep)
                    States[pol]['R_1']=Dec[1]['R_1']
                    
                    #print States[pol]['P']['Tt']
                    ###TRANSITION ACCORDING TO PRE DEFINED SAMPLE PATH
                    ###OR IF NOT DEFINED FOR PROCESS, MAKE A TRANSITION FOR THAT PROCESS
                    for proc in self.processes:
                        if proc.get_name() in processes_to_sample:
                            States[pol][proc.get_name()]=proc_sample_paths[proc.get_name()][j+1]
                        else:
                            if ((States[pol]['T']+1)%proc.get_time_step()==0):
                                States[pol][proc.get_name()]=proc.forward_transition(States[pol])
                    States[pol]['T']=init_time+j+1
                    
        
        ###THE REST IS SIMPLY DATA OUTPUT AND PLOTTING COMMANDS###

        if self.pb:
            pol='PB'
            Results=[]
            OnlineTimePerSamplePath=[]
            for x in xrange(trials):
                Results.append(np.sum(Rewards[pol][:,x]))
                OnlineTimePerSamplePath.append(np.mean(Pol_online_times[pol][:,x]))
                f.write(str(Results[x])+'\n')
            AvgOnlineTimePerSamplePath=np.mean(OnlineTimePerSamplePath)
            AvgValPB=np.mean(Results)
            StdDev=np.std(Results)
            Str='Policy: '+pol+'; Trials: '+str(trials)            
            Str=Str+'; Avg. Value: ' +str(AvgValPB)+'; Std. Dev.: ' +str(StdDev)
            Str=Str+'; Offline Time: '+str(Pol_offline_times[pol])+' sec'
            Str=Str+'; Avg. Online Time per Decision: '+str(AvgOnlineTimePerSamplePath)+' sec \n'
            print(Str)
            f.write(Str)
        BEST=-np.inf
        for pol in self.Pols:
            if not pol=='PB':
                Results=[]
                OnlineTimePerSamplePath=[]
                for x in xrange(trials):
                    Results.append(np.sum(Rewards[pol][:,x]))
                    OnlineTimePerSamplePath.append(np.mean(Pol_online_times[pol][:,x]))
                    f.write(str(Results[x])+'\n')
                AvgOnlineTimePerSamplePath=np.mean(OnlineTimePerSamplePath)
                AvgVal=np.mean(Results)
                if AvgVal>BEST:
                    BEST=AvgVal
                    Pars=pol
                if self.pb:
                    Percent_of_PB=(float(AvgVal)/AvgValPB)*100.0
                StdDev=np.std(Results)
                Str='Policy: '+pol+'; Trials: '+str(trials)            
                Str=Str+'; Avg. Value: ' +str(AvgVal)
                if self.pb:
                    Str=Str+'; Percent vs Posterior Bound: ' +str(Percent_of_PB)
                Str=Str+'; Std. Dev.: ' +str(StdDev)
                Str=Str+'; Offline Time: '+str(Pol_offline_times[pol])+' sec'
                Str=Str+'; Avg. Online Time per Decision: '+str(AvgOnlineTimePerSamplePath)+' sec \n'
                print(Str)
                f.write(Str)
        print(Pars)
        print(BEST)
        f.close()
        colors=['b','r','g','y','c','m']
        figure_num=0
        if self.evp:
            for i in xrange(len(self.evp)):
                e_var=self.evp[i]
                figure_num+=1
                if self.evp_avg_or_ind[i]:              
                    plt.figure(figure_num)
                    plt.clf()
                    plt.hold(True)
                    cnum=0
                    if e_var=='E':
                        cl=colors[cnum%(len(colors))]
                        plt.plot(E_forc,color=cl,label='Forecast')
                        cnum+=1
                    for pol in self.Pols:
                        cl=colors[cnum%(len(colors))]
                        cnum+=1
                        plt.plot(Energy_Plots[pol][e_var],color=cl, label=pol)
                    if e_var not in processes_to_sample: 
                        plt.legend(loc='best')
                    plt.title(e_var)
                    plt.xlabel('t')
                    plt.ylabel('Power Unit')
                    plt.hold(False)
                else:
                    plt.figure(figure_num)
                    plt.clf()
                    plt.hold(True)
                    cnum=0
                    if e_var=='E':
                        cl=colors[cnum%(len(colors))]
                        plt.plot(E_forc,color=cl,label='Forecast')
                        cnum+=1
                    for pol in self.Pols:
                        cl=colors[cnum%(len(colors))]
                        cnum+=1
                        PltVec=[]
                        for t in xrange(time):
                            PltVec.append(np.mean(Energy_Plots[pol][e_var][t,:])/12)
                        plt.plot(PltVec, color=cl, label=pol)
                    if e_var not in processes_to_sample: 
                        plt.legend(loc='best')
                    plt.title(e_var)
                    plt.xlabel('t')
                    plt.ylabel('Power Unit')
                    plt.hold(False)
        if self.pvp:
            for i in xrange(len(self.pvp)):
                p_var=self.pvp[i]
                figure_num+=1
                if self.pvp_avg_or_ind[i]:              
                    plt.figure(figure_num)
                    plt.clf()
                    plt.hold(True)
                    cnum=0
                    if p_var=='P':
                        cl=colors[cnum%(len(colors))]
                        plt.plot(P_forc,color=cl,label='Time Dependent Mean')
                        cnum+=1
                    for pol in self.Pols:
                        cl=colors[cnum%(len(colors))]
                        cnum+=1
                        plt.plot(Price_Plots[pol][p_var],color=cl, label=pol)
                    if p_var not in processes_to_sample: 
                        plt.legend(loc='best')
                    plt.title(p_var)
                    plt.xlabel('t')
                    plt.ylabel('Price')
                    plt.hold(False)
                else:
                    plt.figure(figure_num)
                    plt.clf()
                    plt.hold(True)
                    cnum=0
                    if p_var=='P':
                        cl=colors[cnum%(len(colors))]
                        plt.plot(P_forc,color=cl,label='Time Dependent Mean')
                        cnum+=1
                    for pol in self.Pols:
                        cl=colors[cnum%(len(colors))]
                        cnum+=1
                        PltVec=[]
                        for t in xrange(time):
                            PltVec.append(np.mean(Price_Plots[pol][p_var][t,:]))
                        plt.plot(PltVec, color=cl, label=pol)
                    if p_var not in processes_to_sample: 
                        plt.legend(loc='best')
                    plt.title(p_var)
                    plt.xlabel('t')
                    plt.ylabel('Price')
                    plt.hold(False)
        if self.fvp:
            for i in xrange(len(self.fvp)):
                f_var=self.fvp[i]
                figure_num+=1
                if self.fvp_avg_or_ind[i]:              
                    plt.figure(figure_num)
                    plt.clf()
                    plt.hold(True)
                    cnum=0
                    for pol in self.Pols:
                        cl=colors[cnum%(len(colors))]
                        cnum+=1
                        plt.plot(Flow_Plots[pol][f_var],color=cl, label=pol)
                    if f_var not in processes_to_sample: 
                        plt.legend(loc='best')
                    plt.title(f_var)
                    plt.xlabel('t')
                    plt.ylabel('Power Unit')
                    plt.hold(False)
                else:
                    plt.figure(figure_num)
                    plt.clf()
                    plt.hold(True)
                    cnum=0
                    for pol in self.Pols:
                        cl=colors[cnum%(len(colors))]
                        cnum+=1
                        PltVec=[]
                        for t in xrange(time):
                            PltVec.append(np.mean(Flow_Plots[pol][f_var][t,:]))
                        plt.plot(PltVec, color=cl, label=pol)
                    if f_var not in processes_to_sample: 
                        plt.legend(loc='best')
                    plt.title(f_var)
                    plt.xlabel('t')
                    plt.ylabel('Power Unit')
                    plt.hold(False)
       
        figure_num+=1
        plt.figure(figure_num)
        plt.clf()
        plt.hold(True)
        cnum=0
        for pol in self.Pols:
            cl=colors[cnum%(len(colors))]
            cnum+=1
            PltVec=[]
            for t in xrange(time):
                PltVec.append(np.mean(Rewards[pol][t,:]))
            plt.plot(PltVec, color=cl, label=pol) 
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.ylabel('Reward')
        plt.title('Average Reward at Time t')
        plt.hold(False)
        figure_num+=1
        plt.figure(figure_num)
        plt.clf()
        plt.hold(True)
        cnum=0
        for pol in self.Pols:
            cl=colors[cnum%(len(colors))]
            cnum+=1
            PltVec=[]
            for t in xrange(time):
                PltVec.append(np.mean(Cumulative_Rewards[pol][t,:]))
            plt.plot(PltVec, color=cl, label=pol) 
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.ylabel('Reward')
        plt.title('Average Cumulative Reward at Time t')
        plt.hold(False)
        