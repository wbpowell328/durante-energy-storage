import numpy as np
import timeit
import copy
import matplotlib.pyplot as plt
import os
import random
    
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

class Policy_Tester:

    def __init__(self, Policies,params,GLB_VARS,InputString):
        self.Pols=Policies
        self.glb_vars=GLB_VARS
        self.evp=params[1]
        self.fvp=params[3]
        self.nodes=GLB_VARS.get_global_variable('Nodes')
        self.evp_avg_or_ind=params[2]
        self.fvp_avg_or_ind=params[4]
        self.r_fnct=GLB_VARS.get_global_variable('Reward_Function')
        self.InputString=InputString
        
    def test_policies(self, trials, file_name):
        
        '''
        Initial state held constant,
        Sample paths conducted at start time defined by initial state
        and carried out for T=time steps.
        Sequential decision process made for each policy on the same set
        of exogenous process sample paths for k trials
        Output results plotted according to inputs and written to file
        specified by file_name
        '''

        Pol_offline_times={}
        Pol_online_times={}
        Rewards={}
        Cumulative_Rewards={}
        Node_Plots={}
        Flow_Plots={}
        os.chdir("..")
        os.chdir("data")
        f=open(file_name,'w')
        f.write('Policy Comparisons'+os.linesep)
        f.write(self.InputString)
        time=self.glb_vars.get_global_variable('Horizon')
        
        
        ###INITIALIZE DATA COLLECTION OBJECTS
        
        for pol_name in self.Pols:
            if self.evp:
                Node_Plots[pol_name]={}
                for e_var in self.evp:
                    Node_Plots[pol_name][e_var]=np.zeros((time, trials))
            if self.fvp:
                Flow_Plots[pol_name]={}
                for f_var in self.fvp:
                    Flow_Plots[pol_name][f_var]=np.zeros((time, trials))
            Pol_online_times[pol_name]=np.zeros((time,trials))
            Rewards[pol_name]=np.zeros((time,trials))
            Cumulative_Rewards[pol_name]=np.zeros((time,trials))
            T0=timeit.default_timer()
            self.Pols[pol_name].offline_stage()
            T1=timeit.default_timer()
            Pol_offline_times[pol_name]=T1-T0
        
        for i in xrange(trials):
            #print i
            state={}
            for node in self.nodes:
                state[node]=self.nodes[node].get_initial_preds()
            state['T']=0
            States={}
            for pol in self.Pols:
                States[pol]=copy.copy(state)
            for j in xrange(time):
                #Same randomn outcomes for each node in a trial for a 'fair' comparison
                SharedRandomSeeds={}
                for node in self.nodes:
                    SharedRandomSeeds[node]=random.randint(1,1000000)
                for pol in self.Pols:
                    #print States[pol]['G']
                    #print self.nodes['G'].get_preds_value(States[pol])
                    ###COLLECT PLOTTING DATA
                    if self.evp:
                        for e_var in self.evp:
                            Node_Plots[pol][e_var][j,i]=self.nodes[e_var].get_preds_value(States[pol])
                    ###MAKE DECISION ACCORDING TO POLICY
                    T2=timeit.default_timer()
                    Dec=self.Pols[pol].decision(States[pol])
                    T3=timeit.default_timer()
                    ###TIME DECISION
                    Pol_online_times[pol][j,i]=T3-T2
                    ###COLLECT PLOTTING DATA ON FLOW VARIABLES
                    if self.fvp:
                        for f_var in self.fvp:
                            Flow_Plots[pol][f_var][j,i]=Dec[f_var]
                            
                    ###RECIEVE REWARD AND ACCUMULATE
                    Rewards[pol][j,i]=self.r_fnct(States[pol],Dec,self.nodes)
                    
                    ###Learn if desired/possible
                    if self.Pols[pol].get_learn_after_each_decision():
                        self.Pols[pol].learn_after_decision(States[pol],Dec,Rewards[pol][j,i])
                    
                    if j==0:
                        Cumulative_Rewards[pol][j,i]=Rewards[pol][j,i]
                    else:
                        Cumulative_Rewards[pol][j,i]=Cumulative_Rewards[pol][j-1,i]+Rewards[pol][j,i]
                    if j==(time-1):
                        print pol+':'+str(Cumulative_Rewards[pol][j,i])
                        f.write(pol+':'+str(Cumulative_Rewards[pol][j,i])+os.linesep)
                    
                    ###Make pre-post decision transition
                    PostDecSt={}
                    for node in self.nodes:
                        PostDecSt[node]=self.nodes[node].pre_to_post_ds_transition(States[pol],Dec)
                    States[pol]=PostDecSt
                    States[pol]['T']=j
                    ###Make post-pre decision transition
                    PreDecSt={}
                    for node in self.nodes:
                        self.nodes[node].set_random_seed(SharedRandomSeeds[node])
                        PreDecSt[node]=self.nodes[node].post_to_pre_ds_transition(States[pol])
                    States[pol]=PreDecSt
                    States[pol]['T']=j+1
                    
            for pol in self.Pols:
                if self.Pols[pol].get_learn_after_each_trial():
                    self.Pols[pol].learn_after_trial(Cumulative_Rewards[pol][time-1,i])
                
                    
        
        ###THE REST IS SIMPLY DATA OUTPUT AND PLOTTING COMMANDS###

        for pol in self.Pols:
            Results=[]
            OnlineTimePerSamplePath=[]
            for x in xrange(trials):
                Results.append(np.sum(Rewards[pol][:,x]))
                OnlineTimePerSamplePath.append(np.mean(Pol_online_times[pol][:,x]))
                f.write(str(Results[x])+'\n')
            AvgOnlineTimePerSamplePath=np.mean(OnlineTimePerSamplePath)
            AvgVal=np.mean(Results)
            StdDev=np.std(Results)
            Str='Policy: '+pol+'; Trials: '+str(trials)            
            Str=Str+'; Avg. Value: ' +str(AvgVal)
            Str=Str+'; Std. Dev.: ' +str(StdDev)
            Str=Str+'; Offline Time: '+str(Pol_offline_times[pol])+' sec'
            Str=Str+'; Avg. Online Time per Decision: '+str(AvgOnlineTimePerSamplePath)+' sec \n'
            print Str
            f.write(Str)
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
                    for pol in self.Pols:
                        cl=colors[cnum%(len(colors))]
                        cnum+=1
                        plt.plot(Node_Plots[pol][e_var],color=cl, label=pol)
                    plt.title(e_var)
                    plt.xlabel('t')
                    plt.ylabel('Value')
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
                            PltVec.append(np.mean(Node_Plots[pol][e_var][t,:]))
                        plt.plot(PltVec, color=cl, label=pol)
                    plt.title(e_var)
                    plt.xlabel('t')
                    plt.ylabel('Value')
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
                    plt.title(f_var)
                    plt.xlabel('t')
                    plt.ylabel('Flow')
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
                    plt.title(f_var)
                    plt.xlabel('t')
                    plt.ylabel('Flow')
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
        