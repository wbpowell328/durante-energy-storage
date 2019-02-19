import csv
import numpy as np
from Node import Node
from collections import Counter
import bisect
import os
import random

def detState(Val,binVector):
    Ind=bisect.bisect_right(binVector,Val)
    Ind-=1
    if (Ind==(len(binVector)-1)) or (Ind==(len(binVector))):
        Ind=len(binVector)-2
    if Ind<0:
        Ind=0
    return(Ind)
    

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

def round_to_array_ind(Vec,Val):
    Ind=bisect.bisect_right(Vec,Val)-1
    if Ind<0:
        Ret=0
    elif Ind==len(Vec)-1:
        Ret=Ind
    else:
        diff=abs(Val-Vec[Ind])-abs(Val-Vec[Ind+1])
        if diff>0:
            Ret=Ind+1
        else:
            Ret=Ind
    return Ret

        
def my_own_rand_choice(vec,probs):
    CumProbs=[]
    for i in range(len(probs)):
        if i==0:
            CumProbs.append(probs[i])
        else:
            CumProbs.append(probs[i]+CumProbs[i-1])
    R=random.random()
    choice_ind=0
    while R>CumProbs[choice_ind]:
        choice_ind+=1
    return vec[choice_ind]
        
def rle2(s):
    r = {'lengths':[],'inds':[],'values':[]}
    L = len(s)
    if L == 0:
        return ""
    if L == 1:
        return {'lengths':[1],'inds':[1],'values':[s]}
    cnt = 1
    i = 1
    while i < L:
        if s[i] == s[i - 1]: # check it is the same letter
            cnt += 1
        else:
            r['lengths'].append(cnt) # if not, store the previous data
            r['inds'].append(i-1)
            r['values'].append(s[i-1])
            cnt = 1
        i += 1
    r['lengths'].append(cnt) # if not, store the previous data
    r['inds'].append(i-1)
    r['values'].append(s[i-1])
    return r 

def try_parse(string):
    try:
        return float(string)
    except Exception:
        return np.nan

class exogenous_CSHSMM_node(Node):
    
    """
    Refer to the pdf in docs: CrossingTimeMarkovModels.pdf
    for an explanation of this stochastic model for prices
    """
    
    def __init__(self, name, filePath, headers,params):

        self.name=name
        
        Plen=len(params)
        self.d_int=1.0
        self.ts=1
        self.run_lens_states=3
        self.transMatDim=3
        self.max_wind=1.0
        self.opt_start=0
        
        if Plen>=1:
            self.d_int=float(params[0])
        if Plen>=2:
            self.ts=int(params[1])
        if Plen>=3:
            if not params[2]=='ObsMax':
                self.max_wind=float(params[2])
        if Plen>=4:
            self.run_len_states=params[3]
        if Plen>=5:
            self.transMatDim=params[4]
        
        transMatDim=self.transMatDim
        RLQs=self.run_len_states
        CSTransDim=2*RLQs
        self.CSTransDim=CSTransDim
        actuals=[]
        forecasts=[]
        
        first_step=1
        os.chdir('..')
        os.chdir('input')
        reader=csv.DictReader(open(filePath),fieldnames=headers)
        for row in reader:
            Act=row['Actual']
            Act=Act.replace(',','')
            Forc=row['Fcst']
            Forc=Forc.replace(',','')
            if first_step==1:
                first_step=0
                First_Point=try_parse(Act)
                if not np.isnan(First_Point):
                    actuals.append(First_Point)
                    forecasts.append(try_parse(Forc))
            else:
                actuals.append(try_parse(Act))           
                forecasts.append(try_parse(Forc))
        
        self.original_max=max(actuals)
        if params[2]=='ObsMax':
            self.max_wind=self.original_max
        actuals=np.array(actuals)/self.original_max
        forecasts=np.array(forecasts)/self.original_max
        actuals=actuals*self.max_wind
        forecasts=forecasts*self.max_wind
        
                
        ok = ~np.isnan(actuals)
        xp = ok.ravel().nonzero()[0]
        fp = actuals[~np.isnan(actuals)]
        x  = np.isnan(actuals).ravel().nonzero()[0]
        actuals[np.isnan(actuals)] = np.interp(x, xp, fp)        
                
        ok = ~np.isnan(forecasts)
        xp = ok.ravel().nonzero()[0]
        fp = forecasts[~np.isnan(forecasts)]
        x  = np.isnan(forecasts).ravel().nonzero()[0]
        forecasts[np.isnan(forecasts)] = np.interp(x, xp, fp)
        
        rawErrors = actuals - forecasts
        minErr=min(rawErrors)
        maxErr=max(rawErrors)
        
        self.all_states=np.arange(0.0, max(actuals)+self.d_int, self.d_int)
        
        error_states=np.insert(-self.all_states, np.arange(len(self.all_states)), self.all_states)
        error_states=list(set(error_states))
        error_states.sort()
        MinErrInd=round_to_array_ind(error_states,minErr)-1
        MaxErrInd=round_to_array_ind(error_states,maxErr)+1
        
        ErrorStates=[]
        for i in range(MinErrInd,MaxErrInd+1):
            ErrorStates.append(error_states[i])
        
        self.all_error_states=ErrorStates
        self.maximum=max(self.all_states)
#        asd
#        discretized_data=[]
#        for d in self.wind:
#            discretized_data.append(min(self.all_states, key=lambda x:abs(x-d)))
#        actuals=np.array(discretized_data)
#        
#        discretized_forecasts=[]
#        for d in self.wind_forecasts:
#            discretized_forecasts.append(min(self.all_states, key=lambda x:abs(x-d)))
#        forecasts=np.array(discretized_forecasts)
        
        self.actuals=actuals
        self.forecasts=forecasts
        self.previousCrossingState=0
        
        
        Len=len(actuals)
        
        OUState=np.zeros((Len,),dtype=np.int)
         
        for i in xrange(Len):
            if (rawErrors[i]>0):
                OUState[i]=OUState[i]+1
       
        RunLens=rle2(OUState)

        OverRunLens=[]
        UnderRunLens=[]
         
        for i in xrange(len(RunLens['lengths'])):
            if (RunLens['values'][i]==0):
                UnderRunLens.append(RunLens['lengths'][i])
            else:
                OverRunLens.append(RunLens['lengths'][i])
        
        Percentiles=[]
        for x in xrange(RLQs):
            Percentiles.append((float(100)*x)/RLQs)
        Percentiles.append(float(100))
        
        ORLensQuants=map(int, np.percentile(OverRunLens,Percentiles))
        URLensQuants=map(int, np.percentile(UnderRunLens,Percentiles))
        self.ORQuants=ORLensQuants
        self.URQuants=URLensQuants
        
        
        RunLensSplit=[[] for i in range(CSTransDim)]
        RunLensCurr=[[] for i in range(CSTransDim)]
        StartingPoints=[[] for i in range(CSTransDim)]
        ForwardCounts=np.zeros((CSTransDim,CSTransDim))
        ForwardProb=np.zeros((CSTransDim,CSTransDim))
        OverallTimeinState=[0 for i in range(CSTransDim)]
        t_0_Probabilities=np.zeros(CSTransDim)
         
        Errors=[[] for i in range(CSTransDim)]
         
        for i in xrange(len(RunLens['lengths'])-1):
            if(RunLens['values'][i]==0):
                CurrSt=detState(RunLens['lengths'][i],URLensQuants)
                NextSt=detState(RunLens['lengths'][i+1],ORLensQuants)+RLQs
            else:
                CurrSt=detState(RunLens['lengths'][i],ORLensQuants)+RLQs
                NextSt=detState(RunLens['lengths'][i+1],URLensQuants)
           
            OverallTimeinState[CurrSt]=OverallTimeinState[CurrSt]+RunLens['lengths'][i]
            RunLensSplit[CurrSt].append(RunLens['lengths'][i+1])
            RunLensCurr[CurrSt].append(RunLens['lengths'][i])
            ForwardCounts[CurrSt,NextSt]+=1
           
            if (i>0):
                Errors[CurrSt].extend(rawErrors[(RunLens['inds'][i-1]+1):RunLens['inds'][i]+1])
                StartingPoints[CurrSt].append(rawErrors[RunLens['inds'][i-1]+1])
            else:
                Errors[CurrSt].extend(rawErrors[0:RunLens['inds'][i]+1])
                #StartingPoints[CurrSt].append(rawErrors[0])
          
         
        for i in xrange(CSTransDim):
            S1=sum(ForwardCounts[i,:])
            for j in xrange(CSTransDim):
                if (S1>0):
                    ForwardProb[i,j]=float(ForwardCounts[i,j])/S1
        
        S3=sum(OverallTimeinState)
        for i in xrange(CSTransDim):
            t_0_Probabilities[i]=float(OverallTimeinState[i])/S3
        
        CSQuantiles=[]
        Errors_tplus1=[[[] for j in range(transMatDim)] for i in range(CSTransDim)]
        Errors_t=[[[] for j in range(transMatDim)] for i in range(CSTransDim)]
        Init_Errors_t=[[[] for j in range(transMatDim)] for i in range(CSTransDim)]
        Initial_Errors=[[] for i in range(CSTransDim)]

         
        ForwardCountsWithinState=[np.zeros((transMatDim,transMatDim)) for i in range(CSTransDim)]
        ForwardProbWithinState=[np.zeros((transMatDim,transMatDim)) for i in range(CSTransDim)]
        
        InitialCounts=[np.zeros((transMatDim)) for i in range(CSTransDim)]
        InitialProbs=[np.zeros((transMatDim)) for i in range(CSTransDim)]
        
        Wt_Percentiles=[]
        for x in xrange(transMatDim):
            Wt_Percentiles.append((float(100)*x)/transMatDim)
        Wt_Percentiles.append(float(100))        
        
        for i in xrange(CSTransDim):
            CSQuantiles.append(np.percentile(Errors[i],Wt_Percentiles))
           
            for j in xrange(len(StartingPoints[i])):
                SSt=detState(StartingPoints[i][j],CSQuantiles[i])
                Init_Errors_t[i][SSt].append(StartingPoints[i][j])
                InitialCounts[i][SSt]=InitialCounts[i][SSt]+1
                Initial_Errors[i].append(StartingPoints[i][j])

           
            ###REMEMBER THIS CHANGE FROM R SCRIPT           
            R=0
            for j in xrange(len(RunLensCurr[i])):
                if ((RunLensCurr[i][j])>1):
                    for k in xrange((RunLensCurr[i][j])-1):
                        t=R+k
                        St=detState(Errors[i][t],CSQuantiles[i])
                        NSt=detState(Errors[i][t+1],CSQuantiles[i])
                        Errors_tplus1[i][St].append(Errors[i][t+1])
                        Errors_t[i][St].append(Errors[i][t])
                        ForwardCountsWithinState[i][St,NSt]+=1
                    St=detState(Errors[i][t+1],CSQuantiles[i])
                    Errors_t[i][St].append(Errors[i][t+1])
                else:
                    St=detState(Errors[i][R],CSQuantiles[i])
                    Errors_t[i][St].append(Errors[i][R])
                R=R+(RunLensCurr[i][j])
           
            for j in xrange(transMatDim):
                S1=sum(InitialCounts[i])
                if (S1>0):
                    InitialProbs[i][j]=float(InitialCounts[i][j])/S1

                S3=sum(ForwardCountsWithinState[i][j,])
                for k in xrange(transMatDim):
                    if (S3>0):
                        ForwardProbWithinState[i][j,k]=float(ForwardCountsWithinState[i][j,k])/S3
        
        
        for i in xrange(CSTransDim):
            RunLensCurr[i].sort()
            Errors[i].sort()
            for j in xrange(transMatDim):
                Errors_t[i][j].sort()
                Errors_tplus1[i][j].sort()
        
        self.Time_in_State=OverallTimeinState
        self.t_0_Probabilities=t_0_Probabilities
        self.Forward_SC=ForwardProb       
        self.Run_Lengths=RunLensCurr
        self.Initial_Probs=InitialProbs
        self.Forwards_Probs_in_State=ForwardProbWithinState
        self.Quantiles=CSQuantiles    
        self.State_Errors=Errors_t
        self.Next_Errors=Errors_tplus1
        self.Larger_State_Errors=Errors
        self.initial_state_errors=Init_Errors_t
        self.initial_errors=Initial_Errors
        
        for i in range(len(self.forecasts)):
            Ind=bisect.bisect_right(self.all_states,self.forecasts[i])-1
            if Ind<0:
                self.forecasts[i]=self.all_states[0]
            elif Ind==len(self.all_states)-1:
                self.forecasts[i]=self.all_states[Ind]
            else:
                diff=abs(self.forecasts[i]-self.all_states[Ind])-abs(self.forecasts[i]-self.all_states[Ind+1])
                if diff>0:
                    self.forecasts[i]=self.all_states[Ind+1]
                else:
                    self.forecasts[i]=self.all_states[Ind]

           
#        TransProbs=[]
#        for i in xrange(CSTransDim):
#            Pstay=float(sum(self.Run_Lengths[i]))/(sum(self.Run_Lengths[i])+len(self.Run_Lengths[i]))
#            Pswitch=1-Pstay
#            for j in xrange(transMatDim):
#                IndividTransProbs=[]
#                for k in xrange(CSTransDim):
#                    if i==k:
#                        IntraStateTrans=[x*Pstay for x in self.Forwards_Probs_in_State[i][j,]]
#                        IndividTransProbs.extend(IntraStateTrans)
#                    else:
#                        InterStateTrans=[x*Pswitch*self.Forward_SC[i][k] for x in self.Initial_Probs[k]]
#                        IndividTransProbs.extend(InterStateTrans)
#                TransProbs.append(IndividTransProbs)
#        self.Transition_Probabilities=TransProbs
                    
        TransProbs=[]
        for i in xrange(CSTransDim):
            Pstay=float(sum(self.Run_Lengths[i]))/(sum(self.Run_Lengths[i])+len(self.Run_Lengths[i]))
            Pswitch=1-Pstay
            ThisRow=np.zeros(CSTransDim)
            for k in xrange(CSTransDim):
                if i==k:
                    ThisRow[k]=Pstay
                else:
                    ThisRow[k]=Pswitch*self.Forward_SC[i][k]
            TransProbs.append(ThisRow)
        self.Transition_Probabilities=TransProbs
        
        #Set up inds
        self.preds_vals=[]
        LenErrStates=len(self.all_error_states)
        for i in xrange(CSTransDim):
            for ii in xrange(transMatDim):
                for j in xrange(LenErrStates):
                    self.preds_vals.append(self.all_error_states[j])
        
        self.StateErrorProbs=[[{} for j in range(transMatDim)] for i in range(CSTransDim)]
        self.InitialErrorProbs=[[{} for j in range(transMatDim)] for i in range(CSTransDim)]
        for i in xrange(CSTransDim):
            for ii in xrange(transMatDim):
                E=[]
                for err in self.State_Errors[i][ii]:
                    E.append(round_to_array_ind(self.all_error_states,err))
                S=len(E)
                Counts=Counter(E)
                for key in Counts:
                    Prob=float(Counts[key])/S
                    self.StateErrorProbs[i][ii][key]=Prob
        
        for i in xrange(CSTransDim):
            E=[]
            for err in self.initial_errors[i]:
                E.append(round_to_array_ind(self.all_error_states,err))
            S=len(E)
            Counts=Counter(E)
            for key in Counts:
                Prob=float(Counts[key])/S
                St=detState(self.all_error_states[key],CSQuantiles[i])
                self.InitialErrorProbs[i][St][key]=Prob
        
        #Used for full transition function
        self.elapsed_time=0
        self.opt_time_t=-1
        self.consistent_p_stay=0.0
        
        os.chdir('..')
        os.chdir('input')        
        
#    def get_possible_states(self,t):
#        """
#        Return list of possible states of exogenous process at a certain time
#        """
#        Possible_States=[]
#        t=self.opt_start+t/self.ts
#        fcst=self.forecasts[t]
#        for SC in range(self.CSTransDim):
#            for Wt in range(self.transMatDim):
#                S=[]
#                for d in self.State_Errors[SC][Wt]:
#                    Val=fcst+d
#                    Ind=bisect.bisect_right(self.all_states,Val)-1
#                    if Ind<0:
#                        S.append(self.all_states[0])
#                    elif Ind==len(self.all_states)-1:
#                        S.append(self.all_states[Ind])
#                    else:
#                        diff=abs(Val-self.all_states[Ind])-abs(Val-self.all_states[Ind+1])
#                        if diff>0:
#                            S.append(self.all_states[Ind+1])
#                        else:
#                            S.append(self.all_states[Ind])
#                S2=set(S)
#                for Val in S2:
#                    D={'SC':SC,'Wt':Wt,self.name:Val}
#                    Possible_States.append(collections.OrderedDict(sorted(D.items())))
#        return Possible_States
#        
#    def get_possible_states_sampled(self,t,sample_prob):
#        """
#        Return a sampled list of possible pre-decision 
#        states of exogenous process at a certain time
#        """
#        Possible_States=[]
#        t=self.opt_start+t/self.ts
#        fcst=self.forecasts[t]
#        for SC in range(self.CSTransDim):
#            for Wt in range(self.transMatDim):
#                S=[]
#                for d in self.State_Errors[SC][Wt]:
#                    Val=fcst+d
#                    Ind=bisect.bisect_right(self.all_states,Val)-1
#                    if Ind<0:
#                        S.append(self.all_states[0])
#                    elif Ind==len(self.all_states)-1:
#                        S.append(self.all_states[Ind])
#                    else:
#                        diff=abs(Val-self.all_states[Ind])-abs(Val-self.all_states[Ind+1])
#                        if diff>0:
#                            S.append(self.all_states[Ind+1])
#                        else:
#                            S.append(self.all_states[Ind])
#                S2=list(set(S))
#                Size=int(np.ceil(len(S2)*sample_prob))
#                S3=np.random.choice(S2,size=Size,replace=False)
#                for Val in S3:
#                    D={'SC':SC,'Wt':Wt,self.name:Val}
#                    Possible_States.append(collections.OrderedDict(sorted(D.items())))
#        return Possible_States
#        
#    def get_possible_states_sampled_2(self,t,sample_prob,SC,WT):
#        """
#        Return a sampled list of possible pre-decision 
#        states of exogenous process at a certain time
#        """
#        Possible_States=[]
#        t=self.opt_start+t/self.ts
#        fcst=self.forecasts[t]
#        S=[]
#        for d in self.State_Errors[SC][WT]:
#            Val=fcst+d
#            Ind=bisect.bisect_right(self.all_states,Val)-1
#            if Ind<0:
#                S.append(self.all_states[0])
#            elif Ind==len(self.all_states)-1:
#                S.append(self.all_states[Ind])
#            else:
#                diff=abs(Val-self.all_states[Ind])-abs(Val-self.all_states[Ind+1])
#                if diff>0:
#                    S.append(self.all_states[Ind+1])
#                else:
#                    S.append(self.all_states[Ind])
#        #JOE YOU ARE HEREEEEE
#        #max
#        S2=list(set(S))
#        Size=int(np.ceil(len(S2)*sample_prob))
#        S3=np.random.choice(S2,size=Size,replace=False)
#        for Val in S3:
#            D={'SC':SC,'Wt':WT,self.name:Val}
#            Possible_States.append(collections.OrderedDict(sorted(D.items())))
#        return Possible_States
        
#    def get_possible_states_2(self,t,SC,WT):
#        """
#        Return a sampled list of possible pre-decision 
#        states of exogenous process at a certain time
#        """
#        Possible_States=[]
#        t=self.opt_start+t/self.ts
#        fcst=self.forecasts[t]
#        S=[]
#        for d in self.State_Errors[SC][WT]:
#            Val=fcst+d
#            Ind=bisect.bisect_right(self.all_states,Val)-1
#            if Ind<0:
#                S.append(self.all_states[0])
#            elif Ind==len(self.all_states)-1:
#                S.append(self.all_states[Ind])
#            else:
#                diff=abs(Val-self.all_states[Ind])-abs(Val-self.all_states[Ind+1])
#                if diff>0:
#                    S.append(self.all_states[Ind+1])
#                else:
#                    S.append(self.all_states[Ind])
#        #JOE YOU ARE HEREEEEE
#        #max
#        S2=list(set(S))
#        return S2
#        
#    def get_possible_post_decision_states(self,t):
#        """
#        Return list of possible post decision states of exogenous process at a certain time
#        """
#        Possible_States=[]
#        for SC in range(self.CSTransDim):
#            for Wt in range(self.transMatDim):
#                D={'SC':SC,'Wt':Wt}
#                Possible_States.append(collections.OrderedDict(sorted(D.items())))
#        return Possible_States
#    
#    def post_decision_form(self,state,T):
#        """
#        After a decision is made, return only the information in the process
#        necessary to transition to next pre decision state
#        """
#        del state[self.name]
#        return state
     
    def get_name(self):
        """
        Return name of exogenous process
        """
        return self.name
    
    def get_time_step(self):
        """
        Return time step of process in integer multiple of quickest changing process
        """
        return self.ts
    
    def get_discretization_interval(self):
        """
        Return the interval which this process is discretized by
        """
        return self.d_int    
     
    def get_forecast(self, t):
        """
        Return forecast of exogenous process at time t
        """
        return self.forecasts[self.opt_start+t/self.ts]
    
    def set_new_forecast(self, new_forecast):
        self.forecasts=new_forecast*self.max_wind/self.original_max
     
#    def get_forward_probabilities(self,state,T):
#        """
#        Return a 2 element list of the forward states (list comprising first
#        element) and their probabilities (list comprising second element)
#        
#        Used for possibly simplified Markov chain representation of process 
#        """
#        Forward_States=[]
#        Transition_Probs=[]
#        st=state[self.name]
#        FS=self.transMatDim*st['SC']+st['Wt']
#        t=(T/self.ts)+1+self.opt_start
#        for i in xrange(len(self.Transition_Probabilities[FS])):
#            p=self.Transition_Probabilities[FS][i]
#            if p>0.0:                
#                fcst=self.forecasts[t]
#                S_C=i/self.transMatDim
#                W_t=i%self.transMatDim
#                E=[]
#                for err in self.State_Errors[S_C][W_t]:
#                    Val=fcst+err
#                    Ind=bisect.bisect_right(self.all_states,Val)-1
#                    if Ind<0:
#                        E.append(self.all_states[0])
#                    elif Ind==len(self.all_states)-1:
#                        E.append(self.all_states[Ind])
#                    else:
#                        diff=abs(Val-self.all_states[Ind])-abs(Val-self.all_states[Ind+1])
#                        if diff>0:
#                            E.append(self.all_states[Ind+1])
#                        else:
#                            E.append(self.all_states[Ind])
#                S=len(E)
#                Counts=Counter(E)
#                for key in Counts:
#                    Prob=p*float(Counts[key])/S
#                    D={'SC':S_C,'Wt':W_t,self.name:key}
#                    Forward_States.append(collections.OrderedDict(sorted(D.items())))
#                    Transition_Probs.append(Prob)
#        return [Forward_States, Transition_Probs]
        
#    def get_forward_probabilities_sampled(self,st,T,sample_prob):
#        """
#        Return a 2 element list of the forward states (list comprising first
#        element) and their probabilities (list comprising second element)
#        
#        Used for possibly simplified Markov chain representation of process 
#        """
#        Forward_States=[]
#        Transition_Probs=[]
#        FS=self.transMatDim*st['SC']+st['Wt']
#        t=(T/self.ts)+1+self.opt_start
#        for i in xrange(len(self.Transition_Probabilities[FS])):
#            p=self.Transition_Probabilities[FS][i]
#            if p>0.0:                
#                fcst=self.forecasts[t]
#                S_C=i/self.transMatDim
#                W_t=i%self.transMatDim
#                E=[]
#                for err in self.State_Errors[S_C][W_t]:
#                    Val=fcst+err
#                    Ind=bisect.bisect_right(self.all_states,Val)-1
#                    if Ind<0:
#                        E.append(self.all_states[0])
#                    elif Ind==len(self.all_states)-1:
#                        E.append(self.all_states[Ind])
#                    else:
#                        diff=abs(Val-self.all_states[Ind])-abs(Val-self.all_states[Ind+1])
#                        if diff>0:
#                            E.append(self.all_states[Ind+1])
#                        else:
#                            E.append(self.all_states[Ind])
#                S2=list(set(E))
#                Size=int(np.ceil(len(S2)*sample_prob))
#                S3=np.random.choice(S2,size=Size,replace=False)    
#                S4 = [item for item in E if item in S3]
#                S=len(S4)
#                Counts=Counter(S4)
#                for key in Counts:
#                    Prob=p*float(Counts[key])/S
#                    D={'SC':S_C,'Wt':W_t,self.name:key}
#                    Forward_States.append(collections.OrderedDict(sorted(D.items())))
#                    Transition_Probs.append(Prob)
#        return [Forward_States, Transition_Probs]
#        
#    def get_forward_probabilities_sampled_3(self,sc,wt,T, sample_prob):
#        """
#        Return a 2 element list of the forward states (list comprising first
#        element) and their probabilities (list comprising second element)
#        
#        Used for possibly simplified Markov chain representation of process 
#        """
#        Forward_States=[]
#        Transition_Probs=[]
#        FS=self.transMatDim*sc+wt
#        t=(T/self.ts)+1+self.opt_start
#        for i in xrange(len(self.Transition_Probabilities[FS])):
#            p=self.Transition_Probabilities[FS][i]
#            if p>0.0:                
#                fcst=self.forecasts[t]
#                S_C=i/self.transMatDim
#                W_t=i%self.transMatDim
#                E=[]
#                for err in self.State_Errors[S_C][W_t]:
#                    Val=fcst+err
#                    Ind=bisect.bisect_right(self.all_states,Val)-1
#                    if Ind<0:
#                        E.append(self.all_states[0])
#                    elif Ind==len(self.all_states)-1:
#                        E.append(self.all_states[Ind])
#                    else:
#                        diff=abs(Val-self.all_states[Ind])-abs(Val-self.all_states[Ind+1])
#                        if diff>0:
#                            E.append(self.all_states[Ind+1])
#                        else:
#                            E.append(self.all_states[Ind])
#                S2=list(set(E))  
#                S4 = [item for item in E if item in S2]
#                S=len(S4)
#                Counts=Counter(S4)
#                for key in Counts:
#                    Prob=p*float(Counts[key])/S
#                    D=(S_C,W_t,key)
#                    Forward_States.append(D)
#                    Transition_Probs.append(Prob)
#        Choice=np.random.choice(range(len(Transition_Probs)),size=int(np.ceil(len(Transition_Probs)*sample_prob)),replace=False)
#        Choice2=np.zeros(len(Transition_Probs))
#        for c in Choice:
#            Choice2[c]=1
#        return [Forward_States, Transition_Probs,Choice2]
#        
#    def get_forward_probabilities_sampled_2(self,sc,wt,T,sample_prob):
#        """
#        Return a 2 element list of the forward states (list comprising first
#        element) and their probabilities (list comprising second element)
#        
#        Used for possibly simplified Markov chain representation of process 
#        """
#        Forward_States=[]
#        Transition_Probs=[]
#        FS=self.transMatDim*sc+wt
#        t=(T/self.ts)+1+self.opt_start
#        for i in xrange(len(self.Transition_Probabilities[FS])):
#            p=self.Transition_Probabilities[FS][i]
#            if p>0.0:                
#                fcst=self.forecasts[t]
#                S_C=i/self.transMatDim
#                W_t=i%self.transMatDim
#                E=[]
#                for err in self.State_Errors[S_C][W_t]:
#                    Val=fcst+err
#                    Ind=bisect.bisect_right(self.all_states,Val)-1
#                    if Ind<0:
#                        E.append(self.all_states[0])
#                    elif Ind==len(self.all_states)-1:
#                        E.append(self.all_states[Ind])
#                    else:
#                        diff=abs(Val-self.all_states[Ind])-abs(Val-self.all_states[Ind+1])
#                        if diff>0:
#                            E.append(self.all_states[Ind+1])
#                        else:
#                            E.append(self.all_states[Ind])
#                S2=list(set(E))
#                Size=int(np.ceil(len(S2)*sample_prob))
#                S3=np.random.choice(S2,size=Size,replace=False)    
#                S4 = [item for item in E if item in S3]
#                S=len(S4)
#                Counts=Counter(S4)
#                for key in Counts:
#                    Prob=p*float(Counts[key])/S
#                    D=(S_C,W_t,key)
#                    Forward_States.append(D)
#                    Transition_Probs.append(Prob)
#        return [Forward_States, Transition_Probs]

#    def forward_transition(self, state):
#        """
#        Make one forward transition (full transition function) and return 
#        a dictionary defining the next state in the stochastic process.
#        
#        Actual transition function used here
#        """
#        CRL=state[self.name]['RL']
#        Time_in_Run=state[self.name]['TiR']
#        SC=state[self.name]['SC']
#        t=state['T']/self.ts+1+self.opt_start
#        New_St={}
#        if (Time_in_Run==CRL):
#            New_St['TiR']=0
#            SC2=np.random.choice(range(self.CSTransDim),p=self.Forward_SC[SC,:])
#            New_St['SC']=SC2
#            New_St['RL']=(np.random.choice(self.Run_Lengths[SC2]))-1
#            Wt=np.random.choice(range(self.transMatDim),p=self.Initial_Probs[SC2])
#            while not self.State_Errors[SC2][Wt]:
#                Wt=np.random.choice(range(self.transMatDim),p=self.Initial_Probs[SC2])
#            New_St['Wt']=Wt
#            E=np.random.choice(self.State_Errors[SC2][Wt])
#            sam=E+self.forecasts[t]
#            Ind=bisect.bisect_right(self.all_states,sam)-1
#            if Ind<0:
#                New_St[self.get_name()]=self.all_states[0]
#            elif Ind==len(self.all_states)-1:
#                New_St[self.get_name()]=self.all_states[Ind]
#            else:
#                diff=abs(sam-self.all_states[Ind])-abs(sam-self.all_states[Ind+1])
#                if diff>0:
#                    New_St[self.get_name()]=self.all_states[Ind+1]
#                else:
#                    New_St[self.get_name()]=self.all_states[Ind] 
#        else:
#            New_St['TiR']=Time_in_Run+1
#            New_St['SC']=SC
#            New_St['RL']=CRL
#            Wt=state[self.name]['Wt']
#            if not self.Next_Errors[SC][Wt]:
#                E=np.random.choice(self.Larger_State_Errors[SC])
#            else:
#                E=np.random.choice(self.Next_Errors[SC][Wt])
#            New_St['Wt']=detState(E,self.Quantiles[SC])
#            sam=E+self.forecasts[t]
#            Ind=bisect.bisect_right(self.all_states,sam)-1
#            if Ind<0:
#                New_St[self.get_name()]=self.all_states[0]
#            elif Ind==len(self.all_states)-1:
#                New_St[self.get_name()]=self.all_states[Ind]
#            else:
#                diff=abs(sam-self.all_states[Ind])-abs(sam-self.all_states[Ind+1])
#                if diff>0:
#                    New_St[self.get_name()]=self.all_states[Ind+1]
#                else:
#                    New_St[self.get_name()]=self.all_states[Ind]
#
#
#        return collections.OrderedDict(sorted(New_St.items()))
        
#    def forward_transition_2(self, state):
#        """
#        Make one forward transition (full transition function) and return 
#        a dictionary defining the next state in the stochastic process.
#        
#        Actual transition function used here
#        """
#        CRL=state[self.name]['RL']
#        Time_in_Run=state[self.name]['TiR']
#        SC=state[self.name]['SC']
#        t=state['T']/self.ts+1+self.opt_start
#        New_St={}
#        if (Time_in_Run==CRL):
#            New_St['TiR']=0
#            SC2=np.random.choice(range(self.CSTransDim),p=self.Forward_SC[SC,:])
#            New_St['SC']=SC2
#            New_St['RL']=(np.random.choice(self.Run_Lengths[SC2]))-1
#            Wt=np.random.choice(range(self.transMatDim),p=self.Initial_Probs[SC2])
#            while not self.State_Errors[SC2][Wt]:
#                Wt=np.random.choice(range(self.transMatDim),p=self.Initial_Probs[SC2])
#            New_St['Wt']=Wt
#            E=np.random.choice(self.State_Errors[SC2][Wt])
#            sam=E+self.forecasts[t]
#            Ind=bisect.bisect_right(self.all_states,sam)-1
#            if Ind<0:
#                New_St[self.get_name()]=self.all_states[0]
#            elif Ind==len(self.all_states)-1:
#                New_St[self.get_name()]=self.all_states[Ind]
#            else:
#                diff=abs(sam-self.all_states[Ind])-abs(sam-self.all_states[Ind+1])
#                if diff>0:
#                    New_St[self.get_name()]=self.all_states[Ind+1]
#                else:
#                    New_St[self.get_name()]=self.all_states[Ind] 
#        else:
#            New_St['TiR']=Time_in_Run+1
#            New_St['SC']=SC
#            New_St['RL']=CRL
#            Wt=state[self.name]['Wt']
#            if not self.Next_Errors[SC][Wt]:
#                E=np.random.choice(self.Larger_State_Errors[SC])
#            else:
#                E=np.random.choice(self.Next_Errors[SC][Wt])
#            New_St['Wt']=detState(E,self.Quantiles[SC])
#            E=np.random.choice(self.State_Errors[SC][New_St['Wt']])
#            sam=E+self.forecasts[t]
#            Ind=bisect.bisect_right(self.all_states,sam)-1
#            if Ind<0:
#                New_St[self.get_name()]=self.all_states[0]
#            elif Ind==len(self.all_states)-1:
#                New_St[self.get_name()]=self.all_states[Ind]
#            else:
#                diff=abs(sam-self.all_states[Ind])-abs(sam-self.all_states[Ind+1])
#                if diff>0:
#                    New_St[self.get_name()]=self.all_states[Ind+1]
#                else:
#                    New_St[self.get_name()]=self.all_states[Ind]
#
#
#        return collections.OrderedDict(sorted(New_St.items()))
    
#    def get_initial_state(self, t):
#        """
#        Return an initial state at time t, full state variable used here, used
#        with full transition function
#        """
#        self.elapsed_time=0
#        t=t/self.ts+self.opt_start
#        SC=np.random.choice(range(self.CSTransDim))
#        Wt=np.random.choice(range(self.transMatDim)) 
#        while not self.State_Errors[SC][Wt]:
#            SC=np.random.choice(range(self.CSTransDim))
#            Wt=np.random.choice(range(self.transMatDim)) 
#        RL=np.random.choice(self.Run_Lengths[SC])
#        E=np.random.choice(self.State_Errors[SC][Wt])+self.forecasts[t]
#        Ind=bisect.bisect_right(self.all_states,E)-1
#        if Ind<0:
#            E2=self.all_states[0]
#        elif Ind==len(self.all_states)-1:
#            E2=self.all_states[Ind]
#        else:
#            diff=abs(E-self.all_states[Ind])-abs(E-self.all_states[Ind+1])
#            if diff>0:
#                E2=self.all_states[Ind+1]
#            else:
#                E2=self.all_states[Ind]
#        D={'SC':SC,'Wt':Wt,'RL':RL,'TiR':0,self.name:E2}
#        return collections.OrderedDict(sorted(D.items()))
        
    def detInfoStateIndex(self, crossingState, errorQuant):
        index=(crossingState*(self.transMatDim)+errorQuant)
        return index
        
    def detInfoStateIndex2(self, crossingState, rawError):
        errorQuant=detState(rawError,self.Quantiles[crossingState])
        index=self.detInfoStateIndex(crossingState, errorQuant)  
        return index
        
    def detCrossingState(self, infoStateIndex):
        crossingState=infoStateIndex/self.transMatDim
        return crossingState
        
    def detErrorState(self,infoStateIndex):
        errorQuant=infoStateIndex%self.transMatDim
        return errorQuant
        
    def probTgreaterOrEqualt_primeGivenInfoState(self, t_prime, crossingState, errorQuant):
        ind=bisect.bisect_right(self.Run_Lengths[crossingState],(t_prime))
        prob=1-(float(ind)/len(self.Run_Lengths[crossingState]))
        return prob
        
    def det_CS_given_sign_switch(self, t_prime, prev_sign):
        if prev_sign==1:
            Bin=detState(t_prime,self.ORQuants)
            CS=self.run_len_states+Bin
        else:
            Bin=detState(t_prime,self.URQuants)
            CS=Bin
        return CS

    def probTgreaterOrEqualt_tildeGivenCurrentInfoStateIndex(self, t_prime, currentInfoStateIndex):
        crossingState=self.detCrossingState(currentInfoStateIndex)
        errorQuant=self.detErrorState(currentInfoStateIndex)
        prob=self.probTgreaterOrEqualt_primeGivenInfoState(t_prime, crossingState, errorQuant)
        return prob
        
    def transition_prob_tildeGivenCurrentInfoStateIndex(self, t_prime, currentInfoStateIndex):
        crossingState=self.detCrossingState(currentInfoStateIndex)
        errorQuant=self.detErrorState(currentInfoStateIndex)
        prob=self.probTgreaterOrEqualt_primeGivenInfoState(t_prime, crossingState, errorQuant)
        return prob
    
    def probObservableErrorGivenInfoState(self, obsRelError, crossingState, errorQuant, t):
        indexInterval=self.getObservableErrorIndexInterval(t)
        indexE=self.determineErrorIndex(obsRelError)
        if (indexE<=indexInterval[0]):
            prob=self.cdfErrorGivenCrossingStateAndErrorQuants[crossingState][errorQuant][indexInterval[0]]
        elif (indexE>=indexInterval[1]):
            prob=1-self.cdfErrorGivenCrossingStateAndErrorQuants[crossingState][errorQuant][indexE-1]
        else:
            prob=self.cdfErrorGivenCrossingStateAndErrorQuants[crossingState][errorQuant][indexE]-self.cdfErrorGivenCrossingStateAndErrorQuants[crossingState][errorQuant][indexE-1]
        return prob
    
    	def getObservableErrorIndexInterval(self, T):
         interval = []
         t=T/self.ts+self.opt_start
         minError=self.all_error_states[0]
         maxError=self.all_error_states[len(self.all_error_states)-1]
         minObservableError=max(minError, -self.forecasts[t])
         maxObservableError=min(maxError,self.maximum-(self.forecasts[t]))
         interval.append(self.determineErrorIndex(minObservableError))
         interval.append(self.determineErrorIndex(maxObservableError))
         return interval
    
    def determineErrorIndex(self, rawError):
        Ind=bisect.bisect_right(self.all_error_states,rawError)-1
        if Ind<0:
            return 0
        elif Ind==len(self.all_states)-1:
            return Ind
        else:
            diff=abs(rawError-self.all_states[Ind])-abs(rawError-self.all_states[Ind+1])
            if diff>0:
                return Ind+1
            else:
                return Ind
                
    def return_prob_of_wind_power_in_each_state(self,Error,WP):
        JS=self.CSTransDim*self.transMatDim
        probs=np.zeros(JS)
        rawError=Error
        errorInterval=float(self.d_int)/2
        minError=rawError-errorInterval
        maxError=rawError+errorInterval
        for js in xrange(JS):
            cs=js/self.transMatDim
            wt=js%self.transMatDim
            if WP==0.0:
                ind2=bisect.bisect_right(self.State_Errors[cs][wt],maxError)
                prob=float(ind2)/len(self.State_Errors[cs][wt])
            elif WP==self.maximum:
                ind1=bisect.bisect_right(self.State_Errors[cs][wt],minError)
                prob=float(len(self.State_Errors[cs][wt])-ind1)/len(self.State_Errors[cs][wt])
            else:            
                ind1=bisect.bisect_right(self.State_Errors[cs][wt],minError)
                ind2=bisect.bisect_right(self.State_Errors[cs][wt],maxError)
                prob=float(ind2-ind1)/len(self.State_Errors[cs][wt])
            probs[js]=prob
        return probs
    
    def return_forward_prob_within_state(self,js,tostate):
        cs=js/self.transMatDim
        wt=js%self.transMatDim
        cs2=tostate/self.transMatDim
        wt2=tostate%self.transMatDim
        if not cs==cs2:
            return 0.0
        else:
            return self.Forwards_Probs_in_State[cs][wt][wt2]
    
    def return_prob_of_wind_power(self,WP,T,js):
        t=T/self.ts+self.opt_start
        rawError=WP-self.forecasts[t]
        errorInterval=float(self.d_int)/2
        minError=rawError-errorInterval
        maxError=rawError+errorInterval
        cs=js/self.transMatDim
        wt=js%self.transMatDim
        if WP==0.0:
            ind2=bisect.bisect_right(self.State_Errors[cs][wt],maxError)
            prob=float(ind2)/len(self.State_Errors[cs][wt])
        elif WP==self.maximum:
            ind1=bisect.bisect_right(self.State_Errors[cs][wt],minError)
            prob=float(len(self.State_Errors[cs][wt])-ind1)/len(self.State_Errors[cs][wt])
        else:            
            ind1=bisect.bisect_right(self.State_Errors[cs][wt],minError)
            ind2=bisect.bisect_right(self.State_Errors[cs][wt],maxError)
            prob=float(ind2-ind1)/len(self.State_Errors[cs][wt])
        return prob
        
    def get_error_state(self,Error,CS):
        return detState(Error,self.Quantiles[CS])

    
    def updateObservableState(self, rawError):
        currObservableRelError=rawError
        #First we need to check if the error has changed signs - meaning it changed crossing states
        if ((self.previousCrossingState<self.run_lens_states and currObservableRelError>0) or
        (self.previousCrossingState>=self.run_lens_states and currObservableRelError<=0)):
            #Determine in which quantile the elapsedTime is
            if (currObservableRelError<=0):
                self.previousCrossingState=self.determineBin(self.elapsedTime,self.URQuants)
            else:
                timeBin=self.determineBin(self.elapsedTime,self.ORQuants)
                self.previousCrossingState=self.run_lens_states+timeBin
            self.elapsedTime=1
        else:
            self.elapsedTime+=1
            
       
    def get_postds_value(self, postds):
        """
        Return value of node (the state which the index represents)
        """
        return postds[self.name]
    
    def get_preds_value(self, preds):
        """
        Return value of node (the state which the index represents)
        """
        T=preds['T']
        Forc=self.get_forecast(T)
        Err=self.preds_vals[preds[self.name]]
        return round_to_array(self.all_states,Forc+Err)
    
    def get_postds(self,postds):
        """
        Return post decision state value of node (index, an int)
        """
        return postds[self.name]
    
    def get_preds(self,preds):
        """
        Return pre decision state value of node (index, an int)
        """
        return preds[self.name]
        
    def get_max(self):
        """
        Return maximum value of node (float)
        """
        return self.maximum
    
    def get_min(self):
        """
        Return minimum value of node (float)
        """
        return 0.0
    
    def get_possible_postds(self, t):
        """
        Return list of possible post decision states of node at time t
        """
        return range(self.CSTransDim*self.transMatDim)
        
    def get_possible_preds(self, t):
        """
        Return list of possible pre decision states of node at time t
        """
        return range(len(self.preds_vals))
    
    def get_postds_to_preds_probabilities(self, postds):
        """
        Return a 2 element list of the forward states (list comprising first
        element) and their probabilities (list comprising second element)
        
        Possible simplified Markov chain representation of process 
        """
        
        Forward_States=[]
        Transition_Probs=[]
        SCHere=postds[self.name]/self.transMatDim
        WtHere=postds[self.name]%self.transMatDim
        for i in xrange(self.CSTransDim*self.transMatDim):
            S_C=i/self.transMatDim
            W_t=i%self.transMatDim
            p=self.Transition_Probabilities[SCHere][S_C]
            if p>0.0:
                if S_C==SCHere:
                    p_trans_in_state=self.Forwards_Probs_in_State[S_C][WtHere,W_t]
                    if p_trans_in_state>0:
                        for key in self.StateErrorProbs[S_C][W_t]:
                            Forward_States.append(i*len(self.all_error_states)+key)
                            Transition_Probs.append(p*p_trans_in_state*self.StateErrorProbs[S_C][W_t][key])
                else:
                    if bool(self.InitialErrorProbs[S_C][W_t]):
                        for key in self.InitialErrorProbs[S_C][W_t]:
                            Forward_States.append(i*len(self.all_error_states)+key)
                            Transition_Probs.append(p*self.InitialErrorProbs[S_C][W_t][key])
        return [Forward_States, Transition_Probs]
    
    def pre_to_post_ds_transition(self,preds,dec):
        """
        After a decision is made, return post decision state (return int index)
        """
        return preds[self.name]/len(self.all_error_states)
    
    def post_to_pre_ds_transition(self, postds):
        """
        Make transition from post to pre decision state (return int index)
        """
        
        pds=postds[self.name]
        p_stay=self.transition_prob_tildeGivenCurrentInfoStateIndex(self.elapsed_time, pds)
        SC=pds/self.transMatDim
        WT=pds%self.transMatDim
        R=random.random()
        #In case there are multiple policies, need this condition so that
        #Transitions are done consistently accross polcies, slight workaround
        PrevT=self.opt_time_t
        self.opt_time_t=postds['T']
        ChangeElapsedTime=False
        if not PrevT==self.opt_time_t:
            ChangeElapsedTime=True
            self.consistent_p_stay=p_stay
        if R<self.consistent_p_stay:
            #In case there are multiple policies, need this condition so that
            #Transitions are done correctly
            if ChangeElapsedTime:
                self.elapsed_time+=1
            SC2=SC
            Wt=my_own_rand_choice(range(self.transMatDim),self.Forwards_Probs_in_State[SC2][WT,:])
            E=random.choice(self.State_Errors[SC2][Wt])
            Eind=round_to_array_ind(self.all_error_states,E)
        else:
            self.elapsed_time=0
            SC2=my_own_rand_choice(range(self.CSTransDim),self.Forward_SC[SC,:])
            E=random.choice(self.initial_errors[SC2])
            Eind=round_to_array_ind(self.all_error_states,E)
            Wt=detState(self.all_error_states[Eind],self.Quantiles[SC2])
        NewInfoSt=SC2*self.transMatDim+Wt
        return NewInfoSt*len(self.all_error_states)+Eind
    
    def get_initial_preds(self):
        """
        Return an initial pre decision state at time t
        """
        self.elapsed_time=0
        
        return random.choice(range(len(self.preds_vals)))
    
    def set_random_seed(self, rint):
        """
        set specific random seed of random number generator
        """
        random.seed(rint)
