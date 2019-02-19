import statsmodels.api as sm
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

class exogenous_CSHSMM_Prices_node(Node):
    
    """
    Refer to the pdf in docs: CrossingTimeMarkovModels.pdf
    for an explanation of this stochastic model for prices
    """
    
    def __init__(self, name, filePath, headers, params):
        
        self.name=name
        self.max_price=1.0
        self.seas_bins=[0.0, 0.75, 1.0]
        self.trend_bins=[0,.3,.8,1]
        self.transMatDim=4
        self.minutes_per_time_step=5
        self.d_int=1.0
        self.ts=1
        RLQs=1
        
        Plen=len(params)
        self.opt_start=0
        
        if Plen>=1:
            self.d_int=float(params[0])
        if Plen>=2:
            self.ts=int(params[1])
        if Plen>=3:
            if not params[2]=='ObsMax':
                self.max_price=float(params[2])
        if Plen>=4:
            self.seas_bins=params[3]
        if Plen>=5:
            self.trend_bins=params[4]
        if Plen>=6:
            self.transMatDim=params[5]
        if Plen>=7:
            self.minutes_per_time_step=params[6]
        
        self.num_temp_seas_bins=len(self.seas_bins)-1
        self.num_temp_trend_bins=len(self.trend_bins)-1
        self.numTransMats=2*self.num_temp_trend_bins*self.num_temp_seas_bins
        
        CSTransDim=2*RLQs
        self.CSTransDim=CSTransDim
        
        os.chdir('..')
        os.chdir('input')
        
        actuals=[]
        Temp=[]
        
        reader=csv.DictReader(open(filePath), fieldnames=headers)
        first_step=1
        for row in reader:
            Act=row['Price']
            Act=Act.replace(',','')
            Tem=row['Temp']
            Tem=Tem.replace(',','')
            if first_step==1:
                first_step=0
                First_Point=try_parse(Act)
                if not np.isnan(First_Point):
                    actuals.append(First_Point)
                    Temp.append(try_parse(Tem))
            else:
                A=try_parse(Act)
                actuals.append(A)
                Temp.append(try_parse(Tem))
                
        self.orig_max=max(actuals)
        if params[2]=='ObsMax':
            self.max_price=self.orig_max
        actuals=np.array(actuals)*self.max_price/self.orig_max
        
        Temp=np.array(Temp)
       
        #Deal with missing data        
        ok = ~np.isnan(actuals)
        xp = ok.ravel().nonzero()[0]
        fp = actuals[~np.isnan(actuals)]
        x  = np.isnan(actuals).ravel().nonzero()[0]
        actuals[np.isnan(actuals)] = np.interp(x, xp, fp)        

        self.all_states=np.arange(-1.25*abs(min(actuals)),1.25*max(actuals)+self.d_int,self.d_int)
        lmp=actuals
                
        ok = ~np.isnan(Temp)
        xp = ok.ravel().nonzero()[0]
        fp = Temp[~np.isnan(Temp)]
        x  = np.isnan(Temp).ravel().nonzero()[0]
        Temp[np.isnan(Temp)] = np.interp(x, xp, fp) 
        
        Freq=24*60/self.minutes_per_time_step  
        temp_decomp = sm.tsa.seasonal_decompose(Temp, freq=Freq)
        temp_seas=temp_decomp.seasonal
        temp_trend=temp_decomp.trend
        
        lmp_decomp = sm.tsa.seasonal_decompose(actuals, freq=Freq)
        lmp_seas=lmp_decomp.seasonal
        actuals=lmp-lmp_seas

        k=0
        while np.isnan(temp_trend[k]):
            k+=1
        j=len(temp_trend)-1
        while np.isnan(temp_trend[j]):
            j-=1
        j+=1
        
        for i in xrange(k):
            temp_trend[i]=np.mean(Temp[0:(i+k)]-temp_seas[0:(i+k)])
            
        ma_len=len(temp_trend)-j
        for i in xrange(j,len(temp_trend)):
            temp_trend[i]=np.mean(Temp[(i-ma_len):len(temp_trend)]-temp_seas[(i-ma_len):len(temp_trend)])
            
#        temp_seas=temp_seas[i:j]
#        temp_trend=temp_trend[i:j]        
#        Temp=Temp[i:j]
#        actuals=actuals[i:j] 
#        lmp_seas=lmp_seas[i:j]
        Len=len(actuals)
        forecasts=np.array([np.mean(actuals) for kk in xrange(Len)])        
        
        self.actuals=actuals
        self.maximum=max(actuals)
        self.forecasts=forecasts
        self.temp_seas=temp_seas
        self.temp_trend=temp_trend
        self.lmp_seas=lmp_seas
        
        
        rawErrors = actuals - forecasts
        self.rawErrors=rawErrors
        minErr=min(self.rawErrors)
        maxErr=max(self.rawErrors)
        
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
        self.minimum=min(self.all_states)

    
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
        
        RunLensCurr=[[] for i in range(CSTransDim)]
         
        OverallTimeinState=[0 for i in range(CSTransDim)]
         
         
        #Errors=[[] for i in range(CSTransDim)]
         
        for i in xrange(len(RunLens['lengths'])-1):
            if(RunLens['values'][i]==0):
                CurrSt=0
            else:
                CurrSt=1
           
            OverallTimeinState[CurrSt]=OverallTimeinState[CurrSt]+RunLens['lengths'][i]
            RunLensCurr[CurrSt].append(RunLens['lengths'][i])
        
        self.t_0_Probabilities=np.zeros(CSTransDim)
        S3=sum(OverallTimeinState)
        for i in xrange(CSTransDim):
            self.t_0_Probabilities[i]=float(OverallTimeinState[i])/S3
        
        for i in xrange(CSTransDim):
            RunLensCurr[i].sort()
        self.Run_Lengths=RunLensCurr
        
        self.Stay_Switch_Mat=np.zeros((2,2))
        self.Stay_Switch_Mat[0,0]=float(sum(self.Run_Lengths[0]))/(sum(self.Run_Lengths[0])+len(self.Run_Lengths[0]))
        self.Stay_Switch_Mat[0,1]=1-self.Stay_Switch_Mat[0,0]
        self.Stay_Switch_Mat[1,1]=float(sum(self.Run_Lengths[1]))/(sum(self.Run_Lengths[1])+len(self.Run_Lengths[1]))
        self.Stay_Switch_Mat[1,0]=1-self.Stay_Switch_Mat[1,1]

        
        binVals_ts=np.array(self.seas_bins)*(max(temp_seas)-min(temp_seas))+min(temp_seas)

        binVals_tt=np.array(self.trend_bins)*(max(temp_trend)-min(temp_trend))+min(temp_trend)        
        
        self.tempseasbins=binVals_ts
        self.temptrendbins=binVals_tt        
        
        Errors=[[] for ii in range(self.numTransMats)]
        JSQuantiles=[]
        Errors_t=[[[]for jj in range(self.transMatDim)] for ii in range(self.numTransMats)]

        for i in xrange(Len-1):
            OUS=OUState[i]
            TSS=detState(temp_seas[i],binVals_ts)
            TTS=detState(temp_trend[i],binVals_tt)
            St=OUS*(self.num_temp_trend_bins*self.num_temp_seas_bins)+(TSS)*self.num_temp_trend_bins+TTS
            Errors[St].append(rawErrors[i])

        Wt_Percentiles=[]
        for x in xrange(self.transMatDim):
            Wt_Percentiles.append((float(100)*x)/self.transMatDim)
        Wt_Percentiles.append(float(100))        
        
        for i in xrange(self.numTransMats):
            JSQuantiles.append(np.percentile(Errors[i],Wt_Percentiles))
            Errors[i].sort()
        
        ProbsofNextBininTempState=[np.zeros((self.transMatDim,self.transMatDim)) for i in range(self.numTransMats)]
        CountsofNextBininTempState=[np.zeros((self.transMatDim,self.transMatDim)) for i in range(self.numTransMats)]
        ErrsinThisState=[[[] for jj in range(self.transMatDim)] for ii in range(self.numTransMats)]
        ProbsinThisState=[[[] for jj in range(self.transMatDim)] for ii in range(self.numTransMats)]
        SetErrsinThisState=[[[] for jj in range(self.transMatDim)] for ii in range(self.numTransMats)]
        ErrIndsinThisState=[[[] for jj in range(self.transMatDim)] for ii in range(self.numTransMats)]

        for i in xrange(Len-1):
           # if(OUState[i]==OUState[i+1]): 
            OUS=OUState[i]
            TSS=detState(temp_seas[i],binVals_ts)
            TTS=detState(temp_trend[i],binVals_tt)
            St=OUS*(self.num_temp_trend_bins*self.num_temp_seas_bins)+(TSS)*self.num_temp_trend_bins+TTS
            EState=detState(rawErrors[i],JSQuantiles[St])
            Errors_t[St][EState].append(rawErrors[i])
            OUS_p1=OUState[i+1]
            TSS_p1=detState(temp_seas[i+1],binVals_ts)
            TTS_p1=detState(temp_trend[i+1],binVals_tt)
            if ((OUS_p1==OUS) and (TSS_p1==TSS) and (TTS_p1==TTS)):
                EState_p1=detState(rawErrors[i+1],JSQuantiles[St])
                CountsofNextBininTempState[St][EState,EState_p1]+=1
            ErrsinThisState[St][EState].append(round_to_array(self.all_error_states,rawErrors[i]))
            ErrIndsinThisState[St][EState].append(round_to_array_ind(self.all_error_states,rawErrors[i]))

            
        for i in xrange(self.numTransMats):
            for j in xrange(self.transMatDim):
                S1=0
                Counts=Counter(ErrIndsinThisState[i][j])
                for key in Counts:
                    S1+=Counts[key]
                for key in Counts:
                    SetErrsinThisState[i][j].append(key)
                    ProbsinThisState[i][j].append(float(Counts[key])/S1)
                S2=0
                for k in xrange(self.transMatDim):
                    S2+=CountsofNextBininTempState[i][j,k]
                if S2>0:
                    for k in xrange(self.transMatDim):
                        ProbsofNextBininTempState[i][j,k]=float(CountsofNextBininTempState[i][j,k])/S2
        

        self.Larger_State_Errors=Errors    
        self.State_Errors=SetErrsinThisState
        self.State_Errors_Probs=ProbsinThisState
        self.Intra_State_Tmats=ProbsofNextBininTempState
        self.Quantiles=JSQuantiles
        
        self.tt_states=[]
        self.ts_states=[]
        for ii in xrange(Len):
            self.tt_states.append(detState(self.temp_trend[ii],self.temptrendbins))
            self.ts_states.append(detState(self.temp_seas[ii],self.tempseasbins))
        
#        
#        
#        
#        
#        
#        
#        
#        Len=len(actuals)
#        
#        OUState=np.zeros((Len,),dtype=np.int)
#         
#        for i in xrange(Len):
#            if (rawErrors[i]>0):
#                OUState[i]=OUState[i]+1
#       
#        RunLens=rle2(OUState)
#
#        OverRunLens=[]
#        UnderRunLens=[]
#         
#        for i in xrange(len(RunLens['lengths'])):
#            if (RunLens['values'][i]==0):
#                UnderRunLens.append(RunLens['lengths'][i])
#            else:
#                OverRunLens.append(RunLens['lengths'][i])
#        
#        Percentiles=[]
#        for x in xrange(RLQs):
#            Percentiles.append((float(100)*x)/RLQs)
#        Percentiles.append(float(100))
#        
#        ORLensQuants=map(int, np.percentile(OverRunLens,Percentiles))
#        URLensQuants=map(int, np.percentile(UnderRunLens,Percentiles))
#        self.ORQuants=ORLensQuants
#        self.URQuants=URLensQuants
#        
#        
#        RunLensSplit=[[] for i in range(CSTransDim)]
#        RunLensCurr=[[] for i in range(CSTransDim)]
#        StartingPoints=[[] for i in range(CSTransDim)]
#        ForwardCounts=np.zeros((CSTransDim,CSTransDim))
#        ForwardProb=np.zeros((CSTransDim,CSTransDim))
#        OverallTimeinState=[0 for i in range(CSTransDim)]
#        t_0_Probabilities=np.zeros(CSTransDim)
#         
#        Errors=[[] for i in range(CSTransDim)]
#         
#        for i in xrange(len(RunLens['lengths'])-1):
#            if(RunLens['values'][i]==0):
#                CurrSt=detState(RunLens['lengths'][i],URLensQuants)
#                NextSt=detState(RunLens['lengths'][i+1],ORLensQuants)+RLQs
#            else:
#                CurrSt=detState(RunLens['lengths'][i],ORLensQuants)+RLQs
#                NextSt=detState(RunLens['lengths'][i+1],URLensQuants)
#           
#            OverallTimeinState[CurrSt]=OverallTimeinState[CurrSt]+RunLens['lengths'][i]
#            RunLensSplit[CurrSt].append(RunLens['lengths'][i+1])
#            RunLensCurr[CurrSt].append(RunLens['lengths'][i])
#            ForwardCounts[CurrSt,NextSt]+=1
#           
#            if (i>0):
#                Errors[CurrSt].extend(rawErrors[(RunLens['inds'][i-1]+1):RunLens['inds'][i]+1])
#                StartingPoints[CurrSt].append(rawErrors[RunLens['inds'][i-1]+1])
#            else:
#                Errors[CurrSt].extend(rawErrors[0:RunLens['inds'][i]+1])
#                #StartingPoints[CurrSt].append(rawErrors[0])
#          
#         
#        for i in xrange(CSTransDim):
#            S1=sum(ForwardCounts[i,:])
#            for j in xrange(CSTransDim):
#                if (S1>0):
#                    ForwardProb[i,j]=float(ForwardCounts[i,j])/S1
#        
#        S3=sum(OverallTimeinState)
#        for i in xrange(CSTransDim):
#            t_0_Probabilities[i]=float(OverallTimeinState[i])/S3
#        
#        CSQuantiles=[]
#        Errors_tplus1=[[[] for j in range(self.transMatDim)] for i in range(CSTransDim)]
#        Errors_t=[[[] for j in range(self.transMatDim)] for i in range(CSTransDim)]
#        Init_Errors_t=[[[] for j in range(self.transMatDim)] for i in range(CSTransDim)]
#        Initial_Errors=[[] for i in range(CSTransDim)]
#
#         
#        ForwardCountsWithinState=[np.zeros((self.transMatDim,self.transMatDim)) for i in range(CSTransDim)]
#        ForwardProbWithinState=[np.zeros((self.transMatDim,self.transMatDim)) for i in range(CSTransDim)]
#        
#        InitialCounts=[np.zeros((self.transMatDim)) for i in range(CSTransDim)]
#        InitialProbs=[np.zeros((self.transMatDim)) for i in range(CSTransDim)]
#        
#        Wt_Percentiles=[]
#        for x in xrange(self.transMatDim):
#            Wt_Percentiles.append((float(100)*x)/self.transMatDim)
#        Wt_Percentiles.append(float(100))        
#        
#        for i in xrange(CSTransDim):
#            CSQuantiles.append(np.percentile(Errors[i],Wt_Percentiles))
#           
#            for j in xrange(len(StartingPoints[i])):
#                SSt=detState(StartingPoints[i][j],CSQuantiles[i])
#                Init_Errors_t[i][SSt].append(StartingPoints[i][j])
#                InitialCounts[i][SSt]=InitialCounts[i][SSt]+1
#                Initial_Errors[i].append(StartingPoints[i][j])
#
#           
#            ###REMEMBER THIS CHANGE FROM R SCRIPT           
#            R=0
#            for j in xrange(len(RunLensCurr[i])):
#                if ((RunLensCurr[i][j])>1):
#                    for k in xrange((RunLensCurr[i][j])-1):
#                        t=R+k
#                        St=detState(Errors[i][t],CSQuantiles[i])
#                        NSt=detState(Errors[i][t+1],CSQuantiles[i])
#                        Errors_tplus1[i][St].append(Errors[i][t+1])
#                        Errors_t[i][St].append(Errors[i][t])
#                        ForwardCountsWithinState[i][St,NSt]+=1
#                    St=detState(Errors[i][t+1],CSQuantiles[i])
#                    Errors_t[i][St].append(Errors[i][t+1])
#                else:
#                    St=detState(Errors[i][R],CSQuantiles[i])
#                    Errors_t[i][St].append(Errors[i][R])
#                R=R+(RunLensCurr[i][j])
#           
#            for j in xrange(self.transMatDim):
#                S1=sum(InitialCounts[i])
#                if (S1>0):
#                    InitialProbs[i][j]=float(InitialCounts[i][j])/S1
#
#                S3=sum(ForwardCountsWithinState[i][j,])
#                for k in xrange(self.transMatDim):
#                    if (S3>0):
#                        ForwardProbWithinState[i][j,k]=float(ForwardCountsWithinState[i][j,k])/S3
#        
#        
#        for i in xrange(CSTransDim):
#            RunLensCurr[i].sort()
#            Errors[i].sort()
#            for j in xrange(self.transMatDim):
#                Errors_t[i][j].sort()
#                Errors_tplus1[i][j].sort()
        
        self.Time_in_State=OverallTimeinState     
        self.Run_Lengths=RunLensCurr
        
        for i in range(len(self.forecasts)):
            self.forecasts[i]=round_to_array(self.all_states,self.forecasts[i]+self.lmp_seas[i])

           
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
        
        #Set up inds
        self.preds_vals=[]
        LenErrStates=len(self.all_error_states)
        for i in xrange(self.numTransMats):
            for ii in xrange(self.transMatDim):
                for j in xrange(LenErrStates):
                    self.preds_vals.append(self.all_error_states[j])
        
#        self.StateErrorProbs=[[{} for j in range(self.transMatDim)] for i in range(CSTransDim)]
#        self.InitialErrorProbs=[[{} for j in range(self.transMatDim)] for i in range(CSTransDim)]
#        for i in xrange(CSTransDim):
#            for ii in xrange(self.transMatDim):
#                E=[]
#                for err in self.State_Errors[i][ii]:
#                    E.append(round_to_array_ind(self.all_error_states,err))
#                S=len(E)
#                Counts=Counter(E)
#                for key in Counts:
#                    Prob=float(Counts[key])/S
#                    self.StateErrorProbs[i][ii][key]=Prob
#        
#        for i in xrange(CSTransDim):
#            E=[]
#            for err in self.initial_errors[i]:
#                E.append(round_to_array_ind(self.all_error_states,err))
#            S=len(E)
#            Counts=Counter(E)
#            for key in Counts:
#                Prob=float(Counts[key])/S
#                St=detState(self.all_error_states[key],CSQuantiles[i])
#                self.InitialErrorProbs[i][St][key]=Prob
        
        #Used for full transition function
        self.elapsed_time=0
        self.opt_time_t=-1
        self.consistent_p_stay=0.0
        
        os.chdir('..')
        os.chdir('input')
     
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
        return self.forecasts[t/self.ts]
    
    def set_new_forecast(self, new_forecast):
        self.forecasts=new_forecast*self.max_price/self.original_max
    
    #EXTRA FUNCTION HERE
    def set_new_temp_forecast(self, new_temp_forecast):
        Temp=np.array(new_temp_forecast)      
                
        ok = ~np.isnan(Temp)
        xp = ok.ravel().nonzero()[0]
        fp = Temp[~np.isnan(Temp)]
        x  = np.isnan(Temp).ravel().nonzero()[0]
        Temp[np.isnan(Temp)] = np.interp(x, xp, fp) 
        
        Freq=24*60/self.minutes_per_time_step  
        temp_decomp = sm.tsa.seasonal_decompose(Temp, freq=Freq)
        temp_seas=temp_decomp.seasonal
        temp_trend=temp_decomp.trend

        k=0
        while np.isnan(temp_trend[k]):
            k+=1
        j=len(temp_trend)-1
        while np.isnan(temp_trend[j]):
            j-=1
        j+=1
        
        for i in xrange(k):
            temp_trend[i]=np.mean(Temp[0:(i+k)]-temp_seas[0:(i+k)])
            
        ma_len=len(temp_trend)-j
        for i in xrange(j,len(temp_trend)):
            temp_trend[i]=np.mean(Temp[(i-ma_len):len(temp_trend)]-temp_seas[(i-ma_len):len(temp_trend)])
            
        self.temp_seas=temp_seas
        self.temp_trend=temp_trend
        
        
    def transition_prob_tildeGivenCurrentInfoStateIndex(self, crossingState,t_prime):
        if crossingState>=(self.numTransMats*self.transMatDim/2):
            OUS=1
        else:
            OUS=0
        ind=bisect.bisect_right(self.Run_Lengths[OUS],t_prime)
        prob=1-(float(ind)/len(self.Run_Lengths[OUS]))
        return prob
        
       
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
        return self.minimum
    
    def get_possible_postds(self, t):
        """
        Return list of possible post decision states of node at time t
        """
        tt=detState(self.temp_trend[t],self.temptrendbins)
        ts=detState(self.temp_seas[t],self.tempseasbins)
        ret_states=[]
        for OUS in xrange(self.CSTransDim):
            St=OUS*(self.num_temp_trend_bins*self.num_temp_seas_bins)+(ts)*self.num_temp_trend_bins+tt
            StInd=St*self.transMatDim
            for k in xrange(self.transMatDim):
                ret_states.append(StInd+k)
        return ret_states
        
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
        for i in xrange(self.numTransMats*self.transMatDim):
            S_C=i/self.transMatDim
            W_t=i%self.transMatDim
            if S_C==SCHere:
                if S_C>=self.numTransMats/2:
                    p=self.Stay_Switch_Mat[1,1]
                else:
                    p=self.Stay_Switch_Mat[0,0]
                p_trans_in_state=self.Intra_State_Tmats[S_C][WtHere,W_t]
                if p_trans_in_state>0:
                    for ind in xrange(len(self.State_Errors[S_C][W_t])):
                        Forward_States.append(i*len(self.all_error_states)+self.State_Errors[S_C][W_t][ind])
                        Transition_Probs.append(p*p_trans_in_state*self.State_Errors_Probs[S_C][W_t][ind])
            else:
                if S_C>=self.numTransMats/2:
                    p=self.Stay_Switch_Mat[0,1]
                    if S_C==(SCHere+self.numTransMats/2):
                        p_trans_out_state=float(1.0)/self.transMatDim
                        for ind in xrange(len(self.State_Errors[S_C][W_t])):
                            Forward_States.append(i*len(self.all_error_states)+self.State_Errors[S_C][W_t][ind])
                            Transition_Probs.append(p*p_trans_out_state*self.State_Errors_Probs[S_C][W_t][ind])
                else:
                    p=self.Stay_Switch_Mat[1,0]
                    if S_C==(SCHere-self.numTransMats/2):
                        p_trans_out_state=float(1.0)/self.transMatDim
                        for ind in xrange(len(self.State_Errors[S_C][W_t])):
                            Forward_States.append(i*len(self.all_error_states)+self.State_Errors[S_C][W_t][ind])
                            Transition_Probs.append(p*p_trans_out_state*self.State_Errors_Probs[S_C][W_t][ind])
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
        T=postds['T']
        p_stay=self.transition_prob_tildeGivenCurrentInfoStateIndex(pds,self.elapsed_time)
        SC=pds/self.transMatDim
        WT=pds%self.transMatDim
        tt=self.get_temptrendvar(T)
        ts=self.get_tempseasvar(T)
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
            if SC>=self.numTransMats/2:
                OUS=1
            else:
                OUS=0
            SC2=OUS*(self.num_temp_trend_bins*self.num_temp_seas_bins)+(ts)*self.num_temp_trend_bins+tt
            Estate=my_own_rand_choice(range(self.transMatDim),self.Intra_State_Tmats[SC][WT,:])
            ind=my_own_rand_choice(self.State_Errors[SC2][Estate],self.State_Errors_Probs[SC2][Estate])
            i=SC2*self.transMatDim+Estate                
        else:
            self.elapsed_time=0
            if SC>=self.numTransMats/2:
                OUS=0
            else:
                OUS=1
            SC2=OUS*(self.num_temp_trend_bins*self.num_temp_seas_bins)+(ts)*self.num_temp_trend_bins+tt
            Estate=random.choice(range(self.transMatDim))
            i=SC2*self.transMatDim+Estate
            ind=my_own_rand_choice(self.State_Errors[SC2][Estate],self.State_Errors_Probs[SC2][Estate])
        return i*len(self.all_error_states)+ind
    
    def get_initial_preds(self):
        """
        Return an initial pre decision state at time t
        """
        self.elapsed_time=0
        
        tt=self.get_temptrendvar(0)
        ts=self.get_tempseasvar(0)
        OUS=my_own_rand_choice(range(2), self.t_0_Probabilities)
        St=OUS*(self.num_temp_trend_bins*self.num_temp_seas_bins)+(ts)*self.num_temp_trend_bins+tt
        Estate=random.choice(range(self.transMatDim))
        i=St*self.transMatDim+Estate
        ind=my_own_rand_choice(self.State_Errors[St][Estate],self.State_Errors_Probs[St][Estate])
        return i*len(self.all_error_states)+ind
    
    def set_random_seed(self, rint):
        """
        set specific random seed of random number generator
        """
        random.seed(rint)

    ## EXTRA HELPER FUNCTIONS
    def get_temptrendvar(self,t):
        return self.tt_states[t/self.ts]
    
    def get_tempseasvar(self,t):
        return self.ts_states[t/self.ts]