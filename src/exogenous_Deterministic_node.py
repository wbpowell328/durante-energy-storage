from Node import Node
import csv
import numpy as np
import bisect
import random

def try_parse(string):
    try:
        return float(string)
    except Exception:
        return np.nan
    
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


def readcsv(filename,Rows):	
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=";")

    rownum = 0	
    a = []
    
    for row in reader:
        row2=row[0].split(',')
        b=[]
        for r in row2:
            b.append(try_parse(r))
        a.append(b)
        rownum += 1   
    if not Rows:
        a=map(list, zip(*a))
    
    ifile.close()
    return a

class exogenous_Deterministic_node(Node):
    
    '''
    Deterministic process where the value at time t is input from csv file
    '''
        
    def __init__(self, name, filePath, headers, params):        
        
        Plen=len(params)
        self.max_val=1.0
        self.d_int=1.0
        self.ts=1
        if Plen>=1:
            self.d_int=float(params[0])
        if Plen>=2:
            self.ts=int(params[1])
        if Plen>=3:
            if not params[2]=='ObsMax':
                self.max_val=float(params[2])

        self.name=name
        self.RowsorCols=False
        self.samplePathIdx=0
        
        if not (headers[0]=='Rows' or headers[0]=='Columns'):      
            load_data=[]
            first_step=1
            reader=csv.DictReader(open(filePath), fieldnames=headers)
            for row in reader:
                Str=row['Data']
                Str=Str.replace(',','')
                if first_step==1:
                    first_step=0
                    First_Point=try_parse(Str)
                    if not np.isnan(First_Point):
                        load_data.append(First_Point)
                else:
                    load_data.append(try_parse(Str))
                    
            self.original_max=max(load_data)
            if params[2]=='ObsMax':
                self.max_val=self.original_max

            deterministic_data=np.array(load_data)*self.max_val/self.original_max
            
            #DEAL WITH MISSING DATA WHILE READING IN
            ok = ~np.isnan(deterministic_data)
            xp = ok.ravel().nonzero()[0]
            fp = deterministic_data[~np.isnan(deterministic_data)]
            x  = np.isnan(deterministic_data).ravel().nonzero()[0]
            deterministic_data[np.isnan(deterministic_data)] = np.interp(x, xp, fp)
            MaxDD=max(deterministic_data)
            MinDD=min(deterministic_data)
            
        else:
            self.RowsorCols=True
            if headers[0]=='Rows':
                deterministic_data=readcsv(filePath,True)
                MaxDD=max(max(deterministic_data))
                MinDD=min(min(deterministic_data))
            else:
                deterministic_data=readcsv(filePath,False)
                MaxDD=max(max(deterministic_data))
                MinDD=min(min(deterministic_data))        
        
        self.all_states=np.arange(MinDD, MaxDD+self.d_int, self.d_int)
        self.maximum=max(self.all_states)
        self.minimum=min(self.all_states)    
        self.discretized_data=[]
        ###DISCRETIZE EACH DATA POINT TO CLOSEST POINT IN STATES
        if not (headers[0]=='Rows' or headers[0]=='Columns'):      
            for sam in deterministic_data:
                self.discretized_data.append(round_to_array(self.all_states,sam))
            self.data=self.discretized_data
        else:
            for row in deterministic_data:
                dd_row=[]
                for r in row:
                    dd_row.append(round_to_array(self.all_states,r))
                self.discretized_data.append(dd_row)
            self.data=self.discretized_data[self.samplePathIdx]
        
    def get_name(self):
        """
        Return name of exogenous process
        """
        return self.name
    
    def get_discretization_interval(self):
        """
        Return the interval which this process is discretized by
        """
        return self.d_int
    
    def get_time_step(self):
        """
        Return time step of process in integer multiple of quickest changing process
        """
        return self.ts
        
    def set_new_forecast(self,new_forecast):
        
        deterministic_data=np.array(new_forecast)*self.max_val/self.original_max
        
        #DEAL WITH MISSING DATA WHILE READING IN
        ok = ~np.isnan(deterministic_data)
        xp = ok.ravel().nonzero()[0]
        fp = deterministic_data[~np.isnan(deterministic_data)]
        x  = np.isnan(deterministic_data).ravel().nonzero()[0]
        deterministic_data[np.isnan(deterministic_data)] = np.interp(x, xp, fp)  
        
        
        self.all_states=np.arange(0.0, max(deterministic_data)+self.d_int, self.d_int)
        self.maximum=max(self.all_states)
        
        discretized_data=[]
        for sam in deterministic_data:
            discretized_data.append(round_to_array(self.all_states,sam))
        self.data=discretized_data
    
    def get_max(self):
        """
        Return maximum value of exogenous process
        """
        return self.maximum
    
    def get_min(self):
        """
        Return minimum value of node (float)
        """
        return self.minimum        
    
    def get_postds_value(self, postds):
        """
        Return value of node (the state which the index represents)
        """
        return self.data[postds['T']/self.ts]
    
    def get_preds_value(self, preds):
        """
        Return value of node (the state which the index represents)
        """
        return self.data[preds['T']/self.ts]
    
    def get_postds(self,postds):
        """
        Return post decision state value of node (index, an int)
        """
        return 0
    
    def get_preds(self,preds):
        """
        Return pre decision state value of node (index, an int)
        """
        return 0

    def get_forecast(self, t):
        """
        Return forecast of node at time t (int)
        """
        return self.data[t/self.ts]
    

    def get_possible_postds(self, t):
        """
        Return list of possible post decision states of node at time t
        """
        return [0]
        
    def get_possible_preds(self, t):
        """
        Return list of possible pre decision states of node at time t
        """
        return [0]
        
    def get_postds_to_preds_probabilities(self, postds):
        """
        Return a 2 element list of the forward states (list comprising first
        element) and their probabilities (list comprising second element)
        
        Possible simplified Markov chain representation of process 
        """
        return [[0],[1.0]]     
    
    def pre_to_post_ds_transition(self,preds,dec):
        """
        After a decision is made, return post decision state (return int index)
        """
        return 0
    
    def post_to_pre_ds_transition(self, postds):
        """
        Make transition from post to pre decision state (return int index)
        """
        return 0
    
    def get_initial_preds(self):
        """
        Return an initial pre decision state at time t
        """
        if self.RowsorCols:
            self.data=self.discretized_data[self.samplePathIdx]
            self.samplePathIdx+=1   
        return 0   
    
    def set_random_seed(self, rint):
        """
        set specific random seed of random number generator
        """
        random.seed(rint)