from Globals import Globals

from policy1 import policy1
from policy2 import policy2
from policy3 import policy3
from policy4 import policy4
from policy5 import policy5
from policy_PFA import policy_PFA
from policy_CFA import policy_CFA
from policy_DLA import policy_DLA
from policy_PostDS_LU_table import policy_PostDS_LU_table
from policy_PreDS_Lin_VFA import policy_PreDS_Lin_VFA
from policy_PostDS_Lin_VFA import policy_PostDS_Lin_VFA

#from policyVFA_Par import policyPar

from exogenous_Deterministic_node import exogenous_Deterministic_node
from exogenous_CSHSMM_node import exogenous_CSHSMM_node
from exogenous_CSHSMM_Prices_node import exogenous_CSHSMM_Prices_node
from basic_resource_node import basic_resource_node
from Custom_Node_1 import Custom_Node_1
from Custom_Node_2 import Custom_Node_2
from Custom_Node_3 import Custom_Node_3

from custom_constraint_and_reward_entry import custom_constraint_and_reward_entry
from Example_Constraint_and_Reward_Entry import Example_Constraint_and_Reward_Entry

from custom_decision_space import custom_decision_space
from basic_decision_space_A import basic_decision_space_A
from basic_decision_space_B import basic_decision_space_B
from basic_decision_space_C import basic_decision_space_C
from basic_decision_space_D import basic_decision_space_D
from Example_Decision_Space import Example_Decision_Space

from full_backward_MDP_solver_new import full_backward_MDP_solver_new
from backward_ADP_solver_Post_Decision_State_Lookup_Table import backward_ADP_solver_Post_Decision_State_Lookup_Table
from backward_ADP_solver_Pre_Decision_State_Lin_VFA import backward_ADP_solver_Pre_Decision_State_Lin_VFA
from backward_ADP_solver_Post_Decision_State_Lin_VFA import backward_ADP_solver_Post_Decision_State_Lin_VFA

from Policy_Comparison import Policy_Tester
from parameter_optimizer import parameter_optimizer
from ForwardADP_API import ForwardADP_API

import numpy as np
import csv
import ast
import os

def try_parse(string):
    try:
        return float(string)
    except Exception:
        return np.nan

def try_parse_int(string):
    try:
        return int(string)
    except Exception:
        return np.nan

os.chdir("..")
os.chdir("input")

#USER INPUT
InputFile="Example_User_Input_CSV_Read_Test.csv"


InputString=""
f=open(InputFile)
file_data=f.readlines()
for row in file_data:
    InputString+=row.rstrip()+os.linesep
InputString+='End Input Parameters'+os.linesep
print(InputString)

reader=[]
f=open(InputFile)
reader1=csv.DictReader(f)
for row in reader1:
    reader.append(row)
f.close()
    
GLB_VARS=Globals()
GLB_VARS.set_global_variable('Discount_Factor',1.0)
GLB_VARS.set_global_variable('Horizon',1)

Nodes={}
Policies={}

for row in reader:
    ID=row['Identifier']
    Type=row['Type']
    Name=row['Name']
    FilePath=row['FilePath']
    Headers=row['Headers']
    Headers=Headers.replace(" ","")
    Params=row['Parameters']
    Params=Params.replace(" ","")
    if (ID=='discount factor' or ID=='discount Factor' or ID=='Discount Factor' or ID=='Discount factor'):
        df=float(ast.literal_eval(Params))
        if df>1 or df<0:
            print 'Discount factor not in the interval [0,1]'
        else:
            GLB_VARS.set_global_variable('Discount_Factor', df)
    elif (ID=='Action Space' or ID=='Action space' or ID=='action Space' or ID=='action space'):
        pass
    elif (ID=='policy' or ID=='Policy'):
        pass
    elif (ID=='New Forecast' or ID=='new forecast' or ID=='New forecast' or ID=='new Forecast'):
        pass
    elif (ID=='decision' or ID=='Decision'):
        pass
    elif (ID=='module' or ID=='Module'):
        pass
    elif(ID=='Constraints_and_Rewards'):
        pass
    elif(ID=='Horizon'):
        GLB_VARS.set_global_variable('Horizon',int(Params))
    elif(ID=='Node' or ID=='node'):
        pass
    else:
        print 'Throw Exception A'

for row in reader:
    ID=row['Identifier']
    Type=row['Type']
    Name=row['Name']
    FilePath=row['FilePath']
    Headers=row['Headers']
    Headers=Headers.replace(" ","")
    Params=row['Parameters']
    Params=Params.replace(" ","")
    if (ID=='exogenous' or ID=='Exogenous'):
        pass
    elif (ID=='discount factor' or ID=='discount Factor' or ID=='Discount Factor' or ID=='Discount factor'):
        pass
    elif (ID=='Action Space' or ID=='Action space' or ID=='action Space' or ID=='action space'):
        pass
    elif (ID=='policy' or ID=='Policy'):
        pass
    elif (ID=='New Forecast' or ID=='new forecast' or ID=='New forecast' or ID=='new Forecast'):
        pass
    elif (ID=='decision' or ID=='Decision'):
        pass
    elif (ID=='module' or ID=='Module'):
        pass
    elif(ID=='Constraints_and_Rewards'):
        pass
    elif(ID=='Node' or ID=='node'):
        if Type=='Basic Resource Node':
            headers=Headers.split(';')
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            ThisNode=basic_resource_node(Name,FilePath,headers,params)
            Nodes[Name]=ThisNode
        elif Type=='Exogenous Deterministic Node':
            headers=Headers.split(';')
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            ThisNode=exogenous_Deterministic_node(Name,FilePath,headers,params)
            Nodes[Name]=ThisNode
        elif Type=='Exogenous Crossing State HSMM':
            headers=Headers.split(';')
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            ThisNode=exogenous_CSHSMM_node(Name,FilePath,headers,params)
            Nodes[Name]=ThisNode
        elif Type=='Exogenous Crossing State HSMM Prices':
            headers=Headers.split(';')
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            ThisNode=exogenous_CSHSMM_Prices_node(Name,FilePath,headers,params)
            Nodes[Name]=ThisNode
        elif Type=='Custom Node 1':
            headers=Headers.split(';')
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            ThisNode=Custom_Node_1(Name,FilePath,headers,params)
            Nodes[Name]=ThisNode
        elif Type=='Custom Node 2':
            headers=Headers.split(';')
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            ThisNode=Custom_Node_2(Name,FilePath,headers,params)
            Nodes[Name]=ThisNode
        elif Type=='Custom Node 3':
            headers=Headers.split(';')
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            ThisNode=Custom_Node_3(Name,FilePath,headers,params)
            Nodes[Name]=ThisNode
        
        else:
            print 'Throw Exception B'
    elif(ID=='Horizon'):   
        pass
    else:
        print 'Throw Exception'



for row in reader:
    ID=row['Identifier']
    Type=row['Type']
    Name=row['Name']
    FilePath=row['FilePath']
    Headers=row['Headers']
    Headers=Headers.replace(" ","")
    Params=row['Parameters']
    Params=Params.replace(" ","")
    if (ID=='exogenous' or ID=='Exogenous'):
        pass
    elif (ID=='New Forecast' or ID=='new forecast' or ID=='New forecast' or ID=='new Forecast'):
        headers=Headers.split(';')
        forecasts=[]
        first_step=1
        reader2=csv.DictReader(open(FilePath),fieldnames=headers)
        for row in reader2:
            Forc=row['Fcst']
            Forc=Forc.replace(',','')
            if first_step==1:
                first_step=0
                First_Point=try_parse(Forc)
                if not np.isnan(First_Point):
                    forecasts.append(First_Point)
            else:
                forecasts.append(try_parse(Forc))
        
        forecasts=np.array(forecasts)
        Nodes[Name].set_new_forecast(forecasts)
        
    elif (ID=='resource' or ID=='Resource'):
        pass
    elif (ID=='discount factor' or ID=='discount Factor' or ID=='Discount Factor' or ID=='Discount factor'):
        pass
    elif (ID=='Action Space' or ID=='Action space' or ID=='action Space' or ID=='action space'):
        pass
    elif (ID=='policy' or ID=='Policy'):
        pass
    elif (ID=='decision' or ID=='Decision'):
        pass
    elif (ID=='module' or ID=='Module'):
        pass
    elif(ID=='Constraints_and_Rewards'):
        pass
    elif(ID=='Horizon'):
        pass
    elif(ID=='Node'):
        pass
    else:
        print 'Throw Exception C'

Decision_Vars=[]
Decisions_Discretized={}
for row in reader:
    ID=row['Identifier']
    Type=row['Type']
    Name=row['Name']
    FilePath=row['FilePath']
    Headers=row['Headers']
    Headers=Headers.replace(" ","")
    Params=row['Parameters']
    Params=Params.replace(" ","")
    if (ID=='exogenous' or ID=='Exogenous'):
        pass
    elif (ID=='resource' or ID=='Resource'):
        pass
    elif (ID=='New Forecast' or ID=='new forecast' or ID=='New forecast' or ID=='new Forecast'):
        pass
    elif (ID=='discount factor' or ID=='discount Factor' or ID=='Discount Factor' or ID=='Discount factor'):
        pass
    elif (ID=='policy' or ID=='Policy'):
        pass
    elif (ID=='Action Space' or ID=='Action space' or ID=='action Space' or ID=='action space'):
        pass
    elif (ID=='module' or ID=='Module'):
        pass
    elif (ID=='decision' or ID=='Decision'):
        Decision_Vars.append(Name)
        Decisions_Discretized[Name]=try_parse_int(Params)
    elif(ID=='Constraints_and_Rewards'):
        pass
    elif(ID=='Horizon'):
        pass
    elif(ID=='Node'):
        pass
    else:
        print 'Throw Exception D'

C_and_R_entry=None

for row in reader:
    ID=row['Identifier']
    Type=row['Type']
    Name=row['Name']
    FilePath=row['FilePath']
    Headers=row['Headers']
    Headers=Headers.replace(" ","")
    Params=row['Parameters']
    Params=Params.replace(" ","")
    if (ID=='exogenous' or ID=='Exogenous'):
        pass
    elif (ID=='resource' or ID=='Resource'):
        pass
    elif (ID=='discount factor' or ID=='discount Factor' or ID=='Discount Factor' or ID=='Discount factor'):
        pass
    elif (ID=='policy' or ID=='Policy'):
        pass
    elif (ID=='New Forecast' or ID=='new forecast' or ID=='New forecast' or ID=='new Forecast'):
        pass
    elif (ID=='Action Space' or ID=='Action space' or ID=='action Space' or ID=='action space'):
        pass
    elif (ID=='module' or ID=='Module'):
        pass
    elif (ID=='decision' or ID=='Decision'):
        pass
    elif(ID=='Horizon'):
        pass
    elif(ID=='Node'):
        pass
    elif(ID=='Constraints_and_Rewards'):
        if Type=='Custom':
            C_and_R_entry=custom_constraint_and_reward_entry()
        elif Type=='Example Constraints and Rewards':
            C_and_R_entry=Example_Constraint_and_Reward_Entry()
        else:
            print 'Throw ExceptionE1'
        R=C_and_R_entry.reward_function()
        Inequality_Constraints=C_and_R_entry.remaining_constraints()
        Equality_Constraints=C_and_R_entry.equality_constraints()
        equality_decision_variables=C_and_R_entry.equality_constraint_variables()
        GLB_VARS.set_global_variable('Reward_Function', R)
        GLB_VARS.set_global_variable('Inequality_Constraints', Inequality_Constraints)
        GLB_VARS.set_global_variable('Equality_Constraints', Equality_Constraints)
        GLB_VARS.set_global_variable('Equality_Decision_Variables',equality_decision_variables)
    else:
        print 'Throw Exception E'

GLB_VARS.set_global_variable('Decision_Vars', Decision_Vars)
GLB_VARS.set_global_variable('Nodes',Nodes)

for row in reader:
    ID=row['Identifier']
    Type=row['Type']
    Name=row['Name']
    FilePath=row['FilePath']
    Headers=row['Headers']
    Headers=Headers.replace(" ","")
    Params=row['Parameters']
    Params=Params.replace(" ","")
    if (ID=='exogenous' or ID=='Exogenous'):
        pass
    elif (ID=='resource' or ID=='Resource'):
        pass
    elif (ID=='discount factor' or ID=='discount Factor' or ID=='Discount Factor' or ID=='Discount factor'):
        pass
    elif (ID=='New Forecast' or ID=='new forecast' or ID=='New forecast' or ID=='new Forecast'):
        pass
    elif (ID=='Action Space' or ID=='Action space' or ID=='action Space' or ID=='action space'):
        if Type=='Custom':
            print 'Custom Action Space'
            params=[]
            if Params:
                Pars=Params.split(';')  
                for p in Pars:
                    params.append(ast.literal_eval(p))
            Decision_Space=custom_decision_space(GLB_VARS,params)
        ###NOTE TYPE 1-4 Not Converted to New Package Yet
#        elif Type=='1':
#            Decision_Space=basic_decision_space_A(GLB_VARS)
#        elif Type=='2':
#            Decision_Space=basic_decision_space_B(GLB_VARS,Decisions_Discretized)
#        elif Type=='3':
#            Decision_Space=basic_decision_space_C(GLB_VARS,Decisions_Discretized)
#        elif Type=='4':
#            Decision_Space=basic_decision_space_D(GLB_VARS)
        elif Type=='Example Decision Space':
            Decision_Space=Example_Decision_Space(GLB_VARS)
        else:
            print 'Throw Exception F1'
        GLB_VARS.set_global_variable('Decision_Space',Decision_Space)

    elif (ID=='policy' or ID=='Policy'):
        pass
    elif (ID=='decision' or ID=='Decision'):
        pass
    elif (ID=='module' or ID=='Module'):
        pass
    elif(ID=='Constraints_and_Rewards'):
        pass
    elif(ID=='Horizon'):
        pass
    elif(ID=='Node'):
        pass
    else:
        print 'Throw Exception F'

for row in reader:   
    ID=row['Identifier']
    Type=row['Type']
    Name=row['Name']
    FilePath=row['FilePath']
    Headers=row['Headers']
    Headers=Headers.replace(" ","")
    Params=row['Parameters']
    Params=Params.replace(" ","")
    if (ID=='exogenous' or ID=='Exogenous'):
        pass
    elif (ID=='resource' or ID=='Resource'):
        pass
    elif (ID=='discount factor' or ID=='discount Factor' or ID=='Discount Factor' or ID=='Discount factor'):
        pass
    elif (ID=='New Forecast' or ID=='new forecast' or ID=='New forecast' or ID=='new Forecast'):
        pass
    elif (ID=='decision' or ID=='Decision'):
        pass
    elif (ID=='Action Space' or ID=='Action space' or ID=='action Space' or ID=='action space'):
        pass
    elif (ID=='module' or ID=='Module'):
        pass
    elif(ID=='Constraints_and_Rewards'):
        pass
    elif(ID=='Horizon'):
        pass
    elif(ID=='Node'):
        pass
    elif (ID=='policy' or ID=='Policy'):       
        if Type=='1':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            Policies[Name]=policy1(Name,FilePath,GLB_VARS,params)
        elif Type=='2':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            Policies[Name]=policy2(Name,FilePath,GLB_VARS,params)
        elif Type=='3':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            Policies[Name]=policy3(Name,FilePath,GLB_VARS,params)
        elif Type=='4':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            Policies[Name]=policy4(Name,FilePath,GLB_VARS,params)
        elif Type=='5':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            Policies[Name]=policy5(Name,FilePath,GLB_VARS,params)
        elif Type=='PFA':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            Policies[Name]=policy_PFA(Name,FilePath,GLB_VARS,params)
        elif Type=='CFA':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            Policies[Name]=policy_CFA(Name,FilePath,GLB_VARS,params)
        elif Type=='DLA':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            Policies[Name]=policy_DLA(Name,FilePath,GLB_VARS,params)
        elif Type=='VFA_Post_Lookup':
            Pars=Params.split(';')
            params=[]
            Policies[Name]=policy_PostDS_LU_table(Name,FilePath,GLB_VARS,params)
        elif Type=='VFA_Post_Parametric':
            Pars=Params.split(';')
            params=[]
            Policies[Name]=policy_PostDS_Lin_VFA(Name,FilePath,GLB_VARS,params)
        elif Type=='VFA_Pre_Parametric':
            Pars=Params.split(';')
            params=[]
            Policies[Name]=policy_PreDS_Lin_VFA(Name,FilePath,GLB_VARS,params)
        else:
            print 'Throw Exception G1'
    else:
        print 'Throw Exception G'

GLB_VARS.set_global_variable('Policies', Policies)
for row in reader:
    ID=row['Identifier']
    Type=row['Type']
    Name=row['Name']
    FilePath=row['FilePath']
    Headers=row['Headers']
    Headers=Headers.replace(" ","")
    Params=row['Parameters']
    Params=Params.replace(" ","")
    if (ID=='exogenous' or ID=='Exogenous'):
        pass
    elif (ID=='resource' or ID=='Resource'):
        pass
    elif (ID=='discount factor' or ID=='discount Factor' or ID=='Discount Factor' or ID=='Discount factor'):
        pass
    elif (ID=='decision' or ID=='Decision'):
        pass
    elif (ID=='Action Space' or ID=='Action space' or ID=='action Space' or ID=='action space'):
        pass
    elif (ID=='policy' or ID=='Policy'):
        pass
    elif(ID=='Constraints_and_Rewards'):
        pass
    elif (ID=='New Forecast' or ID=='new forecast' or ID=='New forecast' or ID=='new Forecast'):
        pass
    elif(ID=='Horizon'):
        pass
    elif(ID=='Node'):
        pass
    elif (ID=='module' or ID=='Module'):
        if Type=='Custom':
            print 'Executing Custom Module'
            ###CUSTOM MODULES SHOULD BE PLACED DOWN BELOW THIS AREA###
            
            
            
            
            
            
            
            
            
            
            
            ###End CUSTOM Modules
        elif Type=='Policy Compare':
            Pars=Params.split(';')
            params=[]
            for x in xrange(len(Pars)):
                params.append(ast.literal_eval(Pars[x]))
            pol_test=Policy_Tester(Policies, params, GLB_VARS, InputString)
            pol_test.test_policies(params[0], FilePath)
        elif Type=='Exact Backward DP':
            B_MDP_solver=full_backward_MDP_solver_new(GLB_VARS,InputString)
            B_MDP_solver.backward_MDP(FilePath)  
        elif Type=='Backward ADP Post-Decision Lookup Table':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            B_MDP_solver=backward_ADP_solver_Post_Decision_State_Lookup_Table(GLB_VARS,InputString)
            B_MDP_solver.backward_MDP(params[0],FilePath)
        elif Type=='Backward ADP Pre-Decision Linear Parametric':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            B_MDP_solver=backward_ADP_solver_Pre_Decision_State_Lin_VFA(GLB_VARS,InputString)
            B_MDP_solver.backward_MDP(params[0],FilePath)
        elif Type=='Backward ADP Post-Decision Linear Parametric':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))
            B_MDP_solver=backward_ADP_solver_Post_Decision_State_Lin_VFA(GLB_VARS,InputString)
            B_MDP_solver.backward_MDP(params[0],FilePath)
        elif Type=='Forward ADP API':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                params.append(ast.literal_eval(p))  
            F_MDP_solver=ForwardADP_API(GLB_VARS,InputString)
            F_MDP_solver.API(params[0],params[1],FilePath)
        elif Type=='System Parameter Grid Search':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                if not str.isalpha(p[0]):
                    params.append(ast.literal_eval(p))
                else:
                    params.append(p)
            Param_Optimizer=parameter_optimizer(GLB_VARS,InputString)
            Param_Optimizer.optimize_system_parameters(params,FilePath)
        elif Type=='Grid Search':
            Pars=Params.split(';')
            params=[]
            for p in Pars:
                if not str.isalpha(p[0]):
                    params.append(ast.literal_eval(p))
                else:
                    params.append(p)
            Param_Optimizer=parameter_optimizer(GLB_VARS,InputString)
            Param_Optimizer.optimize_policy_parameters(params, FilePath)
        else:
            print 'Throw Exception H1'
    else:
        print 'Throw Exception H'
        


    
    
