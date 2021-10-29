# -*- coding : utf-8 -*-
# @Time      : 2021/10/2910:51
# @Author    : wkb

from __future__ import print_function
from __future__ import division,print_function
from gurobipy import *
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from matplotlib.lines import lineStyles
import time
import random
import itertools

starttime = time.time()

#function to read data from .txt files
def readData(path,nodeNum):
    nodeNum = nodeNum
    cor_X = []
    cor_Y = []

    f=open(path,'r')
    lines = f.readlines()
    count = 0

    #reaad the information
    for line in lines:
        count = count + 1
        if(count >=10 and count <= 10+ nodeNum):
            line = line[:-1]
            str = re.split(r" +",line)
            cor_X.append(float(str[2]))
            cor_Y.append(float(str[3]))

    #compute the distance matrix
    disMatrix = [([0]* nodeNum) for p in range(nodeNum)]
    for i in range(0,nodeNum):
        for j in range(0,nodeNum):
            temp = (cor_X[i] - cor_X[j])**2 + (cor_Y[i] - cor_Y[j])**2
            disMatrix[i][j] = (int)(math.sqrt(temp))
            temp = 0
    return disMatrix

def printData(disMatrix):
    print("----------------------cost matrix----------------\n")
    for i in range(len(disMatrix)):
        for j in range(len(disMatrix)):
            print("%6.1f"%(disMatrix[i][j]),end = "")
        print()

def reportMIP(model,Routes):
    if model.status == GRB.OPTIMAL:
        print("Best MIP solution:",model.objVal,"\n")
        var = model.getVars()
        for i in range(model.numVars):
            if (var[i].x > 0):
                print(var[i].varName,"=",var[i].x)
                print("optimal route:",Routes[i])

def reportRMP(model):
    if model.status == GRB.OPTIMAL:
        print('travel diatance id ',model.objVal,"\n")
        var = model.getVars()
        for i in range(model.numVars):
            if(var[i].x >0):
                print(var[i].varname,"=",var[i].x)
                print("\n")
        con = model.getConstrs()
        for i in range(model.numConstrs):
            print(con[i].constrName,"=",con[i].pi)
            print('\n')

def reportSUB(model):
    if model.status == GRB.OPTIMAL:
        print("reducde cost :",model.objVal,"\n")
    if model.objVal <= 1e-6:
        var = model.getVars()
        for i in range(mdoel.numVars):
            if(var[i].x>0):
                print(var[i].varName,"=",var[i].x)
                print('\n')

def getValue(model,nodeNum):
    x_value = np.zeros([nodeNum,nodeNum])
    for m in model.getVars():
        if(m.varName.startswith('x')):
            a = (int)(m.varName.split('_')[1])
            b = (int)(m.varName.split('_')[2])
            x_value[a][b] = m.x
    return x_value

def getRoute(x_value):
    x = copy.deepcopy(x_value)
    lastPoint = 0
    arcs = []
    for i in range(nodeNum):
        for j in range(nodeNum):
            if(x[i][j]>0):
                route_temp.append([i,j])
    return route_temp

#callback - use lazy constraints eliminate sub-tours

def subtourelim(subProblem,where):
    # make a list of edge selected in the solution
    print('subProblem._vars',subProblem._vars)
    #vars = subProblem.cbGetSolution(subProblem._vars)
    x_values = np.zeros([nodeNum,nodeNum])
    for m in subProblem.getVars():
        if(m.varName.startswith('x')):
            a = (int)(m.varName.split('_')[1])
            b = (int)(m.varName.split('_')[2])
            x_values[a][b] = subProblem.cbGetSolution(m)
    print("solution = ",x_values)

    #find the shortest cycle in the selected edge list
    tour = subtour(x_values)
    print('tour = ',tour)
    if(len(tour)< nodeNum):
        #add subtour elimination constraint for every pair of cities in tour
        print('---add sub tour elimination constraint--')
        subProblem.cbLazy(quicksum(subProblem._vars[i][j] for i in tour for j in tour if i !=j )<=len(tour)-1)
        LinExpr = quicksum(subProblem._vars[i][j] for i in tour for j in tour  if i != j )
        print('LinExpr = ',LinExpr)
        print('RHS = ',len(tour)-1)

#give a tuplelist of edge, find the shortes subtour
def subtour(graph):
    degree = computeDegree(graph)
    unvisited = []
    for i in range(1,len(degree)):
        if (degree[i]>=2):
            unvisited.append(i)
    cycle = range(0,nodeNum+1)

    edges = findEdges(graph)
    edges = tuplelist(edges)
    print(edges)
    while(unvisited):
        thiscycle =[]
        neighbors = unvisited
        while(neighbors):
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i,j in edges.select(current,'*') if j in unvisited]
            neighbors2 = [i for i ,j in edges.select('*') if i in unvisited]
            if (neighbors2):
                neighbors.extend(neighbors2)
                print('current:',current,'\n neighbor',neighbors)
    isLink = ((thiscycle[0], thiscycle[-1]) in edges) or (thiscycle[-1],thiscycle[-1],thiscycle[0] in edges)
    if (len(cycle)>len(thiscycle) and len(thiscycle)>=3 and isLink):
        print('in = ',(thiscycle[0],thiscycle[-1] in edges) or (thiscycle[-1],thiscycle[0] in edges))
        cycle = thiscycle
    return cycle

def computeDegree(graph):
    degree = np.zeros(len(graph))
    for i in range(len(graph)):
        for j in range(len(graph)):
            if(graph[i][j] >  0.5):
                degree[i] = degree[i] + 1
                degree[j] = degree[j] + 1
    print('degree',degree)
    return degree

def findEdges(graph):
    edges = []
    for i in range(1,len(edges)):
        for j in range(1,len(graph)):
            if(graph[i][j]>0.5):
                edges.append((i,j))
    return edges



# cost =[[0,7,2,1,5],
#        [7,0,3,6,8],
#        [2,3,0,4,2],
#        [1,6,0,4,9],
#        [5,8,2,9,0]]
nodeNum = 6
path = "./"
cost = readData(path,nodeNum)
print(cost)

#nodeNunm = len(cost)
eps = -1e-4
big_M = nodeNum - 1
Routes = []
route_temp = []
obj_RMP = []
print(eps)

#iniital route:1-2-3-4-5-1: 7+3+4+9+5=28
#degree of each vertex:2
masterProblem = Model('Master Problrm')
subProblem = Model('subProblem')

masterProblem.setParam("OutputFlag",0)
subProblem.setParam("OutputFlag",0)

#initialize the rout:1-2-3-...-n-1
distance_initial = 0
for i in range(nodeNum-1):
    distance_initial = distance_initial + cost[i][i+1]
distance_initial = distance_initial + cost[0][nodeNum-1]
print('distance of initial route:',distance_initial)

#constuct RMP
rmp_var = []
rmp_var.append(masterProblem.addVar(lb=0.0,ub=GRB.INFINITY,obj = distance_initial,vtype=GRB.CONTINUOUS,name = 'y_'+str(0)))

rmp_con = []
column_coef_degree = [0.0]* nodeNum
for i in range(nodeNum):
    column_coef_degree[i] = 2
    rmp_con.append(masterProblem.addConstr(lhs = rmp_var[0] * column_coef_degree[0],sense = '==',rhs = 2,name="rmp_con"+str(i)))

rmp_con.append(masterProblem.addConstr(lhs=rmp_var[0],sense="==",rhs = 1,name="rmp_con_"+str(5)))

masterProblem.setAttr("ModelSense",GRB.MINIMIZE)

#end RMP

#add initial route into Routes
route_initial = []
for i in range(len(cost)):
    route_initial.append(i)
route_initial.append(0)
Routes.append(route_initial)

#solve RMP and get duals of each constraint
masterProblem.optimize()
obj_RMP.append(masterProblem.objVal)
masterProblem.write('RMP.lp')
masterProblem.write('RMP.bas')
print("RMP num vars:",masterProblem.numVars)
print("RMP objective :",masterProblem.objVal)
rmp_dual = masterProblem.getAttr("Pi",masterProblem.getConstrs())
print("RMP duals:",rmp_dual)
rmp_dual_pi = rmp_dual[0:len(cost)]
rmp_dual_mu = rmp_dual[-1]
print("RMP duals pi:",rmp_dual_pi)
print("RMP duals mu:",rmp_dual_mu)


#construct SUB
#x_ij
X = [[[] for i in range(nodeNum)] for j in range(nodeNum)]
for i in range(nodeNum):
    for j in range(nodeNum):
        if(i != j):
            x[i][j] = (subProblem.addVar(lb=0.0,ub=1,obj=cost[i][j]-rmp_dual_pi[i] - rmp_dual_pi[j],vtype=GRB.BINARY,name="x_"+str(i)+"_"+str(j)))

#constraint 1: node 1 with degree of 2
expr = LinExpr(0)
for i in range(nodeNum):
    for j in range(nodeNum):
        if ((i != j and i ==0) or (i !=j and j ==0)):
            expr.addTerms(1,X[i][j])
subProblem.addConstr(expr == 2,"c1")
expr.clear()

#constraint 2: each node must be visited
expr = LinExpr(0)
for k in range(nodeNum):
    for i in range(nodeNum):
        for j in range(nodeNum):
            if ((i !=j and i ==k)or (i !=j and j ==k)):
                expr.addTerms(1,X[i][j])
    subProblem.addConstr(expr >= 1,"c2")
    expr.clear()

#constraint 3: aviod subtour x_ij + x_ji <=1
for i in range(0,nodeNum):
    for j in range(i,nodeNum):
        if (i !=j):
            subProblem.addConstr(X[i][j]+X[j][i]  <=1,"c3")

#constraint 4:sum(x_ij)==n
expr = LinExpr(0)
for i in range(nodeNum):
    for j in range(nodeNum):
        if(i!=j):
            expr.addTerms(1,X[i][j])
subProblem.addConstr(expr == nodeNum,"c4")
expr.clear()

subProblem.setAttr("ModelSense",GRB.MINIMIZE)


#set lazy constraints
subProblem._vars = X
subProblem.Params.lazyConstraints = 1
subProblem.optimize(subtourelim)
#subproblem.optimize()

subProblem.write('SUB.lp')

print('\n\n-------cg_iter:',1,"----------------")
print("sub objective:",subProblem.objVal)
print('route:')
for m in subProblem.getVars():
    if (m.x==1 and m.varName.startswith('x')):
        print("%s \t %d"%(m.varName,m.x))

#column_coef_degree = [0]*nodeNum
#print(column_coef_degree[4])

#column generation algorithm
cg_iter = 1
while(subProblem.objVal<eps+rmp_dual_mu):
    route_temp = []
    cg_iter = cg_iter+1
    if(cg_iter ==1):
        print("-------------------------------------------------------------------")
        print("*                          column generation starts                *")
        print("-------------------------------------------------------------------")
    column_coef_degree = [0]*nodeNum
    sub_distance = 0
    for m in subProblem.getVars():
        if(m.x>0 and m.varName.startswith('x')):
            i = (int)(m.varName.split('_')[1])
            j = (int)(m.varName.split('_')[2])
            column_coef_degree[i] = column_coef_degree[i]+1
            column_coef_degree[j] = column_coef_degree[j]+1
            sub_distance = sub_distance + cost[i][j]

    print('column_coef_degree:',column_coef_degree)
    print('sub_distance:',sub_distance)

    #add new route into Routes
    route_temp = []
    x_value = getValue(subProblem,nodeNum)
    route_temp = getRoute(x_value)
    Routes.append(route_temp)
    print('subproblem route_temp:',route_temp)
    print('column_coef_degree:',column_coef_degree)
    print('sub_diatance:',sub_distance)

    #add new column into RMP
    column_coef = column_coef_degree[0:]
    column_coef.append(1)
    print('column_coef:',column_coef)
    rmp_col = Column(column_coef,rmp_con)
    rmp_var.append(masterProblem.addVar(lb=0.0,ub=GRB.INFINITY,ojb=sub_distance,vtype=GRB.CONTINUOUS,name="y_"+str(len(rmp_var)),column=rmp_col))


    #solve RMP and get duals of each constraint
    print("RMP num vars:",masterProblem.numVars)
    masterProblem.optimize()
    obj_RMP.append(masterProblem.objVal)
    masterProblem.write('RMP.lp')
    print('\n\n---RMP postive varibles:------')
    for m in masterProblem.getVars():
        if(m.x>0):
            print("%s = \t %3.2f"%(m.varName,m.x))
    print('\n')

    print("RMP  objective:",masterProblem.objVal)
    rmp_dual = masterProblem.getAttr("Pi",masterProblem.getConstrs())
    rmp_dual_pi = rmp_dual[0:len(cost)]
    rmp_dual_mu = rmp_dual[-1]
    print("RMP duals pi:",rmp_dual_pi)
    print("RMP duals mu:",rmp_dual_mu)


    #reconstruct the objective function of subproblem
    obj = LinExpr(0)
    for i in range(nodeNum):
        for j in range(nodeNum):
            if(i != j):
                obj.addTerms(cost[i][i] - rmp_dual_pi[i] - rmp_dual_pi[j],X[i][j])
    subProblem.setObjective(obj,GRB.MINIMIZE)


    #solve the subproblem and get new 1-tree
    print('\n\n----------cg_iter:',cg_iter,'----------------\n\n')

    #set lazy constraints
    subProblem.update()
    subProblem._vars = X
    subProblem.Params.lazyConstraints=1

    subProblem.optimize(subtourelim)
    subProblem.write('SUB.lp')
    print("sub objective:",subProblem.objVal - rmp_dual_mu)
    print('subprolem route:')
    for m in subProblem.getVars():
        if(m.x>0 and m.varName.startswith('x')):
            print("%s =  \t %3.2f"%(m.varName,m.x))
    print('\n\n-----------------------U------------------------\n\n')

    for m in subProblem.getVars():
        if(m.x>0 and m.varName.startswith('u')):
            print("%s = \t %3.2f"%(m.varName,m.x))

print("-----------------------------------------------------------")
print("-----------------------column generation end---------------")
print("-----------------------------------------------------------")


#solve final RMP
#you can change the varible to integer or solve the RMP directly

mip_var = masterProblem.getVars()
for i in range(masterProblem.numVars):
    mip_var[i].setAttr("Vtype ",GRB.CONTINUOUS)
masterProblem.optimize()
obj_opt = masterProblem.objVal
reportMIP(masterProblem,Routes)

#long running
endtime = time.time()
print("\n\n-------------running time",(endtime - starttime),"seconds---------------\n\n")


#compute the difference between iteration t and iteration t+1

obj_RMP_dif = []
obj_RMP_dif_ration = []
for i in range(len(obj_RMP)-1):
    obj_RMP_dif.append(obj_RMP[i]-obj_RMP[i+1])
    if (obj_RMP[i] != obj_opt):
        obj_RMP_dif_ration.append((obj_RMP[i] - obj_RMP[i+1])/(obj_RMP[i] - obj_opt))
    else:
        obj_RMP_dif_ration.append(0)

#plot the change of objective of RMP
#plt.plot(obj_RMP)
index = range(len(obj_RMP))
plt.figure(figsize=(6,4))
plt.grid(ls='--')
plt.plot(index,obj_RMP,label = r'objective of RMP',color = 'black',linewidth = 1)
plt.xlabel("column generation iterations",fontsize = 10)
plt.ylabel("Objective of RMP",fontsize = 10)
titlename = 'objective of RMP\n'
fig_name1 = 'objective of RMP.png'
plt.savefig(fig_name1)
plt.show()


print(obj_RMP_dif)
print(obj_RMP_dif_ration)
plt.figure(figsize=(6,4))
plt.grid(ls = '--')
plt.scatter(x = index[0:-1],
            y=obj_RMP_dif,
            s=50,
            c='blue',
            marker = '+',
            linewidths=2)
plt.xlabel("column generation iteration",fontsize = 10)
plt.ylabel("objective of RMP",fontsize = 10)
titlename = 'objective change of RMP\n'
fig_name2 = 'objective change of RMP.png'
plt.savefig(fig_name2)
plt.show()

plt.figure(figsize=(6,4))
plt.grid(ls = '--')
plt.scatter(x = index[0:-1],
            y=obj_RMP_dif_ration,
            s=50,
            c='blue',
            marker = '+',
            linewidths=2)
plt.xlabel("column generation iteration",fontsize = 10)
plt.ylabel("objective change ration of RMP",fontsize = 10)
titlename = 'objective change ration of RMP\n'
fig_name3 = 'objective change ration of RMP.png'
plt.savefig(fig_name3)
plt.show()