
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:17:37 2018

@author: lazar
"""

from multiprocessing import Pool
import time
import pandas as pd
import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from IPython.display import Audio, display
from sklearn.cluster.bicluster import SpectralCoclustering
from collections import Counter
import collections
from sklearn import preprocessing
flatten = lambda l: [item for sublist in l for item in sublist]
import weighted_aco_lib as lib
import imp
from scipy.spatial import distance
from scipy import stats
import scipy.spatial.distance as ssd
from scipy.cluster import hierarchy
import seaborn as sns; sns.set(color_codes=True)
path_expr = "lung_cancer_full.csv"
path_ppi = "networks/biogrid.human.entrez.tsv"
#path_ppi = "networks/iid.human.all.entrez.tsv"

#%%


#%%  
path_expr = "lung_cancer_full.csv"
path_ppi = "networks/biogrid.human.entrez.tsv"
imp.reload(lib)
# 198 secs for the whole network
col = "cancer_type"
val1 = "Lung cancer ADK"
val2 = 'Lung Squamous Cell Carcinoma'
patients_info = ["cancer_type","tumor","survival","gender","relapse","live_status"]
log2 = True
start = time.time()  
B,G,H,n,m,GE,A_g,group1,group2,labels_B,rev_labels_B = lib.aco_preprocessing(path_expr, path_ppi, col, val1,val2, patients_info, log2, gene_list = None, size = 2000, HI_calc = "corr",sample = 0.8)
end = time.time()
print(end-start)

#%%  
path_expr = "breast_cancer_expr.csv"
path_ppi = "networks/biogrid.human.entrez.tsv"
imp.reload(lib)
# 198 secs for the whole network
col = "case"
val1 = 1
val2 = 0
patients_info = ["case"]
log2 = True
start = time.time()  
B,G,H,n,m,GE,A_g,group1,group2,labels_B,rev_labels_B = lib.aco_preprocessing(path_expr, path_ppi, col, val1,val2, patients_info, log2, gene_list = None, size = 2000, HI_calc = "corr",sample = None)
end = time.time()
print(end-start)
#%%
pos = nx.spring_layout(G)
plt.figure(figsize=(10,10))
nx.draw(G,with_labels = True,node_size=200,alpha=1.0, pos = pos)
#%%
#TRUE groups
grouping_p = []
p_num = list(GE.columns)

for p in p_num:
    if p in group1:
        grouping_p.append(1)
    else:
        grouping_p.append(2)
grouping_p = pd.DataFrame(grouping_p,index = p_num)
grouping_g = []

        

species = grouping_p[0]
lut = {1: '#2f12a5', 2: '#f4f404'}
row_colors = species.map(lut)

g = sns.clustermap(GE.T, row_colors = row_colors,figsize=(15, 10))


#%%
fig = plt.figure(figsize=(18,12))
ax = fig.add_subplot(111)
cax = ax.matshow(H, interpolation='nearest',cmap=plt.cm.GnBu)
plt.colorbar(cax)
plt.title("HI")

    
#%%
cols = ["dataset","method","time" ,"param","size","genes"]
lines = []
scores_br1000 = []
#%%
imp.reload(lib)
# =============================================================================
# #GENERAL PARAMETERS:
# =============================================================================
clusters = 2
K = 20 # number of ants
eps = 0.02
b = 1 #HI significance
times = 35
t_min =5
# =============================================================================
# #NETWORK SIZE PARAMETERS:
# =============================================================================
cost_limit = 20
L_g_min = 10 # minimum # of genes per group
L_g_max = 20 # minimum # of genes per group

th = 1# the coefficient to define the search radipus which is supposed to be bigger than 
#mean(heruistic_information[patient]+th*std(heruistic_information[patient])
#bigger th - less genes are considered (can lead to empty paths if th is too high)

# =============================================================================
# #PHERAMONES PARAMETERS:
# =============================================================================
evaporation  = 0.5
a = 2 #pheramone significance
w_ge = 1 #weight ge in the score

start = time.time()
solution,t_best,sc,conv= lib.ants(a,b,n,m,H,GE,G,clusters,cost_limit,K,evaporation,th,L_g_min,L_g_max,w_ge,eps,times,opt= None,pts = False,show_pher = False,show_plot = True, print_runs = False, save = None)
end = time.time()
print(str(round((end - start)/60,2))+ " minutes")
#%%
ft = (start-end) 
scores_br1000.append(sc)
genes = "|".join([str(labels_B[x]) for x in solution[0][0]+solution[0][1]])
lines.append(["GSE20685","ACO",round(ft/60,2),cost_limit,L_g_min,genes])
#%%
cols = ["dataset","time","size" ,"genes1","genes2","patients1","patients2"]
lines = []
scores_2000 = []

B,G,H,n,m,GE,A_g,group1,group2,labels_B,rev_labels_B = lib.aco_preprocessing(path_expr, path_ppi, col, val1,val2, patients_info, log2, gene_list = None, size = 2000, HI_calc = "corr")

th = 1
L_g_min = 75
L_g_max = 100
times = 70
cost_limit = 25
st1 = time.time()
solution,t_best,sc,conv= lib.ants(a,b,n,m,H,GE,G,clusters,cost_limit,K,evaporation,th,L_g_min,L_g_max,w_ge,eps,times,opt= None,pts = False,show_pher = False,show_plot = True, print_runs = False, save = None)
end1 = time.time()
genes_new = [str(labels_B[x]) for x in solution[0][0]+solution[0][1]]
ft = (end1-st1)
if len(genes_new)> 90:
    B,G,H,n,m,GE,A_g,group1,group2,labels_B,rev_labels_B = lib.aco_preprocessing(path_expr, path_ppi, gene_list = genes_new, size = None, HI_calc = "corr")
    cost_limit = 10
    th = 0
    times = 50
    L_g_min = 75
    L_g_max = 85
    st2 = time.time()
    solution,t_best,sc,conv= lib.ants(a,b,n,m,H,GE,G,clusters,cost_limit,K,evaporation,th,L_g_min,L_g_max,w_ge,eps,times,opt= None,pts = False,show_pher = False,show_plot = True, print_runs = False, save = None)
    end2 = time.time()
    ft = (end1-st1) +(end2-st2)
scores_2000.append(sc)
genes = "|".join([str(labels_B[x]) for x in solution[0][0]+solution[0][1]])
lines.append(["GSE30219","ACO",round(ft/60,2),cost_limit,L_g_min,genes])
#%%    
res = pd.DataFrame(lines,columns = cols)
res.to_csv("C:/Users/lazar/Desktop/quick_check/results/results_3000.csv", index = False)    
#%%
cols = ["dataset","method","time" ,"param","size","genes"]
lines = []
scores_2000 = []
B,G,H,n,m,GE,A_g,group1,group2,labels_B,rev_labels_B = lib.aco_preprocessing(path_expr, path_ppi, col, val1,val2, patients_info, log2, gene_list = None, size = 2000, HI_calc = "corr")

for i in range(5):
    print("simulation 1000 genes # "+str(i))
    L_g_min = 75
    L_g_max = 85
    cost_limit = 20
    th = 1
    st1 = time.time()
    solution,t_best,sc,conv= lib.ants(a,b,n,m,H,GE,G,clusters,cost_limit,K,evaporation,th,L_g_min,L_g_max,w_ge,eps,times,opt= None,pts = False,show_pher = False,show_plot = True, print_runs = False, save = None)
    end1 = time.time()
    ft = end1-st1
    genes = "|".join([str(labels_B[x]) for x in solution[0][0]+solution[0][1]])
    scores_2000.append(sc)
    lines.append(["GSE20685","ACO",round(ft/60,2),cost_limit,L_g_min,genes])

res = pd.DataFrame(lines,columns = cols)
res.to_csv("C:/Users/lazar/Desktop/quick_check/results/results_2000_br.csv", index = False)   



 #%%
res = pd.DataFrame(lines,columns = cols)
res.to_csv("C:/Users/lazar/Desktop/quick_check/results/results_1000.csv", index = False)

#%%
res1 = pd.read_csv("C:/Users/lazar/Desktop/quick_check/results/results_2000_br.csv",sep = ",")
#res2 = pd.read_csv("C:/Users/lazar/Desktop/quick_check/results/results_2000.csv",sep = ",")

genes1 = [x.split("|") for x in res1["genes"]]
#genes2 = [x.split("|") for x in res2["genes"]]

#genes2 = [x.split("|") for x in res[res["method"] == "std"]["genes"]]
#data = [genes1,genes2] 
data = [genes1] 

lib.stability_plot(data, ["2000"]) 
#%%
cols = ["dataset","method","time" ,"param","size","genes"]
lines = []
scores_generank = []
B,G,H,n,m,GE,A_g,group1,group2,labels_B,rev_labels_B = lib.aco_preprocessing(path_expr, path_ppi, gene_list = new_genes, th = None, HI_calc = "corr") 
for i in range(5):
    
    name = str(i)+" generank"
    print(name)
    start = time.time()
    solution,t_best,sc = lib.ants(a,b,n,m,H,GE,A_g,G,clusters,cost_limit,K,evaporation,th,L_g,L_p,w_ppi,w_ge,eps,times,t_max,False,show_pher = False,show_plot = True, print_runs = False, save =name )
    end = time.time()
    scores_generank.append(sc)
    genes = "|".join([str(labels_B[x]) for x in solution[0][0]+solution[0][1]])
    #genes = "|".join([str(x) for x in solution_hoan[0][0]+solution_hoan[0][1]])
    lines.append(["GSE30219","GF",round((end - start)/60,2),cost_limit,L_g,genes])
B,G,H,n,m,GE,A_g,group1,group2,labels_B,rev_labels_B = lib.aco_preprocessing(path_expr, path_ppi, gene_list = None, th = 1, HI_calc = "corr")
res = pd.DataFrame(lines,columns = cols)
res.to_csv("results_generank.csv", index = False)
lines = []
scores_sd = []
for i in range(5):
    start = time.time()
    name = str(i)+" std"
    print(name)
    solution,t_best,sc = lib.ants(a,b,n,m,H,GE,A_g,G,clusters,cost_limit,K,evaporation,th,L_g,L_p,w_ppi,w_ge,eps,times,t_max,False,show_pher = False,show_plot = True, print_runs = False, save = name)
    end = time.time()
    genes = "|".join([str(labels_B[x]) for x in solution[0][0]+solution[0][1]])
    scores_sd.append(sc)
    #genes = "|".join([str(x) for x in solution_hoan[0][0]+solution_hoan[0][1]])
    lines.append(["GSE30219","GF",round((end - start)/60,2),cost_limit,L_g,genes])
res = pd.DataFrame(lines,columns = cols)
res.to_csv("results_std.csv", index = False)
 #%%
res1 = pd.read_csv("results_generank.csv",sep = ",")
res2 = pd.read_csv("results_std.csv",sep = ",")
genes1 = [x.split("|") for x in res1["genes"]]
genes2 = [x.split("|") for x in res2["genes"]]
#genes2 = [x.split("|") for x in res[res["method"] == "std"]["genes"]]
data = [genes1,genes2] 
# genes '718|727|730|1361|3075|3426|3881|3852|3854|3853|3860|3861'
#%%
lib.stability_plot(data, ["generank"]) 
#%%
fig, ax = plt.subplots(figsize=(10, 10))
bplot1 = ax.boxplot([scores_generank,scores_sd],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=["generank","std"])  # will be used to label x-ticks
ax.set_title('Scores')
plt.show()
#%%
solutions = []
timing = []
scores = []
Ts = [10,20,25,30]
for t_max in Ts:
    start = time.time()
    solution,t_best,sc = lib.ants(a,b,n,m,H,GE,A_g,G,clusters,cost_limit,K,evaporation,th,L_g,L_p,w_ppi,w_ge,eps,times,t_max,False,show_pher = True,show_plot = True, print_runs = False, save = str(t_max))
    end = time.time()
    timing.append(round((end - start)/60,2))
    solutions.append(solution)
    scores.append(sc)
#print(str(round((end - start)/60,2))+ " minutes")

        


        

#%% 
 
#(0.9550561797752809, 0.9344262295081968)

#%%
grouping_p = []
p_num = list(GE.columns)
for p in p_num:
    if p in solution[1][0]:
        grouping_p.append(1)
    else:
        grouping_p.append(2)
grouping_p = pd.DataFrame(grouping_p,index = p_num)
grouping_g = []
g_num = list(GE.index)
for g in g_num:
    if g in solution[0][0]:
        grouping_g.append(1)
    elif  g in solution[0][1]:
        grouping_g.append(2)
    else:
        grouping_g.append(3)
        
grouping_g = pd.DataFrame(grouping_g,index = g_num)
genes1 = solution[0][0]
genes2 = solution[0][1]
nodes = genes1+genes2
species = grouping_g[grouping_g[0]!=3][0]
lut = {1: '#A52A2A', 2: '#7FFFD4'}
col_colors = species.map(lut)

species = grouping_p[0]
lut = {1: '#A52A2A', 2: '#7FFFD4'}
row_colors = species.map(lut)
g = sns.clustermap(GE.T[nodes], row_colors=row_colors,col_colors = col_colors,figsize=(15, 10))



#%%
# ALL TOGETHER
genes1,genes2 = solution[0]
patients1, patients2 = solution[1]

means1 = [np.mean(GE[patients1].loc[gene])-np.mean(GE[patients2].loc[gene]) for gene in genes1]
means2 = [np.mean(GE[patients1].loc[gene])-np.mean(GE[patients2].loc[gene]) for gene in genes2]

G_small = nx.subgraph(G,genes1+genes2)

plt.figure(figsize=(15,10))
cmap=plt.cm.viridis
vmin = min(means1+means2)
vmax = max(means1+means2)
pos = nx.spring_layout(G_small)
ec = nx.draw_networkx_edges(G_small,pos)
nc1 = nx.draw_networkx_nodes(G_small,nodelist =genes1, pos = pos,node_color=means1, node_size=200,alpha=1.0,
                             vmin=vmin, vmax=vmax,node_shape = "^",cmap =plt.cm.viridis)
nc2 = nx.draw_networkx_nodes(G_small,nodelist =genes2, pos = pos,node_color=means2, node_size=200,
                             alpha=1.0,
                             vmin=vmin, vmax=vmax,node_shape = "o",cmap =plt.cm.viridis)
nx.draw_networkx_labels(G_small,pos)
plt.colorbar(nc1)
plt.axis('off')
plt.show()

#%%
solution_hoan  = [[['1825','1830','3852','3853','3854','3860','3861','5317','6278','6698','6699','6707'],
           ['718','722','972','1512','3113','3122','6440','6441','7080','9476','117156']  ],
             [["GSM748239","GSM748056","GSM748059","GSM748061","GSM748062","GSM748063","GSM748065","GSM748067","GSM748072","GSM748074","GSM748076","GSM748084","GSM748088","GSM748093","GSM748095","GSM748097","GSM748098","GSM748105","GSM748106","GSM748109","GSM748110","GSM748112","GSM748115","GSM748117","GSM748119","GSM748120","GSM748132","GSM748136","GSM748141","GSM748145","GSM748148","GSM748151","GSM748154","GSM748155","GSM748156","GSM748159","GSM748163","GSM748165","GSM748172","GSM748177","GSM748182","GSM748183","GSM748191","GSM748192","GSM748193","GSM748195","GSM748280","GSM748281","GSM748282","GSM748283","GSM748284","GSM748285","GSM748286","GSM748288","GSM748312","GSM1465993","GSM1465994","GSM1465998"],
              ["GSM748053","GSM748054","GSM748055","GSM748057","GSM748058","GSM748064","GSM748068","GSM748071","GSM748075","GSM748077","GSM748078","GSM748079","GSM748080","GSM748081","GSM748083","GSM748085","GSM748086","GSM748087","GSM748089","GSM748090","GSM748091","GSM748092","GSM748094","GSM748100","GSM748101","GSM748102","GSM748103","GSM748104","GSM748108","GSM748111","GSM748113","GSM748116","GSM748121","GSM748122","GSM748123","GSM748124","GSM748125","GSM748126","GSM748127","GSM748128","GSM748130","GSM748131","GSM748134","GSM748135","GSM748137","GSM748138","GSM748139","GSM748140","GSM748142","GSM748143","GSM748146","GSM748147","GSM748150","GSM748152","GSM748157","GSM748161","GSM748162","GSM748164","GSM748166","GSM748167","GSM748169","GSM748170","GSM748171","GSM748173","GSM748175","GSM748178","GSM748179","GSM748180","GSM748181","GSM748185","GSM748186","GSM748187","GSM748188","GSM748189","GSM748239","GSM748240","GSM748242","GSM748243","GSM748264","GSM748271","GSM748276","GSM748277","GSM748278","GSM748279","GSM1465989","GSM748056","GSM748066","GSM748099","GSM748133","GSM748287"]]]
sol_hoan = []
for l_big in solution_hoan:
    group_sol = []
    for group in l_big:
        tr = [rev_labels_B[x] for x in group]
        group_sol.append(tr)
    sol_hoan.append(group_sol)
    

#%%
print(lib.score(G,sol_hoan[1],sol_hoan[0],n,m,GE.values,A_g,L_g,L_p,w_ppi,w_ge))
print(lib.score(G,solution[1],solution[0],n,m,GE.values,A_g,L_g,L_p,w_ppi,w_ge))
#11.1979
#9.83
#%%
sol_mine = solution
#%%
solution = sol_mine
#%%
H_small = pd.DataFrame(H[:n,:n],columns = np.arange(n),index = np.arange(n))
grouping_g = []
g_num = list(GE.index)
for g in g_num:
    if g in solution[0][0]:
        grouping_g.append(1)
    elif  g in solution[0][1]:
        grouping_g.append(2)
    else:
        grouping_g.append(3)
        
grouping_g = pd.DataFrame(grouping_g,index = g_num)
species = grouping_g[0]
lut = {1: '#A52A2A', 2: '#7FFFD4', 3:'#FAEBD7'}
col_colors = species.map(lut)

g = sns.clustermap(H_small,row_colors=col_colors,col_colors = col_colors,figsize=(30, 30))
g.savefig("solution_mine.png")