import copy
from random import shuffle
from deepwalk import graph
import random
import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm


def remove_dups(data):
    df = pd.DataFrame(data).astype(int)
    dup_index = df[df.duplicated(subset = df.columns)].index.values.tolist()
    df.drop_duplicates(subset = df.columns,inplace = True,ignore_index=True)
    newlist = df.values.tolist()
    num_dups = len(data)-len(newlist)
    print('Dups removed','\n'+'Num_Dups:', num_dups)
    if num_dups != 0:
        print('Dup_Index:',dup_index)
    return newlist,dup_index

def clean_list(data,dup_index):
    newlist = copy.deepcopy(data)
    for i,j in enumerate(dup_index):
        del newlist[j - i]
    return newlist

def global2df(data,colsname):
    new_df =  pd.DataFrame(list(data.items())[:len(list(data.items()))//2], columns=colsname)
    return new_df
def inner2df(data,colsname,typename):
    new_df =  pd.DataFrame(list(data.items())[:len(list(data.items()))//2], columns=colsname)
    new_df['type'] = [typename] * len(new_df)
    return new_df   


def get_inner_type(dataset,walk_list):

    news_index = np.load(f"../Data-TransFD/{dataset}/graph/nodes/news_index.npy", allow_pickle=True).item()
    entity_index = np.load(f"../Data-TransFD/{dataset}/graph/nodes/entity_index.npy", allow_pickle=True).item()
    topic_index = np.load(f"../Data-TransFD/{dataset}/graph/nodes/topic_index.npy", allow_pickle=True).item()

    global_index1 = np.load(f"../Data-TransFD/{dataset}/graph/nodes/global_index_graph1.npy", allow_pickle=True).item()

    global_df1 = global2df(global_index1,["name", "g_id"])
    news_df = inner2df(news_index,["name", "inner_id"],0)
    entity_df = inner2df(entity_index,["name", "inner_id"],1)
    topic_df = inner2df(topic_index,["name", "inner_id"],2)
    inner_df1 = pd.concat([news_df,entity_df,topic_df],ignore_index = True)
    final_df = pd.merge(global_df1, inner_df1)
    
    inner_list = []
    type_list = []
    for walk in tqdm(walk_list,desc="getting inner_list & type_list ..."):
        inners = []
        types = []
        for j in walk:
            item = final_df[final_df.g_id == int(j)]
            inner = item['inner_id'].values.item()
            type_n  = item['type'].values.item()
            inners.append(inner)
            types.append(type_n)
        inner_list.append(inners)
        type_list.append(types)
    return inner_list,type_list


def rand_walk(dataset, restart, num_laps = 1, walk_length = 5):
    G = graph.load_edgelist(f"../Data-TransFD/{dataset}/graph/edges/{dataset}.edgelist", undirected=True)
    df= pd.read_excel(f"../Data-TransFD/{dataset}/news_final.xlsx")
    num_news = len(df['news_id'].tolist())
    label = df['label'].tolist()
    labels = label * num_laps  
    print('num_laps:',num_laps,'walk_length:',walk_length,'num_news:',num_news)
   
    walk_list = []
    for i in tqdm(range(num_laps),desc = 'news random walk...'):
        for j in range(num_news):
            walk = G.random_walk(j, walk_length, alpha = restart, rand=random.Random())
            walk_list.append(walk)
    
    walk_list_,dup_index = remove_dups(walk_list)
    labels_ = clean_list(labels,dup_index)      
    inner_list,type_list = get_inner_type(dataset,walk_list)
    return walk_list_,labels_,inner_list,type_list


