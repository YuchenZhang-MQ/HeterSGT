import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData
import argparse


def load_edge(dataset,node):
    if node == 'news':
        df = pd.read_csv(f'../Data/{dataset}/graph/edges/news2news.csv',sep=',',encoding='utf-8')
    else:
        df = pd.read_excel(f'../Data/{dataset}/graph/edges/news2{node}.xlsx')
    pair = df.values.tolist()
    news_index = np.load(f'../Data/{dataset}/graph/nodes/news_index.npy',allow_pickle= True).item()
    index_dict = np.load(f'../Data/{dataset}/graph/nodes/{node}_index.npy',allow_pickle= True).item()
    edges = []
    edges_ = []
    for i in pair:
        head = news_index[i[0]]
        tail = index_dict[str(i[1])]
        edge = [head, tail]
        edge_ = [tail, head]
        edges.append(edge)
        edges_.append(edge_) 
    return edges,edges_

def build_graph(dataset,hiddenSize):
    
    news_attr = np.load(f'../Data/{dataset}/graph/nodes/news_embeddings_{hiddenSize}_final.npy')   
    news_attr = torch.from_numpy(news_attr)
    
    entity_attr = np.load(f'../Data/{dataset}/graph/nodes/entity_embeddings_{hiddenSize}.npy')
    entity_attr = torch.from_numpy(entity_attr)
    
    topic_attr = np.load(f'../Data/{dataset}/graph/nodes/topic_embeddings_{hiddenSize}.npy')
    topic_attr = torch.from_numpy(topic_attr)

    news2entity, news2entity_ = load_edge(dataset,'entity')
   
    news2topic, news2topic_ = load_edge(dataset,'topic')

    news2news, news2news_ = load_edge(dataset,'news')
    
    df_news = pd.read_excel(f'../Data/{dataset}/news_final.xlsx')
    label = df_news['label'].tolist()
    
    data = HeteroData()

    data['news'].x = news_attr
    data['entity'].x = entity_attr
    data['topic'].x = topic_attr
 
    data['news', 'has', 'entity'].edge_index = torch.tensor(news2entity, dtype=torch.long).t().contiguous()
    data['entity', 'has_1', 'news'].edge_index = torch.tensor(news2entity_, dtype=torch.long).t().contiguous()

    data['news', 'belongs', 'topic'].edge_index = torch.tensor(news2topic, dtype=torch.long).t().contiguous()
    data['topic', 'belongs_1', 'news'].edge_index = torch.tensor(news2topic_, dtype=torch.long).t().contiguous()
    
    data['news', 'links', 'news'].edge_index = torch.tensor(news2news, dtype=torch.long).t().contiguous()
    data['news', 'links_', 'news'].edge_index = torch.tensor(news2news_, dtype=torch.long).t().contiguous()

    data['news'].y = torch.tensor(label,dtype = torch.long)
    

    print('='*60)
    print('HeteroGraph:',dataset,'\n',data)
    print(' num_nodes:',data.num_nodes,'\n','num_edges:',data.num_edges,'\n','Data has isolated nodes:',data.has_isolated_nodes(),'\n','Data is undirected:',data.is_undirected())
    print('='*60,'\n')
    torch.save(data,f'../Data/{dataset}/graph/{dataset}_{hiddenSize}_final.pt')

    return data

def class2global(edgelist,global_index,classindex):
    indices_g = []
    for i in edgelist:
        ID = classindex[i]
        index_g = global_index[ID]
        indices_g.append(index_g)
    return indices_g

def get_edgeList(dataset,hiddenSize):

    news_index = np.load(f'../Data/{dataset}/graph/nodes/news_index.npy', allow_pickle=True).item()
    entity_index = np.load(f'../Data/{dataset}/graph/nodes/entity_index.npy', allow_pickle=True).item()
    topic_index = np.load(f'../Data/{dataset}/graph/nodes/topic_index.npy', allow_pickle=True).item()
    data = torch.load(f'../Data/{dataset}/graph/{dataset}_{hiddenSize}_final.pt')

    del data['entity', 'has_1', 'news']
    del data['topic', 'belongs_1', 'news']
    del data['news', 'links_', 'news']
    
    newsList0 = data['news', 'has', 'entity'].edge_index.tolist()[0]
    entityList = data['news', 'has', 'entity'].edge_index.tolist()[1]
    
    newsList1 = data['news', 'belongs', 'topic'].edge_index.tolist()[0]
    topicList = data['news', 'belongs', 'topic'].edge_index.tolist()[1]
    
    news_List_h = data['news', 'links', 'news'].edge_index.tolist()[0]
    news_List_t = data['news', 'links', 'news'].edge_index.tolist()[1]
    
    global_index = np.load(f'../Data/{dataset}/graph/nodes/global_index_graph1.npy', allow_pickle=True).item()
   
    news0_g = class2global(newsList0,global_index,news_index)
    entity_g = class2global(entityList,global_index,entity_index)
    
    news1_g = class2global(newsList1,global_index,news_index)
    topic_g = class2global(topicList,global_index,topic_index)   
    
    news_h_g = class2global(news_List_h,global_index,news_index)
    news_t_g = class2global(news_List_t,global_index,news_index)
    
    node_head = news0_g + news1_g + news_h_g
    node_tail = entity_g + topic_g + news_t_g
    
    edgeList_rw = []
    for i in range(len(node_head)):
        head = node_head[i]
        tail = node_tail[i]
        edge_rw = str(head)+' '+str(tail)
        edgeList_rw.append(edge_rw)
    with open(f'../Data/{dataset}/graph/edges/{dataset}.edgelist','w',encoding = 'utf-8') as f:
        for i in edgeList_rw:
            f.write(str(i)+'\n')
        f.close()
    return edgeList_rw


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='choose dataset & hiddenSize')
    parser.add_argument('--dataset', type=str, default='MM COVID')
    parser.add_argument('--hiddenSize', type=int, default=600, help="news_emb_size")

    args = parser.parse_args()
    dataset = args.dataset
    hiddenSize = args.hiddenSize
    
    data_sum_graph = build_graph(dataset,hiddenSize)
    edgeList_rw = get_edgeList(dataset,hiddenSize)
    
    print(f'graph & edgelist for {dataset} done') 
