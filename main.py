import argparse
import torch
from torch_geometric import data
from torch.utils.data import Dataset, random_split, DataLoader
from model.model1 import Model1, train
import json
from Data import Vocab, Data, FakenewsDataset
import news_RandomWalk as rw
from utils import load_data


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class Config():
    def __init__(self):
        self.name = "model config"
    
    def print_config(self):
        for attr in self.attribute:
            print(attr)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='MM COVID')
    parser.add_argument('--hiddenSize', type=int, default=600)
    parser.add_argument('--config', type=str, default='./config/HeteroSGT.json', help='configuration file name.')
    parser.add_argument('--model', type=str, default='Model1')
    parser.add_argument('--num_laps', type=int, default=1, help="num_laps")
    parser.add_argument('--walk_length', type=int, default=5, help="walk length")
    parser.add_argument('--round', type=int, default=1, help='test round')
    parser.add_argument('--num_layers', type=int, default=5, help="num_layers")
    parser.add_argument('--case2', type=str, default="no", help="yes or no for case study II")
    parser.add_argument('--case3', type=str, default="h0", help="h0, mean, or max for case study III")
    parser.add_argument('--restart', type=float, default=0.1, help=" probability of restarts in rw")
    parser.add_argument('--topn', type=int, default=3, help=" num_topics each news linked to")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = arg_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"Training Device: {device}, Dataset: {args.dataset},  Model : {args.model}, Test Round: {args.round}, num_laps: {args.num_laps}, walk_length: {args.walk_length}, hiddenSize: {args.hiddenSize},num_layers: {args.num_layers}")

    with open(args.config, 'r') as f:
        config_dicts = json.load(f)
    configs = {}
    for config in config_dicts:
        conf = Config()
        for key, value in config.items():
            setattr(conf, key, value)
        configs.update({
            config["dataset"] : conf
        })
    config = configs[args.dataset]
    
    
    graph, train_data, test_data = load_data(args)
    
    train_dataloader = DataLoader(train_data, batch_size=train_data.__len__(), shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_data.__len__(), shuffle=True)

    config.query_size, config.key_size, config.value_size = args.hiddenSize, args.hiddenSize, args.hiddenSize
    config.norm_shape = (args.walk_length, config.num_hiddens)
    config.news_size = graph["news"].x.shape[0]
    if graph["entity"] != {}:
        config.entity_size = graph["entity"].x.shape[0]
    else:
        config.entity_size = 0 
    if graph["topic"] != {}:
        config.topic_size = graph["topic"].x.shape[0]
    else:
        config.topic_size = 0
    config.walk_length = args.walk_length
    config.num_hiddens = args.hiddenSize
    args.dropout = config.dropout
    train_label = train_data.labels
    test_label = test_data.labels

    model = globals()[args.model](config)
    model = model.double()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_label = train_data.labels
    test_label = test_data.labels
    train(model, train_data, train_label, test_data, test_label, args.epochs, optimizer, device, args)
