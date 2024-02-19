import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from utils import test_once, print_results_once, save_results
from model.Base import TransformerEncoder, FNclf, genX


class Model1(nn.Module):
    def __init__(self, config, **kwargs) -> None:
      
        super(Model1,self).__init__(**kwargs)
    
        key_size = config.key_size
        query_size = config.query_size
        value_size = config.value_size
        num_hiddens = config.num_hiddens
        norm_shape = config.norm_shape
        ffn_num_input = config.ffn_num_input 
        ffn_num_hiddens = config.ffn_num_hiddens
        num_heads = config.num_heads
        num_layers = config.num_layers 
        dropout = config.dropout
        out_dim = config.out_dim
        
        news_size = config.news_size
        entity_size = config.entity_size
        topic_size = config.topic_size
        self.walk_length = config.walk_length


        self.Transformer = TransformerEncoder(query_size, key_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, self.walk_length, dropout)
        self.decoder = nn.Linear(num_hiddens, news_size)
        self.clf = FNclf(num_hiddens, out_dim, dropout)

    def forward(self, X, device, args):
        h = self.Transformer(X, self.walk_length, args)
        news_pred = 0
        scores, h_emb = self.clf(h)
        if args.case3 == "h0":
            scores = scores[:,0,:]
        elif args.case3 == "mean":
            scores = scores.mean(dim=1)
        elif args.case3 == "max":
            scores = scores.max(dim=1)[0]

        preds = torch.argmax(F.softmax(scores), dim=-1)

        return scores, preds, news_pred, h_emb[:,0,:]

    
def train(model, train_data, train_label, test_data, test_label, epochs, optimizer, device, args):
    epochs = tqdm(range(epochs))
    train_loss_all = []
    test_loss_all = []

    criteria_sup = torch.nn.CrossEntropyLoss()

    train_label = train_label.to(device)
    test_label = test_label.to(device)

    train_X = genX(train_data, device)
    test_X = genX(test_data, device)

    best_f1 = 0
    for epoch in epochs:
        optimizer.zero_grad()  
        train_scores, train_preds, train_news_decode, h_train = model(train_X, device, args)
        loss = criteria_sup(train_scores, train_label)
        train_loss_all.append(loss.item())
        loss.backward()  
        optimizer.step()  
        epochs.set_description(f"Training Epoch: {epoch}, Loss : {loss.item()}")
        with torch.no_grad():
            test_scores, test_preds, test_news_decode,h_test = model(test_X, device, args)
            test_loss = criteria_sup(test_scores, test_label) 
            test_loss_all.append(test_loss.item())
            test_res,auc_test = test_once(test_preds, test_scores, test_label)
           
            if test_res["test_f1_macro"] > best_f1:
                best_f1 = test_res["test_f1_macro"]
                best_test = test_res
                torch.save(auc_test, f"./results/{args.model}/auc/{args.dataset}_r{args.round}_auc.pt")

    with torch.no_grad():
        train_res,auc_trian = test_once(train_preds, train_scores, train_label)
        print_results_once(train_res, "train")
    with torch.no_grad():
        test_res,auc_test = test_once(test_preds, test_scores, test_label)
        print_results_once(test_res, "test")
        print("Best test results on Macro-F1:")
        print_results_once(best_test, "test")
    save_results(args, best_test)