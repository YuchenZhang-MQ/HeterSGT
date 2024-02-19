from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, roc_curve, auc
import torch
import math
from tqdm import tqdm
import time
import news_RandomWalk as rw
from Data import Vocab, Data, FakenewsDataset
import pandas as pd
import numpy as np

def load_data(args):
    graph = torch.load(f"../Data-TransFD/{args.dataset}/graph/{args.dataset}_{args.hiddenSize}_final.pt", map_location=torch.device('cpu'))
    print(graph)
    walk_list,labels,inner_list,type_list = rw.rand_walk(args.dataset, args.restart, args.num_laps, args.walk_length)
    test_ratio = 0.1
    dataset = FakenewsDataset(walk_list, inner_list, type_list, labels, graph)
    train_data, test_data = dataset.get_train_id(test_ratio)
    
    return graph, train_data, test_data

def test_once(preds, scores, y):

    preds = preds.cpu()
    scores = scores.cpu()
    y = y.cpu()
    test_acc = accuracy_score(y, preds)
    """ Binary """
    test_pre_b = precision_score(y, preds, average="binary")
    test_recall_b = recall_score(y, preds, average="binary")
    test_f1_b = f1_score(y, preds, average="binary")
    test_auc_b  = roc_auc_score(y, scores[:,1])

    """ Micro """
    test_pre_micro = precision_score(y, preds, average="micro")
    test_recall_micro = recall_score(y, preds, average="micro")
    test_f1_micro = f1_score(y, preds, average="micro")
    test_auc_micro  = roc_auc_score(y, scores[:,1], average="micro")

    """ Macro """
    test_pre_macro = precision_score(y, preds, average="macro")
    test_recall_macro = recall_score(y, preds, average="macro")
    test_f1_macro = f1_score(y, preds, average="macro")
    test_auc_macro  = roc_auc_score(y, scores[:,1], average="macro")

    fpr, tpr, thresholds = roc_curve(y, scores[:,1])
    roc_auc = auc(fpr, tpr)
    auc_list = [fpr, tpr, thresholds, roc_auc]

    test_results = {"test_acc" : test_acc,
                "test_pre_b" : test_pre_b,
                "test_pre_micro" : test_pre_micro,
                "test_pre_macro" : test_pre_macro,
                "test_recall_b" : test_recall_b,
                "test_recall_micro" : test_recall_micro,
                "test_recall_macro" : test_recall_macro,
                "test_f1_b" : test_f1_b,
                "test_f1_micro" : test_f1_micro,
                "test_f1_macro" : test_f1_macro,
                "test_auc_b" : test_auc_b,
                "test_auc_micro" : test_auc_micro,
                "test_auc_macro" : test_auc_macro
                }
    return test_results,auc_list


def print_results_once(train_result, stage="train"):

    train_acc, train_pre_b, train_pre_micro, train_pre_macro, train_recall_b, train_recall_micro, train_recall_macro, train_f1_b, train_f1_micro, train_f1_macro, \
        train_auc_b, train_auc_micro, train_auc_macro = train_result.values()

    print(f"Avg = Binary \n"
    f"{stage} Acc: {train_acc:.4f}, {stage} Pre: {train_pre_b:.4f}, {stage} Recall: {train_recall_b:.4f}, {stage} f1: {train_f1_b:.4f}, {stage} auc: {train_auc_b:.4f} \n"
    f"Avg = Micro \n"
    f"{stage} Acc: {train_acc:.4f}, {stage} Pre: {train_pre_micro:.4f}, {stage} Recall: {train_recall_micro:.4f}, {stage} f1: {train_f1_micro:.4f}, {stage} auc: {train_auc_micro:.4f} \n"
    f"Avg = Macro \n"
    f"{stage} Acc: {train_acc:.4f}, {stage} Pre: {train_pre_macro:.4f}, {stage} Recall: {train_recall_macro:.4f}, {stage} f1: {train_f1_macro:.4f}, {stage} auc: {train_auc_macro:.4f} \n"
    )

def save_results(args, train_result):
    df = pd.DataFrame([train_result])
    df.to_excel(f"./results/{args.dataset}_{args.hiddenSize}_R{args.round}_WL{args.walk_length}_dp{args.dropout}_{args.num_layers}_layers_case2{args.case2}_case3{args.case3}_restart_{args.restart}_topn_{args.topn}.xlsx", index = False, encoding = 'utf-8')
