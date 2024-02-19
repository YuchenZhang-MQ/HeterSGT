# Heterogeneous Subgraph Transformer for Fake News Detection

### This repository is the official implementation of "Heterogeneous Subgraph Transformer for Fake News Detection" (WWW' 24)

## Datasets
To access the datasets used in this study, please use the following links:

MM COVID: https://github.com/bigheiniu/MM-COVID

ReCOVery: http://coronavirus-fakenews.com

MC Fake: https://github.com/qwerfdsaplking/MC-Fake

PAN2020: https://pan.webis.de/data.html

LIAR: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

## To Run

Build heterogeneous graphs for each dataset:
```
python build_graph.py --dataset <dataset_name>
```

Run HeteroSGT for fake news detection
```
python main.py --dataset <dataset_name>
```
