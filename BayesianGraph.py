from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
from pgmpy.models import BayesianModel
from pgmpy.readwrite.BIF import BIFWriter
import pandas as pd
import numpy as np
from time import time
import graphviz as gv

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

train = pd.read_csv('../msnbcWithHeader.csv', sep=',')
train = train[train.sum(axis=1)<200]
train[train>1] = 1

train_start = time()

bic = BicScore(train)
hc = HillClimbSearch(train, scoring_method=bic)
best_model = hc.estimate(prog_bar=True)
edges = best_model.edges()
model = BayesianModel(edges)

model.fit(train,estimator=BayesianEstimator, prior_type="BDeu")
variables = model.nodes()

print(model.edges())
train_end = time() - train_start
print("train time " + str(train_end))

my_graph = gv.Digraph(format='png')
for node in variables:
    my_graph.node(node)
for edge in edges:
    my_graph.edge(edge[0],edge[1])
filename = my_graph.render('../graph', view=True)

writer = BIFWriter(model)
writer.write_bif(filename='../model_extracted.bif')
