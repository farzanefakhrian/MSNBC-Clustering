import numpy as np
import pandas as pd
import csv
import pprint
import progressbar

NUM_CLUSTERS=4
NUM_ATTRIBS=17
NUM_DISTINCT_VALUES=2
NUM_SAMPLES=-1
data = pd.read_csv("msnbcWithHeader.csv", sep=',')
data = data[data.sum(axis=1)<200]
data[data>1] = 1
vals=np.array(data.get_values())
NUM_SAMPLES = vals.shape[0]

clustersSamples = np.zeros(shape=(NUM_SAMPLES,)) - 1
AGGR_CPD = np.ones(shape=(NUM_CLUSTERS, NUM_ATTRIBS, NUM_DISTINCT_VALUES))
CLUSTERED_COUNT = np.ones(shape=(NUM_CLUSTERS,))

for i in range(NUM_CLUSTERS):
    clustersSamples[i] = i
    AGGR_CPD[i][np.arange(NUM_ATTRIBS), vals[i]] += 1
    print("Cluster_" + str(i) + " : " + str(vals[i]))

t = 0
print("--- " + str(NUM_SAMPLES) + " SAMPLES LOADED ---")
clusterCounts = np.zeros(shape=(NUM_CLUSTERS))
with progressbar.ProgressBar(max_value=NUM_SAMPLES) as bar:
    for sample_ind in range(NUM_CLUSTERS, NUM_SAMPLES):
        t +=1
        if t == 500:
            t=0
            bar.update(sample_ind)
        sample = vals[sample_ind]
        shuffles= np.arange(NUM_CLUSTERS)
        np.random.shuffle(shuffles)
        for cluster in shuffles:
            counts = np.product(AGGR_CPD[cluster][np.arange(NUM_ATTRIBS), sample], dtype=np.float)
            denom = (CLUSTERED_COUNT[cluster] + NUM_CLUSTERS) ** NUM_ATTRIBS
            PClusterI = 1. /(NUM_CLUSTERS)
            clusterCounts[cluster] = float(PClusterI * counts / denom)
        cluster_ind = np.argmax(clusterCounts)
        CLUSTERED_COUNT[cluster_ind] += 1
        AGGR_CPD[cluster_ind][np.arange(NUM_ATTRIBS), sample] += 1
        clustersSamples[sample_ind] = cluster_ind

print(np.asarray(CLUSTERED_COUNT/NUM_SAMPLES*100, dtype=np.int))
np.savetxt(open("clusters.csv", 'w'), clustersSamples)
