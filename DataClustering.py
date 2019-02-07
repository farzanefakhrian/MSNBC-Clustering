import numpy as np
import pandas as pd
import csv
import pprint
import progressbar

NUM_CLUSTERS=4
NUM_ATTRIBS=17
NUM_DISTINCT_VALUES=2
NUM_SAMPLES=-1

X = pd.read_csv("msnbcWithHeader.csv", sep=',')
X = X[X.sum(axis=1)<200]
X[X>1] = 1
Y=np.array(X.get_values())
NUM_SAMPLES = Y.shape[0]

clusters_of_samples = np.zeros(shape=(NUM_SAMPLES,)) - 1

AGGR_CPD = np.ones(shape=(NUM_CLUSTERS, NUM_ATTRIBS, NUM_DISTINCT_VALUES))
CLUSTERED_COUNT = np.ones(shape=(NUM_CLUSTERS,))

for i in range(NUM_CLUSTERS):
    clusters_of_samples[i] = i
    AGGR_CPD[i][np.arange(NUM_ATTRIBS), Y[i]] += 1
    print("Cluster_" + str(i) + " : " + str(Y[i]))


t = 0
print("--- " + str(NUM_SAMPLES) + " SAMPLES LOADED ---")
tmp_probs = np.zeros(shape=(NUM_CLUSTERS))
with progressbar.ProgressBar(max_value=NUM_SAMPLES) as bar:
    for sample_ind in range(NUM_CLUSTERS, NUM_SAMPLES):
        t +=1
        if t == 500:
            t=0
            bar.update(sample_ind)
        sample = Y[sample_ind]
        shuff_clstrs= np.arange(NUM_CLUSTERS)
        np.random.shuffle(shuff_clstrs)
        for cluster in shuff_clstrs:
            counts = np.product(AGGR_CPD[cluster][np.arange(NUM_ATTRIBS), sample], dtype=np.float)
            denom = (CLUSTERED_COUNT[cluster] + NUM_CLUSTERS) ** NUM_ATTRIBS
            P_Ci = 1. / (NUM_CLUSTERS)
            tmp_probs[cluster] = float(P_Ci * counts / denom)

        cluster_ind = np.argmax(tmp_probs)

        CLUSTERED_COUNT[cluster_ind] += 1
        AGGR_CPD[cluster_ind][np.arange(NUM_ATTRIBS), sample] += 1
        clusters_of_samples[sample_ind] = cluster_ind


print(np.asarray(CLUSTERED_COUNT/NUM_SAMPLES*100, dtype=np.int))
np.savetxt(open("clusters.csv", 'w') , clusters_of_samples)
