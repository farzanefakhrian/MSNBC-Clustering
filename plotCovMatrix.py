import bnpy
import numpy as np
import os
import pandas as pd

from matplotlib import pylab
import seaborn as sns

FIG_SIZE = (3, 3)
pylab.rcParams['figure.figsize'] = FIG_SIZE

dataset = np.asarray(pd.read_csv("msnbc_wh.csv", sep= ','))

dataset[np.sum(dataset,axis =1 )>200]
dataset[dataset>1]=1


def show_clusters_over_time(task_output_path=None,query_laps=[0, 1, 2, 5, 10, None],nrows=2):

    ncols = int(np.ceil(len(query_laps) // float(nrows)))
    fig_handle, ax_handle_list = pylab.subplots(
        figsize=(FIG_SIZE[0] * ncols, FIG_SIZE[1] * nrows),
        nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    for plot_id, lap_val in enumerate(query_laps):
        cur_model, lap_val = bnpy.load_model_at_lap(task_output_path, lap_val)
        # Plot the current model
        cur_ax_handle = ax_handle_list.flatten()[plot_id]
        bnpy.viz.PlotComps.plotCompsFromHModel(
            cur_model, Data=dataset, )#ax_handle=cur_ax_handle)
        cur_ax_handle.set_xticks([-2, -1, 0, 1, 2])
        cur_ax_handle.set_yticks([-2, -1, 0, 1, 2])
        cur_ax_handle.set_xlabel("lap: %d" % lap_val)
    pylab.tight_layout()
    pylab.savefig("results/covMat1.png")
    pylab.waitforbuttonpress()
    pylab.show()


K25_trained_model, K25_info_dict = bnpy.run(
    "msnbc_wh.csv", 'FiniteMixtureModel', 'Gauss', 'EM',
    output_path='results/',
    nLap=500, nTask=1, nBatch=1,
    sF=0.1,
    moves='birth,merge,shuffle',
    K=10,)

show_clusters_over_time(K25_info_dict['task_output_path'])
