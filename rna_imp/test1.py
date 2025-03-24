
import scicone
import numpy as np
import pickle
import subprocess
import pandas as pd
import matplotlib as plt
import seaborn as sns
import os

def main():
    install_path = '/cluster/work/bewi/members/andress/SCICoNE_lab/build/'
    temporary_outpath = './'

    seed = 42 # for reproducibility

    np.random.seed(seed)

    # Create SCICoNE object
    sci = scicone.SCICoNE(install_path, temporary_outpath, verbose=False)


    sim = sci.simulate_data(n_cells=200, n_nodes=5, n_bins=1000, n_regions=40,
                        n_reads=2000, nu=5.0, ploidy=2, seed=seed)

    sim['tree'].plot_tree()

    bps = sci.detect_breakpoints(sim['d_mat'], threshold=3.0)
    inferred_tree = sci.learn_tree(sim['d_mat'], bps['segmented_region_sizes'], n_reps=4, seed=seed)
    inferred_tree.plot_tree(node_labels=True, node_sizes=False)

if __name__ == '__main__':
    main()
   