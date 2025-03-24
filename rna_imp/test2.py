
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

    n_cells = [20, 40, 100, 200, 400]
    tumor_fractions = [0.01, 0.05, 0.1, 0.2, 0.5, 0.7]
    reps = [0, 1, 2]
    deltas = []
    baseline_deltas = []
    n_clusters = []
    true_n_clusters = []
    cnvs_rows = []
    clusters_rows = []
    np.random.seed(42)

    # List to track failed iterations
    failed_iterations = []

    for N in n_cells:
        for tumor_fraction in tumor_fractions:
            for rep in reps:
                try:
                    seed = np.random.randint(10000)

                    # Simulate data
                    sim = sci.simulate_data(n_cells=N, n_nodes=5, n_bins=1000, n_regions=10,
                                            n_reads=10000, nu=10.0, ploidy=2, seed=seed)

                    # Transform data to desired proportions
                    clone_a_cells_idx = np.where(np.all(sim['tree'].outputs['inferred_cnvs'] == np.ones((1000,)) * 2, axis=1))[0]
                    clone_a_cells_counts = sim['d_mat'][clone_a_cells_idx]
                    tumor_cells_data = np.copy(sim['d_mat'])
                    tumor_cells_cn = np.copy(sim['tree'].outputs['inferred_cnvs'])
                    is_a = np.zeros((sim['d_mat'].shape[0],))
                    is_a[clone_a_cells_idx] = 1.
                    tumor_cells_data = tumor_cells_data[~is_a.astype(bool), :]
                    tumor_cells_cn = tumor_cells_cn[~is_a.astype(bool), :]
                    T = int(N * tumor_fraction)
                    D = N - T
                    selected_tumor_cells = np.random.randint(tumor_cells_data.shape[0], size=T)
                    subsampled_tumor_cells = tumor_cells_data[selected_tumor_cells, :]
                    subsampled_tumor_cells_cn = tumor_cells_cn[selected_tumor_cells, :]
                    augmented_data = np.concatenate([subsampled_tumor_cells, clone_a_cells_counts[
                                        np.random.randint(clone_a_cells_counts.shape[0], size=D), :]], axis=0)

                    # True copy numbers
                    true_cn = np.ones((D, clone_a_cells_counts.shape[1])) * 2
                    true_cn = np.concatenate([subsampled_tumor_cells_cn, true_cn], axis=0)

                    # Call breakpoints
                    bps = sci.detect_breakpoints(augmented_data, threshold=3.0, window_size=int(0.01 * augmented_data.shape[1]))
                    # Call copy number clones
                    inferred_tree = sci.learn_tree(augmented_data, bps['segmented_region_sizes'], n_reps=4, full=False,
                                                seed=seed)

                    # Compare true vs called CNVs
                    deltas.append(np.sqrt(np.mean((inferred_tree.outputs['inferred_cnvs'] - true_cn) ** 2)))
                    baseline_deltas.append(np.sqrt(np.mean((np.ones(true_cn.shape) * 2 - true_cn) ** 2)))
                    n_clusters.append(len(np.unique(inferred_tree.outputs['inferred_cnvs'], axis=0)))
                    true_n_clusters.append(len(np.unique(true_cn, axis=0)))

                    cnvs_rows.append(dict(cnv_delta=deltas[-1], method='SCICoNE',
                                        rep=rep, tumor_fraction=tumor_fraction, n_cells=N))
                    cnvs_rows.append(dict(cnv_delta=baseline_deltas[-1], method='Diploid',
                                        rep=rep, tumor_fraction=tumor_fraction, n_cells=N))

                    clusters_rows.append(dict(clusters_delta=n_clusters[-1] - true_n_clusters[-1], method='SCICoNE',
                                            rep=rep, tumor_fraction=tumor_fraction, n_cells=N))

                except Exception as e:
                    # Record the failed iteration
                    failed_iterations.append((N, tumor_fraction, rep, str(e)))
                    print(f"Failed iteration: N={N}, tumor_fraction={tumor_fraction}, rep={rep}")
                    print(f"Error: {e}")

    # After the loops, you can print out which iterations failed
    if failed_iterations:
        print("\nFailed iterations:")
        for fail in failed_iterations:
            print(f"N={fail[0]}, tumor_fraction={fail[1]}, rep={fail[2]} -> Error: {fail[3]}")
    else:
        print("All iterations completed successfully.")

    cnvs_deltas = pd.DataFrame(cnvs_rows)
    num_clusters = pd.DataFrame(clusters_rows)

    ax = sns.catplot(x="tumor_fraction", y="cnv_delta", hue="method", col="n_cells",
                    data=cnvs_deltas, palette="Dark2", kind='box')
    plt.show()

    ax = sns.catplot(x="tumor_fraction", y="clusters_delta", hue="method", col="n_cells",
                    data=num_clusters, palette="Dark2", kind='box')
    plt.show()
    
if __name__ == '__main__':
    main()
   