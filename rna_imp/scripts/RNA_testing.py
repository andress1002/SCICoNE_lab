# Import matplotlib and set non-interactive backend first
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot

#import libs
import scicone
import numpy as np
import pickle
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import io
import scanpy as sc
import anndata
import pyranges
import gseapy as gp
import json


def main():

    # Set up SCICoNE
    install_path = '/cluster/work/bewi/members/andress/SCICoNE_lab/build/'
    temporary_outpath = './'
    temporary_p = '/cluster/work/bewi/members/andress/SCICoNE_lab/rna_imp/output2'
    adatas_path = '/cluster/work/bewi/members/andress/SCICoNE_lab/rna_imp/adatas'
    # Make sure the output directory exists
    os.makedirs(temporary_outpath, exist_ok=True)
    
    # Add debug message
    print(f"Output will be saved to: {temporary_outpath}")

    seed = 42 
    np.random.seed(seed)
    
    # Rest of your code remains the same...

    # Create SCICoNE object
    sci = scicone.SCICoNE(install_path, temporary_outpath, verbose=False)

    gr_annotations = pd.read_csv('/cluster/work/bewi/members/andress/SCICoNE_lab/rna_imp/gr_annotations.csv')


    # Load the JSON file as a dictionary
    json_file_path = '/cluster/work/bewi/members/andress/SCICoNE_lab/rna_imp/chromosome_stops.json'
    with open(json_file_path, 'r') as file: 
        chromosome_stops = json.load(file)

# Use chromosome_stops directly without further processing
    chromosome_stops_list = sorted(list(chromosome_stops.values()))

    import matplotlib.pyplot as plt
    file_list = ['adata_clusters_median', 'adata_clusters_mean', 'adata_clusters_sum', 'clusters_mean_transformed',
                'clusters_sum_transformed', 'clusters_median_transformed']
    # Initialize a dictionary to store results
    breakpoints_data = {}

    for cluster_file in file_list:
        breakpoints_data[cluster_file] = []
        for window_size in range(3, 300, 1):
            try:
                adata = anndata.read_h5ad(f'{adatas_path}/{cluster_file}.h5ad')
                data = adata.X
                # Run SCICoNE analysis
                matrix_plot_filename = f"{temporary_p}/matrix_plot_{cluster_file}_{window_size}.png"
                bps = sci.detect_breakpoints(data=data, window_size=window_size, threshold=3, input_breakpoints=chromosome_stops_list)
                breakpoints_data[cluster_file].append((window_size, len(bps['segmented_regions'])))
                scicone.plotting.plot_matrix(data, bps = bps['segmented_regions'],
                                                    chr_stops_dict=chromosome_stops,
                                                    cbar_title='Normalized\n  counts', vmax=2, cluster=False)
                if window_size in [3, 5, 10, 20, 50, 100, 200, 300]:
                    plt.title(f"Matrix Plot for {cluster_file} with Window Size {window_size}")    
                    plt.savefig(matrix_plot_filename, dpi=300, bbox_inches='tight')
                    plt.close()

                    #Learn and save the tree plot
                    tree_plot_filename = f"{temporary_p}/tree_plot_{cluster_file}_{window_size}.png"
                    #Render the tree plot using graphviz's render method
                    inferred_tree = sci.learn_tree(data, bps["segmented_region_sizes"], n_reps = 4, seed = seed, max_tries = 1)
                    inferred_tree.plot_tree(gene_labels=True, node_labels=True, node_sizes=True, event_fontsize=8, nodesize_fontsize=10)

            
                    plt.savefig(tree_plot_filename, dpi=300, bbox_inches='tight')
                    plt.close()  # Close the figure to free memory
                    print(f"Tree plot saved to {tree_plot_filename}")

            except Exception as e:
                print(f"Error processing {cluster_file} with window size {window_size}: {e}")
                continue

    # Plot the results
    for cluster_file, data_points in breakpoints_data.items():
        if data_points:
            window_sizes, breakpoints_counts = zip(*data_points)
            plt.figure(figsize=(8, 6))
            plt.plot(window_sizes, breakpoints_counts, marker='o', label=cluster_file)
            plt.title(f'Breakpoints Detected vs Window Size for {cluster_file}')
            plt.xlabel('Window Size')
            plt.ylabel('Number of Breakpoints Detected')
            plt.legend()
            plt.grid(True)
            plt.show()
            #save plots
            plt.savefig(f"{temporary_p}/breakpoints_plot_{cluster_file}.png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()