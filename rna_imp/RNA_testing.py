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

    seed = 42 
    np.random.seed(seed)

    # Create SCICoNE object
    sci = scicone.SCICoNE(install_path, temporary_outpath, verbose=False)
    scrna_path = "/cluster/work/bewi/members/andress/SCICoNE_lab/rna_imp/clonealign-processed-data/SA501/10X/20171026_SA501X2XB00096/outs/filtered_gene_bc_matrices/hg19/"
    
    gr_annotations = pd.read_csv('/cluster/work/bewi/members/andress/SCICoNE_lab/rna_imp/gr_annotations.csv')

    df_annotations = gr_annotations.set_index('ensembl_gene_id')

    df_exp_annotations = df_annotations.loc[df_annotations.index.intersection(adata.var['ensembl_gene_id'])]\
                            .reset_index().rename(columns={'index':'ensembl_gene_id'})

    adata.var_names = adata.var['ensembl_gene_id'].values

    adata = adata[:,df_exp_annotations['ensembl_gene_id']]
    

    df_exp_annotations_sorted = df_exp_annotations.sort_values(by=['Chromosome', 'End'])


#Read the cluster files

    mean_clusters = anndata.read_h5ad(f'{temporary_outpath}/mean_clusters_sorted.h5ad')
    median_clusters = anndata.read_h5ad(f'{temporary_outpath}/median_clusters_sorted.h5ad')
    sum_clusters = anndata.read_h5ad(f'{temporary_outpath}/sum_clusters_sorted.h5ad')


    chr_var_names = dict()
    for chromosome in df_exp_annotations_sorted['Chromosome'].unique():
        chr_var_names[chromosome] = df_exp_annotations_sorted.query(f' Chromosome=="{chromosome}" ')\
                                        .sort_values('Start')['ensembl_gene_id'].values

    with open(f'{temporary_outpath}/chromosome_stops.json', 'r') as f:
        chromosome_stops = json.load(f)

    for cluster_file in [mean_clusters, median_clusters, sum_clusters]:
        try:
            cluster_file.var['Chromosome'] = cluster_file.var['Chromosome'].astype(str)
            sorted_var = cluster_file.var.sort_values(by=['Chromosome', 'End'])

            # Reorder adata.X columns based on the sorted var index
            cluster_file = cluster_file[:, sorted_var.index]
            sci.detect_breakpoints(cluster_file, window_size=100, threshold=3, input_breakpoints = sorted(list(chromosome_stops.values())))

            scicone.plotting.plot_matrix(cluster_file, bps=sci.bps['segmented_regions'],
            chr_stops_dict=chromosome_stops,
            cbar_title='Normalized\n  counts', vmax=2, cluster=False)

            sci.learn_tree(ful=False)
            fig = sci.plot_tree()
            fig.savefig('scicone_tree.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"An error occurred while processing the cluster file: {e}")
            continue


    #Apply SCICoNE and passing chromosome as known regions

#     scicone.plotting.plot_matrix(gene_by_cells, bps=sci.bps['segmented_regions'],
#                                 chr_stops_dict=sci.data['filtered_chromosome_stops'],
#              
#                    cbar_title='Normalized\n  counts', vmax=2, cluster=False)


if __name__ == '__main__':
    main()