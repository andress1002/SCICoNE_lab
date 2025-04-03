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
    install_path = '/cluster/work/bewi/members/andress/SCICoNE/build/'
    temporary_outpath = './'

    seed = 42 
    np.random.seed(seed)

    # Create SCICoNE object
    sci = scicone.SCICoNE(install_path, temporary_outpath, verbose=False)
    scrna_path = "/home/andress/pylabs/SCICoNE_lab/rna_imp/clonealign-processed-data/SA501/10X/20171026_SA501X2XB00096/outs/filtered_gene_bc_matrices/hg19/"

    mat = io.mmread(f'{scrna_path}/matrix.mtx').toarray()
    barcodes = pd.read_csv(f'{scrna_path}/barcodes.tsv', sep='\t', header=None)
    genes = pd.read_csv(f'{scrna_path}/genes.tsv', sep='\t', header=None)

    #PRE PROCESSING

    #Matrix gene by cells required by SCICoNE
    adata = anndata.AnnData(pd.DataFrame(mat.T, index=barcodes[0].values, columns=genes[0].values))
    adata.var_names_make_unique()
    adata.var['ensembl_gene_id'] = genes[0].values
    adata.var['gene_id'] = genes[1].values


    #basic filtering

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    #filter 

    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < 2500, :] #doublets 
    adata = adata[adata.obs.pct_counts_mt < 5, :] #mitochondrial quality control
    
    adata.raw = adata #checkpoint 
    #save checkpoint
    adata.write_h5ad('checkpoint.h5ad')

    # Further QC filtering
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

    sc.pp.scale(adata, max_value=10)
    


    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata)
    sc.tl.umap(adata)
    

    #Get gene anotations

    annot = sc.queries.biomart_annotations(
        "hsapiens",
        ["ensembl_gene_id", "start_position", "end_position", "chromosome_name"],
    ).set_index("ensembl_gene_id")

    annot = annot.reset_index()
    annot = annot.rename(columns={'start_position':'Start', 'end_position': 'End', 'chromosome_name': 'Chromosome'})

    gr_annotations = pyranges.from_dict(annot.to_dict())
        # Get gene coordinates
    df_annotations = gr_annotations.df.set_index('ensembl_gene_id')

    df_exp_annotations = df_annotations.loc[df_annotations.index.intersection(adata.var['ensembl_gene_id'])]\
                            .reset_index().rename(columns={'index':'ensembl_gene_id'})

    adata.var_names = adata.var['ensembl_gene_id'].values

    adata = adata[:,df_exp_annotations['ensembl_gene_id']]
    

    df_exp_annotations_sorted = df_exp_annotations.sort_values(by=['Chromosome', 'End'])


    # Create a clean copy with corrected chromosome names
    df_exp_annotations_2 = df_exp_annotations_sorted.copy()
    df_exp_annotations_2['Chromosome'] = df_exp_annotations_2['Chromosome'].astype(str).str.strip()

    standard_chromosomes = [str(i) for i in range(1, 23)] + ["X", "Y"]
    filtered_df = df_exp_annotations_2[df_exp_annotations_2['Chromosome'].isin(standard_chromosomes)]

    idx_max = filtered_df.groupby('Chromosome')['End'].idxmax()

    # Sort by chromosome order first, then by End
    chromosome_stops_df = filtered_df.loc[idx_max, ['Chromosome', 'End', 'ensembl_gene_id']].reset_index(drop=True)

    # Create custom chromosome ordering
    chromosome_order = {str(i): i for i in range(1, 23)}
    chromosome_order.update({'X': 23, 'Y': 24})
    chromosome_stops_df['chr_order'] = chromosome_stops_df['Chromosome'].map(chromosome_order)

    # Sort by chromosome order
    chromosome_stops_df = chromosome_stops_df.sort_values(by='chr_order').drop('chr_order', axis=1).reset_index(drop=True)

    chromosome_stops = []

    for gene_id in chromosome_stops_df['ensembl_gene_id']:
        index = np.where(adata.var_names == gene_id)[0][0]
        chromosome_stops.append(int(index))
        
    print(chromosome_stops)
    print(len(chromosome_stops))


    chr_var_names = dict()
    for chromosome in df_exp_annotations_sorted['Chromosome'].unique():
        chr_var_names[chromosome] = df_exp_annotations_sorted.query(f' Chromosome=="{chromosome}" ')\
                                        .sort_values('Start')['ensembl_gene_id'].values



    sci.detect_breakpoints(adata, window_size=100, threshold=3, input_breakpoints=chromosome_stops)

    scicone.plotting.plot_matrix(adata, bps=sci.bps['segmented_regions'],
    chr_stops_dict=chromosome_stops,
    cbar_title='Normalized\n  counts', vmax=2, cluster=False)

    sci.learn_tree(ful=False)
    fig = sci.plot_tree()
    fig.savefig('scicone_tree.png', dpi=300, bbox_inches='tight')
    plt.show()


    #Apply SCICoNE and passing chromosome as known regions

#     scicone.plotting.plot_matrix(gene_by_cells, bps=sci.bps['segmented_regions'],
#                                 chr_stops_dict=sci.data['filtered_chromosome_stops'],
#              
#                    cbar_title='Normalized\n  counts', vmax=2, cluster=False)


if __name__ == '__main__':
    main()