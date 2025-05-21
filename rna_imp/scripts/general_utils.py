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


#Get annotation


def get_annotation(adata, temporary_outpath = './'):

    annot = sc.queries.biomart_annotations(
            "hsapiens",
            ["ensembl_gene_id", "start_position", "end_position", "chromosome_name"],
        ).set_index("ensembl_gene_id")

    annot = annot.reset_index()
    annot = annot.rename(columns={'start_position':'Start', 'end_position': 'End', 'chromosome_name': 'Chromosome'})

    gr_annotations = pyranges.from_dict(annot.to_dict())

    df_annotations = gr_annotations.df.set_index('ensembl_gene_id')

    df_exp_annotations = df_annotations.loc[df_annotations.index.intersection(adata.var['ensembl_gene_id'])]\
                            .reset_index().rename(columns={'index':'ensembl_gene_id'})

    adata.var_names = adata.var['ensembl_gene_id'].values

    adata = adata[:,df_exp_annotations['ensembl_gene_id']]

    df_exp_annotations_sorted = df_exp_annotations.sort_values(by=['Chromosome', 'End'])

    df_exp_annotations_sorted.head()

    merged_var = adata.var.merge(
        df_annotations[['Chromosome', 'Start', 'End']],
        how='left',
        left_on='ensembl_gene_id',
        right_index=True
    )

    # Ensure the index matches the original var index
    merged_var.index = adata.var.index
    adata.var = merged_var

    return adata

def extract_chromosome_stops_dict(adata, temporary_outpath = './'):
    """
    Extracts the stop positions (max end positions) for each chromosome from an AnnData object
    and saves them as a dictionary with 'Start', 'End', and chromosome numbers as keys.

    :param adata: AnnData object containing gene annotations in `adata.var`.
    :param temporary_outpath: Path to save the stop indexes JSON file.
    :return: Sorted dictionary with 'Start', 'End', and chromosome numbers as keys and stop positions as values.
    """
    # Ensure chromosome names are clean
    adata.var['Chromosome'] = adata.var['Chromosome'].astype(str).str.strip()

    # Filter for standard chromosomes
    standard_chromosomes = [str(i) for i in range(1, 23)] + ["X", "Y"]
    filtered_var = adata.var[adata.var['Chromosome'].isin(standard_chromosomes)]

    # Sort chromosomes
    sorted_chromosomes = sort_chromosomes(filtered_var['Chromosome'].unique())

    # Find the max end position for each chromosome
    chromosome_stops = {}
    for chromosome in sorted_chromosomes:
        chromosome_data = filtered_var[filtered_var['Chromosome'] == chromosome]
        if not chromosome_data.empty:
            max_end_idx = chromosome_data['End'].idxmax()
            max_gene_id = chromosome_data.loc[max_end_idx, 'ensembl_gene_id']
            column_position = int(np.where(adata.var['ensembl_gene_id'] == max_gene_id)[0][0])
            chromosome_stops[chromosome] = column_position

    # Add 'Start' and 'End' keys
    chromosome_stops['Start'] = 0
    chromosome_stops['End'] = adata.shape[1] - 1

    # Sort the dictionary by chromosome order
    chromosome_order = {str(i): i for i in range(1, 23)}
    chromosome_order.update({'X': 23, 'Y': 24, 'Start': 0, 'End': 25})
    chromosome_stops = dict(sorted(chromosome_stops.items(), key=lambda x: chromosome_order.get(x[0], float('inf'))))

    # Save the chromosome stops dictionary to a JSON file
    with open(os.path.join(temporary_outpath, f'chromosome_stops_{adata}.json'), 'w') as f:
        json.dump(chromosome_stops, f)

    print("Chromosome stops saved to JSON:", chromosome_stops)
    return chromosome_stops

def sort_chromosomes(chromosome_list):
    """
    Sorts a list of unordered chromosome names.
    :param chromosome_list: list of unordered characters denoting chromosomes '1', '2', ..., 'X', 'Y'.
    """
    # Replace X and Y with 23 and 24
    sorted_chromosome_list = np.array(chromosome_list)
    sorted_chromosome_list[np.where(sorted_chromosome_list == "X")[0]] = 23
    sorted_chromosome_list[np.where(sorted_chromosome_list == "Y")[0]] = 24

    # Convert everything to integer
    sorted_chromosome_list = sorted_chromosome_list.astype(int)

    # Sort
    sorted_chromosome_list = np.sort(sorted_chromosome_list)

    # Convert back to string
    sorted_chromosome_list = sorted_chromosome_list.astype(str)
    sorted_chromosome_list[np.where(sorted_chromosome_list == "23")[0]] = "X"
    sorted_chromosome_list[np.where(sorted_chromosome_list == "24")[0]] = "Y"

    return sorted_chromosome_list