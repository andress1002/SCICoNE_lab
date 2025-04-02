import os
import numpy as np
import scanpy as sc
import logging

def save_adata_objects(adata, output_dir="output"):
    """
    Save copies of adata.raw and smoothed average adata locally.
    
    Parameters:
        adata (AnnData): The AnnData object containing single-cell RNA data.
        output_dir (str): Directory to save the files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save adata.raw if it exists
    if adata.raw is not None:
        adata_raw_copy = adata.raw.to_adata()
        adata_raw_copy.write(os.path.join(output_dir, "adata_raw.h5ad"))
        print("Saved adata.raw as adata_raw.h5ad")
    
    # Save the smoothed average adata
    adata.write(os.path.join(output_dir, "adata_smoothed.h5ad"))
    print("Saved smoothed adata as adata_smoothed.h5ad")

def cluster_cells(adata, resolution=1.0):
    """
    Cluster cells in a copy of the AnnData object using the Leiden algorithm.
    
    Parameters:
        adata (AnnData): The AnnData object containing single-cell RNA data.
        resolution (float): Resolution parameter for the Leiden clustering.
    
    Returns:
        AnnData: A new AnnData object (adata_cluster) with clustering results added.
    """
    # Create a copy of the adata object
    adata_cluster = adata.copy()
    
    # Perform PCA
    sc.tl.pca(adata_cluster)
    
    # Compute the neighborhood graph
    sc.pp.neighbors(adata_cluster, n_neighbors=15, use_rep='X_pca')
    
    # Perform clustering using the Leiden algorithm
    sc.tl.leiden(adata_cluster, resolution=resolution)
    
    # Add clustering results to adata_cluster.obs
    print("Clustering completed. Results stored in adata_cluster.obs['leiden']")
    return adata_cluster

def smooth_expression(adata, var_names=None, window_size=10, clip=3):
    """
    Apply expression smoothing to an AnnData object.
    
    Parameters:
        adata (AnnData): The AnnData object containing gene expression data.
        var_names (list, dict, or None): Gene names to include in smoothing. If dict, keys are chromosome names and values are gene lists.
        window_size (int): Size of the sliding window for smoothing.
        clip (float): Threshold for clipping extreme values.
        
    Returns:
        None: Adds 'scaled' and 'smoothed' layers to the adata object.
    """
    logger = logging.getLogger(__name__)
    
    # Scale the data first
    mat = sc.pp.scale(adata.X, copy=True)
    adata.layers["scaled"] = mat
    
    # For stand-alone smoothing (no chromosome regions)
    if var_names is None or not isinstance(var_names, dict):
        # Apply clipping
        clipped_mat = np.clip(mat, -np.abs(clip), np.abs(clip))
        
        # Apply smoothing
        half_window = int(window_size / 2)
        smoothed = np.zeros(clipped_mat.shape)
        
        for ii in range(clipped_mat.shape[1]):
            left = max(0, ii - half_window)
            right = min(ii + half_window, clipped_mat.shape[1] - 1)
            if left != right:
                smoothed[:, ii] = np.mean(clipped_mat[:, left:right], axis=1)
        
        adata.layers["smoothed"] = smoothed
    
    # For chromosome-specific smoothing
    else:
        smoothed_mat = []
        
        for region in var_names:
            region_slice = adata[:, var_names[region]].layers["scaled"]
            
            # Apply clipping
            clipped = np.clip(region_slice, -np.abs(clip), np.abs(clip))
            
            # Apply smoothing
            half_window = int(window_size / 2)
            smoothed = np.zeros(clipped.shape)
            
            for ii in range(clipped.shape[1]):
                left = max(0, ii - half_window)
                right = min(ii + half_window, clipped.shape[1] - 1)
                if left != right:
                    smoothed[:, ii] = np.mean(clipped[:, left:right], axis=1)
            
            smoothed_mat.append(smoothed)
        
        adata.layers["smoothed"] = np.concatenate(smoothed_mat, axis=1)
    
    logger.info(f'Smoothed gene expression is stored in `adata.layers["smoothed"]`')