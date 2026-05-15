import scanpy as sc
import numpy as np
import pandas as pd
import gseapy
import matplotlib.pyplot as plt
import warnings
import os
import urllib.request


def map_ensembl_to_gene_name(adata, gtf_file_path):
    """
    Maps the ensembl gene ids in anndata var_names to the gene names using gtf file
    Parameters:
    adata (AnnData): The AnnData object containing the gene ids in var_names.
    gtf_file_path (str): The path to the gtf file used for mapping.
    Returns:
    adata (AnnData): The AnnData object with the gene names in var_names.
    """
    gtf = pd.read_csv(gtf_file_path, sep="\t", header=None, comment="#")
    gtf_genes = gtf[gtf[2] == "gene"].copy()  # only select rows with gene annotation, and skip transcript, exon annotations
    gtf_genes["ensembl_id"] = gtf_genes[8].str.extract(r'gene_id "([^"]+)"')
    gtf_genes["gene_name"] = gtf_genes[8].str.extract(r'gene_name "([^"]+)"')
    # Create a dictionary for mapping, ensuring scalar values (and not index objects)
    mapping_dict = dict(zip(gtf_genes["ensembl_id"].values, gtf_genes["gene_name"].values))
    new_var_names = pd.Index([mapping_dict.get(name, name) for name in adata.var_names])
    adata.var_names = new_var_names
    return adata

# if __name__ == "__main__":
#     # load the anndata object for mice hindlimb
#     adata = sc.read_h5ad("/ocean/projects/cis240075p/asachan/datasets/TA_muscle/human_SKM_ageing_atlas_2024/mice_hindlimb.h5ad")
#     # subset the anndata to have only cell types of myofibers
#     myofiber_adata = adata[adata.obs["cell_type"].isin(["skeletal muscle satellite stem cell", "type II muscle cell", "type IIa muscle cell", "type IIb muscle cell", "type I muscle cell"])]
#     print(myofiber_adata.var_names)
#     myofiber_adata = map_ensembl_to_gene_name(myofiber_adata, "/ocean/projects/cis240075p/asachan/datasets/mouse_genome_files/refdata-gex-mm10-2020-A/genes/genes.gtf")
#     print(myofiber_adata.var_names)




