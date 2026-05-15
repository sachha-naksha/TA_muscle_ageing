import scvelo as scv
import scanpy as sc
import velocyto as vcy
import cellrank as cr
import loompy as lp
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import anndata as ad


def read_loom_file(loom_file):
    return scv.read(loom_file, cache=True)

def main(args):
    scv.settings.verbosity = 3
    cr.settings.verbosity = 2

    # Load the anndata object and loom files
    adata_rna = sc.read_h5ad(args.anndata)
    loom_files = args.loom_files

    # Use multiprocessing to read loom files in parallel
    with mp.Pool(processes=64) as pool:
        ldata_list = pool.map(read_loom_file, loom_files)

    # Rename barcodes in order to merge
    suffixes = ['_15', '_11']
    for ldata, suffix in zip(ldata_list, suffixes):
        barcodes = [bc.split(':')[1] for bc in ldata.obs.index.tolist()]
        barcodes = [bc[0:len(bc)-1] + suffix for bc in barcodes]
        ldata.obs.index = barcodes
        ldata.var_names_make_unique()

    # Concatenate the looms
    ldata = ldata_list[0].concatenate(ldata_list[1:])

    # Merge matrices into the original adata object {the order of cells for merging matters > for matching?}
    adata_with_vcy = scv.utils.merge(adata_rna, ldata)

    # Save the new adata object with loom data
    adata_with_vcy.write(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get loom plus anndata")
    parser.add_argument("--anndata", type=str, required=True, help="Path to the anndata file.")
    parser.add_argument("--loom_files", type=str, nargs='+', required=True, help="Paths to the loom files.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file.")
    
    args = parser.parse_args()
    main(args)
