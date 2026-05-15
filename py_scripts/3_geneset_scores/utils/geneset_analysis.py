import sys
import os
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import gseapy as gp
import decoupler as dc
import scanpy as sc

from pathlib import Path
from itertools import chain, repeat

##### DEG ANALYSIS UTILS #####

def perform_deg_analysis(adata_subset, output_dir, layer='log1p_norm_cb', condition_column='condition', reference='WT', comparison='KO', 
                        sex_label=''):
    """
    Perform differential gene expression analysis between conditions
    
    Parameters:
    -----------
    adata_subset : AnnData object
        Subsetted data (e.g., adata_male or adata_female)
    output_dir : str
        Directory to save the results
    layer : str
        Counts layer to use for the analysis
    condition_column : str
        Column name in adata.obs that contains condition information
    reference : str
        Reference condition (e.g., 'WT')
    comparison : str
        Comparison condition (e.g., 'KO')
    sex_label : str
        Label for the sex subset ('Male' or 'Female')
    """
    
    print(f"=== Performing DEG Analysis: {sex_label} {comparison} vs {reference} ===")
    
    # Make a copy to avoid modifying original data
    adata_work = adata_subset.copy()
    
    # Ensure we have the right conditions
    conditions = adata_work.obs[condition_column].unique()
    print(f"Available conditions: {conditions}")
    
    if reference not in conditions or comparison not in conditions:
        print(f"Warning: {reference} or {comparison} not found in {condition_column}")
        return None, None
    
    # Filter to only include the two conditions of interest
    mask = adata_work.obs[condition_column].isin([reference, comparison])
    adata_filtered = adata_work[mask].copy()
    
    print(f"Cells in analysis - {reference}: {(adata_filtered.obs[condition_column] == reference).sum()}")
    print(f"Cells in analysis - {comparison}: {(adata_filtered.obs[condition_column] == comparison).sum()}")
    
    # Perform differential expression using Wilcoxon rank-sum test
    sc.tl.rank_genes_groups(
        adata_filtered,
        groupby=condition_column,
        groups=[comparison],  # Compare KO
        reference=reference,   # Against WT
        method='wilcoxon',
        pts=True,  # Calculate percentage of cells expressing the gene
        tie_correct=True,
        use_raw=False,
        layer=layer
    )
    
    # Extract results
    result_df = sc.get.rank_genes_groups_df(adata_filtered, group=comparison)
    
    # Add additional statistics
    result_df['log2FC'] = result_df['logfoldchanges']
    result_df['-log10(pval)'] = -np.log10(result_df['pvals'])
    result_df['significant'] = (result_df['pvals_adj'] < 0.05) & (np.abs(result_df['logfoldchanges']) > 0.5)
    
    print(f"Total genes analyzed: {len(result_df)}")
    print(f"Significantly upregulated genes (padj < 0.05, |log2FC| > 0.5): {(result_df['significant'] & (result_df['log2FC'] > 0)).sum()}")
    print(f"Significantly downregulated genes (padj < 0.05, |log2FC| > 0.5): {(result_df['significant'] & (result_df['log2FC'] < 0)).sum()}")
    
    return adata_filtered, result_df

# filter the degs using thresholds
def filter_degs(degs, log2fc_thresh=0.05, pval_thresh=0.01):
    filtered_degs = degs[(degs['logfoldchanges'].abs() > log2fc_thresh) & (degs['pvals_adj'] < pval_thresh)]
    # sort the rows by absolute logfoldchanges
    filtered_degs = filtered_degs.sort_values(by='logfoldchanges', key=lambda x: x.abs(), ascending=False)
    return filtered_degs

def get_common_degs(degs_males, degs_females):
    # get the names of the common DEGs between sexes (just names column)
    common_deg_names = degs_males[degs_males['names'].isin(degs_females['names'])]['names'].unique()
    # slice the rows of the common degs df from the males df
    common_deg_df_male = degs_males[degs_males['names'].isin(common_deg_names)]
    common_deg_df_female = degs_females[degs_females['names'].isin(common_deg_names)]
    common_degs = pd.merge(common_deg_df_male, common_deg_df_female, on='names', how='inner', suffixes=('_male', '_female'))
    return common_degs

def get_opposite_sign_degs(common_degs):
    # get the degs that have opposite signs in common male and female degs
    opposite_sign_degs = common_degs[common_degs['logfoldchanges_male'] * common_degs['logfoldchanges_female'] < 0]
    return opposite_sign_degs

def get_same_sign_degs(common_degs):
    # get the degs that have same signs in common male and female degs
    same_sign_degs = common_degs[common_degs['logfoldchanges_male'] * common_degs['logfoldchanges_female'] > 0]
    return same_sign_degs

def visualize_deg_results(result_df, sex_label, output_dir):
    """
    Create comprehensive visualizations for DEG results
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import os
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Differential Gene Expression Analysis - {sex_label} Cells (KO vs WT)', fontsize=16, fontweight='bold')
    
    # 1. Volcano plot
    ax1 = axes[0, 0]
    
    # Color points based on significance and fold change
    colors = []
    for _, row in result_df.iterrows():
        if row['pvals_adj'] < 0.05 and row['log2FC'] > 0.5:
            colors.append('red')  # Upregulated
        elif row['pvals_adj'] < 0.05 and row['log2FC'] < -0.5:
            colors.append('blue')  # Downregulated
        else:
            colors.append('gray')  # Not significant
    
    scatter = ax1.scatter(result_df['log2FC'], result_df['-log10(pval)'], 
                         c=colors, alpha=0.6, s=20)
    
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=-0.5, color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Log2 Fold Change (KO vs WT)')
    ax1.set_ylabel('-Log10(p-value)')
    ax1.set_title('Volcano Plot')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Upregulated'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Downregulated'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Not significant')]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # 2. MA plot - Use scores for x-axis since we only have pct_nz_group
    ax2 = axes[0, 1]
    
    # Use scores as the mean expression measure
    mean_expr = result_df['scores']
    xlabel_text = 'Gene Scores'
    
    ax2.scatter(mean_expr, result_df['log2FC'], c=colors, alpha=0.6, s=20)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.5)
    ax2.set_xlabel(xlabel_text)
    ax2.set_ylabel('Log2 Fold Change (KO vs WT)')
    ax2.set_title('MA Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Top upregulated genes
    ax3 = axes[1, 0]
    top_up = result_df[(result_df['significant']) & (result_df['log2FC'] > 0)].head(10)
    if len(top_up) > 0:
        ax3.barh(range(len(top_up)), top_up['log2FC'], color='red', alpha=0.7)
        ax3.set_yticks(range(len(top_up)))
        ax3.set_yticklabels(top_up['names'], fontsize=10)
        ax3.set_xlabel('Log2 Fold Change')
        ax3.set_title('Top 10 Upregulated Genes')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No significantly\nupregulated genes', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Top 10 Upregulated Genes')
    
    # 4. Top downregulated genes
    ax4 = axes[1, 1]
    top_down = result_df[(result_df['significant']) & (result_df['log2FC'] < 0)].head(10)
    if len(top_down) > 0:
        ax4.barh(range(len(top_down)), top_down['log2FC'], color='blue', alpha=0.7)
        ax4.set_yticks(range(len(top_down)))
        ax4.set_yticklabels(top_down['names'], fontsize=10)
        ax4.set_xlabel('Log2 Fold Change')
        ax4.set_title('Top 10 Downregulated Genes')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No significantly\ndownregulated genes', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Top 10 Downregulated Genes')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/DEG_analysis_{sex_label.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved at: {filename}")
    
    return fig

def create_volcano_plot(degs, cell_type, top_n=10, xlim=(-5, 5), title='', save=True):
    """
    Create volcano plot for DEGs
    
    Parameters:
    -----------
    degs : DataFrame
        DEG results from scanpy
    cell_type : str
        Name of cell type
    top_n : int
        Number of top genes to label
    title : str
        Title of the plot
    save : bool
        Whether to save the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate -log10(p-value)
    degs['-log10_padj'] = -np.log10(degs['pvals_adj'].replace(0, 1e-300))  # Handle p=0
    
    # Define significance thresholds
    pval_thresh = 0.05
    fc_thresh = 0.5
    
    # Classify genes
    degs['significant'] = 'Not Significant'
    degs.loc[
        (degs['pvals_adj'] < pval_thresh) & (degs['logfoldchanges'] > fc_thresh),
        'significant'
    ] = 'Upregulated'
    degs.loc[
        (degs['pvals_adj'] < pval_thresh) & (degs['logfoldchanges'] < -fc_thresh),
        'significant'
    ] = 'Downregulated'
    
    # Color map
    colors = {'Upregulated': '#d62728', 'Downregulated': '#1f77b4', 'Not Significant': '#7f7f7f'}
    
    # Plot points
    for category, color in colors.items():
        mask = degs['significant'] == category
        ax.scatter(
            degs.loc[mask, 'logfoldchanges'],
            degs.loc[mask, '-log10_padj'],
            c=color,
            label=category,
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
    
    # Add threshold lines
    ax.axhline(-np.log10(pval_thresh), color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(fc_thresh, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(-fc_thresh, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Label top genes
    sig_degs = degs[degs['significant'] != 'Not Significant'].copy()
    if xlim:
        sig_degs = sig_degs[(sig_degs['logfoldchanges'] >= xlim[0]) & 
                            (sig_degs['logfoldchanges'] <= xlim[1])]
    if len(sig_degs) > 0:
        # Top upregulated
        top_up = sig_degs[sig_degs['logfoldchanges'] > 0].nlargest(top_n, 'logfoldchanges')
        # Top downregulated
        top_down = sig_degs[sig_degs['logfoldchanges'] < 0].nsmallest(top_n, 'logfoldchanges')
        # Combine
        top_genes = pd.concat([top_up, top_down])
        
        for idx, row in top_genes.iterrows():
            ax.annotate(
                row['names'],
                xy=(row['logfoldchanges'], row['-log10_padj']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )
    
    # Count genes
    n_up = (degs['significant'] == 'Upregulated').sum()
    n_down = (degs['significant'] == 'Downregulated').sum()
    n_total = len(degs)
    
    # Labels and title
    ax.set_xlabel('Log2 Fold Change (KO / WT)', fontsize=12, fontweight='bold')
    ax.set_ylabel('-Log10(Adjusted P-value)', fontsize=12, fontweight='bold')
    ax.set_title(f'{cell_type} + {title}\nKO vs WT', fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis limits
    if xlim:
        ax.set_xlim(xlim)

    # Add counts to legend
    legend_labels = [
        f'Upregulated ({n_up})',
        f'Downregulated ({n_down})',
        f'Not Significant ({n_total - n_up - n_down})'
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    if save:
        filename = f"volcano_KO_vs_WT_{cell_type.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    
    plt.show()
    plt.close()
    
    return fig, ax

def save_deg_results(result_df, sex_label, output_dir):
    """
    Save DEG results to CSV files
    """    
    # Filter for significant results
    sig_results = result_df[result_df['significant']].copy()
    #sort the rows by pvals_adj
    sig_results = sig_results.sort_values('pvals_adj')
    # separate the rows by upregualtion or downregulation and save to different files
    sig_results_up = sig_results[sig_results['log2FC'] > 0]
    sig_results_down = sig_results[sig_results['log2FC'] < 0]
    if len(sig_results_up) > 0:
        sig_results_file_up = f'{output_dir}/DEG_results_{sex_label.lower()}_up.csv'
        sig_results_up.to_csv(sig_results_file_up, index=False)
        print(f"Significant DEG results saved at: {sig_results_file_up}")
    if len(sig_results_down) > 0:
        sig_results_file_down = f'{output_dir}/DEG_results_{sex_label.lower()}_down.csv'
        sig_results_down.to_csv(sig_results_file_down, index=False)
        print(f"Significant DEG results saved at: {sig_results_file_down}")
    
    return sig_results

##### GSEA ANALYSIS UTILS #####

def convert_genes(gene_list, direction='human_to_mouse'):
    """
    Convert gene symbols between human and mouse formats.
    """
    direction = direction.lower()
    
    if direction in ['human_to_mouse', 'h2m']:
        # Human to mouse: capitalize (first letter uppercase, rest lowercase)
        return [gene.capitalize() for gene in gene_list]
    
    elif direction in ['mouse_to_human', 'm2h']:
        # Mouse to human: all uppercase
        return [gene.upper() for gene in gene_list]
    
    else:
        raise ValueError(
            f"Invalid direction: {direction}. "
            "Use 'human_to_mouse'/'h2m' or 'mouse_to_human'/'m2h'"
        )
        
def create_ranked_genelist(deg_df, log2fc_col='avg_log2FC', pval_col='p_val_adj', gene_col='gene_name', min_pval=1e-300):
    """
    Create a ranked gene list based on signed log2FC * -log10(adjusted p-value)
    """
    df = deg_df.copy()
    # Clip p-values
    df[pval_col] = df[pval_col].clip(lower=min_pval)
    # Calculate components and final score
    df['neg_log10_pval'] = -np.log10(df[pval_col])
    df['ranking_score'] = df[log2fc_col] * df['neg_log10_pval']
    # Sort by absolute ranking score
    ranked_df = df.sort_values('ranking_score', ascending=False)
    # Create final DataFrame with gene names as columns
    ranked_list = ranked_df.set_index(gene_col)[['ranking_score']].T
    return ranked_list

def simplify_term(term):
    # Convert to lowercase and split into words
    words = term.lower().replace('_', ' ').split()
    # Remove common words that don't add meaning
    stop_words = {'regulation', 'of', 'positive', 'negative', 'mediated', 'dependent', 
                 'activity', 'process', 'gobp', 'gomf', 'wp', 'reactome'}
    words = [w for w in words if w not in stop_words]
    return ' '.join(sorted(words))  # Sort words to match similar terms

def get_geneset_genes(df, msigdb_mice, output_file='geneset_genes.csv'):
    """
    Extract genes from genesets and save to CSV
    """
    # Create empty dictionary to store results
    geneset_genes = {}
    # For each pathway in the input dataframe
    for idx, row in df.iterrows():
        term = row['Term']
        # Get genes for this term from GSEA genesets (MSigDB)
        try:
            # Get leading edge genes if available
            genes = msigdb_mice[msigdb_mice['geneset'] == term]['genesymbol'].values
            geneset_genes[term] = {
                'Group': row['Group'],
                'Significance': row['-log10(pval)'],
                'Genes': ';'.join(genes)  # Join genes with semicolon for CSV
            }
        except KeyError:
            print(f"Warning: Could not find genes for {term}")
            continue
    
    # Convert to DataFrame
    result_df = pd.DataFrame.from_dict(geneset_genes, orient='index')
    result_df.index.name = 'Pathway'
    result_df.reset_index(inplace=True)
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    
    return result_df

from pathlib import Path

def gmt_to_decoupler(pth: Path) -> pd.DataFrame:
    """Parse a gmt file to a decoupler pathway dataframe."""
    from itertools import chain, repeat

    pathways = {}

    with Path(pth).open("r") as f:
        for line in f:
            name, _, *genes = line.strip().split("\t")
            pathways[name] = genes

    return pd.DataFrame.from_records(
        chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
        columns=["geneset", "genesymbol"],
    )

def gmt_to_decoupler_multiple_pathways(gmt_paths, geneset_name=None, genesymbol_name=None):
    """Parse multiple gmt files and return a combined decoupler pathway dataframe."""
    all_records = []
    for pth in gmt_paths:
        with Path(pth).open("r") as f:
            for line in f:
                name, _, *genes = line.strip().split("\t")
                all_records.extend(zip(repeat(name), genes))
    return pd.DataFrame.from_records(all_records, columns=[geneset_name, genesymbol_name])
    
##### CELL TYPE ANALYSIS UTILS #####

# Calculate percentages for each condition
def get_cell_type_percentages(adata, cell_type_label='cell_type'):
    wt_cells = adata[adata.obs['condition'] == 'WT'].obs[cell_type_label].value_counts(normalize=True) * 100
    ko_cells = adata[adata.obs['condition'] == 'KO'].obs[cell_type_label].value_counts(normalize=True) * 100
    df = pd.DataFrame({
        'Cell Type': wt_cells.index,
        'WT%': wt_cells.values.round(2),
        'ΔERCC1 KO%': ko_cells.values.round(2)
    })
    return df

def get_cell_type_percentages_by_sex(adata, cell_type_label='cell_type'):
    # Calculate percentages for each combination of condition and sex
    wt_female = adata[(adata.obs['condition'] == 'WT') & (adata.obs['sex'] == 'F')].obs[cell_type_label].value_counts(normalize=True) * 100
    wt_male = adata[(adata.obs['condition'] == 'WT') & (adata.obs['sex'] == 'M')].obs[cell_type_label].value_counts(normalize=True) * 100
    ko_female = adata[(adata.obs['condition'] == 'KO') & (adata.obs['sex'] == 'F')].obs[cell_type_label].value_counts(normalize=True) * 100
    ko_male = adata[(adata.obs['condition'] == 'KO') & (adata.obs['sex'] == 'M')].obs[cell_type_label].value_counts(normalize=True) * 100
    df = pd.DataFrame({
        'Cell Type': wt_female.index,
        'WT F%': wt_female.values.round(2),
        'ΔERCC1 KO F%': ko_female.values.round(2),
        'WT M%': wt_male.values.round(2),
        'ΔERCC1 KO M%': ko_male.values.round(2)
    })
    return df

##### SCORE ANALYSIS UTILS #####

def calculate_sc_score(data, up_genes=None, down_genes=None, condition_col='condition'):
    """
    Calculate geneset signature scores for each sample/single-cell based on gene-sets with directionality
    
    Parameters:
    -----------
    data : pd.DataFrame or AnnData
        Expression data. If DataFrame, genes should be rows and samples columns.
        If AnnData, will be converted automatically (cells x genes -> genes x cells)
    up_genes : list, optional
        List of upregulated genes in the signature
    down_genes : list, optional
        List of downregulated genes in the signature
    condition_col : str, optional
        Name of the condition column in adata.obs to include in results (default: 'condition')
    """
    import scipy.sparse as sp
    
    if up_genes is None and down_genes is None:
        raise ValueError("At least one of up_genes or down_genes must be provided")
    
    # Store condition info if AnnData
    condition_info = None
    
    # Check if input is AnnData and convert to DataFrame
    if hasattr(data, 'X'):  # Check if it's an AnnData object
        print("Converting AnnData to DataFrame...")
        
        # Extract condition information before conversion
        if condition_col in data.obs.columns:
            condition_info = data.obs[condition_col].copy()
            print(f"Extracted '{condition_col}' column from obs")
        
        # Convert sparse matrix to dense if needed
        if sp.issparse(data.X):
            expr_matrix = data.X.toarray()
        else:
            expr_matrix = data.X
        
        # Create DataFrame with genes as rows, cells as columns (transpose)
        data = pd.DataFrame(
            expr_matrix.T,  # Transpose: genes x cells
            index=data.var_names,  # Gene names
            columns=data.obs_names  # Cell/sample names
        )
        print(f"Converted to DataFrame with shape: {data.shape} (genes x cells)")
    
    # Set index to gene names if not already done
    if "NAME" in data.columns:
        data = data.set_index("NAME")
    if "Description" in data.columns:
        data = data.drop("Description", axis=1)
    
    # Check which genes are present in the expression data
    available_genes = set(data.index)
    
    # Process up-regulated genes
    if up_genes is not None:
        up_genes_set = set(up_genes)
        up_genes = list(up_genes_set.intersection(available_genes))
        print(f"Using {len(up_genes)} upregulated genes")
    
    # Process down-regulated genes
    if down_genes is not None:
        down_genes_set = set(down_genes)
        down_genes = list(down_genes_set.intersection(available_genes))
        print(f"Using {len(down_genes)} downregulated genes")
    
    # Check if we have enough genes to proceed
    if (up_genes is None or len(up_genes) == 0) and (
        down_genes is None or len(down_genes) == 0
    ):
        raise ValueError(
            "No genes from the gene sets were found in the expression data"
        )
    
    sample_names = data.columns
    expr_matrix = data.select_dtypes(include=[np.number])
    
    # Z-standardize the expression values across samples
    z_standardized = (
        expr_matrix - expr_matrix.mean(axis=1).values.reshape(-1, 1)
    ) / expr_matrix.std(axis=1).values.reshape(-1, 1)
    
    # Calculate scores for each sample
    scores = pd.Series(0, index=sample_names)
    
    # Calculate total gene set size for normalization
    total_genes = 0
    if up_genes:
        total_genes += len(up_genes)
    if down_genes:
        total_genes += len(down_genes)
    
    # Calculate combined score with size normalization
    if up_genes:
        up_score = z_standardized.loc[up_genes].sum()
        scores += up_score
    if down_genes:
        down_score = z_standardized.loc[down_genes].sum()
        scores -= down_score  # Subtract because these are down-regulated
    
    # Normalize by square root of gene set size
    scores = scores / np.sqrt(total_genes)
    
    # Final z-score normalization across samples
    scores = (scores - scores.mean()) / scores.std()
    
    # Convert to DataFrame with meaningful column name
    scores_df = pd.DataFrame(scores, columns=["geneset_score"])
    
    # Add condition column if available
    if condition_info is not None:
        scores_df[condition_col] = condition_info.values
    
    return scores_df
    
def calculate_pairwise_significance(data, groups, x_var, y_var):
    """
    Calculate pairwise significance between all groups
    Returns a dictionary of p-values and significance levels
    """
    from scipy import stats
    results = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1 = data[data[x_var] == groups[i]][y_var]  # Changed from 'category' and 'senescence_score'
            group2 = data[data[x_var] == groups[j]][y_var]  # Changed from 'category' and 'senescence_score'
            
            # Perform Mann-Whitney U test
            statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Add significance stars
            if pvalue < 0.001:
                sig = '***'
            elif pvalue < 0.01:
                sig = '**'
            elif pvalue < 0.05:
                sig = '*'
            else:
                sig = 'ns'
                
            results[(i, j)] = {'p-value': pvalue, 'significance': sig}
    
    return results

def plot_violin_box_combo(data, x_var, y_var, title=None, x_ticks=None,
                          palette=None, rotation=45, show_scatter=True,
                          figsize=(5, 6), scatter_size=4, scatter_alpha=0.6,
                          violin_width=0.7, box_width=0.35, jitter=0.15,
                          show_pvalue=True):
    """
    Create a combined violin-box plot with optional scatter points.

    Parameters
    ----------
    show_scatter : bool, default True
        Overlay individual data points as a strip plot.
    figsize : tuple, default (5, 6)
        Figure dimensions in inches.
    scatter_size : float, default 4
        Marker size for scatter points.
    scatter_alpha : float, default 0.6
        Marker transparency (0–1).
    violin_width : float, default 0.7
        Width of the violin bodies.
    box_width : float, default 0.35
        Width of the box plots.
    jitter : float, default 0.15
        Horizontal jitter for scatter points.
    show_pvalue : bool, default True
        If True, display the numerical p-value alongside significance symbols.
        If False, show only the asterisk symbols (*, **, ***).
    """
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.12, top=0.88)

    # ── y-axis limits ──────────────────────────────────────────────
    y_min, y_max = data[y_var].min(), data[y_var].max()
    y_range = y_max - y_min
    padding = y_range * 0.10
    y_min_plot = y_min - padding
    y_max_plot = y_max + padding

    if y_range > 1.0:
        y_min_plot = np.floor(y_min_plot)
        y_max_plot = np.ceil(y_max_plot)
    else:
        y_min_plot = max(0, y_min_plot)

    ax.set_ylim(y_min_plot, y_max_plot)

    # ── smart tick interval based on range ─────────────────────────
    if y_range < 0.1:
        tick_interval = 0.02
    elif y_range < 0.5:
        tick_interval = 0.05
    elif y_range < 2.0:
        tick_interval = 0.25
    elif y_range < 5.0:
        tick_interval = 0.5
    elif y_range < 15.0:
        tick_interval = 2.0
    elif y_range < 30.0:
        tick_interval = 5.0
    else:
        tick_interval = 10.0

    ax.yaxis.set_major_locator(plt.MultipleLocator(tick_interval))

    # ── category order ─────────────────────────────────────────────
    if x_ticks is not None:
        categories = x_ticks
    else:
        categories = sorted(
            data[x_var].unique(),
            key=lambda x: float(x) if str(x).replace('.', '').isdigit() else x,
        )

    # ── violin plot ────────────────────────────────────────────────
    sns.violinplot(
        data=data, x=x_var, y=y_var,
        order=categories, palette=palette,
        inner=None, linewidth=0, saturation=1.0,
        alpha=0.45, width=violin_width, cut=0, ax=ax,
    )

    # ── box plot (skeleton only – we colour it below) ──────────────
    sns.boxplot(
        data=data, x=x_var, y=y_var,
        order=categories, width=box_width, linewidth=1.2,
        flierprops={'marker': ' '}, showmeans=False,
        boxprops={'facecolor': 'none', 'edgecolor': 'none'},
        whiskerprops={'color': 'none'},
        medianprops={'color': 'none'},
        showcaps=False, ax=ax,
    )

    # ── colour boxes per category ──────────────────────────────────
    import matplotlib.patches as mpatches

    num_boxes = len(categories)
    lines_per_box = len(ax.lines) // num_boxes if num_boxes else 0

    for i, (name, box) in enumerate(zip(categories, ax.patches)):
        color = palette[name]

        box.set_facecolor(color)
        box.set_edgecolor('none')
        box.set_alpha(0.45)
        box.set_zorder(1)

        edges = mpatches.PathPatch(
            box.get_path(), facecolor='none',
            edgecolor=color, linewidth=1.2, alpha=1.0, zorder=2,
        )
        ax.add_patch(edges)

        for line in ax.lines[i * lines_per_box:(i + 1) * lines_per_box]:
            line.set_color(color)
            line.set_alpha(1.0)
            line.set_linewidth(1.2)
            line.set_zorder(2)

    # ── scatter overlay ────────────────────────────────────────────
    if show_scatter:
        sns.stripplot(
            data=data, x=x_var, y=y_var,
            order=categories, palette=palette,
            size=scatter_size, alpha=scatter_alpha,
            linewidth=0, jitter=jitter, zorder=3, ax=ax,
        )

    # ── significance bars ──────────────────────────────────────────
    significance_info = calculate_pairwise_significance(
        data, categories, x_var, y_var,
    )

    current_ymin, current_ymax = ax.get_ylim()
    y_range_plot = current_ymax - current_ymin
    bar_spacing = y_range_plot * 0.08
    bar_tips = y_range_plot * 0.02
    bar_height = current_ymax + bar_spacing * 0.5

    def add_significance_bar(start, end, height, p_value, sig_symbol):
        ax.plot(
            [start, start, end, end],
            [height, height + bar_tips, height + bar_tips, height],
            color='black', linewidth=0.8,
        )
        if not show_pvalue:
            text = sig_symbol
        elif p_value < 0.00005:
            text = f'p = {p_value:.2e} {sig_symbol}'
        else:
            text = f'p = {p_value:.4f} {sig_symbol}'
        ax.text(
            (start + end) * 0.5, height + bar_tips,
            text, ha='center', va='bottom', fontsize=8,
        )

    for (g1, g2), sig_data in significance_info.items():
        if sig_data['significance'] != 'ns':
            add_significance_bar(g1, g2, bar_height,
                                 sig_data['p-value'],
                                 sig_data['significance'])
            bar_height += bar_spacing

    ax.set_ylim(current_ymin, bar_height + bar_spacing * 0.5)

    # ── titles & labels ────────────────────────────────────────────
    if title:
        plt.title(title, pad=20)

    if x_ticks is None:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
    else:
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks, rotation=rotation, ha='right')
        ax.spines['bottom'].set_visible(True)

    # ── spine / tick cosmetics ─────────────────────────────────────
    ax.minorticks_off()
    ax.tick_params(axis='both', which='minor',
                   bottom=False, top=False, left=False, right=False)
    ax.tick_params(axis='x', which='major', top=False)
    ax.tick_params(axis='y', which='major', right=False, width=0.8)

    ax.spines['left'].set_linewidth(0.8)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_tick_params(width=0.8)

    plt.setp(ax.get_yticklabels(), weight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.yaxis.grid(False)

    sns.despine(offset=5, trim=True,
                bottom=(x_ticks is None), right=True)

    if x_ticks is not None:
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')

    plt.close()
    return fig

def plot_pathway_dotplot(
    df_cell_level,
    score_cols,  # List of pathway score column names (y-axis)
    sample_col='sample',  # Column name for sample IDs (x-axis)
    annotation_col='Annotation',
    target_annotation=None,
    sample_order=None,  # Optional: list of samples in desired order
    figsize=(10, 6),
    min_dot_size=20,
    max_dot_size=500,
    dot_size_scale_factor=1.0,
    cmap_name="coolwarm",
    value_legend_title="Mean Score",
    size_legend_title="# Cells",
    ylabel="Pathways",
    xlabel="Samples"
):
    """
    Plots a dot plot where dot color is mean pathway score and dot size is number of cells.
    
    Parameters:
        df_cell_level (pd.DataFrame): Cell-level data with scores, sample, annotation.
        score_cols (list): List of score column names (pathways) - will be y-axis.
        sample_col (str): Column name for sample IDs - will be x-axis.
        annotation_col (str): Column name for annotations.
        target_annotation (str or None): If provided, subset to this annotation.
        sample_order (list or None): Order of samples for x-axis. If None, uses sorted order.
        figsize (tuple): Figure size.
        min_dot_size (int): Minimum size for dots.
        max_dot_size (int): Maximum size for dots.
        dot_size_scale_factor (float): Multiplier for raw cell counts before scaling to dot size.
        cmap_name (str): Colormap for the scores.
        value_legend_title (str): Title for the colorbar.
        size_legend_title (str): Title for the size legend.
        ylabel (str): Label for y-axis (pathways).
        xlabel (str): Label for x-axis (samples).
    """
    plot_df = df_cell_level.copy()

    # 1. Filter by annotation if specified
    if target_annotation is not None:
        if annotation_col not in plot_df.columns:
            print(f"Warning: Annotation column '{annotation_col}' not found. Cannot filter by '{target_annotation}'.")
            return
        plot_df = plot_df[plot_df[annotation_col] == target_annotation]
        if plot_df.empty:
            print(f"No cells found for annotation '{target_annotation}'.")
            return

    # Check required columns
    required_cols = [sample_col] + score_cols
    for col in required_cols:
        if col not in plot_df.columns:
            print(f"Warning: Required column '{col}' not found. Aborting.")
            return

    plot_df = plot_df.dropna(subset=[sample_col], how='any')
    if plot_df.empty:
        print("No data to plot after initial NaN filtering.")
        return

    # 2. Aggregate: mean scores and cell counts per sample
    grouped = plot_df.groupby(sample_col)
    mean_scores_df = grouped[score_cols].mean()
    cell_counts_series = grouped.size()

    # 3. Determine sample order
    if sample_order is None:
        ordered_samples = sorted(mean_scores_df.index.tolist())
    else:
        # Use provided order, but only include samples that exist in data
        ordered_samples = [s for s in sample_order if s in mean_scores_df.index]
        if not ordered_samples:
            print("None of the specified samples found in data.")
            return

    # 4. Prepare data for plotting (long format)
    plot_data_list = []
    for sample_id in ordered_samples:
        for pathway in score_cols:
            mean_score = mean_scores_df.loc[sample_id, pathway] if sample_id in mean_scores_df.index else np.nan
            cell_count = cell_counts_series.loc[sample_id] if sample_id in cell_counts_series.index else 0
            plot_data_list.append({
                'sample': sample_id,
                'pathway': pathway,
                'mean_score': mean_score,
                'cell_count': cell_count
            })
    
    plot_data_df = pd.DataFrame(plot_data_list)
    plot_data_df = plot_data_df.dropna(subset=['mean_score'])

    if plot_data_df.empty:
        print("No data to plot after aggregation.")
        return

    # 5. Scale cell counts for dot sizes
    min_count = plot_data_df['cell_count'].min()
    max_count = plot_data_df['cell_count'].max()
    
    if max_count == min_count:
        plot_data_df['dot_size'] = min_dot_size if max_count == 0 else (min_dot_size + max_dot_size) / 2
    else:
        scaled_counts = plot_data_df['cell_count'] * dot_size_scale_factor
        min_s_count = scaled_counts.min()
        max_s_count = scaled_counts.max()
        
        if max_s_count == min_s_count:
            plot_data_df['dot_size'] = min_dot_size if max_s_count == 0 else (min_dot_size + max_dot_size) / 2
        else:
            plot_data_df['dot_size'] = min_dot_size + \
                (scaled_counts - min_s_count) / (max_s_count - min_s_count) * (max_dot_size - min_dot_size)

    # 6. Plotting
    fig, ax = plt.subplots(figsize=figsize)

    # Create coordinate mappings
    pathway_y_coords = {name: i for i, name in enumerate(score_cols)}
    sample_x_coords = {name: i for i, name in enumerate(ordered_samples)}

    scatter = ax.scatter(
        x=plot_data_df['sample'].map(sample_x_coords),
        y=plot_data_df['pathway'].map(pathway_y_coords),
        s=plot_data_df['dot_size'],
        c=plot_data_df['mean_score'],
        cmap=cmap_name,
        edgecolors='gray',
        linewidths=0.5
    )

    # X-axis (Samples)
    ax.set_xticks(list(sample_x_coords.values()))
    ax.set_xticklabels(ordered_samples, rotation=45, ha="right")
    ax.set_xlabel(xlabel)

    # Y-axis (Pathways)
    ax.set_yticks(list(pathway_y_coords.values()))
    ax.set_yticklabels(score_cols)
    ax.set_ylabel(ylabel)

    # Colorbar for Mean Score
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.15)
    cbar.set_label(value_legend_title)

    # Legend for Dot Size (# Cells)
    if max_count > 0:
        legend_counts_raw = np.linspace(min_count, max_count, num=4, dtype=int)
        if min_count == 0 and 0 not in legend_counts_raw and len(legend_counts_raw) > 1:
            legend_counts_raw[0] = 0
        legend_counts_raw = np.unique(legend_counts_raw)
    else:
        legend_counts_raw = np.array([min_count]) if min_count > 0 else np.array([0])

    legend_dots = []
    for count_val in legend_counts_raw:
        if max_count == min_count:
            size_val = min_dot_size if max_count == 0 else (min_dot_size + max_dot_size) / 2
        else:
            scaled_c = count_val * dot_size_scale_factor
            size_val = min_dot_size + \
                (scaled_c - (min_count * dot_size_scale_factor)) / \
                ((max_count * dot_size_scale_factor) - (min_count * dot_size_scale_factor)) * \
                (max_dot_size - min_dot_size)
        size_val = max(min_dot_size, min(max_dot_size, size_val))
        legend_dots.append(plt.scatter([], [], s=size_val, c='gray', label=f"{int(count_val)}"))

    size_leg = ax.legend(
        handles=legend_dots,
        title=size_legend_title,
        bbox_to_anchor=(1.18, 0.4),
        loc='center left',
        labelspacing=1.5,
        borderpad=1,
        frameon=True,
        handletextpad=1.5,
        scatterpoints=1
    )

    # Layout adjustments
    fig_title = f'Pathway Activity Dot Plot (Annotation: {target_annotation})' if target_annotation else 'Pathway Activity Dot Plot'
    plt.suptitle(fig_title, fontsize=16, y=1.02)
    plt.subplots_adjust(bottom=0.15, right=0.8)
    plt.grid(True, linestyle='--', alpha=0.3, axis='both')
    ax.tick_params(axis='both', which='major', pad=7)

    plt.tight_layout()
    plt.show()