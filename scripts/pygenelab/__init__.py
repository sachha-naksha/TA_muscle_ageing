"""
pyGeneLab

a small python library for gene analysis, plotting, and utility functions
"""

__version__ = "0.1.0"

from .data import *
from .images import *
from .plotting import *
from .utils import *
from .geneset_activity import *
from .pseudotime_animation import (
    create_animated_pathway_plot,
    animate_gene_along_pseudotime,
)
from .deg_functional_enrichment import *
from .transcriptional_noise import *
from .LF_viz import *
from .llm_categorize import (
    PATHWAY_TOP_CATEGORIES,
    strip_pathway_label,
    extract_gene_set,
    categorize_pathways_via_llm,
    categorize_enriched_plot_df,
)
# transfer learning / SLIDE cross-prediction (access as pgl.crossprediction.*)
from . import crossprediction