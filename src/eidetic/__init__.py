from eidetic.core.cliques import detect_memorization
from eidetic.core.definitions import (
    EideticMemorizationResult,
    ExtractionResult,
    count_neighbors_within_delta,
)
from eidetic.core.distances import adaptive_distance, l2, tiled_l2
from eidetic.defenses.dedup import DedupReport, audit_duplicates
from eidetic.extraction.black_box import (
    ExtractionConfig,
    PromptExtractionResult,
    extract_from_prompt,
    extract_from_prompts,
)
from eidetic.inpainting.attack import InpaintingResult, inpaint_attack
from eidetic.mia.base import MIAResult, auc_log_log, tpr_at_fpr
from eidetic.mia.lira import LiRAAttack, LiRADistributions
from eidetic.mia.loss_threshold import LossThresholdAttack
from eidetic.mia.strong_lira import (
    StrongLiRAAttack,
    estimate_expected_loss,
    goldilocks_timestep_search,
)

__version__ = "0.1.0"

__all__ = [
    "DedupReport",
    "EideticMemorizationResult",
    "ExtractionConfig",
    "ExtractionResult",
    "InpaintingResult",
    "LiRAAttack",
    "LiRADistributions",
    "LossThresholdAttack",
    "MIAResult",
    "PromptExtractionResult",
    "StrongLiRAAttack",
    "adaptive_distance",
    "auc_log_log",
    "audit_duplicates",
    "count_neighbors_within_delta",
    "detect_memorization",
    "estimate_expected_loss",
    "extract_from_prompt",
    "extract_from_prompts",
    "goldilocks_timestep_search",
    "inpaint_attack",
    "l2",
    "tiled_l2",
    "tpr_at_fpr",
]
