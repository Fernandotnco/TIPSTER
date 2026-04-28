from .ae_loss import AELoss
from .contrastive_losses import triplet_loss, TripletLoss, ev_loss, EVLoss, SoftNearestNeighborLoss, MultiLabelSoftNearestNeighborLoss, SupConLoss, MultiSimilarityLoss, ProxyAnchorLoss, BarlowTwinsLoss, SmoothTripletLoss
from .kernel_losses import MMD_RBF, HSIC_RBF, HSIC_RBF_Multi
from .residual_losses import CosineOrthogonalityLoss, CrossCovarianceLoss, CrossCorrelationLoss, DistanceCorrelationLoss, AntiInfoNCELoss, VarianceFloorLoss
from .normalized_losses import ZScoreMSE
from .ms_spec_loss import MultiScaleSpectrogramLoss, SpecLoss
from .metrics import ExplainedVariance, MRRMetric, DecoderGap, SemanticProbeMetric, CrossHeadPredictabilityMetric, DCIHeadImportanceMetric, ReconstructionDiffMetric
from .norm_loss import NormLoss, NormMSE, MaxNormLoss, GateNormLoss
from .combined_loss import CombinedLoss

import inspect


def _normalize(name: str) -> str:
    return name.replace("-", "").replace("_", "").lower()


# -----------------------------------------------------
# AUTOMATIC REGISTRY (no manual entries)
# -----------------------------------------------------
def _build_registry():
    """
    Scans globals() for any class/function and registers them
    using a normalized (case-insensitive) name.
    """
    reg = {}

    for name, obj in globals().items():
        if name.startswith("_"):
            continue  # skip internals

        # we only want callable losses (functions or classes)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            key = _normalize(name)
            reg[key] = obj

    return reg


_LOSS_REGISTRY = _build_registry()


# -----------------------------------------------------
# FACTORY FUNCTION
# -----------------------------------------------------
def get_loss(name: str):
    key = _normalize(name)

    if key not in _LOSS_REGISTRY:
        raise KeyError(
            f"Unknown loss '{name}'. Available entries: {list(_LOSS_REGISTRY.keys())}"
        )

    return _LOSS_REGISTRY[key]