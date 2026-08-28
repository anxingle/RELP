from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional

class PipelineContext(BaseModel):
    """
    Context Object passed through the pipeline to replace global variables.
    Ensures thread safety and prevents cross-contamination.
    """
    scoring_config: List[str] = Field(default_factory=list)
    gray_scale_params: Dict[str, Any] = Field(default_factory=lambda: {'Enabled': False})
    scoring_setting: Dict[str, Any] = Field(default_factory=dict)
    defect_output_format: str = 'Combined'
    reference_params: Dict[str, Any] = Field(default_factory=dict)
    output_config: List[str] = Field(default_factory=list)

# --- Node Data Schemas for Validation ---
class BaseNodeData(BaseModel):
    model_config = ConfigDict(extra='allow', coerce_numbers_to_str=True)

class LocatorNodeData(BaseNodeData):
    weights: Dict[str, str] = Field(default_factory=dict)

class SamNodeData(BaseNodeData):
    ratio: float = Field(1.0, ge=0.0)
    prompt: str = Field("object")

class BinningNodeData(BaseNodeData):
    color_space: str = Field("Gray Scale")
    binning_rule: str = Field("")
    invert_color: bool = Field(False)

class ScoringNodeData(BaseNodeData):
    maxw: float = Field(100.0)
    decay: float = Field(0.3)
    reverse: bool = Field(False)

class FilterAdaptiveGaussian(BaseNodeData):
    blocksize: int = Field(11, ge=3, alias="block_size")
    c: float = Field(2.0, alias="c_value")
    invert: bool = Field(False)
    gamma: float = Field(1.0)
    contrast: float = Field(1.0)

class FilterArea(BaseNodeData):
    area_pct: float = Field(0.0)
    area_pixel: Optional[int] = Field(None)

class FilterShape(BaseNodeData):
    aspect_min: float = Field(1.0, alias="aspect_ratio_th")
    circle_min: float = Field(0.0)

class FilterMorph(BaseNodeData):
    mode: str = Field("Top-Hat", alias="morph_type")
    kernel_size: int = Field(15, ge=1)
    threshold: int = Field(127, alias="morph_th")
    uniform_light: bool = Field(False)

class FilterColorDistance(BaseNodeData):
    target_rgb: str = Field("0,0,0")
    tolerance: int = Field(50, alias="color_tol")
    proximity: str = Field("Near")

class FilterKNN(BaseNodeData):
    target_rgb: str = Field("0,0,0")
    knn_distance_threshold: float = Field(15.0, alias="tolerance")
    first_knn_cluster_qty: int = Field(2)
    second_knn_cluster_qty: int = Field(3)

class BooleanOpsNodeData(BaseNodeData):
    operation: str = Field("AND")

class MetrologyRefNodeData(BaseNodeData):
    physical_length: float = Field(0.0)
    physical_width: float = Field(0.0)
    unit: str = Field("mm")
    auto_detect: bool = Field(False)

# --- Formal Payload Contract for DAG Execution ---
class NodeMetadata(BaseModel):
    """
    Standardized hidden channel metadata passed between nodes.
    Replaces ad-hoc dictionaries to ensure all downstream nodes 
    know exactly what advanced attributes are available.
    """
    boxes: List[List[float]] = Field(default_factory=list)
    instance_masks: List[Any] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class NodePayload(BaseModel):
    """
    The official output contract for any node.
    Contains both the pixel-level mask and high-level abstract metadata.
    """
    mask: Any = None
    metadata: NodeMetadata = Field(default_factory=NodeMetadata)
    model_config = ConfigDict(arbitrary_types_allowed=True)
