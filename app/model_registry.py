"""
Model registry for managing multiple ML model backends.

Each model must implement scikit-learn API (fit, predict, predict_proba).
"""

from typing import Any, Dict, Optional, Type, Literal
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Base configuration for all models"""
    model_type: str = Field(..., description="Model identifier")
    objective: str = Field(..., description="Training objective")
    
    class Config:
        extra = "allow"  # Allow model-specific parameters


class WarpGBMConfig(ModelConfig):
    """WarpGBM-specific configuration"""
    model_type: Literal["warpgbm"] = "warpgbm"
    objective: str = Field(default="multiclass")
    num_class: Optional[int] = None
    max_depth: int = Field(default=6, ge=1, le=20)
    num_trees: int = Field(default=100, ge=1, le=1000)
    learning_rate: float = Field(default=0.1, gt=0, le=1.0)
    num_bins: int = Field(default=64, ge=2, le=127)
    min_child_weight: int = Field(default=20, ge=0)
    min_split_gain: float = Field(default=0.0, ge=0)
    colsample_bytree: float = Field(default=1.0, gt=0, le=1.0)


class LightGBMConfig(ModelConfig):
    """LightGBM-specific configuration"""
    model_type: Literal["lightgbm"] = "lightgbm"
    objective: str = Field(default="multiclass")
    num_class: Optional[int] = None
    max_depth: int = Field(default=-1, description="-1 means no limit")
    num_iterations: int = Field(default=100, ge=1, le=5000, alias="num_trees")
    learning_rate: float = Field(default=0.1, gt=0, le=1.0)
    num_leaves: int = Field(default=31, ge=2, le=131072)
    min_data_in_leaf: int = Field(default=20, ge=1)
    feature_fraction: float = Field(default=1.0, gt=0, le=1.0, alias="colsample_bytree")
    bagging_fraction: float = Field(default=1.0, gt=0, le=1.0)
    bagging_freq: int = Field(default=0, ge=0)
    lambda_l1: float = Field(default=0.0, ge=0)
    lambda_l2: float = Field(default=0.0, ge=0)
    min_gain_to_split: float = Field(default=0.0, ge=0)
    verbose: int = Field(default=-1)


class ModelAdapter(ABC):
    """Abstract base class for model adapters"""
    
    @abstractmethod
    def create_model(self, config: ModelConfig) -> Any:
        """Create model instance from config"""
        pass
    
    @abstractmethod
    def get_config_class(self) -> Type[ModelConfig]:
        """Return the config class for this model"""
        pass
    
    @abstractmethod
    def supports_early_stopping(self) -> bool:
        """Whether this model supports early stopping"""
        pass
    
    @abstractmethod
    def to_cpu(self, model: Any) -> Any:
        """Convert model to CPU-compatible version if needed"""
        pass


class WarpGBMAdapter(ModelAdapter):
    """Adapter for WarpGBM models"""
    
    def create_model(self, config: WarpGBMConfig) -> Any:
        """Create WarpGBM model instance"""
        try:
            from warpgbm.core import WarpGBM
            
            # WarpGBM has a single class with objective parameter
            model = WarpGBM(
                objective=config.objective,
                n_estimators=config.num_trees,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                num_bins=config.num_bins,
                min_child_weight=config.min_child_weight,
                min_split_gain=config.min_split_gain,
                colsample_bytree=config.colsample_bytree,
                device="cuda",
            )
            
            return model
            
        except ImportError as e:
            # WarpGBM not installed - this should never be reached in CPU path
            # because WarpGBM requires GPU and should be delegated to GPU function
            raise ImportError(
                "WarpGBM library not installed. WarpGBM requires GPU and should be "
                "handled by the GPU training function. If you see this error, "
                "the GPU delegation is not working properly."
            ) from e
    
    def get_config_class(self) -> Type[ModelConfig]:
        return WarpGBMConfig
    
    def supports_early_stopping(self) -> bool:
        return True
    
    def to_cpu(self, model: Any) -> Any:
        """Convert WarpGBM model to CPU if needed"""
        if hasattr(model, "to_cpu"):
            return model.to_cpu()
        
        # Manually move model attributes to CPU
        import torch
        if hasattr(model, 'device'):
            model.device = torch.device('cpu')
        
        # Move all tensor attributes to CPU
        for attr_name in dir(model):
            if not attr_name.startswith('_'):
                attr = getattr(model, attr_name, None)
                if isinstance(attr, torch.Tensor):
                    setattr(model, attr_name, attr.cpu())
                elif isinstance(attr, list):
                    # Handle lists of tensors
                    cpu_list = []
                    for item in attr:
                        if isinstance(item, torch.Tensor):
                            cpu_list.append(item.cpu())
                        else:
                            cpu_list.append(item)
                    if cpu_list:
                        setattr(model, attr_name, cpu_list)
        
        return model


class LightGBMAdapter(ModelAdapter):
    """Adapter for LightGBM models"""
    
    def create_model(self, config: LightGBMConfig) -> Any:
        """Create LightGBM model instance"""
        try:
            import lightgbm as lgb
            
            params = {
                "objective": self._map_objective(config.objective),
                "num_iterations": config.num_iterations,
                "learning_rate": config.learning_rate,
                "num_leaves": config.num_leaves,
                "max_depth": config.max_depth,
                "min_data_in_leaf": config.min_data_in_leaf,
                "feature_fraction": config.feature_fraction,
                "bagging_fraction": config.bagging_fraction,
                "bagging_freq": config.bagging_freq,
                "lambda_l1": config.lambda_l1,
                "lambda_l2": config.lambda_l2,
                "min_gain_to_split": config.min_gain_to_split,
                "verbose": config.verbose,
            }
            
            if config.num_class is not None:
                params["num_class"] = config.num_class
            
            if config.objective == "regression":
                return lgb.LGBMRegressor(**params)
            else:
                return lgb.LGBMClassifier(**params)
                
        except ImportError:
            raise ImportError(
                "LightGBM not installed. Install with: pip install lightgbm"
            )
    
    def _map_objective(self, objective: str) -> str:
        """Map generic objective to LightGBM objective"""
        mapping = {
            "regression": "regression",
            "binary": "binary",
            "multiclass": "multiclass",
        }
        return mapping.get(objective, objective)
    
    def get_config_class(self) -> Type[ModelConfig]:
        return LightGBMConfig
    
    def supports_early_stopping(self) -> bool:
        return True
    
    def to_cpu(self, model: Any) -> Any:
        """LightGBM models are already CPU-compatible"""
        return model


class ModelRegistry:
    """Registry for managing available models"""
    
    def __init__(self):
        self._adapters: Dict[str, ModelAdapter] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default models"""
        self.register("warpgbm", WarpGBMAdapter())
        self.register("lightgbm", LightGBMAdapter())
    
    def register(self, name: str, adapter: ModelAdapter):
        """Register a new model adapter"""
        self._adapters[name] = adapter
    
    def get_adapter(self, model_type: str) -> ModelAdapter:
        """Get adapter for a model type"""
        if model_type not in self._adapters:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(self._adapters.keys())}"
            )
        return self._adapters[model_type]
    
    def list_models(self) -> list[str]:
        """List all registered model types"""
        return list(self._adapters.keys())
    
    def get_config_class(self, model_type: str) -> Type[ModelConfig]:
        """Get config class for a model type"""
        adapter = self.get_adapter(model_type)
        return adapter.get_config_class()
    
    def create_model(self, config: ModelConfig) -> Any:
        """Create a model instance from config"""
        adapter = self.get_adapter(config.model_type)
        return adapter.create_model(config)
    
    def to_cpu(self, model: Any, model_type: str) -> Any:
        """Convert model to CPU-compatible version"""
        adapter = self.get_adapter(model_type)
        return adapter.to_cpu(model)


# Global registry instance
registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry"""
    return registry

