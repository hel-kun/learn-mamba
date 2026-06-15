from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

import yaml


@dataclass(slots=True)
class MambaLMConfig:
    vocab_size: int
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    dt_rank: int | None = None
    num_layers: int = 4
    dropout: float = 0.0
    expand: int = 2
    tie_embeddings: bool = True
    block_size: int = 128

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, int | float | bool | None]) -> "MambaLMConfig":
        return cls(**values)


@dataclass(slots=True)
class DatasetConfig:
    name: str = "SimpleStories/SimpleStories-JA"
    config_name: str | None = None
    text_column: str = "story"
    max_train_samples: int | None = 128
    max_eval_samples: int | None = 32


@dataclass(slots=True)
class TokenizerConfig:
    name: str = "EleutherAI/gpt-neox-20b"


@dataclass(slots=True)
class ModelConfig:
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    dt_rank: int | None = None
    num_layers: int = 4
    dropout: float = 0.0
    expand: int = 2
    tie_embeddings: bool = True
    block_size: int = 128


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 2
    max_steps: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 5
    eval_batches: int = 4


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "auto"
    seed: int = 0


@dataclass(slots=True)
class OutputConfig:
    checkpoint_path: str = "checkpoints/mamba_lm.pt"


T = TypeVar("T")


@dataclass(slots=True)
class TrainConfig:
    dataset: DatasetConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    output: OutputConfig

    @classmethod
    def default(cls) -> "TrainConfig":
        return cls(
            dataset=DatasetConfig(),
            tokenizer=TokenizerConfig(),
            model=ModelConfig(),
            training=TrainingConfig(),
            runtime=RuntimeConfig(),
            output=OutputConfig(),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open() as file:
            values = yaml.safe_load(file) or {}
        if not isinstance(values, dict):
            raise ValueError("Config YAML must contain a mapping at the top level")
        return _dataclass_from_dict(cls, values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_mamba_lm_config(self, vocab_size: int) -> MambaLMConfig:
        return MambaLMConfig(vocab_size=vocab_size, **asdict(self.model))

    def with_overrides(self, overrides: dict[str, Any]) -> "TrainConfig":
        config = self
        for dotted_key, value in overrides.items():
            if value is None:
                continue
            section_name, _, field_name = dotted_key.partition(".")
            if not section_name or not field_name:
                raise ValueError(f"Override key must be '<section>.<field>', got {dotted_key!r}")
            if not hasattr(config, section_name):
                raise ValueError(f"Unknown config section: {section_name}")
            section = getattr(config, section_name)
            if not hasattr(section, field_name):
                raise ValueError(f"Unknown config field: {dotted_key}")
            updated_section = replace(section, **{field_name: value})
            config = replace(config, **{section_name: updated_section})
        return config


def _dataclass_from_dict(cls: type[T], values: dict[str, Any]) -> T:
    field_map = {field.name: field for field in fields(cls)}
    unknown_keys = sorted(set(values) - set(field_map))
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {joined}")

    kwargs: dict[str, Any] = {}
    default_instance = cls.default() if cls is TrainConfig else None
    for name, field in field_map.items():
        raw_value = values.get(name, getattr(default_instance, name) if default_instance is not None else field.default)
        field_type = _unwrap_optional(field.type)
        if is_dataclass(field_type):
            if is_dataclass(raw_value):
                kwargs[name] = raw_value
                continue
            if not isinstance(raw_value, dict):
                raise ValueError(f"Config key {name!r} must be a mapping")
            kwargs[name] = _dataclass_from_dict(field_type, raw_value)
        else:
            kwargs[name] = _coerce_scalar(raw_value, field_type, name)
    return cls(**kwargs)


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    args = get_args(annotation)
    if type(None) in args and len(args) == 2:
        return next(arg for arg in args if arg is not type(None))
    return annotation


def _coerce_scalar(value: Any, target_type: Any, field_name: str) -> Any:
    if value is None:
        return None
    if target_type is float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config key {field_name!r} must be a float, got {value!r}") from exc
    if target_type is int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config key {field_name!r} must be an int, got {value!r}") from exc
    if target_type is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"Config key {field_name!r} must be a bool, got {value!r}")
    if target_type is str:
        if isinstance(value, str):
            return value
        raise ValueError(f"Config key {field_name!r} must be a string, got {value!r}")
    return value
