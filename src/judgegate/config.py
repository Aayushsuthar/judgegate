from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from judgegate.errors import ConfigError


class ExtractionSettings(BaseModel):
    """How to pull the verdict label out of the judge's response text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["json_field", "regex", "label_search"] = "json_field"
    field: str = "label"
    pattern: str | None = None

    @model_validator(mode="after")
    def _check_pattern(self) -> Self:
        if self.method == "regex" and not self.pattern:
            raise ValueError("regex extraction needs a pattern with one capture group")
        return self


class JudgeSettings(BaseModel):
    """The judge endpoint and prompt under evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint: str = "https://api.openai.com/v1"
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    prompt: str = Field(min_length=1)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    retries: int = Field(default=4, ge=0)
    concurrency: int = Field(default=4, ge=1, le=64)


class LabelSettings(BaseModel):
    """The label vocabulary shared by humans and the judge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: tuple[str, ...] = Field(min_length=2)
    ordinal: bool = False

    @model_validator(mode="after")
    def _check_distinct(self) -> Self:
        if len(set(self.values)) != len(self.values):
            raise ValueError("label values must be distinct")
        return self


class GateSettings(BaseModel):
    """Decision policy for the trust verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_kappa: float = Field(default=0.6, gt=-1.0, lt=1.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    power: float = Field(default=0.8, gt=0.0, lt=1.0)
    weighting: Literal["none", "linear", "quadratic"] = "none"
    resamples: int = Field(default=10_000, ge=100)
    seed: int | None = Field(default=None, ge=0)
    min_labels: int = Field(default=20, ge=10)

    @property
    def confidence(self) -> float:
        return 1.0 - self.alpha


class StabilityProbeSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    runs: int = Field(default=3, ge=2, le=10)
    max_flip_rate: float = Field(default=0.10, ge=0.0, le=1.0)


class PositionProbeSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    max_flip_rate: float = Field(default=0.15, ge=0.0, le=1.0)


class VerbosityProbeSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    max_leniency_gap: float = Field(default=0.15, ge=0.0, le=1.0)


class FormatProbeSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    max_flip_rate: float = Field(default=0.10, ge=0.0, le=1.0)


class ProbeSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stability: StabilityProbeSettings = Field(default_factory=StabilityProbeSettings)
    position: PositionProbeSettings = Field(default_factory=PositionProbeSettings)
    verbosity: VerbosityProbeSettings = Field(default_factory=VerbosityProbeSettings)
    format: FormatProbeSettings = Field(default_factory=FormatProbeSettings)


class CacheSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    path: str = ".judgegate-cache.sqlite"


class Config(BaseModel):
    """Top level judgegate configuration, loaded from judge.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    judge: JudgeSettings
    labels: LabelSettings
    gate: GateSettings = Field(default_factory=GateSettings)
    probes: ProbeSettings = Field(default_factory=ProbeSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)


def _format_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors()
    )


def load_config(path: Path) -> Config:
    """Load and validate a judge.yaml configuration file."""
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"could not read config file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {path}: {_format_validation_error(exc)}") from exc


def apply_overrides(config: Config, gate: dict[str, object] | None = None) -> Config:
    """Return a new config with gate overrides applied and re-validated."""
    if not gate:
        return config
    try:
        updated = GateSettings(**{**config.gate.model_dump(), **gate})
    except ValidationError as exc:
        raise ConfigError(f"invalid option value: {_format_validation_error(exc)}") from exc
    return config.model_copy(update={"gate": updated})


def render_prompt(template: str, mapping: dict[str, str]) -> str:
    """Fill {placeholder} slots without treating other braces specially."""
    rendered = template
    for key, value in mapping.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered
