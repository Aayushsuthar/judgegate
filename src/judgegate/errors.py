class JudgegateError(Exception):
    """Base class for all judgegate errors."""


class DataError(JudgegateError):
    """Raised when a labels file cannot be parsed or is unusable."""


class ConfigError(JudgegateError):
    """Raised when configuration is missing, malformed, or invalid."""


class JudgeError(JudgegateError):
    """Raised when the judge endpoint fails or returns unusable output."""


class AnalysisError(JudgegateError):
    """Raised when the data is insufficient or unsuitable for analysis."""
