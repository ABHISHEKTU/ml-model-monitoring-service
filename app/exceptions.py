"""Custom exception hierarchy. Keep errors typed → API layer maps them to clean HTTP codes."""


class MonitoringError(Exception):
    """Base class. Catch this at API boundary as fallback."""
    pass


class ReferenceNotSetError(MonitoringError):
    """Raised when drift check requested but no baseline/reference dataset exists yet."""
    pass


class SchemaMismatchError(MonitoringError):
    """Raised when current data columns don't match reference schema."""

    def __init__(self, missing: list, extra: list):
        self.missing = missing
        self.extra = extra
        super().__init__(f"Schema mismatch. Missing={missing} Extra={extra}")


class InsufficientDataError(MonitoringError):
    """Raised when a feature/sample has too few points for a statistically meaningful test."""

    def __init__(self, feature: str, n: int, min_required: int):
        self.feature = feature
        self.n = n
        self.min_required = min_required
        super().__init__(
            f"Feature '{feature}' has {n} samples, need >= {min_required} for reliable test."
        )