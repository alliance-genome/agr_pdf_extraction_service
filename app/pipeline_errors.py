"""Operational control failures that scientific fallback handlers must not consume."""


class StageControlError(RuntimeError):
    """Stop stage work instead of substituting a scientific fallback."""


class OwnershipLost(StageControlError):
    pass


class PipelineDeadlineExceeded(StageControlError, TimeoutError):
    pass


class ProviderAdmissionError(StageControlError):
    """Provider work was not admitted safely; fail the entire job explicitly."""
