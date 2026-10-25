"""Shared exceptions for py-docstringer."""


class DocstringerError(Exception):
    """Base class for expected py-docstringer failures."""


class ConfigurationError(DocstringerError):
    """Raised when the CLI or environment is misconfigured."""


class BatchError(DocstringerError):
    """Raised when an OpenRouter batch cannot be completed."""


class DocstringApplicationError(DocstringerError):
    """Raised when a generated docstring cannot be applied to source code."""
