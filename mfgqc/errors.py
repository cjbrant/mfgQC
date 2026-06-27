"""Public exception types for mfgQC.

A thin frontend (or any caller) needs to distinguish "you asked for an analysis
but did not supply what it needs" from a genuine programming error, so it can
prompt the user for the missing input instead of crashing.

``MissingPrerequisiteError`` is the catchable signal for that case. It subclasses
both :class:`PyQCError` (so callers can catch every mfgQC-specific error with one
clause) and the built-in :class:`ValueError` (so existing code that catches
``ValueError`` keeps working unchanged -- raising it is not a breaking change).
"""

from __future__ import annotations


class PyQCError(Exception):
    """Base class for all mfgQC-specific exceptions."""


class MissingPrerequisiteError(PyQCError, ValueError):
    """An analysis was requested without an input it requires.

    Examples: capability without a spec limit, gage R&R without the part/
    operator/replicate roles, a p-chart without subgroup sizes.

    Attributes
    ----------
    analysis : str or None
        The analysis that was requested (e.g. ``"capability"``).
    missing : tuple of str
        Machine-readable tokens for what is missing (e.g. ``("spec",)`` or
        ``("role:part", "role:operator")``), so a UI can prompt for exactly
        those inputs.
    """

    def __init__(self, message: str, *, analysis: str | None = None,
                 missing: "list[str] | tuple[str, ...] | None" = None) -> None:
        super().__init__(message)
        self.analysis = analysis
        self.missing = tuple(missing or ())
