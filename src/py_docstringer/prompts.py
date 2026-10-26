"""Prompt construction for docstring generation."""

from __future__ import annotations

from .discovery import Definition

SYSTEM_PROMPT = """\
You are an expert Python engineer who writes accurate Google-style docstrings.

Return only the docstring content. Do not include triple quotes, code fences, the \
def/class line, or any commentary before or after the docstring.

Follow these rules:
- Start with a concise imperative summary ending with a period.
- Leave one blank line after the summary before any extended description.
- Use Google-style sections such as Args, Returns, Yields, Raises, Attributes, and \
Example only when they apply.
- Document every parameter, including *args and **kwargs, but omit self and cls.
- Include types in parentheses when annotations or defaults make them clear.
- Describe only behavior that is visible in the supplied code. Never invent side \
effects, exceptions, or return values.
- For a class, describe its purpose and public attributes. Document constructor \
arguments in __init__ rather than repeating them in the class docstring.
- Do not wrap or manually break lines. Write each paragraph and each Args, Returns, \
Yields, Raises, Attributes, or Example entry as one unbroken line; the caller wraps long \
lines afterward.
- Indent section content with four spaces.
- Use only ASCII characters. Do not use non-ASCII punctuation, symbols, accents, emoji, \
or typographic quotes; use plain ASCII equivalents.
- Do not wrap the result in quotation marks.
"""


def build_prompt(definition: Definition, *, overwrite: bool = False) -> str:
    """Build the user prompt for one definition."""

    source = definition.source
    if not source.endswith("\n"):
        source += "\n"
    existing = ""
    if overwrite and definition.has_docstring:
        existing = (
            "The definition already has a docstring. Replace it with a clearer, "
            "more accurate Google-style docstring.\n\n"
        )
    return (
        existing + f"Definition type: {definition.kind}\n"
        f"Qualified name: {definition.qualname}\n"
        f"File: {definition.path}\n\n"
        "Source code:\n"
        f"{source}"
    )
