# API reference

Auto-generated from the in-code docstrings. For task-oriented help start with the
[User Guide](../user-guide/index.md); for the maths behind a method see the other
[Reference](index.md) pages.

## Loading data

::: mfgqc.data.QCData
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      filters: ["!^_"]

## The result surface

Every analysis returns a frozen result that mixes in `QCResult`. These methods are
available on every result object.

::: mfgqc._result.QCResult
    options:
      show_root_heading: true
      heading_level: 3
      filters: ["!^_"]

## Assumption checks

::: mfgqc.assumptions.AssumptionCheck
    options:
      show_root_heading: true
      heading_level: 3
      filters: ["!^_"]

## The analysis catalog

A frontend or report builder discovers every analysis and its required inputs from
this catalog.

::: mfgqc.registry
    options:
      show_root_heading: true
      heading_level: 3
      members: [Analysis, ANALYSES, list_analyses]
      filters: ["!^_"]

## Errors

::: mfgqc.errors
    options:
      show_root_heading: true
      heading_level: 3
      filters: ["!^_"]
