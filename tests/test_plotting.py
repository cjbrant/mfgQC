"""Smoke tests for the view() hook (standalone Figure and into a provided Axes)."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def test_capability_view_standalone_and_axes(montgomery_qc):
    cap = montgomery_qc.capability()
    fig = cap.view()
    assert isinstance(fig, Figure)
    plt.close(fig)

    fig2, ax = plt.subplots()
    out = cap.view(ax=ax)
    assert isinstance(out, Axes)
    plt.close(fig2)

    fig3 = cap.view(kind="probability")
    assert isinstance(fig3, Figure)
    plt.close(fig3)


def test_control_chart_view(montgomery_qc):
    cc = montgomery_qc.control_chart()
    fig = cc.view()
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_gage_view(aiag_qc):
    grr = aiag_qc.gage_rr()
    fig = grr.view()
    assert isinstance(fig, Figure)
    plt.close(fig)

    fig2, ax = plt.subplots()
    out = grr.view(ax=ax)
    assert isinstance(out, Axes)
    plt.close(fig2)
