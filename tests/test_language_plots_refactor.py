import matplotlib.pyplot as plt
import numpy as np

from src.viz.language_plots import _add_itpc_annotations, _save_and_close, _setup_figure_and_ax


def test_setup_figure_and_ax():
    fig, ax = _setup_figure_and_ax(figsize=(8, 4), title="Test Title", xlabel="X", ylabel="Y")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Test Title"
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"
    plt.close(fig)


def test_save_and_close(tmp_path):
    fig, ax = plt.subplots()
    path = tmp_path / "test_plot.png"
    saved_path = _save_and_close(fig, path)
    assert saved_path == path
    assert path.exists()
    # Check if figure is closed
    assert not plt.fignum_exists(fig.number)


def test_add_itpc_annotations():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 5)
    metrics = {
        "p_word": 0.001,
        "itpc_word": 0.2,
        "p_phrase": 0.01,
        "itpc_phrase": 0.15,
        "p_sentence": 0.05,
        "itpc_sentence": 0.1,
    }
    plot_freqs = np.linspace(0, 5, 100)
    _add_itpc_annotations(ax, metrics, "DFT", 0.5, plot_freqs)

    # Verify some annotations were added (vertical lines)
    vlines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
    assert len(vlines) >= 3
    plt.close(fig)
