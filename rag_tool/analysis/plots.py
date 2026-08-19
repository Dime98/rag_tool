from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def overlay_percentile(percentiles, percentile_values, ax_line, colors):
    for percentile in percentiles:
        value = percentile_values[percentile]
        ax_line(
            value,
            linestyle="--",
            linewidth=1,
            color=colors[percentile],
            label=f"P{percentile}: {value:.0f}",
        )


def plot_distribution(
    values: np.ndarray,
    save_to: Path,
    title: str,
    x_label: str,
    y_label: str,
    bins: int = 50,
    percentiles: list | None = None,
):
    values = np.asarray(values, dtype=np.float64).flatten()
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError(f"No finite values available for plotting '{title}'")

    value_range = values.max() - values.min()
    if value_range < 1e-12:
        bins = 1

    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95, 99, 99.9]

    percentile_values = {percentile: np.percentile(values, percentile) for percentile in percentiles}

    cmap = plt.get_cmap("viridis")
    colors = {percentile: cmap(i / (len(percentiles) - 1)) for i, percentile in enumerate(percentiles)}

    fig, (ax_hist, ax_scatter) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    # Histogram
    ax_hist.hist(values, bins=bins)

    overlay_percentile(
        percentiles=percentiles,
        percentile_values=percentile_values,
        ax_line=ax_hist.axvline,
        colors=colors,
    )

    ax_hist.set_title(f"{title} distribution")
    ax_hist.set_xlabel(x_label)
    ax_hist.set_ylabel(y_label)
    ax_hist.grid(True, alpha=0.3)
    ax_hist.legend(loc="upper right", framealpha=0.7)

    # Values by index
    indices = np.arange(len(values))

    ax_scatter.scatter(
        indices,
        values,
        s=10,
        alpha=1.0,
    )

    overlay_percentile(
        percentiles=percentiles,
        percentile_values=percentile_values,
        ax_line=ax_scatter.axhline,
        colors=colors,
    )

    ax_scatter.set_title(f"{title} by index")
    ax_scatter.set_xlabel("Index")
    ax_scatter.set_ylabel(x_label)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(loc="upper right", framealpha=0.7)

    fig.tight_layout()
    fig.savefig(save_to, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_plots(experiment_folder, chunks_char_len, chunks_words_len, embeddings, embedding_norms, percentiles):
    plot_folder = experiment_folder / "plots"
    plot_folder.mkdir(exist_ok=True)

    plot_distribution(
        chunks_char_len,
        save_to=plot_folder / "chunks_char_distribution.png",
        title="Chunk Character Length",
        x_label="Characters",
        y_label="Number of chunks",
        percentiles=percentiles,
    )

    plot_distribution(
        chunks_words_len,
        save_to=plot_folder / "chunks_word_distribution.png",
        title="Chunk Word Length",
        x_label="Words",
        y_label="Number of chunks",
        percentiles=percentiles,
    )

    print(f"{embeddings.dtype=}")
    print(f"{embedding_norms.dtype=}")
    print(embedding_norms.min(), embedding_norms.max(), embedding_norms.std())
    plot_distribution(
        embedding_norms,
        save_to=plot_folder / "embedding_norm_distribution.png",
        title="Embedding Norm",
        x_label="L2 norm",
        y_label="Number of embeddings",
        percentiles=percentiles,
    )
    plot_distribution(
        embeddings.flatten(),
        save_to=plot_folder / "embedding_value_distribution.png",
        title="Embedding Component Value",
        x_label="Embedding value",
        y_label="Number of values",
        percentiles=percentiles,
    )
