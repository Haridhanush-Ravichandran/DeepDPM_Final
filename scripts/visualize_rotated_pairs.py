import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D


def load_tensor(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return tensor


def to_2d(data: torch.Tensor) -> np.ndarray:
    data_np = data.detach().cpu().numpy()
    if data_np.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(data.shape)}")
    if data_np.shape[1] <= 2:
        return data_np
    return PCA(n_components=2, random_state=42).fit_transform(data_np)


def make_palette(labels: np.ndarray):
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", n_colors=max(len(unique_labels), 1))
    color_map = {label: palette[idx % len(palette)] for idx, label in enumerate(unique_labels)}
    return color_map


def plot_rotated_pairs(train_data: torch.Tensor, train_labels: torch.Tensor, title: str):
    points_2d = to_2d(train_data)
    labels_np = train_labels.detach().cpu().numpy().astype(int)

    color_map = make_palette(labels_np)
    colors = [color_map[label] for label in labels_np]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(points_2d[:, 0], points_2d[:, 1], c=colors, s=18, alpha=0.85, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_aspect("equal", adjustable="datalim")

    handles = []
    labels = []
    for label, color in color_map.items():
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8))
        labels.append(str(label))

    ax.legend(handles, labels, title="Train label", loc="best", frameon=False)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize the rotated-pairs training data and labels.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("pretrained_embeddings/umap_embedded_datasets/ROTATED_PAIRS"),
        help="Directory containing train_data.pt and train_labels.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the figure. If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Rotated Pairs: train data colored by train labels",
        help="Figure title.",
    )
    args = parser.parse_args()

    train_data_path = args.data_dir / "train_data.pt"
    train_labels_path = args.data_dir / "train_labels.pt"

    if not train_data_path.exists():
        raise FileNotFoundError(f"Missing train data file: {train_data_path}")
    if not train_labels_path.exists():
        raise FileNotFoundError(f"Missing train labels file: {train_labels_path}")

    train_data = load_tensor(train_data_path)
    train_labels = load_tensor(train_labels_path)

    if train_data.shape[0] != train_labels.shape[0]:
        raise ValueError(
            f"Mismatched sample counts: train_data has {train_data.shape[0]} rows, "
            f"train_labels has {train_labels.shape[0]} rows"
        )

    fig = plot_rotated_pairs(train_data, train_labels, args.title)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
