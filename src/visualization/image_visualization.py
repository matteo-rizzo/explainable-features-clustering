import torch
from matplotlib import pyplot as plt


def draw_activation(activation_maps, label: str = "Activation"):
    # Create a grid of subplots based on the number of tensors
    num_rows: int = 3
    num_cols: int = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 9))
    im = None
    for i, tensor in enumerate(activation_maps):
        row_idx = i // num_cols
        col_idx = i % num_cols
        ax = axes[row_idx, col_idx]
        im = ax.imshow(tensor, cmap='inferno')
        ax.set_title(f'Heatmap {i + 1}')
        ax.axis('off')

    # Set the overall title using the label parameter
    fig.suptitle(label, fontsize=16)
    # Create a big colorbar on the right side
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cax)
    # plt.tight_layout()
    plt.show()

    plt.close()
    summed_tensor = torch.sum(activation_maps, dim=0)
    plt.imshow(summed_tensor , cmap='inferno')
    plt.title("All summed")
    plt.show()
