import torch
import matplotlib.pyplot as plt

def plot_grid(vae, grid_size=10):
    device = next(vae.parameters()).device  # Get device from model parameters

    # Generate a grid of latent space points
    latent_grid = torch.linspace(-3, 3, grid_size, device=device)
    latent_points = torch.meshgrid(*([latent_grid] * vae.latent2_dim))
    latent_points = torch.stack(latent_points, dim=-1).reshape(-1, vae.latent2_dim)

    # Generate images from the latent space points
    with torch.no_grad():
        decoded_images = vae.sample(latent_points, current_device=device)

    # Plot the decoded images in a grid
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i in range(grid_size):
        for j in range(grid_size):
            ax = axs[i, j]
            ax.imshow(decoded_images[i * grid_size + j].permute(1, 2, 0).cpu().numpy(), cmap='gray')
            ax.axis('off')

    return fig
