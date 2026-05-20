import torch
import numpy as np
import skimage
import matplotlib.pyplot as plt

# NN architecture
# Use the other line if this causes errors
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = torch.load("model2D.pth")
pi = model["reference_points"].to(device)
N = pi.shape[0]

layers = torch.nn.ModuleList()

inDim = 2
hidden = 64
noLayers = 4

for i in range(noLayers):
    layers.append(torch.nn.Linear(inDim, hidden))
    inDim = hidden

finalLayer = torch.nn.Linear(hidden, N)
activation = torch.nn.ReLU()

# Loading the model
for l, w in zip(layers, model["layers"]):
    l.load_state_dict(w)
finalLayer.load_state_dict(model["finalLayer"])

layers = layers.to(device)
finalLayer = finalLayer.to(device)

print("model loaded")

# SDF creation
resolution = 128
bound = 1.2
a = 1.0

xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)

grid = np.zeros((resolution, resolution), np.float32)
voronoi_grid = np.zeros((resolution, resolution), np.float32)

with torch.no_grad():
    X, Y = np.meshgrid(xs, ys, indexing='ij')
        
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    pts = torch.tensor(pts, dtype=torch.float32, device=device)

    out = pts
    for l in layers:
        out = activation(l(out))

    finalLayerData = finalLayer(out)
        
    #Tried both relu and sigmoid as activation on final layer
    numerator = torch.sigmoid(a * finalLayerData)
    # numerator = torch.relu(a * finalLayerData)
    
    denominator = torch.sum(numerator, dim=-1, keepdim=True)

    weights = numerator / torch.clamp(denominator, min=1e-6)

    dist = torch.norm(pts.unsqueeze(1) - pi.unsqueeze(0), dim=2)
        
    max_weight_indices = torch.argmax(weights, dim=1) 
    voronoi_grid = max_weight_indices.view(resolution, resolution).cpu().numpy()

    f_x = torch.sum(weights * dist, dim=1)
       
    grid = f_x.view(resolution, resolution).cpu().numpy()

print("SDF created")  

print(np.min(grid), np.max(grid))
half = (np.min(grid) + np.max(grid))/2
eps = max(1e-4, np.percentile(grid, 1))
contVal = 0.2

# Marching squares
contours = skimage.measure.find_contours(grid, contVal)

world_contours = []
for c in contours:
    c_world = np.zeros_like(c)
    
    c_world[:, 0] = c[:, 0] * (xs[1] - xs[0]) + xs[0]
    c_world[:, 1] = c[:, 1] * (ys[1] - ys[0]) + ys[0]
    
    world_contours.append(c_world)

#polyscope
avgWeights = weights.mean(dim=0).cpu().numpy()
maxWeights = weights.max(dim=0).values.cpu().numpy()
dead = avgWeights < 0.001
anchorsNp = pi.detach().cpu().numpy()


# Diagrams
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Voronoi regions
ax1.imshow(voronoi_grid.T, origin='lower', extent=[-bound, bound, -bound, bound], cmap='nipy_spectral', interpolation='nearest')

# Anchor points
anchors_np = pi.detach().cpu().numpy()

for i, p in enumerate(anchors_np):
    color = 'red' if dead[i] else 'white'
    ax1.scatter(p[0], p[1], c=color, edgecolors='black', s=30, zorder=3)

# mesh contour
im = ax2.imshow(grid.T, origin='lower', extent=[-bound, bound, -bound, bound], cmap='viridis')
plt.colorbar(im, ax=ax2, label='Predicted Distance')

for contour in world_contours:
    ax2.plot(contour[:, 0], contour[:, 1], color='white', linewidth=2, zorder=4)

ax2.set_title(f"Reconstructed Mesh Contour (Level: {contVal:.4f})")

plt.savefig("neural_voronoi_diagram_labeled.png", dpi=300)
plt.show()


#influence plot
cols = 6
rows = (N + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
axes = axes.flatten()

for i in range(N):
    w_i = weights[:, i].view(resolution, resolution).cpu().numpy()
    
    axes[i].imshow(w_i.T, origin='lower', extent=[-bound, bound, -bound, bound], cmap='gray')
    
    anchor_pos = pi[i].cpu().numpy()
    axes[i].scatter(anchor_pos[0], anchor_pos[1], color='red', s=10)
    
    axes[i].set_title(f"Anchor {i}")
    axes[i].axis('off')

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig("anchor_influence_grid.png", dpi=300)
plt.show()