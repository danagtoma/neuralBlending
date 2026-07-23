import torch
import numpy as np
import skimage
import matplotlib.pyplot as plt

# NN architecture
# Use the other line if this causes errors
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = torch.load("model2D.pth")

layers = torch.nn.ModuleList()

inDim = 2
hidden = 64
noLayers = 4
w0 = 30
c = 6

for i in range(noLayers):
    layer = torch.nn.Linear(inDim, hidden)
    with torch.no_grad(): 
        if i == 0: layer.weight.uniform_(-1 / inDim, 1 / inDim)
        else: layer.weight.uniform_(-np.sqrt(c/inDim) / w0, np.sqrt(c/inDim) / w0)
    
    layers.append(layer)
    inDim = hidden

finalLayer = torch.nn.Linear(hidden, 1)
with torch.no_grad(): 
    finalLayer.weight.uniform_(-np.sqrt(c/hidden)/w0, np.sqrt(c/hidden)/w0)

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

xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)

grid = np.zeros((resolution, resolution), np.float32)

with torch.no_grad():
    X, Y = np.meshgrid(xs, ys, indexing='ij')
        
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    pts = torch.tensor(pts, dtype=torch.float32, device=device)

    out = pts
    for l in layers:
        out = torch.sin(w0 * l(out))

    sdf = finalLayer(out)

    grid = sdf.view(resolution, resolution).cpu().numpy()

print("SDF created")  

print(np.min(grid), np.max(grid))
half = (np.min(grid) + np.max(grid))/2
eps = max(1e-4, np.percentile(grid, 1))
contVal = 0.0

# Diagrams
contour_levels = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
fig, ax = plt.subplots(figsize=(10, 8))

# mesh contour
im = ax.imshow(grid.T, origin='lower', extent=[-bound, bound, -bound, bound], cmap='viridis')
plt.colorbar(im, ax=ax, label='Predicted Distance')

for level in contour_levels:
    contours = skimage.measure.find_contours(grid, level)
    
    if np.isclose(level, 0.0):
        color = 'red'      
        linewidth = 3.5     
        label_text = 'Level 0'
    else:
        color = 'white'     
        linewidth = 1.2    
        label_text = None
        
    for i, c in enumerate(contours):
        c_world = np.zeros_like(c)
        c_world[:, 0] = c[:, 0] * (xs[1] - xs[0]) + xs[0]
        c_world[:, 1] = c[:, 1] * (ys[1] - ys[0]) + ys[0]
        
        ax.plot(c_world[:, 0], c_world[:, 1], color=color, linewidth=linewidth, zorder=4,label=label_text if i == 0 else "")

ax.legend(loc='upper right')
plt.savefig("neural_voronoi_diagram_labeled.png", dpi=300)
plt.show()
print("done")