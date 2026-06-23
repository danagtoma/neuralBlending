import torch
import numpy as np
import skimage
import matplotlib.pyplot as plt
import meshio
import polyscope as ps

# NN architecture
# Use the other line if this causes errors
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = torch.load("model3D.pth")
# model = torch.load("modelCE3D.pth")

pi = model["reference_points"].to(device)
N = pi.shape[0]
normals = model["reference_normals"].to(device)

layers = torch.nn.ModuleList()

inDim = 3
hidden = 128
noLayers = 6
for i in range(noLayers):
    layers.append(torch.nn.Linear(inDim, hidden))
    inDim = hidden

finalLayer = torch.nn.Linear(hidden, N)
activation = torch.nn.ReLU()
softmax = torch.nn.Softmax(dim=-1)

# Loading the model
for l, w in zip(layers, model["layers"]):
    l.load_state_dict(w)
finalLayer.load_state_dict(model["finalLayer"])

layers = layers.to(device)
finalLayer = finalLayer.to(device)

print("model loaded")

# SDF creation
influence_threshold = 0.001
resolution = 128
bound = 1.2

xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)
zs = np.linspace(-bound, bound, resolution)

grid = np.zeros((resolution, resolution, resolution), np.float32)
voronoi_grid = np.zeros((resolution, resolution, resolution), np.float32)

with torch.no_grad():
    for i, x in enumerate(xs):
        for j,y in enumerate(ys):
            pts = np.stack([np.full_like(zs, x), np.full_like(zs, y), zs], 1)
            pts = torch.tensor(pts, dtype=torch.float32, device=device)

            out = pts
            for l in layers:
                out = activation(l(out))
            weights = softmax(finalLayer(out)) 

            dist = pts.unsqueeze(1) - pi.unsqueeze(0)
            sdfDist = torch.sum(dist * normals, dim=2) 

            activeAnchors = weights > influence_threshold
            weightsThreshold = torch.where(activeAnchors, weights, torch.zeros_like(weights))

            f_x = torch.sum(weightsThreshold * sdfDist, dim=1)
            
            grid[i,j,:] = f_x.squeeze().cpu().numpy()

print("SDF created")  

print(np.min(grid), np.max(grid))
half = (np.min(grid) + np.max(grid))/2
eps = max(1e-4, np.percentile(grid, 1))
contVal = 0.0

# Marching squares
verts, faces, _, _ = skimage.measure.marching_cubes(grid, 0.0) 

scale = np.array([xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]])
offset = np.array([xs[0], ys[0], zs[0]])
verts = verts * scale + offset

#save mesh
mesh = meshio.Mesh(verts, [("triangle", faces)])
meshio.write("reconstructedMesh.obj", mesh)

print("meshed saved")

#polyscope
avgWeights = weights.mean(dim=0).cpu().numpy()
maxWeights = weights.max(dim=0).values.cpu().numpy()
dead = avgWeights < 0.001
anchorsNp = pi.detach().cpu().numpy()

meshOrg = meshio.read("Meshes/3D/happy.obj")

min_box = meshOrg.points.min(axis=0)
max_box = meshOrg.points.max(axis=0)
center = (min_box + max_box) / 2.0

scale = (max_box - min_box).max() / 2.0
vertsOrg = (meshOrg.points - center) / scale
facesOrg = meshOrg.cells_dict["triangle"]


ps.init()
ps.set_ground_plane_mode("none")
ps.register_surface_mesh("reconstructed mesh", verts, faces, enabled=True)
ps.register_surface_mesh("original mesh", vertsOrg, facesOrg, enabled=False)
ps.show()


# # Diagrams
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# # Voronoi regions
# ax1.imshow(voronoi_grid.T, origin='lower', extent=[-bound, bound, -bound, bound], cmap='nipy_spectral', interpolation='nearest')

# # Anchor points
# anchors_np = pi.detach().cpu().numpy()

# for i, p in enumerate(anchors_np):
#     color = 'red' if dead[i] else 'white'
#     ax1.scatter(p[0], p[1], c=color, edgecolors='black', s=30, zorder=3)

# # mesh contour
# im = ax2.imshow(grid.T, origin='lower', extent=[-bound, bound, -bound, bound], cmap='viridis')
# plt.colorbar(im, ax=ax2, label='Predicted Distance')

# for contour in world_contours:
#     ax2.plot(contour[:, 0], contour[:, 1], color='white', linewidth=2, zorder=4)


# plt.savefig("neural_voronoi_diagram_labeled.png", dpi=300)
# plt.show()


# #influence plot
# cols = 6
# rows = (N + cols - 1) // cols
# fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
# axes = axes.flatten()

# for i in range(N):
#     w_i = (weights[:, i]).view(resolution, resolution).cpu().numpy()
    
#     im = axes[i].imshow(w_i.T, origin='lower', extent=[-bound, bound, -bound, bound], cmap='gray', vmin=0.0, vmax=1.0)
    
#     anchor_pos = pi[i].cpu().numpy()
#     axes[i].scatter(anchor_pos[0], anchor_pos[1], color='red', s=20)
    
#     cbar = fig.colorbar(im, ax=axes[i], orientation='vertical', pad=0.04, shrink=0.8)
#     cbar.ax.tick_params(labelsize=8)
    
#     axes[i].set_title(f"Anchor {i}")
#     axes[i].axis('off')

# for j in range(i + 1, len(axes)):
#     axes[j].axis('off')

# plt.tight_layout()
# plt.savefig("anchor_influence_grid.png", dpi=300)


# # weight*dist influence plot

# cols = 6
# rows = (N + cols - 1) // cols
# fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
# axes = axes.flatten()

# data_list = []
# max_abs = 0.0

# for i in range(N):
#     wd_i = (weights[:, i] * sdfDist[:, i]).view(resolution, resolution).cpu().numpy()
#     data_list.append(wd_i)
    
#     local_max = np.max(np.abs(wd_i))
#     if local_max > max_abs:
#         max_abs = local_max

# for i in range(N):
#     wd_i = data_list[i]

#     im = axes[i].imshow(wd_i.T, origin='lower', extent=[-bound, bound, -bound, bound], cmap='PiYG', vmin=-max_abs, vmax=max_abs)
    
#     anchor_pos = pi[i].cpu().numpy()
#     axes[i].scatter(anchor_pos[0], anchor_pos[1], color='red', s=20)
    
#     cbar = fig.colorbar(im, ax=axes[i], orientation='vertical', pad=0.04, shrink=0.8)
#     cbar.ax.tick_params(labelsize=8)

#     axes[i].set_title(f"Anchor {i}")
#     axes[i].axis('off')

# for j in range(i + 1, len(axes)):
#     axes[j].axis('off')


# plt.tight_layout(pad=2.0)
# plt.savefig("weightDist_influence_grid.png", dpi=300)



# #number of anchors that influnece a point
active_anchors = (weights > influence_threshold).float()
influence_count = torch.sum(active_anchors, dim=1)
influence_count_max = influence_count.max()
print(f"max number of anchors that influnece a point {influence_count_max}") 

with open("influenceMetric.txt", "w") as f:
    f.write(f"max number of anchors influence a point: {influence_count_max}\n")

# influence_grid = influence_count.view(resolution, resolution).cpu().numpy()

# fig, ax = plt.subplots(figsize=(10, 8))

# max_count = int(np.max(influence_grid))
# min_count = int(np.min(influence_grid))

# im = ax.imshow(influence_grid.T, origin='lower', extent=[-bound, bound, -bound, bound], cmap='viridis', interpolation='nearest', vmin=min_count, vmax=max_count)

# cbar = plt.colorbar(im, ax=ax, ticks=range(min_count, max_count + 1))
# cbar.ax.tick_params(labelsize=10)

# anchors_np = pi.detach().cpu().numpy()
# for i, p in enumerate(anchors_np):
#     color = 'red' if dead[i] else 'white'
#     ax.scatter(p[0], p[1], c=color, edgecolors='black', s=30, zorder=4)

# ax.set_xlabel("X")
# ax.set_ylabel("Y")

# plt.savefig("anchor_influence_count.png", dpi=300, bbox_inches='tight')


print("done")