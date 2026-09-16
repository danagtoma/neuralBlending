import torch
import numpy as np
import skimage
import matplotlib.pyplot as plt
import meshio
import polyscope as ps
from scipy.stats import gaussian_kde

# NN architecture
# Use the other line if this causes errors
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = torch.load("abacaOutput/lucy 100k 1mil/model3D.pth")

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

#Read mesh
mesh = meshio.read("abacaOutput/lucy 100k 1mil/reconstructedMesh.obj")

vertsMesh = mesh.points
facesMesh = mesh.cells_dict["triangle"]

#Surface points
def samplingSurfacePoints(no):
    APoints = vertsMesh[facesMesh[:, 0]]
    BPoints = vertsMesh[facesMesh[:, 1]]
    CPoints = vertsMesh[facesMesh[:, 2]]

    cross_prod = np.cross(BPoints - APoints, CPoints - APoints)
    facesAreas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    prob = facesAreas / facesAreas.sum()

    facesSampled = np.random.choice(len(facesMesh), no, p=prob)
    
    face_normals = cross_prod / np.linalg.norm(cross_prod, axis=1, keepdims=True)

    surfPoint = np.zeros((no, 3))
    sampledNormals = face_normals[facesSampled]

    for i, j in enumerate(facesSampled):
        A, B, C = (vertsMesh[v] for v in facesMesh[j])
        u, v = np.random.rand(2)
        surfPoint[i, :] = A + (1 - np.sqrt(u)) * (B - A) + v * np.sqrt(u) * (C - A)
    
    return surfPoint, sampledNormals

#Surface points with noise
def samplingBoundaryPoints(no, noise_ratio=0.875, sigma=0.01):
    surfPoint, _ = samplingSurfacePoints(no)
    
    num_noise = int(no * noise_ratio)

    noise = np.random.normal(0, sigma, (num_noise, 3))
    surfPoint[:num_noise] += noise
    
    return surfPoint 

#Computing weights on points close to boundary
no_samples = 3 * 1000 * 100
boundary_pts = samplingBoundaryPoints(no_samples)
pts_tensor = torch.tensor(boundary_pts, dtype=torch.float32)

batch_size = 2000 

running_max_weights = torch.zeros(N, device=device)
running_sum_weights = torch.zeros(N, device=device)

with torch.no_grad():
    for start_idx in range(0, no_samples, batch_size):
        end_idx = min(start_idx + batch_size, no_samples)
        batch_pts = pts_tensor[start_idx:end_idx].to(device)

        out = batch_pts
        for l in layers:
            out = activation(l(out))

        batch_weights = softmax(finalLayer(out))

        batch_max, _ = torch.max(batch_weights, dim=0)
        running_max_weights = torch.maximum(running_max_weights, batch_max)
        running_sum_weights += torch.sum(batch_weights, dim=0)

max_weights = running_max_weights
mean_weights = running_sum_weights / no_samples

max_weights_np = max_weights.cpu().numpy()
mean_weights_np = mean_weights.cpu().numpy()

print("Weight statistics computed")     

#Pruned inactive anchors
threshold = 0.001   

active_mask = max_weights > threshold
active_indices = torch.nonzero(active_mask).squeeze()
dead_indices = torch.nonzero(~active_mask).squeeze()

print(f"Total anchors: {N}")
print(f"Active anchors: {active_indices.numel()} Pruned anchors: {dead_indices.numel()}")

new_N = active_indices.numel()
pruned_finalLayer = torch.nn.Linear(hidden, new_N).to(device)

with torch.no_grad():
    pruned_finalLayer.weight.copy_(finalLayer.weight[active_mask])
    pruned_finalLayer.bias.copy_(finalLayer.bias[active_mask])

pi_pruned = pi[active_mask]
normals_pruned = normals[active_mask]

resolution = 200
bound = 1.2

#Computing sdf
xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)
zs = np.linspace(-bound, bound, resolution)

grid = np.zeros((resolution, resolution, resolution), np.float32)

X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
total_grid_pts = pts.shape[0]

f_x_pruned = np.zeros(total_grid_pts, dtype=np.float32)

eval_batch_size = 16384  

pts = torch.tensor(pts, dtype=torch.float32)

with torch.no_grad():
    for start_idx in range(0, total_grid_pts, eval_batch_size):
        end_idx = min(start_idx + eval_batch_size, total_grid_pts)
        batch_pts = pts[start_idx:end_idx].to(device)

        out = batch_pts
        for l in layers:
            out = activation(l(out))

        batch_weights = softmax(pruned_finalLayer(out))

        batch_max_idx = torch.argmax(batch_weights, dim=1)

        dist = batch_pts.unsqueeze(1) - pi_pruned.unsqueeze(0)
        sdf_dist = torch.sum(dist * normals_pruned.unsqueeze(0), dim=2) 

        batch_f_x = torch.sum(batch_weights * sdf_dist, dim=1)

        f_x_pruned[start_idx:end_idx] = batch_f_x.cpu().numpy()

grid_pruned = f_x_pruned.reshape(resolution, resolution, resolution)
print(np.min(grid_pruned), np.max(grid_pruned))

print("3D SDF grid computed")

fig, ax_hist = plt.subplots(figsize=(8, 5))

ax_hist.hist(max_weights_np, bins=100, color='royalblue', edgecolor='black', alpha=0.7)
ax_hist.axvline(threshold, color='red', linestyle='--', linewidth=2)

ax_hist.set_xlabel("Max Softmax Weight")
ax_hist.set_ylabel("Anchor Count")
ax_hist.legend()
ax_hist.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("anchor_weight_distribution.png", dpi=300)

# Log scale
fig, ax = plt.subplots(figsize=(8, 5))

sorted_w = np.sort(max_weights_np)
y_percent = np.arange(1, len(sorted_w) + 1) / len(sorted_w) * 100.0

ax.plot(sorted_w, y_percent, color='royalblue', linewidth=2, label='Cumulative Distribution')
ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold})')

active_pct = (max_weights_np > threshold).mean() * 100.0
ax.axhline(100.0 - active_pct, color='gray', linestyle=':', label=f'{100.0 - active_pct:.1f}% Pruned')

ax.set_title("Empirical Cumulative Distribution of Anchor Influence")
ax.set_xlabel("Max Softmax Weight")
ax.set_ylabel("Cumulative Percentage of Anchors (%)")
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0, 102])
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig("anchor_weights_ecdf.png", dpi=300)

# Computing anchors per points
no_anchors = N 
points_per_anchor = torch.zeros(no_anchors, dtype=torch.int64, device=device)

with torch.no_grad():
    for start_idx in range(0, no_samples, batch_size):
        end_idx = min(start_idx + batch_size, no_samples)
        batch_pts = pts_tensor[start_idx:end_idx].to(device)

        out = batch_pts
        for l in layers:
            out = activation(l(out))

        batch_weights = softmax(finalLayer(out))

        batch_anchor_influence = torch.sum(batch_weights > threshold, dim=0)
        points_per_anchor += batch_anchor_influence

points_per_anchor = points_per_anchor.cpu().numpy()
print(f"{no_samples} points")
print(f"Min anchors per point: {points_per_anchor.min()}")
print(f"Max anchors per point: {points_per_anchor.max()}")
print(f"Mean anchors per point: {points_per_anchor.mean():.2f}")

#Histogram
fig, ax = plt.subplots(figsize=(12, 5))

anchor_indices = np.arange(len(points_per_anchor))

ax.hist(points_per_anchor, bins=30, edgecolor="black")

ax.set_ylabel("Anchors")
ax.set_xlabel("Number of points influenced")
ax.legend()
ax.set_yscale("log")

plt.tight_layout()
plt.savefig("points_per_anchor.png", dpi=300)
plt.show()

#sorted
sorted_counts = np.sort(points_per_anchor)
point_indices = np.arange(len(sorted_counts))  

fig, ax = plt.subplots(figsize=(6, 8))

sorted_anchor_counts = np.sort(points_per_anchor)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(sorted_anchor_counts, linewidth=2)

ax.set_xlabel("Sorted anchors")
ax.set_ylabel("Number of points influenced")
ax.legend()

plt.tight_layout()
plt.savefig("ordered_active_counts.png", dpi=300)
plt.show()

# Marching squares
verts, faces, _, _ = skimage.measure.marching_cubes(grid_pruned, 0.0) 

scale = np.array([xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]])
offset = np.array([xs[0], ys[0], zs[0]])
verts = verts * scale + offset

#save mesh
mesh = meshio.Mesh(verts, [("triangle", faces)])
meshio.write("reconstructedMesh.obj", mesh)

anchors_np = pi_pruned.detach().cpu().numpy()
normals_np = normals_pruned.detach().cpu().numpy()
weights_np = max_weights_np[active_mask.cpu().numpy()]

anchor_cloud = meshio.Mesh(
    points=anchors_np,
    cells=[],
    point_data={
        "normals": normals_np,
        "max_weight": weights_np,
    }
)
meshio.write("active_anchors.obj", anchor_cloud)

ps.init()
ps.set_ground_plane_mode("none")

ps.register_surface_mesh("reconstructed mesh", verts, faces, enabled=True)

anchors_active_np = pi_pruned.detach().cpu().numpy()
ps_anchors = ps.register_point_cloud("active anchors", anchors_active_np, radius=0.003, enabled=True)

ps.show()
print("meshed saved")