import torch
import numpy as np
import skimage
import matplotlib.pyplot as plt
import time
import meshio

# NN architecture
# Use the other line if this causes errors
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = torch.load("model2D.pth")
# model = torch.load("modelCE2D.pth")

pi = model["reference_points"].to(device)
N = pi.shape[0]
normals = model["normals"].to(device)
gt_normals = model["gt_normals"].to(device)
gt_pi = model["gt_points"].to(device)

origins_trained = pi.detach().cpu().numpy()
origins_gt = gt_pi.detach().cpu().numpy()
normals_np = normals.detach().cpu().numpy()
gt_normals_np = gt_normals.detach().cpu().numpy()

cos_sim = torch.sum(normals * gt_normals, dim=-1)
print(f"Mean cosine alignment with GT normals: {cos_sim.mean().item():.4f}")

pos_drift = torch.norm(pi - gt_pi, dim=-1)
mean_drift = pos_drift.mean().item()
max_drift = pos_drift.max().item()
print(f"Anchor position drift: Mean: {mean_drift:.6f}, Max: {max_drift:.6f}")

with open("influenceMetric.txt", "a") as f:
    f.write(f"Mean cosine alignment with GT normals: {cos_sim.mean().item():.4f}")
    f.write(f"Anchor position drift -> Mean: {mean_drift:.6f}, Max: {max_drift:.6f}\n")    


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
resolution = 200
bound = 1.2

xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)
zs = np.linspace(-bound, bound, resolution)

grid = np.zeros((resolution, resolution, resolution), np.float32)

startTime = time.time()
max_influence_gpu = torch.tensor(0, dtype=torch.int32, device=device)

with torch.no_grad():
    pi_unsqueezed = pi.unsqueeze(0)       
    normals_unsqueezed = normals.unsqueeze(0)  
    for i, x in enumerate(xs):
        for j,y in enumerate(ys):
            pts_np = np.stack([np.full_like(zs, x), np.full_like(zs, y), zs], 1)
            pts = torch.tensor(pts_np, dtype=torch.float32, device=device)
    
            out = pts
            for l in layers:
                out = activation(l(out))
            weights = softmax(finalLayer(out)) 
    
            activeAnchors = weights > influence_threshold

            # Max number non-zero weights
            influence_counts = torch.sum(activeAnchors, dim=1)
            current_max_gpu = influence_counts.max()
            max_influence_gpu = torch.max(max_influence_gpu, current_max_gpu)
    
            #SDF only for active anchors
            activeAnchors = weights > influence_threshold 
            activeWeights = weights[activeAnchors]
    
            z_indices, anchor_indices = torch.where(activeAnchors)
    
            pts_filtered = pts[z_indices]                      
            pi_filtered = pi_unsqueezed[0, anchor_indices]     
            normals_filtered = normals_unsqueezed[0, anchor_indices]
    
            dist = pts_filtered - pi_filtered                  
            sdfDist = torch.sum(dist * normals_filtered, dim=1)
    
            f_x_row = torch.zeros(resolution, device=device)
            f_x_row.index_add_(0, z_indices, activeWeights * sdfDist)
    
            pred_sdf_row = f_x_row.cpu().numpy()
            grid[i, j, :] = pred_sdf_row
print("SDF created")  

elapsedTime = time.time() - startTime
mins, secs = divmod(elapsedTime, 60) 
print(f"Time taken to create the SDF: {int(mins)}m {int(secs)}s")   

print(np.min(grid), np.max(grid))
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

np.savez(
    "anchor_normals_pc.npz",
    points=origins_trained,
    gt_points=origins_gt,
    trained_normals=normals_np,
    gt_normals=gt_normals_np,
    position_drift=pos_drift.detach().cpu().numpy(),
    cosine_alignment=cos_sim.detach().cpu().numpy()
)

#number of anchors that influnece a point
print(f"max number of anchors that influnece a point lazy computing {max_influence_gpu}") 
with open("influenceMetric.txt", "a") as f:
    f.write(f"Max number of anchors influence a point: lazy computing: {max_influence_gpu}\n")

print("done")