import torch
import numpy as np
import skimage
import meshio
import time
from pyevtk.hl import imageToVTK
import igl
import sys
import os

mesh_name = sys.argv[1]

# NN architecture
# Use the other line if this causes errors
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

os.makedirs(f"outputs/{mesh_name}", exist_ok=True)

meshOrg = meshio.read(f"Meshes/{mesh_name}.obj")

min_box = meshOrg.points.min(axis=0)
max_box = meshOrg.points.max(axis=0)
center = (min_box + max_box) / 2.0

scale_factor = (max_box - min_box).max() / 2.0
vertsMeshOrg = (meshOrg.points - center) / scale_factor
facesMeshOrg = meshOrg.cells_dict["triangle"]

model = torch.load(f"outputs/{mesh_name}/model3D.pth")
# model = torch.load("f"outputs/{mesh_name}/modelCE3D.pth")

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
resolution = 400
bound = 1.2
targetAnchor = 1

xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)
zs = np.linspace(-bound, bound, resolution)

grid = np.zeros((resolution, resolution, resolution), np.float32)
startTime = time.time()

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

dx = xs[1] - xs[0]
dy = ys[1] - ys[0]
dz = zs[1] - zs[0]

grad_x, grad_y, grad_z = np.gradient(grid, dx, dy, dz)

gradienNorm = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

mean_norm = np.mean(gradienNorm)
std_norm = np.std(gradienNorm)
print(f"Gradient Norm: Mean: {mean_norm}, Std Dev: {std_norm}") 
with open(f"outputs/{mesh_name}/influenceMetric.txt", "a") as f:
    f.write(f"Gradient Norm: Mean: {mean_norm}, Std Dev: {std_norm}\n") 

elapsedTime = time.time() - startTime
mins, secs = divmod(elapsedTime, 60) 
print(f"Time taken to create the gradient: {int(mins)}m {int(secs)}s")   
with open(f"outputs/{mesh_name}/influenceMetric.txt", "a") as f:
    f.write(f"Time taken to create the SDF: {int(mins)}m {int(secs)}s\n")

print(np.min(grid), np.max(grid))

#Save vtk file
gradNormClean = np.ascontiguousarray(gradienNorm, dtype=np.float32)
imageToVTK(f"outputs/{mesh_name}/gradienNorm", pointData={"gradienNorm": gradNormClean})

print("done")
