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
targetAnchor = 10

xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)
zs = np.linspace(-bound, bound, resolution)

grid = np.zeros((resolution, resolution, resolution), np.float32)
anchor_influence_grid = np.zeros((resolution, resolution, resolution), np.int32)
one_anchor_influence_grid = np.zeros((resolution, resolution, resolution), np.float32)
argmax_grid = np.zeros((resolution, resolution, resolution), np.int32)

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

            # argmax
            argmax_grid[i, j, :] = torch.argmax(weights, dim=-1).cpu().numpy()

            #One anchor influence
            one_anchor_influence_grid[i, j, :] = weights[:, targetAnchor].cpu().numpy()

            #Anchors influence
            anchor_influence_grid[i, j, :] = torch.sum(activeAnchors, dim=1).cpu().numpy()
            
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

grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
all_pts_np = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1)

gt_sdf_flat, _, _, _ = igl.signed_distance(all_pts_np, vertsMeshOrg, facesMeshOrg)
gt_sdf_grid = gt_sdf_flat.reshape(resolution, resolution, resolution)

error_SDF_grid = np.abs(grid - gt_sdf_grid)

print("SDF error calculated")  

elapsedTime = time.time() - startTime
mins, secs = divmod(elapsedTime, 60) 
print(f"Time taken to create the SDF: {int(mins)}m {int(secs)}s")   
with open(f"outputs/{mesh_name}/influenceMetric.txt", "w") as f:
    f.write(f"Time taken to create the SDF: {int(mins)}m {int(secs)}s\n")

print(np.min(grid), np.max(grid))

#Save vtk file
gridClean = np.ascontiguousarray(grid, dtype=np.float32)
imageToVTK(f"outputs/{mesh_name}/sdf", pointData={"SDF": gridClean})

anchor_influence_grid_clean = np.ascontiguousarray(anchor_influence_grid, dtype=np.float32)
imageToVTK(f"outputs/{mesh_name}/anchorInfluence", pointData={"anchorInfluence": anchor_influence_grid_clean})

one_anchor_influence_grid_clean = np.ascontiguousarray(one_anchor_influence_grid, dtype=np.float32)
imageToVTK(f"outputs/{mesh_name}/oneAnchorInfluence", pointData={"oneAnchorInfluence": one_anchor_influence_grid_clean})

argmax_grid_clean = np.ascontiguousarray(argmax_grid, dtype=np.float32)
imageToVTK(f"outputs/{mesh_name}/argmax", pointData={"argmax": argmax_grid_clean})

error_SDF_grid_clean = np.ascontiguousarray(error_SDF_grid, dtype=np.float32)
imageToVTK(f"outputs/{mesh_name}/errorSdf", pointData={"errorSdf": error_SDF_grid_clean})

# Marching squares
verts, faces, _, _ = skimage.measure.marching_cubes(grid, 0.0) 

scale = np.array([xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]])
offset = np.array([xs[0], ys[0], zs[0]])
verts = verts * scale + offset

#save mesh
mesh = meshio.Mesh(verts, [("triangle", faces)])
meshio.write(f"outputs/{mesh_name}/reconstructedMesh.obj", mesh)

print("meshed saved")


#number of anchors that influnece a point
print(f"max number of anchors that influnece a point lazy computing {max_influence_gpu}") 

with open(f"outputs/{mesh_name}/influenceMetric.txt", "a") as f:
    f.write(f"Max number of anchors influence a point: lazy computing: {max_influence_gpu}\n")

print("done")
