import torch
import numpy as np
import skimage
import meshio
import time
from pyevtk.hl import imageToVTK
import sys
import os
import igl

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

model = torch.load(f"outputs/{mesh_name}/modelSiren3D.pth")

layers = torch.nn.ModuleList()

inDim = 3
hidden = 128
noLayers = 6
w0 = 5
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
influence_threshold = 0.001
resolution = 400
bound = 1.2

xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)
zs = np.linspace(-bound, bound, resolution)

grid = np.zeros((resolution, resolution, resolution), np.float32)

startTime = time.time()

with torch.no_grad(): 
    for i, x in enumerate(xs):
        for j,y in enumerate(ys):
            pts = np.stack([np.full_like(zs, x), np.full_like(zs, y), zs], 1)
            pts = torch.tensor(pts, dtype=torch.float32, device=device)

            out = pts
            for l in layers:
                out = torch.sin(w0 * l(out))
            sdf = finalLayer(out)

            grid[i, j, :] = sdf.squeeze().cpu().numpy()

print("SDF created")  

grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
all_pts_np = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1)

gt_sdf_flat, _, _, _ = igl.signed_distance(all_pts_np, vertsMeshOrg, facesMeshOrg)
gt_sdf_grid = gt_sdf_flat.reshape(resolution, resolution, resolution)

error_SDF_grid = np.abs(grid - gt_sdf_grid)

elapsedTime = time.time() - startTime
mins, secs = divmod(elapsedTime, 60) 
print(f"Time taken to create the SDF: {int(mins)}m {int(secs)}s")   
with open(f"outputs/{mesh_name}/influenceMetric.txt", "w") as f:
    f.write(f"Time taken to create the SDF: {int(mins)}m {int(secs)}s\n")

print(np.min(grid), np.max(grid))

#Save vtk file
gridClean = np.ascontiguousarray(grid, dtype=np.float32)
imageToVTK(f"outputs/{mesh_name}/sdf", pointData={"SDF": gridClean})

error_SDF_grid_clean = np.ascontiguousarray(error_SDF_grid, dtype=np.float32)
imageToVTK(f"outputs/{mesh_name}/errorSdf", pointData={"errorSdf": error_SDF_grid_clean})

# Marching squares
verts, faces, _, _ = skimage.measure.marching_cubes(grid, 0.0) 

scale = np.array([xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]])
offset = np.array([xs[0], ys[0], zs[0]])
verts = verts * scale + offset

#save mesh
mesh = meshio.Mesh(verts, [("triangle", faces)])
meshio.write(f"outputs/{mesh_name}/reconstructedSirenMesh.obj", mesh)

print("meshed saved")

print("done")