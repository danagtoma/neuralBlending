import torch
import numpy as np
import skimage
import matplotlib.pyplot as plt
import meshio
import polyscope as ps
import time

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
resolution = 400
bound = 1.2

xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)
zs = np.linspace(-bound, bound, resolution)

grid = np.zeros((resolution, resolution, resolution), np.float32)
voronoi_grid = np.zeros((resolution, resolution, resolution), np.float32)

startTime = time.time()

with torch.no_grad():
    pi_unsqueezed = pi.unsqueeze(0)       
    normals_unsqueezed = normals.unsqueeze(0)  
    for i, x in enumerate(xs):
        for j,y in enumerate(ys):
            pts = np.stack([np.full_like(zs, x), np.full_like(zs, y), zs], 1)
            pts = torch.tensor(pts, dtype=torch.float32, device=device)

            out = pts
            for l in layers:
                out = activation(l(out))
            weights = softmax(finalLayer(out)) 


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

            grid[i, j, :] = f_x_row.cpu().numpy()

            #SDF for all anchors
            # dist = pts.unsqueeze(1) - pi.unsqueeze(0)
            # sdfDist = torch.sum(dist * normals, dim=2) 

            # f_x = torch.sum(weights * sdfDist, dim=1)
            
            # grid[i,j,:] = f_x.squeeze().cpu().numpy()

print("SDF created")  
elapsedTime = time.time() - startTime
mins, secs = divmod(elapsedTime, 60) 
print(f"Time taken to create the SDF: {int(mins)}m {int(secs)}s")   
with open("influenceMetric.txt", "w") as f:
    f.write(f"Time taken to create the SDF: {int(mins)}m {int(secs)}s\n")

print(np.min(grid), np.max(grid))

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
meshOrg = meshio.read("Meshes/3D/beetle.obj")

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


# #number of anchors that influnece a point
active_anchors = (weights > influence_threshold).float()
influence_count = torch.sum(active_anchors, dim=1)
influence_count_max = influence_count.max()
print(f"max number of anchors that influnece a point {influence_count_max}") 

with open("influenceMetric.txt", "a") as f:
    f.write(f"max number of anchors influence a point: {influence_count_max}\n")

print("done")