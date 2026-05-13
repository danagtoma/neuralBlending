import torch
import numpy as np
import skimage
import meshio
import polyscope as ps

# Use the other line if this causes errors
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# NN architecture
layers = torch.nn.ModuleList()

inDim = 3
hidden = 128
noLayers = 6
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
model = torch.load("models/modelSirenArmadillo.pth")

for l, w in zip(layers, model["layers"]):
    l.load_state_dict(w)
finalLayer.load_state_dict(model["finalLayer"])

layers = layers.to(device)
finalLayer = finalLayer.to(device)

print("model loaded")


# SDF creation
resolution = 128
bound = 1.1

xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)
zs = np.linspace(-bound, bound, resolution)

grid = np.zeros((resolution, resolution, resolution), np.float32)

with torch.no_grad():
    for i, x in enumerate(xs):
        for j,y in enumerate(ys):
            pts = np.stack([np.full_like(zs, x), np.full_like(zs, y), zs], 1)
            pts = torch.tensor(pts, dtype=torch.float32, device=device)

            out = pts
            for l in layers:
                out = torch.sin(w0 * l(out))
            sdf = finalLayer(out)

            grid[i,j,:] = sdf.squeeze().cpu().numpy()

print("SDF created")  

print(np.min(grid), np.max(grid))
level = (np.min(grid) + np.max(grid))/2

# Marching cubes
verts, faces, _, _ = skimage.measure.marching_cubes(grid, 0.0) 

scale = np.array([xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]])
offset = np.array([xs[0], ys[0], zs[0]])
verts = verts * scale + offset

print("marching cubes applied")

#save mesh
mesh = meshio.Mesh(verts, [("triangle", faces)])
meshio.write("reconstructedMesh.obj", mesh)

print("meshed saved")


#polyscope
meshOrg = meshio.read("../Meshes/armadillo.obj")

vertsOrg = (meshOrg.points - meshOrg.points.min(axis=0)) / (meshOrg.points.max(axis=0) - meshOrg.points.min(axis=0)) * 2 - 1
facesOrg = meshOrg.cells_dict["triangle"]


ps.init()
ps.set_ground_plane_mode("none")
ps.register_surface_mesh("reconstructed mesh", verts, faces, enabled=True)
ps.register_surface_mesh("original mesh", vertsOrg, facesOrg, enabled=False)
ps.show()
