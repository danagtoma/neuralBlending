import torch
import numpy as np
import skimage
import meshio
import polyscope as ps
from samplingPoints2D import vertsMeshNorm, facesMesh

# Use the other line if this causes errors
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

layers = torch.nn.ModuleList()

inDim = 3
hidden = 256
noLayers = 6

for i in range(noLayers):
    layers.append(torch.nn.Linear(inDim, hidden))
    inDim = hidden

finalLayer = torch.nn.Linear(hidden, 1)
activation = torch.nn.ReLU()


# Loading the model
model = torch.load("Models/cheval.pth")

for l, w in zip(layers, model["layers"]):
    l.load_state_dict(w)
finalLayer.load_state_dict(model["finalLayer"])

layers = layers.to(device)
finalLayer = finalLayer.to(device)

print("model loaded")


# SDF creation
resolution = 128
bound = 1.0
z_res = 10   
z_bound = 0.01

xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)
zs = np.linspace(-z_bound, z_bound, z_res)

grid = np.zeros((resolution, resolution, z_res), np.float32)

with torch.no_grad():
    for k, z in enumerate(zs):

        X, Y = np.meshgrid(xs, ys)
        Z = np.full_like(X, z)

        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        pts = torch.tensor(pts, dtype=torch.float32, device=device)

        out = pts
        for l in layers:
            out = activation(l(out))
        sdf = finalLayer(out)

        grid[:, :, k] = sdf.view(resolution, resolution).cpu().numpy()

print("SDF created")  

print(np.min(grid), np.max(grid))
level = (np.min(grid) + np.max(grid))/2

#Marching cubes
verts, faces, _, _ = skimage.measure.marching_cubes(grid, 0.0)        

scale = np.array([xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]])
offset = np.array([xs[0], ys[0], zs[0]])
verts = verts[:, [1, 0, 2]]
verts = verts * scale + offset

#save mesh
mesh = meshio.Mesh(verts, [("triangle", faces)])
meshio.write("reconstructedMesh2D.obj", mesh)

print("meshed saved")


#polyscope
ps.init()
ps.set_ground_plane_mode("none")
ps.register_surface_mesh("reconstructed mesh", verts, faces, enabled=True)
ps.register_surface_mesh("original mesh", vertsMeshNorm, facesMesh, enabled=False)

ps.show()
