import os
import sys
import numpy as np
from pyevtk.hl import imageToVTK
import torch

mesh_name = sys.argv[1]

target_anchors = [60, 70, 200, 300, 600, 2000, 5000]

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(f"outputs/{mesh_name}", exist_ok=True)

model = torch.load(f"outputs/{mesh_name}/model3D.pth", map_location=device)
pi = model["reference_points"].to(device)
N = pi.shape[0]

layers = torch.nn.ModuleList()
inDim = 3
hidden = 128
noLayers = 6

for _ in range(noLayers):
    layers.append(torch.nn.Linear(inDim, hidden))
    inDim = hidden

finalLayer = torch.nn.Linear(hidden, N)
activation = torch.nn.ReLU()
softmax = torch.nn.Softmax(dim=-1)

for l, w in zip(layers, model["layers"]):
    l.load_state_dict(w)
finalLayer.load_state_dict(model["finalLayer"])

layers = layers.to(device).eval()
finalLayer = finalLayer.to(device).eval()

print("Model loaded successfully.")

resolution = 400
bound = 1.2

xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)
zs = np.linspace(-bound, bound, resolution)

anchor_grids = {
    anchor_id: np.zeros((resolution, resolution, resolution), dtype=np.float32)
    for anchor_id in target_anchors
}

print(f"Computing activation volumes for anchors: {target_anchors}...")
with torch.no_grad():
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            pts_np = np.stack([np.full_like(zs, x), np.full_like(zs, y), zs], axis=1)
            pts = torch.tensor(pts_np, dtype=torch.float32, device=device)

            out = pts
            for l in layers:
                out = activation(l(out))
            weights = softmax(finalLayer(out)) 

            for anchor_id in target_anchors:
                anchor_grids[anchor_id][i, j, :] = (weights[:, anchor_id].cpu().numpy())

for anchor_id, grid_data in anchor_grids.items():
    clean_grid = np.ascontiguousarray(grid_data, dtype=np.float32)
    save_path = f"outputs/{mesh_name}/anchor_{anchor_id}_influence"

    imageToVTK(
        save_path, pointData={f"anchor_{anchor_id}_influence": clean_grid}
    )
    print(f"Saved: {save_path}.vti")

print("All target anchors exported successfully.")
