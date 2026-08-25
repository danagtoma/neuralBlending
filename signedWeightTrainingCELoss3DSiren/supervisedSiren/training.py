import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os

mesh_name = sys.argv[1]

# Use the other line if this causes errors
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

os.makedirs(f"outputs/{mesh_name}", exist_ok=True)
data = np.load(f"data/{mesh_name}/sampledPointsSiren3d.npz")
allPoint = data["allPoint"]
S_surf = data["S_surf"]

#Training data
pointsTens = torch.from_numpy(allPoint).float().to(device)
sdTens = torch.from_numpy(S_surf).float().to(device)

# Build NN
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

layers.to(device)     
finalLayer.to(device)

params = list(layers.parameters()) + list(finalLayer.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4)

# Training
batchSize = 256
epochs = 100

startTime = time.time()

sdfLossPlot = []

print("Pretraining on Sphere SDF")
sphere_epochs = 20
sphere_radius = 0.5  


for p_epoch in range(sphere_epochs):
    totalBatchSize = 0
    epochLoss = 0

    p_pts = (torch.rand(batchSize, 3, device=device) * 2.4) - 1.2
    gt_sphere_sdf = torch.norm(p_pts, dim=1) - sphere_radius

    batchSizeCurr = p_pts.size(0)
    
    out = p_pts
    for l in layers:
        out = torch.sin(w0 * l(out))
    pred_sphere_sdf = finalLayer(out).squeeze()
        
    sphereLoss = torch.nn.functional.mse_loss(pred_sphere_sdf, gt_sphere_sdf)
    
    optimizer.zero_grad()
    sphereLoss.backward()
    optimizer.step()
    
    epochLoss += sphereLoss.item() * batchSizeCurr
    totalBatchSize += batchSizeCurr

    avgSphereLoss = epochLoss / totalBatchSize

    elapsedTime = time.time() - startTime
    mins, secs = divmod(elapsedTime, 60)    
            
    print(f"pre-train epoch {p_epoch}  sphereLoss: {avgSphereLoss} time {int(mins)}m {int(secs)}s")


print("SIREN training")

for epoch in range(epochs):
    perm = torch.randperm(pointsTens.shape[0])
    totalBatchSize = 0
    epochLoss = 0

    for i in range(0, pointsTens.shape[0], batchSize):
        idx = perm[i:i+batchSize]
        x = pointsTens[idx].clone()
        gt = sdTens[idx]

        batchSizeCurr = x.size(0)

        out = x
        for l in layers:
            out = torch.sin(w0 * l(out))
        pred = finalLayer(out).squeeze()

        MSEloss = torch.nn.functional.mse_loss(pred,gt)

        optimizer.zero_grad()
        MSEloss.backward()
        optimizer.step()

        epochLoss += MSEloss.item() * batchSizeCurr
        totalBatchSize += batchSizeCurr

    avgMSEloss = epochLoss / totalBatchSize
    sdfLossPlot.append(avgMSEloss)

    elapsedTime = time.time() - startTime
    mins, secs = divmod(elapsedTime, 60)    

    print(f"train epoch {epoch} MSEloss {avgMSEloss} time {int(mins)}m {int(secs)}s")

torch.save({
    "layers": [l.state_dict() for l in layers],
    "finalLayer": finalLayer.state_dict()
}, f"outputs/{mesh_name}/modelSiren3D.pth")
print("model saved")

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(sdfLossPlot, color='blue', label='SDF Loss')
ax.set_title("SDF Training Convergence")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
plt.savefig(f"outputs/{mesh_name}/training_loss.png")

fig2, ax_log = plt.subplots(figsize=(10, 8))
ax_log.plot(sdfLossPlot, color='blue', label='SDF Loss')
ax_log.set_yscale('log') 
ax_log.set_title("SDF Training Convergence (Log Scale)")
ax_log.set_xlabel("Epoch")
ax_log.set_ylabel("MSE Loss (Log Scale)")
plt.savefig(f"outputs/{mesh_name}/training_loss_log.png")