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
pointsOnSurface = data["pointsOnSurface"]
pointsOnSurfaceNormals = data["pointsOnSurfaceNormals"]

#Training data
pointsTens = torch.from_numpy(allPoint).float().to(device)
pointsOnSurfaceNormalsTens = torch.from_numpy(pointsOnSurfaceNormals).float().to(device)
pointsOnSurfaceTens = torch.from_numpy(pointsOnSurface).float().to(device)

# Build NN
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

layers.to(device)     
finalLayer.to(device)

params = list(layers.parameters()) + list(finalLayer.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4)

# Training
batchSize = 256
epochs = 150

alpha = 100.0        
lambda_on = 3000.0      
lambda_out = 100.0     
lambda_eik = 50.0
lambda_norm = 100.0

startTime = time.time()

sdfLossPlot = []

print("SIREN training")

num_samples = min(pointsTens.shape[0], pointsOnSurfaceTens.shape[0])

for epoch in range(epochs):
    perm_out = torch.randperm(pointsTens.shape[0])
    perm_on = torch.randperm(pointsOnSurfaceTens.shape[0])
    
    totalBatchSize = 0
    epochLoss = 0

    for i in range(0, num_samples, batchSize):
        idx_out = perm_out[i:i+batchSize]
        x_out = pointsTens[idx_out].clone().detach().requires_grad_(True)
        
        idx_on = perm_on[i:i+batchSize]
        x_on = pointsOnSurfaceTens[idx_on].clone().detach().requires_grad_(True)
        gt_on = pointsOnSurfaceNormalsTens[idx_on]

        batchSizeCurr = x_on.size(0)

        #Surface loss
        out_on = x_on
        for l in layers:
            out_on = torch.sin(w0 * l(out_on))
        pred_on = finalLayer(out_on).squeeze()

        loss_on = torch.abs(pred_on).mean()

        #Off Surface loss
        out_out = x_out
        for l in layers:
            out_out = torch.sin(w0 * l(out_out))
        pred_out = finalLayer(out_out).squeeze()

        loss_out = torch.exp(-alpha * torch.abs(pred_out)).mean()

        #Eikonal loss
        grad_out = torch.autograd.grad(outputs=pred_out, inputs=x_out, grad_outputs=torch.ones_like(pred_out),create_graph=True, retain_graph=True, only_inputs=True)[0]  
        grad_norm = grad_out.norm(2, dim=-1)
        loss_eik = torch.abs(grad_norm - 1.0).mean()

        #Normal loss
        grad_on = torch.autograd.grad(outputs=pred_on, inputs=x_on, grad_outputs=torch.ones_like(pred_on), create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_on_norm = torch.nn.functional.normalize(grad_on, dim=-1)
        dot_product = torch.sum(grad_on_norm * gt_on, dim=-1)
        loss_normals = (1.0 - dot_product).mean()

        total_loss = lambda_norm * loss_normals + lambda_on * loss_on + lambda_out * loss_out + lambda_eik * loss_eik

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epochLoss += total_loss.item() * batchSizeCurr
        totalBatchSize += batchSizeCurr

    avgMSEloss = epochLoss / totalBatchSize
    sdfLossPlot.append(avgMSEloss)

    elapsedTime = time.time() - startTime
    mins, secs = divmod(elapsedTime, 60)    

    print(f"train epoch {epoch} Loss {avgMSEloss} time {int(mins)}m {int(secs)}s")

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