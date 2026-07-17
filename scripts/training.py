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

data = np.load(f"data/{mesh_name}/sampledPoints3d.npz")
allPoint = data["allPoint"]
refPoints = data["refPoints"]
refNormals = data["refNormals"]
S_surf = data["S_surf"]

#Reference points
pi = torch.from_numpy(refPoints).float().to(device)
N = pi.shape[0]

#Training data
pointsTens = torch.from_numpy(allPoint).float()
normalTens = torch.from_numpy(refNormals).float()
sdTens = torch.from_numpy(S_surf).float()

# Build NN
layers = torch.nn.ModuleList()

inDim = 3
hidden = 128
noLayers = 6

for i in range(noLayers):
    layers.append(torch.nn.Linear(inDim, hidden))
    inDim = hidden

finalLayer = torch.nn.Linear(hidden, N)
softmax = torch.nn.Softmax(dim=-1)

layers.to(device)     
finalLayer.to(device)

params = list(layers.parameters()) + list(finalLayer.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4)
activation = torch.nn.ReLU()

ceOptimizer = torch.optim.Adam(params, lr=1e-3)
ceEpochs = 100

# Training
batchSize = 256
epochs = 1000
sigma = 0.01
h = 0.02

startTime = time.time()

labels = torch.zeros(pointsTens.shape[0], dtype=torch.long, device=device)
chunkSize = 10000  

for i in range(0, pointsTens.shape[0], chunkSize):
    pointsChunk = pointsTens[i : i + chunkSize].to(device)
    distChunk = torch.cdist(pointsChunk, pi)
    labels[i : i + chunkSize] = torch.argmin(distChunk, dim=1)

ceLossPlot = []
sdfLossPlot = []

print("initialization")

for epoch in range(ceEpochs):
    perm = torch.randperm(pointsTens.shape[0])
    totalBatchSize = 0
    epochLoss = 0
    
    for i in range(0, pointsTens.shape[0], batchSize):
        idx = perm[i:i+batchSize]
        x = pointsTens[idx].to(device)
        gtLabels = labels[idx].to(device)

        batchSizeCurr = x.size(0)
        
        out = x
        for l in layers:
            out = activation(l(out))

        CEloss = torch.nn.functional.cross_entropy(finalLayer(out), gtLabels)
        
        ceOptimizer.zero_grad()
        CEloss.backward()
        ceOptimizer.step()

        epochLoss += CEloss.item() * batchSizeCurr
        totalBatchSize += batchSizeCurr
    
    avgCEloss = epochLoss / totalBatchSize
    ceLossPlot.append(avgCEloss)

    elapsedTime = time.time() - startTime
    mins, secs = divmod(elapsedTime, 60)    
        
    print(f"pre-train epoch {epoch}  CEloss: {avgCEloss} time {int(mins)}m {int(secs)}s")

torch.save({
    "layers": [l.state_dict() for l in layers],
    "finalLayer": finalLayer.state_dict(),
    "reference_points": pi.cpu(),
    "reference_normals": normalTens.cpu(),
}, f"outputs/{mesh_name}/modelCE3D.pth")


print("SDF training")

for epoch in range(epochs):
    perm = torch.randperm(pointsTens.shape[0])
    totalBatchSize = 0
    epochLoss = 0

    for i in range(0, pointsTens.shape[0], batchSize):
        idx = perm[i:i+batchSize]
        x = pointsTens[idx].clone().to(device).requires_grad_(True)
        normals = normalTens.unsqueeze(0).to(device)
        gt = sdTens[idx].to(device)

        batchSizeCurr = x.size(0)

        out = x
        for l in layers:
            out = activation(l(out))
      
        weights = softmax(finalLayer(out))

        dist = x.unsqueeze(1) - pi.unsqueeze(0)
        sdfDist = torch.sum(dist * normals, dim=2) 

        pred = torch.sum(weights * sdfDist, dim=1)
       
        MSEloss = torch.nn.functional.mse_loss(pred, gt)

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
    "finalLayer": finalLayer.state_dict(),
    "reference_points": pi.cpu(),
    "reference_normals": normalTens.cpu(),
}, f"outputs/{mesh_name}/model3D.pth")
print("model saved")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(ceLossPlot, color='orange', label='CE Loss')
ax1.set_title("Pre-training Convergence")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")

ax2.plot(sdfLossPlot, color='blue', label='SDF Loss')
ax2.set_title("SDF Training Convergence")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE Loss")
plt.savefig(f"outputs/{mesh_name}/training_loss.png")


fig2, (ax1_log, ax2_log) = plt.subplots(1, 2, figsize=(12, 5))
ax1_log.plot(ceLossPlot, color='orange', label='CE Loss')
ax1_log.set_yscale('log')
ax1_log.set_title("Pre-training Convergence (Log Scale)")
ax1_log.set_xlabel("Epoch")
ax1_log.set_ylabel("Cross-Entropy Loss (Log Scale)")

ax2_log.plot(sdfLossPlot, color='blue', label='SDF Loss')
ax2_log.set_yscale('log') 
ax2_log.set_title("SDF Training Convergence (Log Scale)")
ax2_log.set_xlabel("Epoch")
ax2_log.set_ylabel("MSE Loss (Log Scale)")
plt.savefig(f"outputs/{mesh_name}/training_loss_log.png")