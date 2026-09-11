import torch
from otherExperiments.signedWeightTrainingNormals.samplingPoints2D import allPoint, refPoints, refNormals, S_surf
import numpy as np
import time

# Use the other line if this causes errors
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#Reference points
pi = torch.from_numpy(refPoints).float().to(device)
N = pi.shape[0]
gt_normals = torch.from_numpy(refNormals).float().to(device)

#Training data
pointsTens = torch.from_numpy(allPoint).float().to(device)
sdTens = torch.from_numpy(S_surf).float().to(device)
anchor_normals = torch.nn.Parameter(gt_normals.clone())

# Build NN
layers = torch.nn.ModuleList()

inDim = 2
hidden = 64
noLayers = 4

for i in range(noLayers):
    layers.append(torch.nn.Linear(inDim, hidden))
    inDim = hidden

finalLayer = torch.nn.Linear(hidden, N)
softmax = torch.nn.Softmax(dim=-1)

layers.to(device)     
finalLayer.to(device)

nn_params = list(layers.parameters()) + list(finalLayer.parameters())
optimizer = torch.optim.Adam(
    nn_params + [anchor_normals], 
    lr=1e-4
)
activation = torch.nn.ReLU()

ceOptimizer = torch.optim.Adam(nn_params, lr=1e-3)
ceEpochs = 50

# Training
batchSize = 128
epochs = 100

startTime = time.time()

dist_matrix = torch.cdist(pointsTens, pi)
labels = torch.argmin(dist_matrix, dim=1)

print("initialization")

for epoch in range(ceEpochs):
    perm = torch.randperm(pointsTens.shape[0])
    totalCEloss = 0
    
    for i in range(0, pointsTens.shape[0], batchSize):
        idx = perm[i:i+batchSize]
        x = pointsTens[idx]
        gtLabels = labels[idx] 
        
        out = x
        for l in layers:
            out = activation(l(out))

        CEloss = torch.nn.functional.cross_entropy(finalLayer(out), gtLabels)
        
        ceOptimizer.zero_grad()
        CEloss.backward()
        ceOptimizer.step()
        
        totalCEloss += CEloss.item()

    elapsedTime = time.time() - startTime
    mins, secs = divmod(elapsedTime, 60)    
        
    print(f"pre-train epoch {epoch}  CEloss: {totalCEloss:.4f} time {int(mins)}m {int(secs)}s")

torch.save({
    "layers": [l.state_dict() for l in layers],
    "finalLayer": finalLayer.state_dict(),
    "reference_points": pi.cpu(),
    "gt_normals": gt_normals.cpu(),
    "normals": torch.nn.functional.normalize(anchor_normals, p=2, dim=-1).detach().cpu(),
}, "modelCE2D.pth")


print("SDF training")

for epoch in range(epochs):
    perm = torch.randperm(pointsTens.shape[0])
    totalLoss = 0

    for i in range(0, pointsTens.shape[0], batchSize):
        idx = perm[i:i+batchSize]
        x = pointsTens[idx].clone().requires_grad_(True)
        gt = sdTens[idx]

        out = x
        for l in layers:
            out = activation(l(out))
      
        weights = softmax(finalLayer(out))

        normed_normals = torch.nn.functional.normalize(anchor_normals, p=2, dim=-1)
        normals_batch = normed_normals.unsqueeze(0)

        dist = x.unsqueeze(1) - pi.unsqueeze(0)
        sdfDist = torch.sum(dist * normals_batch, dim=2) 

        pred = torch.sum(weights * sdfDist, dim=1)
       
        MSEloss = torch.nn.functional.mse_loss(pred, gt)

        loss = MSEloss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

    elapsedTime = time.time() - startTime
    mins, secs = divmod(elapsedTime, 60)    

    print(f"train epoch {epoch} MSEloss {MSEloss:.4f} time {int(mins)}m {int(secs)}s")

torch.save({
    "layers": [l.state_dict() for l in layers],
    "finalLayer": finalLayer.state_dict(),
    "reference_points": pi.cpu(),
    "gt_normals": gt_normals.cpu(),
    "normals": torch.nn.functional.normalize(anchor_normals, p=2, dim=-1).detach().cpu(),
}, "model2D.pth")
print("model saved")