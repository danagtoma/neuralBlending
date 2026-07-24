import torch
from signedWeightTrainingCERegularisation.samplingPoints2D import allPoint, refPoints, refNormals, S_surf
import numpy as np
import time

# Use the other line if this causes errors
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#Reference points
pi = torch.from_numpy(refPoints).float().to(device)
N = pi.shape[0]

#Training data
pointsTens = torch.from_numpy(allPoint).float().to(device)
normalTens = torch.from_numpy(refNormals).float().to(device)
sdTens = torch.from_numpy(S_surf).float().to(device)

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

params = list(layers.parameters()) + list(finalLayer.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4)
activation = torch.nn.ReLU()


# Training
batchSize = 128
epochs = 500
lambdaCe = 0.1


startTime = time.time()

labels = torch.zeros(pointsTens.shape[0], dtype=torch.long, device=device)
chunkSize = 4000

for i in range(0, pointsTens.shape[0], chunkSize):
    pointsChunk = pointsTens[i : i + chunkSize]
    distChunk = torch.cdist(pointsChunk, pi)
    labels[i : i + chunkSize] = torch.argmin(distChunk, dim=1)

print("SDF training")

for epoch in range(epochs):
    perm = torch.randperm(pointsTens.shape[0])
    totalBatchSize = 0
    epochSDFLoss = 0
    epochCeLoss = 0
    totalLoss = 0

    for i in range(0, pointsTens.shape[0], batchSize):
        idx = perm[i:i+batchSize]
        x = pointsTens[idx].clone().requires_grad_(True)
        normals = normalTens.unsqueeze(0)
        gt_sdf = sdTens[idx]
        gt_labels = labels[idx]

        batchSizeCurr = x.size(0)

        out = x
        for l in layers:
            out = activation(l(out))

        logits = finalLayer(out)
        weights = softmax(finalLayer(out))

        dist = x.unsqueeze(1) - pi.unsqueeze(0)
        sdfDist = torch.sum(dist * normals, dim=2) 

        pred = torch.sum(weights * sdfDist, dim=1)
       
        MSEloss = torch.nn.functional.mse_loss(pred, gt_sdf)

        CELoss = torch.nn.functional.cross_entropy(logits, gt_labels)

        loss = MSEloss + lambdaCe * CELoss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epochSDFLoss += MSEloss.item() * batchSizeCurr
        epochCeLoss += CELoss.item() * batchSizeCurr
        totalLoss += loss.item() * batchSizeCurr
        totalBatchSize += batchSizeCurr
        
    avgMSEloss = epochSDFLoss / totalBatchSize
    avgCeLoss = epochCeLoss / totalBatchSize
    avgTotalLoss = totalLoss / totalBatchSize

    elapsedTime = time.time() - startTime
    mins, secs = divmod(elapsedTime, 60)    

    print(f"train epoch {epoch} total loss {avgTotalLoss} MSEloss {avgMSEloss} CELoss {avgCeLoss} time {int(mins)}m {int(secs)}s")

torch.save({
    "layers": [l.state_dict() for l in layers],
    "finalLayer": finalLayer.state_dict(),
    "reference_points": pi.cpu(),
    "reference_normals": normalTens.cpu(),
}, "model2D.pth")
print("model saved")