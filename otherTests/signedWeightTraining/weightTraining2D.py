import torch
from signedWeightTraining.weightSamplingPoints2D import allPoint, refPoints, refNormals, S_surf
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
optimizer = torch.optim.Adam(params, lr=1e-4)#, weight_decay=1e-9)
activation = torch.nn.ReLU()
drop = torch.nn.Dropout(0.3)

# Training
batchSize = 128
epochs = 100

startTime = time.time()

for epoch in range(epochs):
    perm = torch.randperm(pointsTens.shape[0])
    totalLoss = 0

    for i in range(0, pointsTens.shape[0], batchSize):
        idx = perm[i:i+batchSize]
        x = pointsTens[idx].clone().requires_grad_(True)
        normals = normalTens.unsqueeze(0)
        gt = sdTens[idx]

        out = x
        for l in layers:
            out = activation(l(out))

        # out = drop(out)
        weights = softmax(finalLayer(out))

        dist = x.unsqueeze(1) - pi.unsqueeze(0)
        sdfDist = torch.sum(dist * normals, dim=2) 

        pred = torch.sum(weights * sdfDist, dim=1)
       
        # loss = torch.mean(pred)
        loss = torch.nn.functional.mse_loss(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

    elapsedTime = time.time() - startTime
    mins, secs = divmod(elapsedTime, 60)    

    print(f"epoch {epoch} loss {totalLoss:.4f} time {int(mins)}m {int(secs)}s")

torch.save({
    "layers": [l.state_dict() for l in layers],
    "finalLayer": finalLayer.state_dict(),
    "reference_points": pi.cpu(),
    "reference_normals": normalTens.cpu(),
}, "model2D.pth")
print("model saved")