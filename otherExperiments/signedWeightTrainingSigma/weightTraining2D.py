import torch
from signedWeightTrainingSigma.weightSamplingPoints2D import allPoint, refPoints, refNormals, S_surf
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
sdTens = torch.from_numpy(S_surf).float().to(device)
normalTens = torch.from_numpy(refNormals).float().to(device)

# Build NN
layers = torch.nn.ModuleList()

inDim = 2
hidden = 64
noLayers = 4

for i in range(noLayers):
    layers.append(torch.nn.Linear(inDim, hidden))
    inDim = hidden

finalLayer = torch.nn.Linear(hidden, N)

layers.to(device)     
finalLayer.to(device)

params = list(layers.parameters()) + list(finalLayer.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4)
activation = torch.nn.ReLU()

# Training
batchSize = 128
epochs = 100
a = 1.0

startTime = time.time()

for epoch in range(epochs):
    perm = torch.randperm(pointsTens.shape[0])
    totalLoss = 0

    for i in range(0, pointsTens.shape[0], batchSize):
        idx = perm[i:i+batchSize]
        x = pointsTens[idx].clone().requires_grad_(True)
        gt = sdTens[idx]
        normals = normalTens.unsqueeze(0)

        out = x
        for l in layers:
            out = activation(l(out))

        finalLayerData = finalLayer(out)
        
        #Tried both relu and sigmoid as activation on final layer
        numerator = torch.sigmoid(a * finalLayerData)
        # numerator = torch.relu(a * finalLayerData)

        denominator = torch.sum(numerator, dim=-1, keepdim=True)
        weights = numerator / torch.clamp(denominator, min=1e-6)

        dist = x.unsqueeze(1) - pi.unsqueeze(0)
        sdfDist = torch.sum(dist * normals, dim=2)  
        pred = torch.sum(weights * sdfDist, dim=1)
       
        loss = torch.nn.functional.mse_loss(pred,gt)

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