import torch
from signedWeightTrainingCELossSiren.samplingPoints2D import allPoint, S_surf
import numpy as np
import time

# Use the other line if this causes errors
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#Training data
pointsTens = torch.from_numpy(allPoint).float().to(device)
sdTens = torch.from_numpy(S_surf).float().to(device)

# Build NN
layers = torch.nn.ModuleList()

inDim = 2
hidden = 64
noLayers = 4
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
batchSize = 128
epochs = 100

startTime = time.time()

print("SIREN training")

for epoch in range(epochs):
    perm = torch.randperm(pointsTens.shape[0])
    totalLoss = 0

    for i in range(0, pointsTens.shape[0], batchSize):
        idx = perm[i:i+batchSize]
        x = pointsTens[idx].clone().requires_grad_(True)
        gt = sdTens[idx]

        out = x
        for l in layers:
            out = torch.sin(w0 * l(out))
        pred = finalLayer(out).squeeze()
      
        loss = torch.nn.functional.mse_loss(pred,gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

    elapsedTime = time.time() - startTime
    mins, secs = divmod(elapsedTime, 60)    

    print(f"train epoch {epoch} loss1 {loss:.4f} time {int(mins)}m {int(secs)}s")

torch.save({
    "layers": [l.state_dict() for l in layers],
    "finalLayer": finalLayer.state_dict(),
}, "model2D.pth")
print("model saved")