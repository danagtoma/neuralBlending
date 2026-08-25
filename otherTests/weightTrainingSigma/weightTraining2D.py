import torch
from weightTrainingSigma.weightSamplingPoints2D import allPoint, refPoints, U_surf, S_surf
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
udTens = torch.from_numpy(U_surf).float().to(device)
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
epochs = 300
a = 1.0

startTime = time.time()

for epoch in range(epochs):
    perm = torch.randperm(pointsTens.shape[0])
    totalLoss = 0

    for i in range(0, pointsTens.shape[0], batchSize):
        idx = perm[i:i+batchSize]
        x = pointsTens[idx].clone().requires_grad_(True)
        gtU = udTens[idx]
        gtS = sdTens[idx]

        out = x
        for l in layers:
            out = activation(l(out))

        weights = softmax(finalLayer(out))

        # finalLayerData = finalLayer(out)
        
        # #Tried both relu and sigmoid as activation on final layer
        # # numerator = torch.sigmoid(a * finalLayerData)
        # numerator = torch.relu(a * finalLayerData)

        # denominator = torch.sum(numerator, dim=-1, keepdim=True)

        # # weights = numerator / torch.clamp(denominator, min=1e-6) #for sigmoid
        # weights = numerator / torch.clamp(denominator, min=1)  #for relu

        dist = torch.norm(x.unsqueeze(1) - pi.unsqueeze(0), dim=2) 

        pred = torch.sum(weights * dist, dim=1)
       
        sdLoss = torch.nn.functional.mse_loss(pred, gtU)
        
        loss = sdLoss 

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
}, "model2D.pth")
print("model saved")