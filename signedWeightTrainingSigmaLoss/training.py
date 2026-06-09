import torch
from signedWeightTrainingSigmaLoss.samplingPoints2D import allPoint, refPoints, refNormals, S_surf
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
epochs = 300
lm = 1.0
sigma = 0.01
h = 0.2

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
        numerator = torch.sigmoid(finalLayerData)
        # numerator = torch.relu(finalLayerData)

        denominator = torch.sum(numerator, dim=-1, keepdim=True)
        weights = numerator / torch.clamp(denominator, min=1e-6)

        dist = x.unsqueeze(1) - pi.unsqueeze(0)
        sdfDist = torch.sum(dist * normals, dim=2)  
        pred = torch.sum(weights * sdfDist, dim=1)

        loss1 = torch.nn.functional.mse_loss(pred, gt)

        epsilon = torch.randn_like(pi) * sigma 
        x_i = pi + epsilon

        out_local = x_i
        for l in layers:
            out_local = activation(l(out_local))
        finalLayerData_local = finalLayer(out_local)

        numerator_local = torch.sigmoid(finalLayerData_local)
        # numerator_local = torch.relu(finalLayerData_local)

        denominator_local = torch.sum(numerator_local, dim=-1, keepdim=True)
        weights_local = numerator_local / torch.clamp(denominator_local, min=1e-6)
        
        w_i_xi = torch.diagonal(weights_local) 
        
        dist_xi_pi = torch.norm(epsilon, dim=1)
        target_weights = torch.exp(-(dist_xi_pi / h) ** 2)
        
        loss2 = torch.nn.functional.mse_loss(w_i_xi, target_weights)

        loss = loss1 + lm * loss2
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()
        

    elapsedTime = time.time() - startTime
    mins, secs = divmod(elapsedTime, 60)

    print(f"epoch {epoch} loss1 {loss1:.4f} loss2 {loss2:.4f} time {int(mins)}m {int(secs)}s")

torch.save({
    "layers": [l.state_dict() for l in layers],
    "finalLayer": finalLayer.state_dict(),
    "reference_points": pi.cpu(),
    "reference_normals": normalTens.cpu(),
}, "model2D.pth")
print("model saved")