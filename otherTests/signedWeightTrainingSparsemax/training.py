import torch
from signedWeightTrainingSparsemax.samplingPoints2D import allPoint, refPoints, refNormals, S_surf
import numpy as np
import time
# from entmax import sparsemax, entmax15
from signedWeightTrainingSparsemax.sparsemax import Sparsemax

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
sparsemax = Sparsemax(dim=-1)


layers.to(device)     
finalLayer.to(device)

params = list(layers.parameters()) + list(finalLayer.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4)
activation = torch.nn.ReLU()

ceOptimizer = torch.optim.Adam(params, lr=1e-3)
ceEpochs = 10 

# Training
batchSize = 128
epochs = 100
lm1 = 1.0
sigma = 0.01
h = 0.2

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
    "reference_normals": normalTens.cpu(),
}, "modelCE2D.pth")


print("SDF training")

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
      
        # weights = softmax(finalLayer(out))
        weights = sparsemax(finalLayer(out))
        # weights = entmax15(finalLayer(out))

        dist = x.unsqueeze(1) - pi.unsqueeze(0)
        sdfDist = torch.sum(dist * normals, dim=2) 

        pred = torch.sum(weights * sdfDist, dim=1)
       
        MSEloss = torch.nn.functional.mse_loss(pred, gt)

        epsilon = torch.randn_like(pi) * sigma 
        x_i = pi + epsilon

        out_local = x_i
        for l in layers:
            out_local = activation(l(out_local))
            
        # weights_local = softmax(finalLayer(out_local))
        weights_local = sparsemax(finalLayer(out_local))
        # weights_local = entmax15(finalLayer(out_local))

        w_i_xi = torch.diagonal(weights_local) 
        
        dist_xi_pi = torch.norm(epsilon, dim=1)
        target_weights = torch.exp(-(dist_xi_pi / h) ** 2)
        
        loss1 = torch.nn.functional.mse_loss(w_i_xi, target_weights)

        loss = MSEloss + lm1 * loss1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

    elapsedTime = time.time() - startTime
    mins, secs = divmod(elapsedTime, 60)    

    print(f"train epoch {epoch} MSEloss {MSEloss:.4f} loss1 {loss1:.4f} time {int(mins)}m {int(secs)}s")

torch.save({
    "layers": [l.state_dict() for l in layers],
    "finalLayer": finalLayer.state_dict(),
    "reference_points": pi.cpu(),
    "reference_normals": normalTens.cpu(),
}, "model2D.pth")
print("model saved")