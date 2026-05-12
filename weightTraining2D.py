import torch
from weightSamplingPoints2D import allPoint, S_surf, refPoints
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#Reference points
pi = torch.from_numpy(refPoints).float().to(device)
N = pi.shape[0]

#Training data
pointsTens = torch.from_numpy(allPoint).float()
sdTens = torch.from_numpy(np.abs(S_surf)).float()


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

params = list(layers.parameters()) + list(finalLayer.parameters())
optimizer = torch.optim.Adam(params, 1e-4)
activation = torch.nn.ReLU()

# Training
batchSize = 128
epochs = 100

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

        dist = torch.norm(x.unsqueeze(1) - pi.unsqueeze(0), dim=2) 

        pred = torch.sum(weights * dist, dim=1)
        # prob = torch.mean(weights, dim=0)
     

        loss = torch.mean(pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

    print(f"epoch {epoch} loss {totalLoss:.4f}")

torch.save({
    "layers": [l.state_dict() for l in layers],
    "finalLayer": finalLayer.state_dict(),
    "reference_points": pi.cpu(),
}, "model2D.pth")
print("model saved")