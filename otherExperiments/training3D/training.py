import torch
from training3D.samplingPoints import surfPoint, S_surf

# Use the other line if this causes errors
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using {device} device")

pointsTens = torch.from_numpy(surfPoint).float().to(device)
sdTens = torch.from_numpy(S_surf).float().to(device)


# Build NN
layers = torch.nn.ModuleList()

inDim = 3
hidden = 256
noLayers = 6

for i in range(noLayers):
    layers.append(torch.nn.Linear(inDim, hidden))
    inDim = hidden

finalLayer = torch.nn.Linear(hidden, 1)

layers.to(device)     
finalLayer.to(device)

params = list(layers.parameters()) + list(finalLayer.parameters())
optimizer = torch.optim.Adam(params, 1e-4)
activation = torch.nn.ReLU()


# Training
batchSize = 250
epochs = 200

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
        pred = finalLayer(out).squeeze()

        sdLoss = torch.nn.functional.mse_loss(pred,gt)
        
        grad = torch.autograd.grad(pred, x, torch.ones_like(pred), True, True, True)[0]
        gradNorm = torch.linalg.vector_norm(grad, dim = -1)

        lipLoss = torch.mean((gradNorm - 1.0)**2)
        loss = sdLoss + 0.001*lipLoss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

    print(f"epoch {epoch} loss {totalLoss:.4f} sdLoss {sdLoss:.4f} lipLoss {lipLoss:.4f}")

torch.save({
    "layers": [l.state_dict() for l in layers],
    "finalLayer": finalLayer.state_dict()
}, "model.pth")
print("model saved")