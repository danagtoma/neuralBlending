import numpy as np
import matplotlib.pyplot as plt
from signedWeightTrainingCELoss.samplingPoints2D import refPoints, refNormals

resolution = 400
bound = 1.2
xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)

X, Y = np.meshgrid(xs, ys, indexing='ij')
pts = np.stack([X.ravel(), Y.ravel()], axis=1)

#perfect voronoi diagram
fig1, ax1 = plt.subplots(figsize=(10, 10))

dist_to_anchors = np.linalg.norm(pts[:, None, :] - refPoints[None, :, :], axis=2)
closest_anchor_idx = np.argmin(dist_to_anchors, axis=1)
voronoi_grid = closest_anchor_idx.reshape(resolution, resolution)

ax1.imshow(voronoi_grid.T, origin='lower', extent=[-bound, bound, -bound, bound], cmap='nipy_spectral')
ax1.scatter(refPoints[:, 0], refPoints[:, 1], c='white', edgecolors='black', s=40, zorder=5, label='Anchors')
ax1.set_xlim(-bound, bound)
ax1.set_ylim(-bound, bound)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.grid(False)

plt.tight_layout()
plt.savefig("ideal_voronoi_cells.png", dpi=300)
plt.close(fig1)

#exponential implicit diagram
fig2, ax2 = plt.subplots(figsize=(11, 10))

h = 0.1 

dist_matrix = np.linalg.norm(pts[:, None, :] - refPoints[None, :, :], axis=2)

exp_weights = np.exp(- (dist_matrix / h) ** 2)
weight_sum = np.sum(exp_weights, axis=1, keepdims=True)
norm_weights = exp_weights / (weight_sum + 1e-9) 


x_minus_pi = pts[:, None, :] - refPoints[None, :, :] 
sdfDist = np.sum(x_minus_pi * refNormals[None, :, :], axis=2)

F_x = np.sum(norm_weights * sdfDist, axis=1).reshape(resolution, resolution)

im2 = ax2.imshow(F_x.T, origin='lower', extent=[-bound, bound, -bound, bound], cmap='viridis')
ax2.contour(F_x.T, levels=[0], colors='black', extent=[-bound, bound, -bound, bound])
ax2.scatter(refPoints[:, 0], refPoints[:, 1], c='white', edgecolors='black', s=20, zorder=5)
fig2.colorbar(im2, ax=ax2)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")

plt.tight_layout()
plt.savefig("ideal_exponential_weights.png", dpi=300)