import numpy as np
import matplotlib.pyplot as plt
from signedWeightTrainingCELoss.samplingPoints2D import refPoints, refNormals

resolution = 400
bound = 1.2
xs = np.linspace(-bound, bound, resolution)
ys = np.linspace(-bound, bound, resolution)

X, Y = np.meshgrid(xs, ys, indexing='ij')
pts = np.stack([X.ravel(), Y.ravel()], axis=1)

#exponential implicit weights
fig2, ax2 = plt.subplots(figsize=(11, 10))

h = 0.05 

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


#voronoi weights
dist_to_anchors = np.linalg.norm(pts[:, None, :] - refPoints[None, :, :], axis=2)
closest_anchor_idx = np.argmin(dist_to_anchors, axis=1)
voronoi_grid = closest_anchor_idx.reshape(resolution, resolution)

chosen_anchors = refPoints[closest_anchor_idx]  
chosen_normals = refNormals[closest_anchor_idx]  

x_minus_py = pts - chosen_anchors
f_voronoi = np.sum(x_minus_py * chosen_normals, axis=1).reshape(resolution, resolution)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow( voronoi_grid.T, origin='lower', extent=[-bound, bound, -bound, bound], cmap='nipy_spectral',)
ax1.scatter( refPoints[:, 0], refPoints[:, 1], c='white',edgecolors='black', s=25, zorder=5,)

ax1.set_xlim(-bound, bound)
ax1.set_ylim(-bound, bound)
ax1.set_aspect('equal')

im = ax2.imshow( f_voronoi.T, origin='lower', extent=[-bound, bound, -bound, bound], cmap='viridis',)

ax2.contour(X, Y, f_voronoi, levels=[0], colors='white', linewidths=1.8, zorder=6)

ax2.set_xlim(-bound, bound)
ax2.set_ylim(-bound, bound)
ax2.set_aspect('equal')

cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Predicted Distance')

plt.tight_layout()
plt.savefig('voronoi_piecewise_sdf.png', dpi=300, bbox_inches='tight')
plt.show()
