import meshio
import numpy as np
import igl
import os
import sys

mesh_name = sys.argv[1]

noRandomPoints = 20000
noSurfacePoints = 80000

# Read mesh
mesh = meshio.read(f"Meshes/{mesh_name}.obj")

vertsMesh = mesh.points
facesMesh = mesh.cells_dict["triangle"]

# Normalize mesh
min_box = mesh.points.min(axis=0)
max_box = mesh.points.max(axis=0)
center = (min_box + max_box) / 2.0

scale = (max_box - min_box).max() / 2.0
vertsMeshNorm = (mesh.points - center) / scale

# Random points around mesh
rand_points = np.random.uniform(-1.2, 1.2, (noRandomPoints, 3))

# Surface sampling
def samplingSurfacePoints(no):
    APoints = vertsMeshNorm[facesMesh[:, 0]]
    BPoints = vertsMeshNorm[facesMesh[:, 1]]
    CPoints = vertsMeshNorm[facesMesh[:, 2]]

    cross_prod = np.cross(BPoints - APoints, CPoints - APoints)
    facesAreas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    prob = facesAreas / facesAreas.sum()

    facesSampled = np.random.choice(len(facesMesh), no, p=prob)
    
    face_normals = cross_prod / np.linalg.norm(cross_prod, axis=1, keepdims=True)

    surfPoint = np.zeros((no, 3))
    sampledNormals = face_normals[facesSampled]

    for i, j in enumerate(facesSampled):
        A, B, C = (vertsMeshNorm[v] for v in facesMesh[j])
        u, v = np.random.rand(2)
        surfPoint[i, :] = A + (1 - np.sqrt(u)) * (B - A) + v * np.sqrt(u) * (C - A)
    
    return surfPoint, sampledNormals

# Surface sampling with noise
def samplingBoundaryPoints(no, noise_ratio=0.875, sigma=0.01):
    surfPoint, _ = samplingSurfacePoints(no)
    
    num_noise = int(no * noise_ratio)

    noise = np.random.normal(0, sigma, (num_noise, 3))
    surfPoint[:num_noise] += noise
    
    return surfPoint 

# Equally spread refpoints
def farthest_point_sampling(points, k):
    chosen = [np.random.randint(len(points))]
    dist = np.full(len(points), np.inf)

    for _ in range(k - 1):
        last = points[chosen[-1]]
        d = np.linalg.norm(points - last, axis=1)
        dist = np.minimum(dist, d)
        chosen.append(np.argmax(dist))

    return chosen


surfPoint = samplingBoundaryPoints(noSurfacePoints)
allPoint = np.vstack([rand_points, surfPoint])

S_surf, _, _, _ = igl.signed_distance(allPoint, vertsMeshNorm, facesMesh)

os.makedirs(f"data/{mesh_name}", exist_ok=True)

np.savez(f"data/{mesh_name}/sampledPointsSiren3d.npz", allPoint=allPoint, S_surf=S_surf)
