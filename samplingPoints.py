import polyscope as ps
import meshio
import numpy as np
import igl


noRandomPoints = 50000
noSurfacePoints = 10000

#Read mesh
mesh = meshio.read("Meshes/armadillo.obj")

vertsMesh = mesh.points
facesMesh = mesh.cells_dict["triangle"]

#Normalize mesh
vertsMeshNorm = (mesh.points - mesh.points.min(axis=0)) / (mesh.points.max(axis=0) - mesh.points.min(axis=0)) * 2 - 1

#Random points around mesh
rand_points = np.random.uniform(-1, 1, (noRandomPoints, 3))
#Random poins + mesh vertices
points = np.vstack([rand_points, vertsMeshNorm])


#Surface sampling
APoints = vertsMeshNorm[facesMesh[:, 0]]
BPoints = vertsMeshNorm[facesMesh[:, 1]]
CPoints = vertsMeshNorm[facesMesh[:, 2]]

facesAreas = 0.5 * np.linalg.norm(np.cross(BPoints - APoints, CPoints - APoints))
prob = facesAreas / facesAreas.sum()

facesSampled = np.random.choice(len(facesMesh), noSurfacePoints, prob)

surfPoint = np.zeros((len(facesSampled), 3))

for i,j in enumerate(facesSampled):
    A, B, C = (vertsMeshNorm[v] for v in facesMesh[j])
    u,v = np.random.rand(2)
    surfPoint[i, :] = A + (1 - np.sqrt(u))*(B - A) + v*np.sqrt(u)*(C - A)

#Random poins + surface points
surfPoint = np.vstack([rand_points, surfPoint])


#SDF
# S, _, _, _ = igl.signed_distance(points, vertsMeshNorm, facesMesh)
S_surf, _, _, _ = igl.signed_distance(surfPoint, vertsMeshNorm, facesMesh)


#Polyscope
ps.init()
ps.set_ground_plane_mode("none")
ps.register_surface_mesh("mesh", vertsMeshNorm, facesMesh, enabled=False)

# ps_cloud = ps.register_point_cloud("points", points, radius=0.002, enabled=False)
# ps_cloud.add_scalar_quantity("sdf values", S, enabled=True, cmap='coolwarm')

ps_cloudSurf = ps.register_point_cloud("surface points", surfPoint, radius=0.002)
ps_cloudSurf.add_scalar_quantity("sdf values", S_surf, enabled=True)
ps.show()