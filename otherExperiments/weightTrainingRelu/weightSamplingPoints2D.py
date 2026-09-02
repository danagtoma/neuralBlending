import polyscope as ps
import meshio
import numpy as np
import igl

noRefPoints = 35
noRandomPoints = 8000
noSurfacePoints = 2000

#Read mesh
mesh = meshio.read("Meshes/2D/U.obj")

vertsMesh = mesh.points
facesMesh = mesh.cells_dict["triangle"]

#Normalize mesh
xy = mesh.points[:, :2]

center = (xy.min(axis=0) + xy.max(axis=0)) / 2
scale = (xy.max(axis=0) - xy.min(axis=0)).max() / 2

vertsMeshNorm = (xy - center) / scale

#Random points around mesh
rand_points = np.random.uniform(-1.2, 1.2, (noRandomPoints, 2))

#Egde sample
def samplingEdgePoints(no):
    be = igl.boundary_facets(facesMesh)
    if isinstance(be, tuple):
        be = be[0]
    
    be = np.array(be)
    
    v0 = vertsMeshNorm[be[:, 0]]
    v1 = vertsMeshNorm[be[:, 1]]
    
    edge_lengths = np.linalg.norm(v1 - v0, axis=1)
    prob = edge_lengths / edge_lengths.sum()
    
    edges_sampled_idx = np.random.choice(len(be), no, p=prob)
    
    t = np.random.rand(no, 1) 
    sampled_v0 = vertsMeshNorm[be[edges_sampled_idx, 0]]
    sampled_v1 = vertsMeshNorm[be[edges_sampled_idx, 1]]
    
    edgePoints = sampled_v0 + t * (sampled_v1 - sampled_v0)
    return edgePoints

#Surface sampling
def samplingSurfacePoints(no):
    APoints = vertsMeshNorm[facesMesh[:, 0]]
    BPoints = vertsMeshNorm[facesMesh[:, 1]]
    CPoints = vertsMeshNorm[facesMesh[:, 2]]

    facesAreas = 0.5 * np.linalg.norm(np.cross(BPoints - APoints, CPoints - APoints))
    prob = facesAreas / facesAreas.sum()

    facesSampled = np.random.choice(len(facesMesh), no, prob)

    surfPoint = np.zeros((len(facesSampled), 2))

    for i, j in enumerate(facesSampled):
        A, B, C = (vertsMeshNorm[v] for v in facesMesh[j])
        u,v = np.random.rand(2)
        surfPoint[i, :] = A + (1 - np.sqrt(u))*(B - A) + v*np.sqrt(u)*(C - A)
    
    return surfPoint  

#Equally spread refpoints
def farthest_point_sampling(points, k):
    chosen = [np.random.randint(len(points))]
    dist = np.full(len(points), np.inf)

    for _ in range(k - 1):
        last = points[chosen[-1]]
        d = np.linalg.norm(points - last, axis=1)
        dist = np.minimum(dist, d)
        chosen.append(np.argmax(dist))

    return points[chosen]

surfPoint = samplingSurfacePoints(noSurfacePoints)

#Random poins + surface points
allPoint = np.vstack([rand_points, surfPoint])


denseSurf = samplingEdgePoints(5000)
refPoints = farthest_point_sampling(denseSurf[:, :2], noRefPoints)

#Polyscope
# ps.init()
# ps.set_ground_plane_mode("none")
# ps.register_surface_mesh("mesh", vertsMeshNorm, facesMesh, enabled=True)

# ps_cloud = ps.register_point_cloud("reference points", refPoints, radius=0.002, enabled=False)
# ps_cloud = ps.register_point_cloud("rand points", rand_points, radius=0.002, enabled=False)
# ps_cloud = ps.register_point_cloud("surface points", surfPoint, radius=0.002, enabled=False)

# ps_cloudSurf = ps.register_point_cloud("all points", allPoint, radius=0.002)

# ps.show()