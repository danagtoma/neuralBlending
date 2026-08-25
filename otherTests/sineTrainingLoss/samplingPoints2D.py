import polyscope as ps
import meshio
import numpy as np
import igl

noRefPoints = 3
noRandomPoints = 2000
noSurfacePoints = 8000

#Read mesh
mesh = meshio.read("Meshes/2D/dauphin.obj")

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
    
    edge_dirs = (v1 - v0) / edge_lengths[:, None]
    edge_normals = np.zeros_like(edge_dirs)
    edge_normals[:, 0] = edge_dirs[:, 1]
    edge_normals[:, 1] = -edge_dirs[:, 0]

    edges_sampled_idx = np.random.choice(len(be), no, p=prob)
    
    t = np.random.rand(no, 1) 
    sampled_v0 = vertsMeshNorm[be[edges_sampled_idx, 0]]
    sampled_v1 = vertsMeshNorm[be[edges_sampled_idx, 1]]
    
    edgePoints = sampled_v0 + t * (sampled_v1 - sampled_v0)
    sampledNormals = edge_normals[edges_sampled_idx]

    return edgePoints, sampledNormals

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

def samplingBoundaryPoints(no, noise_ratio=0.875, sigma=0.05):
    edgePoints, _ = samplingEdgePoints(no)
    
    num_noise = int(no * noise_ratio)

    noise = np.random.normal(0, sigma, (num_noise, 2))
    edgePoints[:num_noise] += noise
    
    return edgePoints 

#Equally spread refpoints
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

#Random poins + surface points
allPoint = np.vstack([rand_points, surfPoint])
allPoint = np.hstack([allPoint, np.zeros((noRandomPoints + noSurfacePoints, 1))])

def extrude_2d_mesh(verts, faces):
    be = igl.boundary_facets(faces)
    if isinstance(be, tuple):
        be = be[0]
    be = np.array(be, dtype=np.int64)

    nv = len(verts)
    V_3d = np.hstack([verts, np.zeros((nv, 1), dtype=np.float64)])
    
    center = verts.mean(axis=0)
    apex_up = np.array([center[0], center[1], 100.0], dtype=np.float64)
    apex_down = np.array([center[0], center[1], -100.0], dtype=np.float64)
    
    V_extruded = np.vstack([V_3d, apex_up, apex_down])
    idx_up = nv
    idx_down = nv + 1
    
    F_list = []
    for a, b in be:
        F_list.append([a, b, idx_up])
        F_list.append([b, a, idx_down])
    
    V_final = np.array(V_extruded, dtype=np.float64)
    F_final = np.array(F_list, dtype=np.int64)
    
    return V_final, F_final

V_tent, F_tent = extrude_2d_mesh(vertsMeshNorm, facesMesh)
S_surf, _, _, _ = igl.signed_distance(allPoint, V_tent, F_tent, sign_type=0)

denseSurf, denseNormals = samplingEdgePoints(5000)
anchor_indices = farthest_point_sampling(denseSurf[:, :2], noRefPoints)
refPoints = denseSurf[anchor_indices, :2]
refNormals = denseNormals[anchor_indices]
allPoint = allPoint[:, :2]

#Polyscope
# ps.init()
# ps.set_ground_plane_mode("none")
# ps.register_surface_mesh("mesh", vertsMeshNorm, facesMesh, enabled=True)

# ps_cloud = ps.register_point_cloud("reference points", refPoints, radius=0.002, enabled=True)
# ps_cloud.add_vector_quantity("anchor normals", refNormals, enabled=True)
# ps_cloud = ps.register_point_cloud("rand points", rand_points, radius=0.002, enabled=False)
# ps_cloud = ps.register_point_cloud("surface points", surfPoint, radius=0.002, enabled=False)

# ps_cloudSurf = ps.register_point_cloud("all points", allPoint, radius=0.002)
# ps_cloudSurf.add_scalar_quantity("sdf values", S_surf, enabled=True)

# ps.show()