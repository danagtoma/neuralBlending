import meshio

mesh = meshio.read("Meshes/3D/MedievalCastle.obj")

faces = mesh.cells_dict["triangle"]

min_box = mesh.points.min(axis=0)
max_box = mesh.points.max(axis=0)
center = (min_box + max_box) / 2.0

scale = (max_box - min_box).max() / 2.0
vertsMeshNorm = (mesh.points - center) / scale

meshNorm = meshio.Mesh(vertsMeshNorm, [("triangle", faces)])
meshio.write("Meshes/normalized/medievalCastleNorm.obj", meshNorm)