import meshio
import polyscope as ps

meshOrg = meshio.read("Meshes/3D/armadillo.obj")

min_box = meshOrg.points.min(axis=0)
max_box = meshOrg.points.max(axis=0)
center = (min_box + max_box) / 2.0

scale = (max_box - min_box).max() / 2.0
vertsOrg = (meshOrg.points - center) / scale
facesOrg = meshOrg.cells_dict["triangle"]


mesh1 = meshio.read("signedWeightTrainingCELoss3D/img/armadillo/1000a 100k 100e 100ce - big arch/reconstructedMesh.obj")
verts1 = mesh1.points
faces1 = mesh1.cells_dict["triangle"]


mesh2 = meshio.read("signedWeightTrainingCELoss3D/img/armadillo/1000a 100k 300e 100ce - big arch/reconstructedMesh.obj")
verts2 = mesh2.points
faces2 = mesh2.cells_dict["triangle"]

ps.init()
ps.set_ground_plane_mode("none")
ps.register_surface_mesh("reconstructed mesh1", verts1, faces1, enabled=True)
ps.register_surface_mesh("reconstructed mesh2", verts2, faces2, enabled=True)
ps.register_surface_mesh("original mesh", vertsOrg, facesOrg, enabled=False)
ps.show()


print("done")