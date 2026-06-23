import meshio
import polyscope as ps

meshOrg = meshio.read("Meshes/3D/armadillo.obj")
vertsOrg = (meshOrg.points - meshOrg.points.min(axis=0)) / (meshOrg.points.max(axis=0) - meshOrg.points.min(axis=0)) * 2 - 1
facesOrg = meshOrg.cells_dict["triangle"]

mesh = meshio.read("signedWeightTrainingCELoss3D/img/cheburashka/500a 10k 100e 10ce/reconstructedMesh.obj")
verts = mesh.points
faces = mesh.cells_dict["triangle"]

ps.init()
ps.set_ground_plane_mode("none")
ps.register_surface_mesh("reconstructed mesh", verts, faces, enabled=True)
ps.register_surface_mesh("original mesh", vertsOrg, facesOrg, enabled=False)
ps.show()


print("done")