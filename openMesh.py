import meshio
import polyscope as ps
import numpy as np

#org
meshOrg = meshio.read("Meshes/thin.obj")
vertsOrg = meshOrg.points
facesOrg = meshOrg.cells_dict["triangle"]


mesh1 = meshio.read("Meshes/lessThin.obj")
mesh1.points[:, [1, 2]] = mesh1.points[:, [2, 1]]
verts1 = mesh1.points
faces1 = mesh1.cells_dict["triangle"]

# mesh2 = meshio.read("outputs/lucy 100ce 500e/reconstructedMesh.obj")
# verts2 = mesh2.points
# faces2 = mesh2.cells_dict["triangle"]

# Surface error
# meshError = meshio.read("error meshes/lucyError2.obj")

# vertsError = meshError.points[:, :3]
# facesError = meshError.cells_dict["triangle"]

# if meshError.points.shape[1] >= 6:
#     colors = meshError.points[:, 3:6]
# else:
#     colors = meshError.point_data.get("RGB", None)

# hex_color = (0x23 / 255.0, 0xE3 / 255.0, 0x1C / 255.0)
# hex_color2 = (0x1C / 255.0, 0x63 / 255.0, 0xE3 / 255.0)


ps.init()
ps.set_ground_plane_mode("none")
ps.set_window_size(772, 749)
ps.register_surface_mesh("reconstructed mesh1", verts1, faces1, enabled=True)
# ps.register_surface_mesh("reconstructed mesh2", verts2, faces2, enabled=False)
# error = ps.register_surface_mesh("error", vertsError, facesError)
# error.add_color_quantity("RGB", colors, enabled=True)
ps.register_surface_mesh("original mesh", vertsOrg, facesOrg, enabled=False)

ps.show()

print("done")
