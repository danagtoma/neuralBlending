import polyscope as ps
import meshio

# 1. Read the VTK file
mesh = meshio.read("signedWeightTrainingNormals3D/img/anchor_normals.vtk")
points = mesh.points
gt_normals = mesh.point_data["gt_normals"]
trained_normals = mesh.point_data["trained_normals"]
cosine_alignment = mesh.point_data.get("cosine_alignment")

# 2. Initialize Polyscope
ps.init()
ps.set_up_dir("y_up")  # Adjust to "y_up" if your mesh is oriented along Y
ps.set_ground_plane_mode("none")

# 3. Register the points as a Point Cloud
pc = ps.register_point_cloud("Anchor Points", points, radius=0.004)

# 4. Add the vector arrows
q_gt = pc.add_vector_quantity(
    "GT Normals",
    gt_normals,
    enabled=True,
    color=(0.1, 0.4, 0.95),  # Blue
)
q_gt.set_length(0.04)  # Arrow display length (adjust if needed)

q_trained = pc.add_vector_quantity(
    "Trained Normals",
    trained_normals,
    enabled=True,
    color=(0.95, 0.2, 0.1),  # Red
)
q_trained.set_length(0.04)

# 5. Optionally color the points by cosine alignment error
if cosine_alignment is not None:
    pc.add_scalar_quantity(
        "Cosine Alignment", cosine_alignment, enabled=True, cmap="coolwarm"
    )

# 6. Show the interactive window
ps.show()