import numpy as np
import meshio

data = np.load("signedWeightTrainingNormals3D/img/anchors 200a 10k 300e 100ce/anchor_normals_pc.npz")

points = data["points"]
gt_normals = data["gt_normals"]
trained_normals = data["trained_normals"]

cos_sim = np.sum(gt_normals * trained_normals, axis=-1)

cells = [("vertex", np.arange(len(points)).reshape(-1, 1))]

point_data = {
    "gt_normals": gt_normals,
    "trained_normals": trained_normals,
    "cosine_alignment": cos_sim,
}

mesh_anchors = meshio.Mesh(points=points, cells=cells, point_data=point_data)
mesh_anchors.write("anchor_normals.vtk")
