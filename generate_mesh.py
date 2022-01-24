import numpy as np
import mcubes
import os
import trimesh
import open3d as o3d

if True:
    datapath = '/home/lir0b/Code/TransparenceDetection/src/neural_representation/nerf/logs/hotdog_test/renderonly_path_130001'
    voxel_path = os.path.join(datapath, 'voxels.npy')
    weight_path = os.path.join(datapath, 'max_weight.npy')
    voxels = np.load(voxel_path)
    weights = np.load(weight_path)
    print(voxels.shape, weights.shape)

#threshold = 50.
if False:
    threshold = 0.05
    print('fraction occupied', np.mean(weights > threshold))
    vertices, triangles = mcubes.marching_cubes(weights, threshold)
    print('done', vertices.shape, triangles.shape)
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.show()

if False:
    inds = weights > 0.01
    voxels_selected = voxels[inds, :]
    voxels_selected = voxels_selected.reshape(-1, 3)
    pc = trimesh.points.PointCloud(vertices=voxels_selected,
                                   colors=np.random.random(voxels_selected.shape))
    pc.merge_vertices()
    pc.export('test.ply')
    #pcch = pc.convex_hull
    pc.show()

if False:
    import open3d as o3d
    import trimesh
    import numpy as np

    pcd = o3d.io.read_point_cloud("test.ply")
    pcd.estimate_normals()

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius * 2]))

    # create the triangular mesh with the vertices and faces from open3d
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                               vertex_normals=np.asarray(mesh.vertex_normals))

    trimesh.convex.is_convex(tri_mesh)
    tri_mesh.export('test_mesh.ply')

if True:
    #mesh = o3dtut.get_bunny_mesh()
    mesh = o3d.io.read_triangle_mesh("test_mesh.ply")

    # pcd = o3d.io.read_point_cloud("test.ply")
    # pcd.estimate_normals()
    # print('run Poisson surface reconstruction')
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #         pcd, depth=9)
    # print(mesh)
    # o3d.visualization.draw_geometries([mesh],
    #                                   zoom=0.664,
    #                                   front=[-0.4761, -0.4698, -0.7434],
    #                                   lookat=[1.8900, 3.2596, 0.9284],
    #                                   up=[0.2304, -0.8825, 0.4101])


    pcd = mesh.sample_points_poisson_disk(1000)
    o3d.visualization.draw_geometries([pcd])
    alpha = 0.01
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
