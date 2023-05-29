import pye57
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import alphashape
from matplotlib.patches import Polygon
from copy import copy


# To-do: generate IFC (industry foundation classes) file


def read_e57(file_name):
    # read the documentation at https://github.com/davidcaron/pye57
    e57 = pye57.E57(file_name)
    data_raw = e57.read_scan_raw(0)
    print('e57 data imported')
    return data_raw


def e57_data_to_xyz(e57_data, output_file_name):
    x = e57_data['cartesianX']
    y = e57_data['cartesianY']
    z = e57_data['cartesianZ']
    red = e57_data['colorRed']
    green = e57_data['colorGreen']
    blue = e57_data['colorBlue']
    intensity = e57_data['intensity']
    df = pd.DataFrame({'//X': x, 'Y': y, 'Z': z, 'R': red, 'G': green, 'B': blue, 'Intensity': intensity})
    df.to_csv(output_file_name, sep='\t', index=False)
    print('e57 converted to ASCII format, saved as %s' % output_file_name)


def save_xyz(points, output_file_name):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # df = pd.DataFrame({'//X': x, 'Y': y, 'Z': z, 'R': red, 'G': green, 'B': blue, 'Intensity': intensity})
    # df.to_csv(output_file_name, sep='\t', index=False)
    df = pd.DataFrame({'//X': x, 'Y': y, 'Z': z})
    df.to_csv(output_file_name, sep='\t', index=False)
    print('Points saved as %s' % output_file_name)


def load_xyz_file(file_name):
    pcd = np.loadtxt(file_name, skiprows=1)
    xyz = pcd[:, :3]
    rgb = pcd[:, 3:6]

    # show plot of xyz points from top view: (x, y) coordinates and with rgb-colored points
    plot_xyz = False
    if plot_xyz:
        plt.figure(figsize=(8, 5), dpi=150)
        plt.scatter(xyz[:, 0], xyz[:, 1], c=rgb / 255, s=0.05)
        plt.title("Top-View")
        plt.xlabel('X-axis (m)')
        plt.ylabel('Y-axis (m)')
        plt.show()
    return xyz, rgb


def ransac_plane(xyz, threshold=0.04, iterations=1000):
    print('Running RANSAC segmentation of planes')
    inliers, equation = [], []
    n_points = len(xyz)
    i = 1
    percent_limit = 10
    while i < iterations:
        percent_done = int(i / iterations * 100)
        if percent_done >= percent_limit:
            print('Progress: %d %%' % int(percent_done))
            percent_limit += 10
        idx_samples = random.sample(range(n_points), 3)
        # idx = random.sample(range(n_points), 1)
        # idx_samples = [idx[0], idx[0] + 1, idx[0] + 2]
        pts = xyz[idx_samples]
        vec1 = pts[1] - pts[0]
        vec2 = pts[2] - pts[0]
        normal = np.cross(vec1, vec2)
        a, b, c = normal / np.linalg.norm(normal)
        d = -np.sum(normal * pts[1])
        distance = (a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        idx_candidates = np.where(np.abs(distance) <= threshold)[0]
        if len(idx_candidates) > len(inliers):
            equation = [a, b, c, d]
            inliers = idx_candidates
        i += 1
    return equation, inliers


def ransac_find_horiz_surface_plane(xyz, threshold=0.02, steps=1000):
    print('Running RANSAC segmentation of horiz_surface planes (x, y = 0, 0)')
    inliers, equation = [], []
    i = 0
    percent_limit = 10
    z_min, z_max = min(xyz[:, 2]), max(xyz[:, 2])
    while i < steps:
        percent_done = int(i / steps * 100)
        if percent_done >= percent_limit:
            print('Progress: %d %%' % int(percent_done))
            percent_limit += 10
        a, b, c = 0, 0, 1
        d = z_min + steps * (z_max - z_min)
        distance = (a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        idx_candidates = np.where(np.abs(distance) <= threshold)[0]
        if len(idx_candidates) > len(inliers):
            equation = [a, b, c, d]
            inliers = idx_candidates
        i += 1
    return equation, inliers


def create_hull_alphashape(points_3d, concavity_level=1.0):
    points_2d = [[x, y] for x, y, _ in points_3d]
    alpha_shape = alphashape.alphashape(points_2d, concavity_level)
    hull = alpha_shape.convex_hull
    xx, yy = hull.exterior.coords.xy
    x_coords = xx.tolist()
    y_coords = yy.tolist()
    polygon = Polygon(list(zip(x_coords, y_coords)), facecolor='red', alpha=0.2, edgecolor='r', linewidth=3)
    return x_coords, y_coords, polygon


def create_ifc_file(polygon_x_coords, polygon_y_coords, slab_bottom_z_coord, slab_thickness, file_path):
    num_vertices = len(polygon_x_coords)
    vertices = ''
    for i in range(num_vertices):
        vertex = "('{0}','{1}','{2}')".format(polygon_x_coords[i], polygon_y_coords[i], slab_bottom_z_coord)
        vertices += vertex + ',' if i < num_vertices - 1 else vertex
    polygon = '#POLYGON({0})'.format(vertices)
    slab = '''#SLAB('','$',#MATERIAL('','','$',.NOTDEFINED.,.NOTDEFINED.,$,$,$,$,$),
                 $,$,.NOTDEFINED.,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,
                 $,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,
                 $,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,
                 $,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,
                 $,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,
                 $,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,
                 $,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,
                 $,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,
                 $,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,
                 $,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,$,
                 {0},$,{1},.NOTDEFINED.,$,$,$,$,$,$,$,$,$,$)'''.format(polygon, slab_thickness)

    with open(file_path, 'w') as f:
        f.write(polygon)
        f.write('\n')
        f.write(slab)


if __name__ == '__main__':

    # user inputs
    e57_input = False
    e57_file_name = 'input_e57/test_room.e57'
    # xyz_filename = 'input_xyz/test_room.xyz'
    xyz_filename = 'input_xyz/Test_room_merged.xyz'

    ransac_threshold = 0.2  # threshold for ransac identification of points corresponding to a plane (inliers)
    n_iterations = 1000  # number of iterations when identifying inliers

    # read e57 file
    if e57_input:
        imported_e57_data = read_e57(e57_file_name)
        e57_data_to_xyz(imported_e57_data, xyz_filename)

    # read xyz file
    points_xyz, points_rgb = load_xyz_file(xyz_filename)
    points_xyz = np.round(np.array(points_xyz), 3)  # round the xyz coordinates to 3 decimals

    z_step = 0.1
    z_min, z_max = min(points_xyz[:, 2]), max(points_xyz[:, 2])
    n_steps = int((z_max - z_min) / z_step + 1)
    z_array, n_points_array = [], []
    progress_percent_limit = 10
    for i in range(n_steps):
        percent_done = int(i / n_steps * 100)
        if percent_done >= progress_percent_limit:
            print('Progress searching for horiz_surface candidate z-coordinates: %d %%' % int(percent_done))
            progress_percent_limit += 10
        z = z_min + i * z_step
        idx_selected_xyz = np.where((z < points_xyz[:, 2]) & (points_xyz[:, 2] < (z + z_step)))[0]
        z_array.append(z)
        n_points_array.append(len(idx_selected_xyz))

    # plotting histogram of point densities in x-z plane
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(z_array, n_points_array, '-r')
    # ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c='salmon', s=0.02)
    # plt.show()

    horiz_surface_candidates = []
    max_n_points_array = max(n_points_array)
    # extract z-coordinates where the density of points (indicated by a high value on the histogram) exceeds 50%
    # of a maximum -> horiz_surface candidates
    for i in range(len(n_points_array)):
        if n_points_array[i] > 0.5 * max_n_points_array:
            horiz_surface_candidates.append([z_array[i], z_array[i] + z_step])

    horiz_surface_planes, horiz_surface_colors, horiz_surface_polygon, horiz_surface_polygon_x, horiz_surface_polygon_y, horiz_surface_z, horiz_surface_thickness = \
        [], [], [], [], [], [], []
    # extract xyz points within an interval given by horiz_surface_candidates (lie within the range given by the z-coordinates
    # in horiz_surface candidates)
    for i in range(len(horiz_surface_candidates)):
        print('Extracting points for a horiz_surface no. %d of %d.' % (i+1, len(horiz_surface_candidates)))
        horiz_surface_idx = np.where((horiz_surface_candidates[i][0] < points_xyz[:, 2]) & (points_xyz[:, 2] < horiz_surface_candidates[i][1]))[0]
        horiz_surface_planes.append(np.array(points_xyz[horiz_surface_idx]))
        horiz_surface_colors.append(np.array(points_rgb[horiz_surface_idx]) / 255)
    # eq, idx_inliers = ransac_plane(single_horiz_surface_selected, threshold=ransac_threshold, iterations=n_iterations)
    # inliers = points_xyz[idx_inliers]

    # merge lower and upper surface of each horiz_surface and create a hull
    slab = []
    for i in range(len(horiz_surface_candidates)):
        if (i % 2) == 1:
            print('Creating hull for slab no. %d of %d.' % ((i + 1) / 2, len(horiz_surface_candidates) / 2))
            slab_bottom_z_coord = np.median(horiz_surface_planes[i-1][:, 2])
            slab_top_z_coord = np.median(horiz_surface_planes[i][:, 2])
            slab_thickness = slab_top_z_coord - slab_bottom_z_coord
            slab_points = np.concatenate((horiz_surface_planes[i-1], horiz_surface_planes[i]), axis=0)
            x_coords, y_coords, polygon = create_hull_alphashape(slab_points, concavity_level=0.0)  # 0.0 -> convex
            slab.append({'polygon': polygon, 'polygon_x_coords': x_coords, 'polygon_y_coords': y_coords,
                         'slab_bottom_z_coord': slab_bottom_z_coord, 'thickness': slab_thickness})
            print('Slab no. %d: bottom (z-coordinate) = %.3f m, thickness = %0.1f mm'
                  % ((i + 1) / 2, slab_bottom_z_coord, slab_thickness * 1000))

    # plotting the slabs in x-y plane
    for i in range(len(horiz_surface_planes)):
        print('Plotting the horizontal surface no. %d of %d to file %s.' % (i + 1, len(horiz_surface_planes),
                                                                            'images/horiz_surface_%d.jpg' % (i + 1)))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(horiz_surface_planes[i][:, 0], horiz_surface_planes[i][:, 1], c=horiz_surface_colors[i], s=0.02)
        # ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c='salmon', s=0.02)
        # plt.axis('scaled')
        if (i % 2) == 0:
            new_slab_polygon = copy(slab[int((i - 1) / 2)]['polygon'])  # to avoid a RuntimeError
            ax.add_patch(new_slab_polygon)
        else:
            new_slab_polygon = copy(slab[int((i - 1) / 2)]['polygon'])  # to avoid a RuntimeError
            ax.add_patch(new_slab_polygon)
        ax.set_aspect('equal', 'box')
        # plt.show()
        fig.tight_layout()
        plt.savefig('images/horiz_surface_%d.jpg' % (i + 1), dpi=300)
        # plt.savefig('images/horiz_surface_%d.pdf' % (i + 1))
        plt.close(fig)

        save_xyz(horiz_surface_planes[i], 'output_xyz/horiz_surface_%d.xyz' % (i + 1))

    # save_xyz(inliers, 'output_xyz/inliers.txt')
    # mask = np.ones(len(points_xyz), dtype=bool)
    # mask[idx_inliers] = False
    # outliers = points_xyz[mask]

    # plot the segmented plane
    plot_segmented_plane = False
    pcd = []
    if plot_segmented_plane:
        for i in range(len(horiz_surface_planes)):
            pcd.append(o3d.io.read_point_cloud('output_xyz/horiz_surface_%d.xyz' % (i + 1), format='xyz'))
        o3d.visualization.draw_geometries(pcd)

    create_ifc_files = True
    if create_ifc_files:
        for i, s in enumerate(slab):
            create_ifc_file(s['polygon_x_coords'], s['polygon_y_coords'], s['slab_bottom_z_coord'], s['thickness'],
                            'output_IFC/slab_%d.ifc' % (i + 1))
