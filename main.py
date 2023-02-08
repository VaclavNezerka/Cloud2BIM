import pye57
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


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


def ransac_plane(xyz, threshold=0.05, iterations=1000):
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


if __name__ == '__main__':

    # user inputs
    e57_input = True
    e57_file_name = 'input_e57/test_room.e57'
    xyz_filename = 'input_xyz/test_room.xyz'
    ransac_threshold = 0.01  # threshold for ransac identification of points corresponding to a plane (inliers)
    n_iterations = 100  # number of iterations when identifying inliers

    # read e57 file
    if e57_input:
        imported_e57_data = read_e57(e57_file_name)
        e57_data_to_xyz(imported_e57_data, xyz_filename)

    # read xyz file
    points_xyz, points_rgb = load_xyz_file(xyz_filename)

    # get plane equation(s) and list of inliers
    eq, idx_inliers = ransac_plane(points_xyz, threshold=ransac_threshold, iterations=n_iterations)
    inliers = points_xyz[idx_inliers]
    mask = np.ones(len(points_xyz), dtype=bool)
    mask[idx_inliers] = False
    outliers = points_xyz[mask]

    # plot the segmented plane
    plot_segmented_plane = True
    if plot_segmented_plane:
        ax = plt.axes(projection='3d')
        ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], c='cornflowerblue', s=0.02)
        # ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c='salmon', s=0.02)
        plt.show()
