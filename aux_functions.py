import os
import sys
import time
import math
import random
from datetime import datetime
from itertools import islice

import yaml
import cv2
import pandas as pd
from scipy.signal import find_peaks
from skimage.morphology import closing, footprint_rectangle
import open3d as o3d
import e57
from tqdm import tqdm
from plotting_functions import *


def load_config_and_variables():
    """Load YAML config passed as CLI argument, validate required keys, and return configuration variables."""
    if len(sys.argv) < 2:
        # Default fallback path for development
        config_path = "config.yaml"  # nebo celá cesta
        print("[INFO] No argument provided, using input configuration file in directory of project:", config_path)
    else:
        config_path = sys.argv[1]

    if not os.path.isfile(config_path):
        sys.exit(f"[ERROR] File '{config_path}' does not exist.")

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        sys.exit(f"[ERROR] Invalid YAML format in '{config_path}': {e}")
    except Exception as e:
        sys.exit(f"[ERROR] Failed to load configuration file: {e}")

    required_keys = [
        "e57_input", "xyz_files", "exterior_scan",
        "dilute", "dilution_factor", "pc_resolution", "grid_coefficient",
        "bfs_thickness", "tfs_thickness",
        "min_wall_length", "min_wall_thickness", "max_wall_thickness", "exterior_walls_thickness",
        "output_ifc", "ifc_project_name", "ifc_project_long_name", "ifc_project_version",
        "ifc_author_name", "ifc_author_surname", "ifc_author_organization",
        "ifc_building_name", "ifc_building_type", "ifc_building_phase",
        "ifc_site_latitude", "ifc_site_longitude", "ifc_site_elevation",
        "material_for_objects"
    ]

    if config.get("e57_input"):
        required_keys.append("e57_files")

    missing = [key for key in required_keys if key not in config]
    if missing:
        for key in missing:
            print(f"[ERROR] Missing required config key: '{key}'")
        sys.exit(1)

    variables = {
        "e57_input": config["e57_input"],
        "xyz_filenames": config["xyz_files"],
        "exterior_scan": config["exterior_scan"],
        "dilute_pointcloud": config["dilute"],
        "dilution_factor": config["dilution_factor"],
        "pc_resolution": config["pc_resolution"],
        "grid_coefficient": config["grid_coefficient"],
        "bfs_thickness": config["bfs_thickness"],
        "tfs_thickness": config["tfs_thickness"],
        "min_wall_length": config["min_wall_length"],
        "min_wall_thickness": config["min_wall_thickness"],
        "max_wall_thickness": config["max_wall_thickness"],
        "exterior_walls_thickness": config["exterior_walls_thickness"],
        "ifc_output_file": config["output_ifc"],
        "ifc_project_name": config["ifc_project_name"],
        "ifc_project_long_name": config["ifc_project_long_name"],
        "ifc_project_version": config["ifc_project_version"],
        "ifc_author_name": config["ifc_author_name"],
        "ifc_author_surname": config["ifc_author_surname"],
        "ifc_author_organization": config["ifc_author_organization"],
        "ifc_building_name": config["ifc_building_name"],
        "ifc_building_type": config["ifc_building_type"],
        "ifc_building_phase": config["ifc_building_phase"],
        "ifc_site_latitude": tuple(config["ifc_site_latitude"]),
        "ifc_site_longitude": tuple(config["ifc_site_longitude"]),
        "ifc_site_elevation": config["ifc_site_elevation"],
        "material_for_objects": config["material_for_objects"]
    }

    if config["e57_input"]:
        variables["e57_file_names"] = config["e57_files"]

    return variables

def log(message, last_time, filename):
    current_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    elapsed_time = current_time - last_time
    log_message = f"{timestamp} - {message} Elapsed time: {elapsed_time:.2f} s."
    with open(filename, 'a') as f:
        f.write(log_message)
    print(log_message)
    return current_time


def read_e57(file_name):
    # read the documentation at https://github.com/davidcaron/pye57
    e57_array = e57.read_points(file_name)
    return e57_array


def e57_data_to_xyz(e57_data, output_file_name, chunk_size=10000):
    points = e57_data.points
    n_points = points.shape[0]
    colors = e57_data.color
    intensities = e57_data.intensity

    num_chunks = int((n_points - 1) // chunk_size + 1)  # Compute the number of chunks

    for i in tqdm(range(num_chunks)):  # tqdm will display a progress bar
        start = int(i * chunk_size)
        end = int(min((i + 1) * chunk_size, n_points))

        x = points[:, 0][start:end]
        y = points[:, 1][start:end]
        z = points[:, 2][start:end]
        red = colors[:, 0][start:end]
        green = colors[:, 1][start:end]
        blue = colors[:, 2][start:end]
        intensity = intensities[:, 0][start:end]

        df = pd.DataFrame({'X': x, 'Y': y, 'Z': z, 'R': red, 'G': green, 'B': blue, 'Intensity': intensity})
        # Round the DataFrame entries to 3 decimal places
        df = df.round(3)

        # Check if file exists and is not empty
        if os.path.exists(output_file_name) and os.path.getsize(output_file_name) > 0:
            df.to_csv(output_file_name, sep='\t', index=False, header=False, mode='a')
        else:
            df.to_csv(output_file_name, sep='\t', index=False, header=True, mode='a')


def save_xyz(points, output_file_name):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # df = pd.DataFrame({'//X': x, 'Y': y, 'Z': z, 'R': red, 'G': green, 'B': blue, 'Intensity': intensity})
    df = pd.DataFrame({'//X': x, 'Y': y, 'Z': z})
    df.to_csv(output_file_name, sep='\t', index=False)
    print('Points saved as %s' % output_file_name)


def load_selective_lines(filename, step):
    with open(filename, 'r') as file:
        # Skip the first line
        next(file)
        lines = (line.strip().split('\t') for line in islice(file, 0, None, step))
        return [[float(element) for element in line] for line in lines]


def load_xyz_file(file_name, plot_xyz=False, select_ith_lines=True, ith_lines=20):
    if select_ith_lines:
        pcd = np.array(load_selective_lines(file_name, ith_lines))
        xyz = pcd[1:, :3]
        rgb = pcd[1:, 3:6]
    else:
        pcd = np.loadtxt(file_name, skiprows=1)
        xyz = pcd[:, :3]
        rgb = pcd[:, 3:6]
    print('Point cloud consisting of %d points loaded.' % len(xyz))

    # show plot of xyz points from top view: (x, y) coordinates and with rgb-colored points
    if plot_xyz:
        plt.figure(figsize=(8, 5), dpi=150)
        plt.scatter(xyz[:, 0], xyz[:, 1], c=rgb / 255, s=0.05)
        plt.title("Top-View")
        plt.xlabel('X-axis (m)')
        plt.ylabel('Y-axis (m)')
        plt.show()
    return xyz, rgb


def smooth_contour(x_contour, y_contour, epsilon):
    """
    Smooths a contour using the Douglas-Peuckert algorithm.

    Parameters:
        x_contour (ndarray): Array of x-coordinates of the contour.
        y_contour (ndarray): Array of y-coordinates of the contour.
        epsilon (float): Maximum distance for a point to be considered on the approximated line.

    Returns:
        ndarray: Smoothed x-coordinates of the contour.
        ndarray: Smoothed y-coordinates of the contour.
        ndarray: Smoothed contour as ndarray: (n, 2) (x and y)
    """
    points = np.column_stack((x_contour, y_contour)).astype(np.float32)
    epsilon = epsilon * cv2.arcLength(points, True)
    simplified_points = cv2.approxPolyDP(points, epsilon, True)
    simplified_points = np.squeeze(simplified_points, axis=1)
    x_smoothed = simplified_points[:, 0]
    y_smoothed = simplified_points[:, 1]
    return x_smoothed, y_smoothed, simplified_points


def create_hull_from_histogram(points_3d, pointcloud_resolution, grid_coefficient, plot_graphics,
                               dilation_meters, erosion_meters):
    # Project 3D points to 2D
    points_2d = np.array([[x, y] for x, y, _ in points_3d])

    # Parameters for histogram
    pixel_size = pointcloud_resolution * grid_coefficient
    dilation_kernel_size = int(dilation_meters / pixel_size)
    erosion_kernel_size = int(erosion_meters / pixel_size)

    x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
    y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()

    # Extend the domain of the plot
    extension_meters = 1  # extend the domain for histogram generation
    x_min_extended, x_max_extended = x_min - extension_meters, x_max + extension_meters
    y_min_extended, y_max_extended = y_min - extension_meters, y_max + extension_meters

    x_edges = np.arange(x_min_extended, x_max_extended + pixel_size, pixel_size)
    y_edges = np.arange(y_min_extended, y_max_extended + pixel_size, pixel_size)

    # Create 2D histogram and mask
    histogram, _, _ = np.histogram2d(points_2d[:, 0], points_2d[:, 1], bins=(x_edges, y_edges))
    mask = histogram.T > 0  # Threshold to create mask, transposed for correct orientation
    if plot_graphics:
        plot_2d_histogram(mask, x_edges, y_edges)

    # Apply morphological operation
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    mask_eroded = cv2.erode(mask_dilated, kernel, iterations=1)

    # Shift the mask for more accurate contours
    shifted_mask = np.roll(mask_eroded, (-1, -1), axis=(0, 1))
    if plot_graphics:
        plot_shifted_mask(shifted_mask, x_edges, y_edges)

    # Find contours on the eroded mask for correct orientation
    contours, hierarchy = cv2.findContours(shifted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Adjust contour scaling
    contour = np.squeeze(largest_contour, axis=1)  # Remove redundant dimension
    # Adjusting scaling to fully cover the bin extents
    contour_scaled = (contour + 0.5) * pixel_size + [x_min_extended,
                                                     y_min_extended]  # Add 0.5 to shift to the center of the bin
    # Adjusting scaling to the original domain
    polygon = Polygon(contour_scaled, fill=None, edgecolor='blue')
    x_contour = contour_scaled[:, 0].flatten()
    y_contour = contour_scaled[:, 1].flatten()

    # Smooth the contour
    smoothing_factor = 0.0005
    x_contour_smoothed, y_contour_smoothed, simplified_points = smooth_contour(x_contour, y_contour,
                                                                               epsilon=smoothing_factor)
    polygon_smoothed = Polygon(simplified_points, fill=None, edgecolor='red')
    if plot_graphics:
        plot_smoothed_contour(polygon, polygon_smoothed)

    return x_contour_smoothed, y_contour_smoothed, polygon_smoothed


def identify_slabs(points_xyz, points_rgb, bottom_floor_slab_thickness, top_floor_ceiling_thickness,
                   z_step, pc_resolution, plot_segmented_plane=True):
    z_min, z_max = min(points_xyz[:, 2]), max(points_xyz[:, 2])
    n_steps = int((z_max - z_min) / z_step + 1)
    z_array, n_points_array = [], []
    for i in tqdm(range(n_steps), desc="Progress searching for horiz_surface candidate z-coordinates"):
        z = z_min + i * z_step
        idx_selected_xyz = np.where((z < points_xyz[:, 2]) & (points_xyz[:, 2] < (z + z_step)))[0]
        z_array.append(z)
        n_points_array.append(len(idx_selected_xyz))
    max_n_points_array = 0.6 * max(n_points_array)

    # plot_point_cloud_data(points_xyz, n_points_array, z_array, max_n_points_array, z_step)

    # Histogram plotting
    if plot_segmented_plane:
        plt.plot(np.array(n_points_array) / 1000, z_array, '-r', linewidth=0.8)
        plt.plot([max_n_points_array / 1000, max_n_points_array / 1000], [min(z_array), max(z_array)], '--b', linewidth=1.0)
        plt.ylabel(r'Height/z-coordinate (m)')
        plt.xlabel(r'Number of points ($\times 10^3$)')
        plt.show()

    # extract z-coordinates where the density of points (indicated by a high value on the histogram) exceeds 50%
    # of a maximum -> horiz_surface candidates
    h_surf_candidates = []
    start = None
    for i in range(len(n_points_array)):
        if n_points_array[i] > max_n_points_array:
            if start is None:
                start = i
        elif start is not None:
            h_surf_candidates.append([z_array[start], z_array[i - 1] + z_step])
            start = None

    if start is not None:
        h_surf_candidates.append([z_array[start], z_array[-1] + z_step])

    merged_candidates = []
    for interval in h_surf_candidates:
        if len(merged_candidates) == 0 or interval[0] > merged_candidates[-1][1]:
            merged_candidates.append(interval)
        else:
            merged_candidates[-1][1] = interval[1]

    horiz_surface_planes, horiz_surface_colors, horiz_surface_polygon, horiz_surface_polygon_x, \
        horiz_surface_polygon_y, horiz_surface_z, horiz_surface_thickness = [], [], [], [], [], [], []

    # extract xyz points within an interval given by horiz_surface_candidates (lie within the range given by the
    # z-coordinates in horiz_surface candidates)
    for i in tqdm(range(len(h_surf_candidates)), desc="Extracting points for horizontal surfaces"):
        horiz_surface_idx = np.where(
            (h_surf_candidates[i][0] < points_xyz[:, 2]) &
            (points_xyz[:, 2] < h_surf_candidates[i][1]))[0]
        horiz_surface_planes.append(points_xyz[horiz_surface_idx])
        #horiz_surface_colors.append(points_rgb[horiz_surface_idx] / 255)

    # plot_horizontal_surfaces(horiz_surface_planes)

    # merge lower and upper surface of each horiz_surface and create a hull
    slabs = []
    for i in range(len(h_surf_candidates)):
        if i == 0:
            print('Creating hull for slab no. %d of %d.' % ((i + 1), int(len(h_surf_candidates) / 2) + 1))
            slab_top_z_coord = np.median(horiz_surface_planes[i][:, 2])
            slab_bottom_z_coord = slab_top_z_coord - bottom_floor_slab_thickness
            x_coords, y_coords, polygon = create_hull_from_histogram(horiz_surface_planes[i], pc_resolution,
                                                                     grid_coefficient=5, plot_graphics=True,
                                                                     dilation_meters=1.0, erosion_meters=1.0)
            slabs.append({'polygon': polygon, 'polygon_x_coords': x_coords, 'polygon_y_coords': y_coords,
                          'slab_bottom_z_coord': slab_bottom_z_coord, 'thickness': bottom_floor_slab_thickness})
            print('Slab no. %d: bottom (z-coordinate) = %.3f m, thickness = %0.1f mm'
                  % ((i + 1) / 2, slab_bottom_z_coord, bottom_floor_slab_thickness * 1000))

        elif (i % 2) == 0:
            print('Creating hull for slab no. %d of %d.' % ((i + 1) / 2, int(len(h_surf_candidates) / 2) + 1))
            slab_bottom_z_coord = np.median(horiz_surface_planes[i - 1][:, 2])
            slab_top_z_coord = np.median(horiz_surface_planes[i][:, 2])
            slab_thickness = slab_top_z_coord - slab_bottom_z_coord
            slab_points = np.concatenate((horiz_surface_planes[i - 1], horiz_surface_planes[i]), axis=0)

            # create hull for the slab
            x_coords, y_coords, polygon = create_hull_from_histogram(slab_points, pc_resolution,
                                                                     grid_coefficient=5, plot_graphics=True,
                                                                     dilation_meters=1.5, erosion_meters=1.5)
            slabs.append({'polygon': polygon, 'polygon_x_coords': x_coords, 'polygon_y_coords': y_coords,
                          'slab_bottom_z_coord': slab_bottom_z_coord, 'thickness': slab_thickness})
            print('Slab no. %d: bottom (z-coordinate) = %.3f m, thickness = %0.1f mm'
                  % ((i + 1) / 2, slab_bottom_z_coord, slab_thickness * 1000))

        elif (i % 2) == 1 and i == len(h_surf_candidates) - 1:
            print('Creating hull for slab no. %d of %d.' % ((i + 1) / 2, int(len(h_surf_candidates) / 2) + 1))
            slab_bottom_z_coord = np.median(horiz_surface_planes[i][:, 2])

            # create hull for the slab
            x_coords, y_coords, polygon = create_hull_from_histogram(horiz_surface_planes[i], pc_resolution,
                                                                     grid_coefficient=5, plot_graphics=True,
                                                                     dilation_meters=1.5, erosion_meters=1.5)
            slabs.append({'polygon': polygon, 'polygon_x_coords': x_coords, 'polygon_y_coords': y_coords,
                          'slab_bottom_z_coord': slab_bottom_z_coord, 'thickness': top_floor_ceiling_thickness})
            print('Slab no. %d: bottom (z-coordinate) = %.3f m, thickness = %0.1f mm'
                  % ((i + 1) / 2, slab_bottom_z_coord, top_floor_ceiling_thickness * 1000))

        save_xyz(horiz_surface_planes[i], 'output_xyz/horiz_surface_%d.xyz' % (i + 1))

    # plot the segmented plane
    pcd = []
    if plot_segmented_plane:
        for i in range(len(horiz_surface_planes)):
            pcd.append(o3d.io.read_point_cloud('output_xyz/horiz_surface_%d.xyz' % (i + 1), format='xyz'))
        o3d.visualization.draw_geometries(pcd)

    return slabs, horiz_surface_planes


def split_pointcloud_to_storeys(points_xyz, slabs):
    segmented_pointclouds_3d = []

    # Iterate through the slabs and get the regions between consecutive slabs
    safety_margin = 0.1  # safety margin for the point cloud splitting into storeys
    for i in range(len(slabs) - 1):
        bottom_z_of_upper_slab = slabs[i + 1]['slab_bottom_z_coord'] + safety_margin  # upper limit (+ 10 cm of the ceiling slab)
        top_z_of_bottom_slab = slabs[i]['slab_bottom_z_coord'] + slabs[i][
            'thickness'] - safety_margin  # bottom limit (- 10 cm of the floor)

        # Extract points that are between the bottom of the upper slab and the top of the lower slab
        segmented_pointcloud_idx = np.where((top_z_of_bottom_slab < points_xyz[:, 2]) &
                                            (points_xyz[:, 2] < bottom_z_of_upper_slab))[0]

        if len(segmented_pointcloud_idx) > 0:
            segmented_pointcloud_points_in_storey = points_xyz[segmented_pointcloud_idx]
            segmented_pointclouds_3d.append(segmented_pointcloud_points_in_storey)

    return segmented_pointclouds_3d


def visualize_segmented_pointclouds(segmented_pointclouds_3d):
    # Create an Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Iterate through segmented point clouds and add them to the visualizer
    for pointcloud in segmented_pointclouds_3d:
        # Create an Open3D point cloud from the segmented point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)

        # Add the point cloud to the visualizer with a unique color
        color = np.random.rand(3)
        pcd.paint_uniform_color(color)
        vis.add_geometry(pcd)

    # Set the camera position
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_up([0, -1, 0])
    vis.get_view_control().set_zoom(0.8)

    # Run the visualizer
    vis.run()
    vis.destroy_window()


def display_cross_section_plot(segmented_pointclouds_3d, slabs):
    # Create a new matplotlib figure
    plt.figure(figsize=(8, 6))

    # Plot segmented point clouds
    for i, pointcloud in enumerate(segmented_pointclouds_3d):
        plt.scatter(pointcloud[:, 0], pointcloud[:, 2], s=1)

        # Plot lines representing slab limits
        slab = slabs[i]
        bottom_z = slab['slab_bottom_z_coord']
        top_z = bottom_z + slab['thickness']
        min_x = np.min(pointcloud[:, 0])
        max_x = np.max(pointcloud[:, 0])
        plt.plot([min_x - 1, max_x + 1], [bottom_z, bottom_z], 'r--')
        plt.plot([min_x - 1, max_x + 1], [top_z, top_z], 'b--')

    plt.xlabel('X-coordinate (m)')
    plt.ylabel('Z-coordinate (m)')
    plt.legend('Cross Section')
    plt.grid(False)
    plt.axis('equal')
    plt.show()


def save_coordinates_to_xyz(coordinates_list, base_filename):
    for i, coordinates in enumerate(coordinates_list):
        x_coordinates = coordinates[:, 0]
        y_coordinates = coordinates[:, 1]
        z_coordinates = coordinates[:, 2]

    # Construct the filename with a numerical suffix
    filename = f'{base_filename}_{i}.xyz'

    # Combine X, Y, and Z coordinates and save to the XYZ file
    combined_coordinates = np.column_stack((x_coordinates, y_coordinates, z_coordinates))
    np.savetxt(filename, combined_coordinates, delimiter=' ', fmt='%.4f')


# Functions used for identification of walls

# Define a function to get line segments from a contour using Douglas-Peuckert algorithm
def get_line_segments(contour, pixel_size, segment_approximation_tolerance=0.02):
    epsilon = segment_approximation_tolerance / pixel_size
    approx = cv2.approxPolyDP(contour, epsilon, True)
    segments = []
    for i in range(len(approx)):
        segment = [tuple(approx[i - 1][0]), tuple(approx[i][0])]
        segments.append(segment)

    return segments


def random_color():
    """Generate a random color."""
    return random.random(), random.random(), random.random()


def distance_point_to_line(point, line_start, line_end):
    """Calculate the distance from a single point to a line defined by two points."""

    line_start = np.array(line_start)
    line_end = np.array(line_end)
    point = np.array(point)

    # Vector from line_start to line_end
    line_vec = line_end - line_start

    # Vector from line_start to the point
    point_vec = point - line_start

    # Calculate the line length and ensure it's not zero for division
    line_length = np.linalg.norm(line_vec)
    if np.isclose(line_length, 0):
        print("Warning: Line start and end points are the same. Returning NaN.")
        return np.nan

    # Normalize the line vector
    line_vec_normalized = line_vec / line_length

    # Project point_vec onto the line vector (dot product)
    projection_length = np.dot(point_vec, line_vec_normalized)

    # Calculate the closest point on the line to the point
    closest_point = line_start + projection_length * line_vec_normalized

    # Calculate and return the distance from the point to the closest point on the line
    distance = np.linalg.norm(point - closest_point)

    return distance


def distance_points_to_line_np(points, line_start, line_end):
    """Calculate the Euclidean distances from multiple points to a line defined by two points."""
    points = np.asarray(points)
    line_start = np.asarray(line_start)
    line_end = np.asarray(line_end)

    line_vec = line_end - line_start
    line_length = np.linalg.norm(line_vec)

    if np.isclose(line_length, 0):
        return np.full(points.shape[0], np.nan)

    line_vec_normalized = line_vec / line_length
    point_vecs = points - line_start
    projections = np.dot(point_vecs, line_vec_normalized)

    on_segment = (projections >= 0) & (projections <= line_length)
    closest_points = np.outer(projections, line_vec_normalized) + line_start
    perpendicular_distances = np.linalg.norm(points - closest_points, axis=1)

    distances_to_start = np.linalg.norm(points - line_start, axis=1)
    distances_to_end = np.linalg.norm(points - line_end, axis=1)
    distances = np.where(on_segment, perpendicular_distances, np.minimum(distances_to_start, distances_to_end))

    return distances


def distance_between_points(point1, point2):
    """Calculate the Euclidean distance between two points."""
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)


def merge_segments(seg1, seg2):
    """Merge two segments into one."""
    points = seg1 + seg2
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
    return [sorted_points[0], sorted_points[-1]]


def segments_collinearity_check(seg1, seg2, min_thickness, max_distance):
    """Check if two segments are candidates for merging."""
    # Check if the segments are close enough to merge based on maximum wall thickness
    close_enough = any(
        distance_between_points(p1, p2) <= max_distance for p1 in seg1 for p2 in seg2
    )

    # Check if the segments are co-linear
    collinear = any(
        distance_point_to_line(point, seg1[0], seg1[1]) < (min_thickness / 2) for point in seg2
    )

    return close_enough and collinear


def find_furthest_points(all_points):
    def distance(point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    max_distance = -1
    start_point = None
    end_point = None

    # Iterate through each pair of points to find the furthest pair
    for i in range(len(all_points)):
        for j in range(i + 1, len(all_points)):
            dist = distance(all_points[i], all_points[j])
            if dist > max_distance:
                max_distance = dist
                start_point = all_points[i]
                end_point = all_points[j]

    return start_point, end_point


def merge_collinear_segments(segments, min_thickness, max_distance):
    """Merge co-linear segments from the given list using the direct approach we tested."""
    final_segments = []
    counter = 0
    while segments:
        counter += 1
        base_segment = segments[0]
        to_merge = [base_segment]

        for other_segment in segments[1:]:
            if (segments_collinearity_check(base_segment, other_segment, min_thickness, max_distance)
                    and segments_angle(base_segment, other_segment, angle_tolerance=3)):
                to_merge.append(other_segment)

        # Merge all the segments in to_merge into a single segment
        if len(to_merge) > 1:
            all_points = [point for seg in to_merge for point in seg]
            start, end = find_furthest_points(all_points)
            merged_segment = [start, end]
            segments.append(merged_segment)
        else:
            final_segments.append(base_segment)

        for seg in to_merge:
            segments.remove(seg)

    return final_segments


def angle_between_segments(seg1, seg2):
    """Calculate the angle (in degrees) between two segments."""
    dx1 = seg1[1][0] - seg1[0][0]
    dy1 = seg1[1][1] - seg1[0][1]
    dx2 = seg2[1][0] - seg2[0][0]
    dy2 = seg2[1][1] - seg2[0][1]

    dot_product = dx1 * dx2 + dy1 * dy2
    magnitude1 = (dx1 ** 2 + dy1 ** 2) ** 0.5
    magnitude2 = (dx2 ** 2 + dy2 ** 2) ** 0.5

    if magnitude1 * magnitude2 == 0:
        return 90  # Perpendicular

    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle_rad = math.acos(min(1, max(-1, cosine_angle)))  # Clip to avoid out of domain error
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def segments_angle(seg1, seg2, angle_tolerance=3):
    """Check if two segments are approximately parallel within a given angle tolerance (in degrees)."""
    angle = angle_between_segments(seg1, seg2)
    return abs(angle) < angle_tolerance or abs(angle - 180) < angle_tolerance


def perpendicular_distance_between_segments(seg1, seg2):
    """Calculate the shortest perpendicular distance between two parallel segments."""
    if segments_angle(seg1, seg2):
        return distance_point_to_line(seg2[0], seg1[0], seg1[1])
    else:
        return float('inf')


def check_overlap_parallel_segments(seg1, seg2, min_overlap):
    def calculate_angle(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.atan2(dy, dx)

    def rotate_point(point, angle_for_rotation):
        rotation_matrix = np.array([
            [np.cos(angle_for_rotation), -np.sin(angle_for_rotation)],
            [np.sin(angle_for_rotation), np.cos(angle_for_rotation)]
        ])
        return np.dot(rotation_matrix, np.array([point[0], point[1]]))

    def process_and_rotate_segments(seg_1, seg_2, rot_angle):
        return [
            [rotate_point(seg_1[0], -rot_angle), rotate_point(seg_1[1], -rot_angle)],
            [rotate_point(seg_2[0], -rot_angle), rotate_point(seg_2[1], -rot_angle)]
        ]

    def find_x_axis_overlap(rot_seg1, rot_seg2):
        x1_min, x1_max = sorted([rot_seg1[0][0], rot_seg1[1][0]])
        x2_min, x2_max = sorted([rot_seg2[0][0], rot_seg2[1][0]])
        start = max(x1_min, x2_min)
        end = min(x1_max, x2_max)
        return (start, end) if start < end else None

    def calculate_overlap_length(overlay):
        return overlay[1] - overlay[0] if overlay else 0

    # Calculate the rotation angle for the first segment to align with the x-axis
    angle = calculate_angle(seg1[0], seg1[1])

    # Rotate both segments using the calculated angle
    rotated_seg1, rotated_seg2 = process_and_rotate_segments(seg1, seg2, angle)

    # Find overlap along the x-axis
    overlap = find_x_axis_overlap(rotated_seg1, rotated_seg2)
    overlap_length = calculate_overlap_length(overlap)

    # Check if the overlap length meets the minimum requirement
    return overlap_length > min_overlap


def group_segments(segments, max_wall_thickness, wall_label, angle_tolerance=5):
    """Group segments that are parallel with a small tolerance."""
    grouped = []
    wall_labels = []
    facade_wall_candidate = []  # List to hold segments that aren't grouped

    while segments:
        current_segment = segments.pop(0)
        parallel_group = [current_segment]

        i = 0
        while i < len(segments):
            segment = segments[i]
            if (
                    segments_angle(current_segment, segment, angle_tolerance) and
                    any(distance_between_points(p1, p2) <= max_wall_thickness for p1 in current_segment for p2 in
                        segment) and
                    check_overlap_parallel_segments(current_segment, segment, min_overlap=max_wall_thickness)
            ):
                parallel_group.append(segment)
                segments.pop(i)
            else:
                i += 1

        # Save only the groups consisting of two or more segments
        if len(parallel_group) >= 2:
            grouped.append(parallel_group)
            wall_labels.append(wall_label)
        else:
            facade_wall_candidate.append(current_segment)

    return grouped, wall_labels, facade_wall_candidate


def calculate_wall_axis(group):
    """Calculate the axis for a group of parallel segments."""
    if len(group) < 2:
        return None

    # Find the longer segment in the group
    lengths = [distance_between_points(seg[0], seg[1]) for seg in group]
    longer_segment = group[np.argmax(lengths)]
    shorter_segment = group[1 - np.argmax(lengths)]

    # Calculate the direction of the axis based on the longer segment
    direction = [longer_segment[1][0] - longer_segment[0][0], longer_segment[1][1] - longer_segment[0][1]]
    norm = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
    direction = [direction[0] / norm, direction[1] / norm]

    # Calculate the mean distance between the two segments
    mean_distance = np.mean([perpendicular_distance_between_segments(longer_segment, shorter_segment),
                             perpendicular_distance_between_segments(shorter_segment, longer_segment)])
    half_mean_distance = mean_distance / 2

    # Calculate the start and end of the axis (initial position)
    axis_start = [longer_segment[0][0] - half_mean_distance * direction[1],
                  longer_segment[0][1] + half_mean_distance * direction[0]]
    axis_end = [longer_segment[1][0] - half_mean_distance * direction[1],
                longer_segment[1][1] + half_mean_distance * direction[0]]

    # Calculate sum of distances for initial position
    distance_sum_initial = sum([distance_between_points(pt, axis_start) + distance_between_points(pt, axis_end)
                                for pt in longer_segment + shorter_segment])

    # Flip the axis to the other side of the longer segment
    axis_start_flipped = [longer_segment[0][0] + half_mean_distance * direction[1],
                          longer_segment[0][1] - half_mean_distance * direction[0]]
    axis_end_flipped = [longer_segment[1][0] + half_mean_distance * direction[1],
                        longer_segment[1][1] - half_mean_distance * direction[0]]

    # Calculate sum of distances for flipped position
    distance_sum_flipped = sum(
        [distance_between_points(pt, axis_start_flipped) + distance_between_points(pt, axis_end_flipped)
         for pt in longer_segment + shorter_segment])

    # Choose the position that gives a smaller sum
    if distance_sum_flipped < distance_sum_initial:
        axis_start, axis_end = axis_start_flipped, axis_end_flipped

    return [axis_start, axis_end], mean_distance


def line_intersection(line1, line2):
    """Find the intersection point of two lines (if it exists)."""
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Lines don't intersect

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def adjust_intersections(wall_axes, max_wall_thickness):
    """Adjust wall axes to account for intersections."""
    half_max_thickness = max_wall_thickness / 2

    for i, axis1 in enumerate(wall_axes):
        for j, axis2 in enumerate(wall_axes):
            if i == j:
                continue  # Don't compare the segment with itself

            intersection = line_intersection(axis1, axis2)
            if intersection:
                # Check the distance from the intersection to each endpoint of the axes
                for k in range(2):  # Check both endpoints for each axis
                    if distance_between_points(axis1[k], intersection) <= half_max_thickness:
                        axis1[k] = list(intersection)
                    if distance_between_points(axis2[k], intersection) <= half_max_thickness:
                        axis2[k] = list(intersection)

    return wall_axes


def plot_parallel_groups(groups, wall_axes, binary_image, points_2d, x_min, x_max, y_min, y_max, storey):
    fig = plt.figure(figsize=(10, 8))

    # Plot the binary image
    plt.imshow(binary_image, cmap='gray', origin='lower', extent=(x_min, x_max, y_min, y_max), alpha=0.6)

    # Scatter plot of points_2d
    plt.scatter(points_2d[:, 0], points_2d[:, 1], color='green', alpha=0.2, s=1)  # alpha for transparency

    # Plot each group with a unique color
    for idx, group in enumerate(groups):
        color = random_color()
        for segment in group:
            x_values = [segment[0][0], segment[1][0]]
            y_values = [segment[0][1], segment[1][1]]
            plt.plot(x_values, y_values, color=color, linewidth=2)

        # Plot the corresponding wall axis
        axis = wall_axes[idx]
        if axis:
            plt.plot([axis[0][0], axis[1][0]], [axis[0][1], axis[1][1]], color=color, linestyle='--', linewidth=1.5)

    plt.xlabel('x-coordinate (m)')
    plt.ylabel('y-coordinate (m)')
    plt.title("Identified walls and their axes")
    ax = fig.gca()
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.savefig('images/walls_in_storey_%d.jpg' % (storey + 1), dpi=200)
    plt.close(fig)


def swell_polygon(vertices, thickness):
    def compute_normal(pt1, pt2):
        edge = np.array(pt2) - np.array(pt1)
        normal_vector = np.array([-edge[1], edge[0]])  # Rotate 90 degrees to get the normal
        normal_length = np.linalg.norm(normal_vector)
        return normal_vector / normal_length if normal_length != 0 else normal

    def compute_centroid(vertices_local):
        centroid_local = np.mean(vertices_local, axis=0)
        return centroid_local

    centroid = compute_centroid(vertices)
    offset_segments = []

    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        normal = compute_normal(p1, p2)
        midpoint = (np.array(p1) + np.array(p2)) / 2
        direction = midpoint - centroid
        if np.dot(normal, direction) < 0:
            normal = -normal
        offset_p1 = (np.array(p1) + thickness * normal).tolist()
        offset_p2 = (np.array(p2) + thickness * normal).tolist()
        offset_segments.append([offset_p1, offset_p2])

    return offset_segments


def identify_walls(pointcloud, pointcloud_resolution, minimum_wall_length, minimum_wall_thickness,
                   maximum_wall_thickness, z_floor, z_ceiling, grid_coefficient=5, slab_polygon=None,
                   exterior_scan=False, exterior_walls_thickness=0.3):
    x_coords, y_coords, z_coords = zip(*pointcloud)
    z_section_boundaries = [0.85, 1.2]  # percentage of the height for the storey sections

    # Calculate z-coordinate limits
    z_max_in_point_cloud = np.max(z_coords)
    z_min_in_point_cloud = np.min(z_coords)
    z_max = z_min_in_point_cloud + z_section_boundaries[1] * (z_max_in_point_cloud - z_min_in_point_cloud)
    z_min = z_min_in_point_cloud + z_section_boundaries[0] * (z_max_in_point_cloud - z_min_in_point_cloud)

    # Filter points based on z-coordinate limits
    filtered_indices = [i for i, z in enumerate(z_coords) if z_min <= z <= z_max]
    points_2d = np.array([(x_coords[i], y_coords[i]) for i in filtered_indices])

    # Compute 2D histogram from the 2D point cloud
    print("Computing 2D histogram from the 2D point cloud")
    pixel_size = pointcloud_resolution * grid_coefficient
    x_min, y_min = np.min(points_2d, axis=0)
    x_max, y_max = np.max(points_2d, axis=0)
    x_values_full = np.arange(x_min + 0.5 * pixel_size, x_max, pixel_size)
    y_values_full = np.arange(y_min + 0.5 * pixel_size, y_max, pixel_size)
    grid_full, _, _ = np.histogram2d(points_2d[:, 0], points_2d[:, 1], bins=[x_values_full, y_values_full])
    grid_full = grid_full.T
    plot_histogram(grid_full, x_values_full, y_values_full)

    # Convert the 2D histogram to binary (mask) based on a threshold
    threshold = 0.01  # relative point cloud density
    print("Converting the 2D histogram to binary (mask) based on a threshold")
    binary_image = (grid_full > threshold).astype(np.uint8) * 255
    plot_binary_image(binary_image)

    # Pre-process the binary image
    print("Pre-processing the binary image")
    binary_image = closing(binary_image, footprint_rectangle((5, 5))) # closes small holes in the binary mask
    plot_binary_image(binary_image)

    # Find contours in the binary image
    print("Finding contours in the binary image")
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Definition of the shift (1 pixel up and 1 pixel right) for correct position (grid bug fix)
    shift_x = 1  # positive for right
    shift_y = 1  # negative for up

    # Adjust the contour coordinates
    adjusted_contours = []
    for cnt in contours:
        transformation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        adjusted_cnt = cv2.transform(cnt, transformation_matrix)
        adjusted_contours.append(adjusted_cnt)
    plot_contours(adjusted_contours)

    # Extract all segments from contours
    print("Extracting all segments from contours with Douglas-Peuckert algorithm")
    all_segments = []
    for contour in adjusted_contours:
        all_segments.extend(get_line_segments(contour, pixel_size,
                                              segment_approximation_tolerance=0.04))

    # Convert pixel-based segment coordinates to real-world coordinates
    print("Converting pixel-based segment coordinates to real-world coordinates")
    segments_in_world_coords = [[[x[0] * pixel_size + x_min, x[1] * pixel_size + y_min] for x in segment] for segment in
                                all_segments]

    # Filter out segments shorter than the given threshold
    print("Filtering out segments shorter than minimum wall length")
    filtered_segments = [
        segment for segment in segments_in_world_coords
        if distance_between_points(segment[0], segment[1]) >= minimum_wall_length]
    plot_segments_with_random_colors(filtered_segments, name="filtered_wall_segments")

    # Merge the co-linear segments using the updated function
    print("Merging the co-linear segments")
    final_wall_segments = merge_collinear_segments(filtered_segments.copy(), minimum_wall_thickness,
                                                   maximum_wall_thickness)
    plot_segments_with_random_colors(final_wall_segments, name="final_wall_segments")

    # Group parallel segments
    print("Grouping parallel segments")
    parallel_groups, wall_labels, facade_wall_candidates = (
        group_segments(final_wall_segments, maximum_wall_thickness, 'interior'))

    # Create facade wall surfaces into the list of exterior wall candidates
    if not exterior_scan:
        swollen_polygon_segments = swell_polygon(slab_polygon.get_xy(), exterior_walls_thickness)
        facade_wall_candidates.extend(swollen_polygon_segments)
        print("Grouping parallel exterior segments")
        parallel_facade_groups, wall_labels_facade, _ = (
            group_segments(facade_wall_candidates, maximum_wall_thickness, 'exterior'))
        parallel_groups.extend(parallel_facade_groups)
        wall_labels.extend(wall_labels_facade)

    plot_parallel_wall_groups(parallel_groups)
    # plot_segments_with_candidates(facade_wall_candidates)

    wall_axes, wall_thicknesses = [], []
    for group in parallel_groups:
        wall_axis, wall_thickness = calculate_wall_axis(group)
        wall_axes.append(wall_axis)
        wall_thicknesses.append(wall_thickness)

    wall_axes = adjust_intersections(wall_axes, maximum_wall_thickness)
    # plot_parallel_groups(parallel_groups, wall_axes, binary_image, points_2d, x_min, x_max, y_min, y_max, storey)

    start_points, end_points = zip(*wall_axes)
    wall_materials = ['Concrete'] * len(parallel_groups)

    # Calculate direction vectors for each wall axis
    wall_directions = [(axis[1][0] - axis[0][0], axis[1][1] - axis[0][1]) for axis in wall_axes]

    # Assign points to walls
    wall_groups, wall_thicknesses = assign_points_to_walls(x_coords, y_coords, z_coords, wall_axes, parallel_groups,
                                                           z_floor, z_ceiling)

    # Rotate each group of points to the x-z plane
    rotated_wall_groups, rotated_wall_axes = [], []
    wall_counter = 0
    for group, direction in zip(wall_groups, wall_directions):
        rotated_wall = rotate_points_to_xz_plane(group, direction)
        wall_axis = [(wall_axes[wall_counter][0][0], wall_axes[wall_counter][0][1], z_floor + z_ceiling / 2),
                     (wall_axes[wall_counter][1][0], wall_axes[wall_counter][1][1], z_floor + z_ceiling / 2)]
        rotated_wall_axis = rotate_points_to_xz_plane(wall_axis, direction)
        rotated_wall_groups.append(rotated_wall)
        rotated_wall_axes.append(rotated_wall_axis)
        wall_counter += 1

    filtered_rotated_wall_groups = []
    idx = 0
    for wall_group, wall_thickness in zip(rotated_wall_groups, wall_thicknesses):
        threshold = 0.5 * wall_thickness + 0.2 * wall_thickness
        filtered_wall_points = []
        for point in wall_group:
            if abs(point[1] - rotated_wall_axes[idx][0][1]) <= threshold:
                filtered_wall_points.append(point)
        idx += 1
        filtered_rotated_wall_groups.append(filtered_wall_points)

    # Plot the filtered walls
    translated_filtered_rotated_wall_groups = []
    for idx, wall_group in enumerate(filtered_rotated_wall_groups):
        # Translate the wall to start at the origin
        rotated_wall_axis = rotated_wall_axes[idx]
        min_x = min(rotated_wall_axis[0][0], rotated_wall_axis[1][0])
        min_y, min_z = 10e10, 10e10
        for point in wall_group:
            min_y = min(min_y, point[1])
            min_z = min(min_z, point[2])

        translated_wall = [(x - min_x, y - min_y, z - min_z) for x, y, z in wall_group]
        translated_filtered_rotated_wall_groups.append(translated_wall)
        # plot_wall(translated_wall, wall_thicknesses[idx], idx+1)

    return (start_points, end_points, wall_thicknesses, wall_materials, translated_filtered_rotated_wall_groups,
            wall_labels)


def identify_floor_and_ceiling(points, point_cloud_resolution, min_distance=2, plot_histograms_for_floors=False):
    """Identify the z-coordinates of the floor and ceiling surfaces in the wall point cloud."""

    # Extract z-coordinates
    z_coords = [point[2] for point in points]

    # Generate bin edges based on the specified resolution
    z_min, z_max = min(z_coords), max(z_coords)
    bin_edges = np.arange(z_min, z_max, point_cloud_resolution)

    # Create a histogram of z-coordinates
    hist, _ = np.histogram(z_coords, bins=bin_edges)

    # Set the height threshold to a percentage of the maximum histogram value
    height_threshold = 0.5 * max(hist)

    # Find peaks in the histogram
    peaks, properties = find_peaks(hist, distance=min_distance, height=height_threshold,
                                   prominence=0.25 * height_threshold)

    # Check if we have at least 2 peaks
    if len(peaks) < 2:
        print("Warning: Unable to identify both floor and ceiling surfaces.")
        z_floor = z_min
        z_ceiling = z_max
        return z_floor, z_ceiling

    # The lowest peak corresponds to the floor and the highest peak corresponds to the ceiling
    z_floor = bin_edges[peaks[0]] + point_cloud_resolution
    z_ceiling = bin_edges[peaks[-1]] - point_cloud_resolution

    # Plotting
    if plot_histograms_for_floors:
        plt.figure(figsize=(10, 6))
        plt.bar(bin_edges[:-1], hist, width=point_cloud_resolution, align='edge')
        plt.plot(bin_edges[:-1], hist, color='black', lw=1.5)
        plt.scatter(bin_edges[peaks], hist[peaks], color='red', s=100, zorder=3, label='Detected peaks')
        plt.axvline(x=z_floor, color='cyan', linestyle='--', label=f'z_floor: {z_floor:.3f}')
        plt.axvline(x=z_ceiling, color='yellow', linestyle='--', label=f'z_ceiling: {z_ceiling:.3f}')
        plt.xlabel('z-coordinate (m)')
        plt.ylabel('Frequency')
        plt.legend()
        # plt.title('z-coordinate histogram with floor and ceiling peaks')
        plt.savefig('images/wall_outputs_images/identified_floor_and_ceiling_surfaces.jpg', dpi=300)
        plt.savefig('images/wall_outputs_images/identified_floor_and_ceiling_surfaces.pdf')
        plt.show()

    return z_floor, z_ceiling


def identify_wall_faces(wall_number, points, wall_label, point_cloud_resolution, min_distance=25,
                        plot_histograms_for_walls=False):
    """Identify the y-coordinates of the wall surfaces in the wall point cloud."""

    # Extract y-coordinates
    y_coords = [point[1] for point in points]

    # Generate bin edges based on the specified resolution
    y_min, y_max = min(y_coords), max(y_coords)
    bin_edges = np.arange(y_min, y_max + point_cloud_resolution, point_cloud_resolution)

    # Create a histogram of y-coordinates
    hist, bin_edges = np.histogram(y_coords, bins=bin_edges)

    # Set the height threshold to a percentage of the maximum histogram value
    height_threshold = 0.3 * max(hist)

    # Optional plotting of histogram
    if plot_histograms_for_walls:
        plot_histogram_with_threshold(hist, height_threshold)

    # Find peaks in the histogram
    peaks, properties = find_peaks(hist, distance=min_distance, height=height_threshold,
                                   prominence=0.25 * height_threshold)

    # Check if we have at least 2 peaks
    if wall_label == 'interior':
        if len(peaks) >= 2:
            y1 = (bin_edges[peaks[0]] + bin_edges[peaks[0] + 1]) / 2
            y2 = (bin_edges[peaks[1]] + bin_edges[peaks[1] + 1]) / 2
        else:
            # If fewer than two peaks, take the highest points from the first and second halves of the histogram
            half = len(hist) // 2
            first_half_max_index = np.argmax(hist[:half])
            second_half_max_index = np.argmax(hist[half:]) + half
            y1 = (bin_edges[first_half_max_index] + bin_edges[first_half_max_index + 1]) / 2
            y2 = (bin_edges[second_half_max_index] + bin_edges[second_half_max_index + 1]) / 2
            print("No two distinct peaks found. Using highest points from histogram halves.")
    else:
        # Find the highest peak
        if len(peaks) > 0:
            peak_index = np.argmax(hist[peaks])
            peak = peaks[peak_index]
            y1 = (bin_edges[peak] + bin_edges[peak + 1]) / 2
            y2 = y1
        else:
            print(f"No peaks found for wall {wall_number}.")
            return None, None

    # Plotting
    if plot_histograms_for_walls:
        plt.figure(figsize=(10, 6))
        plt.bar(bin_edges[:-1], hist, width=point_cloud_resolution, align='edge')
        plt.plot(bin_edges[:-1], hist, color='black', lw=1.5)
        plt.scatter(bin_edges[peaks], hist[peaks], color='red', s=100, zorder=3, label='Detected peaks')
        plt.axvline(x=y1, color='cyan', linestyle='--', label=f'wall face 1: {y1:.3f}')
        plt.axvline(x=y2, color='yellow', linestyle='--', label=f'wall face 2: {y2:.3f}')
        plt.xlabel('y-coordinate (m)')
        plt.ylabel('Frequency')
        plt.legend()
        # plt.title('y-coordinate histogram with wall faces peaks')
        plt.savefig('images/wall_outputs_images/identified_wall_faces_%d.jpg' % wall_number, dpi=300)
        plt.savefig('images/wall_outputs_images/identified_wall_faces_%d.pdf' % wall_number)
        plt.show()

    return y1, y2


def distance_points_to_line(points, line_start, line_end):
    """Calculation of perpendicular distance from points to a line defined by two points."""
    line_vec = np.array(line_end) - np.array(line_start)
    line_vec = line_vec / np.linalg.norm(line_vec)  # Normalize the line vector
    points = np.array(points) - np.array(line_start)  # Translate points based on line_start
    projected_point = np.dot(points, line_vec)[:, None] * line_vec[None, :]
    perpendicular_vec = points - projected_point
    distances = np.sqrt(np.sum(perpendicular_vec ** 2, axis=1))
    return distances


def compute_wall_thickness(segment_group):
    """Compute the thickness of the wall based on the perpendicular distance between two segments."""
    segments = np.array(segment_group)
    lengths = np.linalg.norm(segments[:, 0, :] - segments[:, 1, :], axis=1)
    longest_indices = np.argsort(lengths)[-2:]
    segment1, segment2 = segments[longest_indices]
    distances = distance_points_to_line([segment1[0], segment1[1]], segment2[0], segment2[1])
    return np.mean(distances)


def assign_points_to_walls(x_coords, y_coords, z_coords, wall_axes, parallel_groups, z_floor, z_ceiling):
    # Stack coordinates into a single array
    points = np.vstack([x_coords, y_coords, z_coords]).T

    # Compute wall thicknesses using a vectorized approach if possible
    wall_thicknesses = np.array([compute_wall_thickness(group) for group in parallel_groups])
    acceptable_distances = 0.5 * wall_thicknesses + 0.2 * wall_thicknesses

    # Filter points by z-coordinates first to reduce computations
    valid_z_mask = (z_floor < z_coords) & (z_coords < z_ceiling)
    valid_points = points[valid_z_mask]

    # Precompute line start and end arrays for vectorized distance calculations
    line_starts = np.array([axis[0] for axis in wall_axes])
    line_ends = np.array([axis[1] for axis in wall_axes])

    all_distances = []

    batch_size = 1_000_000
    n_points = valid_points.shape[0]

    for start, end in zip(line_starts, line_ends):
        distances_for_wall = []
        for i in range(0, n_points, batch_size):
            batch = valid_points[i:i + batch_size, :2]
            distances = distance_points_to_line_np(batch, start, end)
            distances_for_wall.append(distances)
        all_distances.append(np.concatenate(distances_for_wall))

    all_distances = np.stack(all_distances)

    # Determine the closest wall for each point
    min_distances = np.min(all_distances, axis=0)
    min_distance_indices = np.argmin(all_distances, axis=0)

    # Group points based on their closest wall
    wall_groups = [[] for _ in range(len(wall_axes))]
    for idx, point in enumerate(valid_points):
        min_dist_idx = min_distance_indices[idx]
        if min_distances[idx] <= acceptable_distances[min_dist_idx]:
            wall_groups[min_dist_idx].append(point.tolist())

    return wall_groups, wall_thicknesses


def rotate_points_to_xz_plane(points, direction_vector):
    """Rotate a group of points so that the direction vector aligns with the x-axis."""
    # Calculate the angle between the direction vector and the x-axis
    angle = math.atan2(direction_vector[1], direction_vector[0])

    rotated_points = []
    for x, y, z in points:
        # Apply 2D rotation matrix on the x-y plane
        new_x = x * math.cos(angle) + y * math.sin(angle)
        new_y = -x * math.sin(angle) + y * math.cos(angle)
        rotated_points.append((new_x, new_y, z))

    return rotated_points


def export_wall_points_to_txt(wall_groups, output_dir="walls_outputs_txt"):
    """
    Export the xyz coordinates of points for each wall to individual .txt files.

    Parameters:
    - wall_groups: List of lists. Each inner list contains the xyz coordinates of a wall.
    - output_dir: String. Directory where the files will be saved.
    """

    # Check if output directory exists. If not, create it.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each wall group and save its points to a .txt file
    for idx, wall in enumerate(wall_groups, start=1):
        file_path = os.path.join(output_dir, f"wall_{idx}.txt")
        with open(file_path, 'w') as file:
            for point in wall:
                file.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"Exported wall points to {output_dir} directory.")


def identify_openings(wall_number, wall_points, wall_label, resolution, grid_roughness,
                      histogram_threshold=0.7, thickness_for_extraction=0.07,
                      min_opening_width=0.3, min_opening_height=0.3, max_opening_aspect_ratio=4,
                      door_z_max=0.1, door_min_height=1.8, opening_min_z_top=1.6, plot_histograms_for_openings=False):
    """Detect rectangular openings (windows and doors) in the wall."""

    valid_opening_widths, valid_opening_heights, valid_opening_types = [], [], []
    try:
        # Project points within the region of interest onto the x-z plane
        y1, y2 = identify_wall_faces(wall_number, wall_points, wall_label, resolution)
        inner_threshold = y1 - thickness_for_extraction / 2
        outer_threshold = y2 + thickness_for_extraction / 2

        projected_points = [(x, z) for x, y, z in wall_points if
                            (inner_threshold <= y <= (y1 + thickness_for_extraction / 2) or
                             (y2 - thickness_for_extraction / 2) <= y <= outer_threshold)]

        # Project all points onto the x-coordinate
        x_coords, z_coords = zip(*projected_points)

        # Create a histogram with bins of size equal to point_cloud_resolution
        x_min, x_max = min(x_coords), max(x_coords)
        bins = int((x_max - x_min) / (resolution * grid_roughness))
        hist, edges = np.histogram(x_coords, bins=bins, range=(x_min, x_max))

        z_min, z_max = min(z_coords), max(z_coords)
        z_bins = int((z_max - z_min) / (resolution * grid_roughness))

        # Define a threshold to decide if a bin contains an opening or not
        if len(hist) > 10:
            max10 = sorted(hist, reverse=True)[10]  # find the tenth maximum from the histogram
        else:
            max10 = hist[-1]

        x_threshold = max10 * histogram_threshold

        # Identify start and end of openings
        openings = []
        in_opening = False
        start, end = None, None
        for i, count in enumerate(hist):
            if count < x_threshold and not in_opening:
                in_opening = True
                start = edges[i]
            elif count >= x_threshold and in_opening:
                in_opening = False
                end = edges[i]
                if abs(end - start) > min_opening_width and (start >= x_min and end <= x_max):
                    openings.append((start, end))

        # For each valid opening, determine more precise height using z-histogram
        for x_start, x_end in openings:
            middle_x = (x_start + x_end) / 2
            tolerance = min_opening_width
            points_at_middle = [z for x, z in projected_points if (middle_x - tolerance) <= x <= (middle_x + tolerance)]

            z_hist, z_edges = np.histogram(points_at_middle, bins=z_bins, range=(z_min, z_max))
            try:
                max2 = sorted(z_hist, reverse=True)[2]
            except:
                pass
            z_threshold = max2 * 0.2

            candidates = []
            in_opening = False
            refined_z_min, refined_z_max = None, None
            for i, count in enumerate(z_hist):
                if count < z_threshold and not in_opening:
                    in_opening = True
                    refined_z_min = z_edges[i]
                elif count >= z_threshold and in_opening:
                    in_opening = False
                    refined_z_max = z_edges[i + 1]
                    candidates.append((refined_z_min, refined_z_max))
                    refined_z_min, refined_z_max = None, None  # Reset for next potential candidate

            if candidates:
                refined_z_min, refined_z_max = max(candidates, key=lambda pair: pair[1] - pair[0])

                width = x_end - x_start
                height = refined_z_max - refined_z_min

                if (height > min_opening_height and (height / width) < max_opening_aspect_ratio
                        and opening_min_z_top < refined_z_max):

                    if min([refined_z_min, refined_z_max]) > door_z_max:
                        valid_opening_widths.append((x_start, x_end))
                        valid_opening_heights.append((refined_z_min, refined_z_max))
                        valid_opening_types.append('window')
                    elif height > door_min_height:
                        valid_opening_widths.append((x_start, x_end))
                        valid_opening_heights.append((refined_z_min, refined_z_max))
                        valid_opening_heights[-1] = (0.0, refined_z_max)
                        valid_opening_types.append('door')
                    else:
                        pass

            if plot_histograms_for_openings:
                # Plotting
                plt.rc('text', usetex=True)
                plt.rc('font', family='serif', size=11)
                fig = plt.figure(figsize=(8 / 1.2, 5 / 1.2))
                bin_width_x = (x_max - x_min) / bins
                bin_width_z = (z_max - z_min) / z_bins

                # Plot the projected points and the openings
                axs0 = fig.add_subplot(221)
                xs_diluted, zs_diluted = zip(*projected_points[0::50])
                xs, zs = zip(*projected_points)
                axs0.scatter(xs_diluted, zs_diluted, s=2, c='g')
                for (x_start, x_end), (z1, z2), op_type in zip(valid_opening_widths, valid_opening_heights,
                                                               valid_opening_types):
                    z_start = min([z1, z2])
                    z_end = max([z1, z2])
                    if op_type == 'door':
                        axs0.add_patch(
                            plt.Rectangle((x_start, z_start), x_end - x_start, z_end - z_start, edgecolor='r',
                                          facecolor='red', alpha=0.2, linewidth=2, label='door'))
                    else:
                        axs0.add_patch(
                            plt.Rectangle((x_start, z_start), x_end - x_start, z_end - z_start, edgecolor='blue',
                                          facecolor='blue', alpha=0.2, linewidth=2, label='window'))
                axs0.set_xlabel(r'$x$ (m)')
                axs0.set_ylabel(r'$z$ (m)')

                # Plot x-histogram
                axs1 = fig.add_subplot(223)
                axs1.bar(edges[:-1], hist, width=bin_width_x)
                axs1.axhline(y=x_threshold, color='r', linestyle='dashed', label='x-threshold')
                axs1.legend(loc='upper right')
                axs1.set_xlabel(r'$x$ (m)')
                axs1.set_ylabel(r'Frequency')

                # Plot z-histogram for the opening refinement
                axs2 = fig.add_subplot(222)
                axs2.barh(z_edges[:-1], z_hist, height=bin_width_z)
                axs2.axvline(x=z_threshold, color='g', linestyle='dashed', label='z-threshold')
                axs2.legend()
                axs2.set_xlabel(r'Frequency')
                axs2.set_ylabel(r'$z$ (m)')
                plt.tight_layout()
                plt.savefig('images/wall_outputs_images/wall_%d_openings.jpg' % wall_number, dpi=300)
                plt.savefig('images/pdf/wall_%d_opening.pdf' % wall_number)
                plt.show()
            else:
                pass

        return valid_opening_widths, valid_opening_heights, valid_opening_types
    except (TypeError, ValueError):
        print('Problem with wall boundaries identification, no openings detected.')
        return valid_opening_widths, valid_opening_heights, valid_opening_types