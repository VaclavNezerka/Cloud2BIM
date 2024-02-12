import pye57
import pandas as pd
import open3d as o3d
import alphashape
from matplotlib.patches import Polygon
from copy import copy
import time
from datetime import datetime
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing, square
import cv2
import random
import math
from scipy.signal import find_peaks


def log(message, last_time, filename):
    current_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    elapsed_time = current_time - last_time
    log_message = f"{timestamp} - {message} Elapsed time: {elapsed_time:.2f} s."
    with open(filename, 'a') as f:
        f.write(log_message)
    print(log_message)
    return current_time  # Return the current time so it can be used as the "last_time" for the next log


def read_e57(file_name):
    # read the documentation at https://github.com/davidcaron/pye57
    e57 = pye57.E57(file_name)
    data_raw = e57.read_scan_raw(0)
    return data_raw


def e57_data_to_xyz(e57_data_array, output_file_name, chunk_size=10000):
    for idx, e57_data in enumerate(e57_data_array):
        num_chunks = (len(e57_data['cartesianX']) - 1) // chunk_size + 1  # Compute the number of chunks
        print(f"\nProcessing e57 file no. {idx + 1}...")

        for i in tqdm(range(num_chunks)):  # tqdm will display a progress bar
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(e57_data['cartesianX']))

            x = np.array(e57_data['cartesianX'][start:end])
            y = np.array(e57_data['cartesianY'][start:end])
            z = np.array(e57_data['cartesianZ'][start:end])
            red = np.array(e57_data['colorRed'][start:end])
            green = np.array(e57_data['colorGreen'][start:end])
            blue = np.array(e57_data['colorBlue'][start:end])
            intensity = np.array(e57_data['intensity'][start:end])

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


def load_xyz_file(file_name, plot_xyz=False):
    pcd = np.loadtxt(file_name, skiprows=1)
    xyz = pcd[:, :3]
    rgb = pcd[:, 3:6]

    # show plot of xyz points from top view: (x, y) coordinates and with rgb-colored points
    if plot_xyz:
        plt.figure(figsize=(8, 5), dpi=150)
        plt.scatter(xyz[:, 0], xyz[:, 1], c=rgb / 255, s=0.05)
        plt.title("Top-View")
        plt.xlabel('X-axis (m)')
        plt.ylabel('Y-axis (m)')
        plt.show()
    return xyz, rgb


def create_hull_alphashape(points_3d, concavity_level=1.0):
    points_2d = [[x, y] for x, y, _ in points_3d]
    alpha_shape = alphashape.alphashape(points_2d, concavity_level)
    hull = alpha_shape.convex_hull
    xx, yy = hull.exterior.coords.xy
    x_coords = xx.tolist()
    y_coords = yy.tolist()
    polygon = Polygon(list(zip(x_coords, y_coords)), facecolor='red', alpha=0.2, edgecolor='r', linewidth=3)
    return x_coords, y_coords, polygon


def create_hull_from_histogram(points_3d, pointcloud_resolution, grid_coefficient=5, plot_contours=False):
    # Project 3D points to 2D
    points_2d = np.array([[x, y] for x, y, _ in points_3d])

    # Parameters for histogram
    pixel_size = pointcloud_resolution * grid_coefficient
    x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
    y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
    x_edges = np.arange(x_min, x_max + pixel_size, pixel_size)
    y_edges = np.arange(y_min, y_max + pixel_size, pixel_size)

    # Create 2D histogram and mask
    histogram, _, _ = np.histogram2d(points_2d[:, 0], points_2d[:, 1], bins=(x_edges, y_edges))
    mask = histogram.T > 2  # Threshold to create mask, transposed for correct orientation

    # Find contours on the transposed mask for correct orientation
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Plotting
    if plot_contours:
        fig, ax = plt.subplots()
        ax.imshow(mask, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='Greys', alpha=0.5)
        ax.scatter(points_2d[:, 0], points_2d[:, 1], s=1, color='blue')

    # Adjust contour scaling
    for contour in contours:
        contour = np.squeeze(contour, axis=1)  # Remove redundant dimension
        # Adjusting scaling to fully cover the bin extents
        contour_scaled = (contour + 0.5) * pixel_size + [x_min, y_min]  # Add 0.5 to shift to the center of the bin
        polygon = Polygon(contour_scaled, fill=None, edgecolor='red')
        if plot_contours:
            ax.add_patch(polygon)
    x_contour = contour_scaled[:, 0].flatten()
    y_contour = contour_scaled[:, 1].flatten()

    if plot_contours:
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        plt.show()

    return x_contour, y_contour, polygon


def identify_slabs_from_point_cloud(points_xyz, points_rgb, z_step, pointcloud_resolution, plot_segmented_plane=False):
    z_min, z_max = min(points_xyz[:, 2]), max(points_xyz[:, 2])
    n_steps = int((z_max - z_min) / z_step + 1)
    z_array, n_points_array = [], []
    for i in tqdm(range(n_steps), desc="Progress searching for horiz_surface candidate z-coordinates"):
        z = z_min + i * z_step
        idx_selected_xyz = np.where((z < points_xyz[:, 2]) & (points_xyz[:, 2] < (z + z_step)))[0]
        z_array.append(z)
        n_points_array.append(len(idx_selected_xyz))

    horiz_surface_candidates = []
    max_n_points_array = max(n_points_array)

    # extract z-coordinates where the density of points (indicated by a high value on the histogram) exceeds 50%
    # of a maximum -> horiz_surface candidates
    for i in range(len(n_points_array)):
        if n_points_array[i] > 0.5 * max_n_points_array:
            horiz_surface_candidates.append([z_array[i], z_array[i] + z_step])

    horiz_surface_planes, horiz_surface_colors, horiz_surface_polygon, horiz_surface_polygon_x, \
    horiz_surface_polygon_y, horiz_surface_z, horiz_surface_thickness = [], [], [], [], [], [], []

    # extract xyz points within an interval given by horiz_surface_candidates (lie within the range given by the
    # z-coordinates in horiz_surface candidates)
    for i in tqdm(range(len(horiz_surface_candidates)), desc="Extracting points for horizontal surfaces"):
        horiz_surface_idx = np.where(
            (horiz_surface_candidates[i][0] < points_xyz[:, 2]) &
            (points_xyz[:, 2] < horiz_surface_candidates[i][1]))[0]
        horiz_surface_planes.append(points_xyz[horiz_surface_idx])
        horiz_surface_colors.append(points_rgb[horiz_surface_idx] / 255)

    # merge lower and upper surface of each horiz_surface and create a hull
    slabs = []
    for i in range(len(horiz_surface_candidates)):
        if (i % 2) == 1:
            print('Creating hull for slab no. %d of %d.' % ((i + 1) / 2, len(horiz_surface_candidates) / 2))
            slab_bottom_z_coord = np.median(horiz_surface_planes[i - 1][:, 2])
            slab_top_z_coord = np.median(horiz_surface_planes[i][:, 2])
            slab_thickness = slab_top_z_coord - slab_bottom_z_coord
            slab_points = np.concatenate((horiz_surface_planes[i - 1], horiz_surface_planes[i]), axis=0)

            # create hull for the slab
            # x_coords, y_coords, polygon = create_hull_alphashape(slab_points, concavity_level=0.0)  # 0.0 -> convex
            x_coords, y_coords, polygon = create_hull_from_histogram(slab_points, pointcloud_resolution,
                                                                     grid_coefficient=5)

            slabs.append({'polygon': polygon, 'polygon_x_coords': x_coords, 'polygon_y_coords': y_coords,
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
            new_slab_polygon = copy(slabs[int((i - 1) / 2)]['polygon'])  # to avoid a RuntimeError
            ax.add_patch(new_slab_polygon)
        else:
            new_slab_polygon = copy(slabs[int((i - 1) / 2)]['polygon'])  # to avoid a RuntimeError
            ax.add_patch(new_slab_polygon)
        ax.set_aspect('equal', 'box')
        # plt.show()
        fig.tight_layout()
        plt.savefig('images/horiz_surface_%d.jpg' % (i + 1), dpi=200)
        plt.close(fig)

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
    for i in range(len(slabs) - 1):
        bottom_z_of_upper_slab = slabs[i + 1]['slab_bottom_z_coord'] + 0.1  # upper limit (wall + 10 cm of the ceiling slab)
        top_z_of_bottom_slab = slabs[i]['slab_bottom_z_coord'] + slabs[i]['thickness'] - 0.1  # bottom limit (- 10 cm of the floor)

        # Extract points that are between the bottom of the upper slab and the top of the lower slab
        segmented_pointcloud_idx = np.where((top_z_of_bottom_slab < points_xyz[:, 2]) &
                                            (points_xyz[:, 2] < bottom_z_of_upper_slab))[0]

        if len(segmented_pointcloud_idx) > 0:
            segmented_pointcloud_points_in_storey = points_xyz[segmented_pointcloud_idx]
            segmented_pointclouds_3d.append(segmented_pointcloud_points_in_storey)

    return segmented_pointclouds_3d


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
# Define a function to get line segments from a contour using Douglas-Peucker algorithm
def get_line_segments(contour, epsilon_factor=0.01):
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
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
    """Calculate the distance from a point to a line defined by two points."""
    numerator = abs((line_end[1] - line_start[1]) * point[0] -
                    (line_end[0] - line_start[0]) * point[1] + line_end[0] * line_start[1] -
                    line_end[1] * line_start[0])
    denominator = ((line_end[1] - line_start[1]) ** 2 + (line_end[0] - line_start[0]) ** 2) ** 0.5
    return numerator / denominator


def distance_between_points(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def merge_segments(seg1, seg2):
    """Merge two segments into one."""
    points = seg1 + seg2
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
    return [sorted_points[0], sorted_points[-1]]


def segments_within_tolerance_for_merge_corrected(seg1, seg2, max_thickness, max_distance):
    """Check if two segments are candidates for merging."""
    # Check if the segments are close enough to merge based on maximum wall thickness
    close_enough = any(
        distance_between_points(p1, p2) <= max_distance for p1 in seg1 for p2 in seg2
    )

    # Check if the segments are co-linear
    colinear = all(
        distance_point_to_line(point, seg1[0], seg1[1]) < max_thickness for point in seg2
    )

    return close_enough and colinear


def merge_colinear_segments_updated(segments, max_thickness, max_distance):
    """Merge co-linear segments from the given list using the direct approach we tested."""
    final_segments = []

    while segments:
        base_segment = segments[0]
        to_merge = [base_segment]

        for other_segment in segments[1:]:
            if segments_within_tolerance_for_merge_corrected(base_segment, other_segment, max_thickness, max_distance):
                to_merge.append(other_segment)

        # Merge all the segments in to_merge into a single segment
        all_points = [point for seg in to_merge for point in seg]
        sorted_points = sorted(all_points, key=lambda p: (p[0], p[1]))
        merged_segment = [sorted_points[0], sorted_points[-1]]
        final_segments.append(merged_segment)

        # Remove segments that have been merged
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


def segments_are_parallel(seg1, seg2, angle_tolerance=1):
    """Check if two segments are approximately parallel within a given angle tolerance (in degrees)."""
    angle = angle_between_segments(seg1, seg2)
    return abs(angle) < angle_tolerance or abs(angle - 180) < angle_tolerance


def perpendicular_distance_between_segments(seg1, seg2):
    """Calculate the shortest perpendicular distance between two parallel segments."""
    if segments_are_parallel(seg1, seg2):
        return distance_point_to_line(seg2[0], seg1[0], seg1[1])
    else:
        return float('inf')


def group_parallel_segments(segments, min_wall_thickness, max_wall_thickness, angle_tolerance=1):
    """Group segments that are parallel with a small tolerance."""
    grouped = []

    while segments:
        current_segment = segments.pop(0)
        parallel_group = [current_segment]

        i = 0
        while i < len(segments):
            segment = segments[i]
            if segments_are_parallel(current_segment, segment, angle_tolerance):
                min_distance = min(perpendicular_distance_between_segments(current_segment, segment),
                                   perpendicular_distance_between_segments(segment, current_segment))
                if min_wall_thickness < min_distance < max_wall_thickness:
                    parallel_group.append(segment)
                    segments.pop(i)
                else:
                    i += 1
            else:
                i += 1

        # Save only the groups consisting of two or more segments
        if len(parallel_group) >= 2:
            grouped.append(parallel_group)

    return grouped


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
    distance_sum_flipped = sum([distance_between_points(pt, axis_start_flipped) + distance_between_points(pt, axis_end_flipped)
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


def adjust_wall_axes_for_intersections(wall_axes, max_wall_thickness):
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
    plt.imshow(binary_image, cmap='gray', origin='lower', extent=[x_min, x_max, y_min, y_max], alpha=0.6)

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


def identify_walls(pointcloud, pointcloud_resolution, minimum_wall_length, minimum_wall_thickness,
                   maximum_wall_thickness, storey):
    x_coords, y_coords, z_coords = zip(*pointcloud)
    z_section_boundaries = [0.9, 1.0]  # percentage of the height for the storey sections
    grid_coefficient = 5  # computational grid size (multiplies the point_cloud_resolution)

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

    # Convert the 2D histogram to binary (mask) based on a threshold
    threshold = 0.01
    print("Converting the 2D histogram to binary (mask) based on a threshold")
    binary_image = (grid_full > threshold).astype(np.uint8) * 255

    # Pre-process the binary image
    print("Pre-processing the binary image")
    binary_image = closing(binary_image, square(5))  # closes small holes in the binary mask

    # Find contours in the binary image
    print("Finding contours in the binary image")
    # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Extract all segments from contours
    print("Extracting all segments from contours")
    all_segments = []
    for contour in contours:
        all_segments.extend(get_line_segments(contour))

    # Convert pixel-based segment coordinates to real-world coordinates
    print("Converting pixel-based segment coordinates to real-world coordinates")
    segments_in_world_coords = [[[x[0] * pixel_size + x_min, x[1] * pixel_size + y_min] for x in segment] for segment in
                                all_segments]

    # Filter out segments shorter than the given threshold
    print("Filtering out segments shorter than the given threshold")
    filtered_segments = [
        segment for segment in segments_in_world_coords
        if distance_between_points(segment[0], segment[1]) >= minimum_wall_length
    ]

    # Merge the co-linear segments using the updated function
    print("Merging the co-linear segments using the updated function")
    final_wall_segments = merge_colinear_segments_updated(filtered_segments.copy(), minimum_wall_thickness,
                                                          maximum_wall_thickness)
    # Group parallel segments
    print("Grouping parallel segments")
    parallel_groups = group_parallel_segments(final_wall_segments, minimum_wall_thickness, maximum_wall_thickness)
    wall_axes, wall_thicknesses = [], []
    for group in parallel_groups:
        wall_axis, wall_thickness = calculate_wall_axis(group)
        wall_axes.append(wall_axis)
        wall_thicknesses.append(wall_thickness)
    wall_axes = adjust_wall_axes_for_intersections(wall_axes, maximum_wall_thickness)
    plot_parallel_groups(parallel_groups, wall_axes, binary_image, points_2d, x_min, x_max, y_min, y_max, storey)

    start_points, end_points = zip(*wall_axes)
    wall_materials = ['Concrete'] * len(parallel_groups)

    # Calculate direction vectors for each wall axis
    wall_directions = [(axis[1][0] - axis[0][0], axis[1][1] - axis[0][1]) for axis in wall_axes]

    # Filter out the ceiling and floor points
    z_floor, z_ceiling = identify_floor_and_ceiling(list(zip(x_coords, y_coords, z_coords)), pointcloud_resolution)

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
    for wall_group, wall_thickness in zip(rotated_wall_groups, wall_thicknesses):
        threshold = 0.5 * wall_thickness + 0.1 * wall_thickness
        filtered_wall_points = [point for point in wall_group if point[1] <= threshold]
        filtered_rotated_wall_groups.append(filtered_wall_points)

    # Plot the filtered walls
    translated_filtered_rotated_wall_groups = []
    for idx, wall_group in enumerate(filtered_rotated_wall_groups):
        # Translate the wall to start at the origin
        rotated_wall_axis = rotated_wall_axes[idx]
        min_x = min(rotated_wall_axis[0][0], rotated_wall_axis[1][0])
        min_y = min([point[1] for point in wall_group])
        min_z = min([point[2] for point in wall_group])
        translated_wall = [(x - min_x, y - min_y, z - min_z) for x, y, z in wall_group]
        translated_filtered_rotated_wall_groups.append(translated_wall)
        # plot_wall(translated_wall, wall_thicknesses[idx], idx+1)

    return start_points, end_points, wall_thicknesses, wall_materials, grid_coefficient, translated_filtered_rotated_wall_groups


def identify_floor_and_ceiling(points, point_cloud_resolution, min_distance=10, plot_histograms_for_floors=False):
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
                                   prominence=0.25*height_threshold)

    # Check if we have at least 2 peaks
    if len(peaks) < 2:
        print("Warning: Unable to identify both floor and ceiling surfaces.")
        return None, None

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


def identify_wall_faces(wall_number, points, point_cloud_resolution, min_distance=3, plot_histograms_for_walls=False):
    """Identify the z-coordinates of the floor and ceiling surfaces in the wall point cloud."""

    # Extract z-coordinates
    y_coords = [point[1] for point in points]

    # Generate bin edges based on the specified resolution
    y_min, y_max = min(y_coords), max(y_coords)
    bin_edges = np.arange(y_min, y_max, point_cloud_resolution)

    # Create a histogram of z-coordinates
    hist, _ = np.histogram(y_coords, bins=bin_edges)

    # Set the height threshold to a percentage of the maximum histogram value
    height_threshold = 0.5 * max(hist)

    # Find peaks in the histogram
    peaks, properties = find_peaks(hist, distance=min_distance, height=height_threshold,
                                   prominence=0.25*height_threshold)

    # Check if we have at least 2 peaks
    if len(peaks) < 2:
        print("Warning: Unable to identify both floor and ceiling surfaces.")
        return None, None

    # The lowest peak corresponds to the floor and the highest peak corresponds to the ceiling
    y1 = bin_edges[peaks[0]] + point_cloud_resolution
    y2 = bin_edges[peaks[-1]] - point_cloud_resolution

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


def assign_points_to_walls(x_coords, y_coords, z_coords, wall_axes, parallel_groups, z_floor, z_ceiling):
    """Assign each point in the point cloud to a wall based on its proximity to the wall's axis and compute thickness."""

    def compute_wall_thickness(segment_group):
        """Compute the thickness of the wall based on the perpendicular distance between two segments."""

        # Sort the segments based on their lengths and select the two longest ones
        segment_group_sorted = sorted(segment_group, key=lambda seg: distance_between_points(seg[0], seg[1]),
                                      reverse=True)
        segment1 = segment_group_sorted[0]
        segment2 = segment_group_sorted[1]

        # Calculate perpendicular distances from the endpoints of the first segment to the second segment
        distance1 = distance_point_to_line(segment1[0], segment2[0], segment2[1])
        distance2 = distance_point_to_line(segment1[1], segment2[0], segment2[1])

        # Average the two distances to get the wall thickness
        thickness = (distance1 + distance2) / 2

        return thickness

    wall_groups = [[] for _ in wall_axes]
    wall_thicknesses = [compute_wall_thickness(seg_group) for seg_group in parallel_groups]

    for x, y, z in zip(x_coords, y_coords, z_coords):
        # Check if z-coordinate is within floor and ceiling bounds
        if not (z_floor <= z <= z_ceiling):
            continue

        min_distance = float('inf')
        assigned_wall_idx = None

        for idx, axis in enumerate(wall_axes):
            dist = distance_point_to_line((x, y), axis[0], axis[1])
            acceptable_distance = 0.5 * wall_thicknesses[idx] + 0.1 * wall_thicknesses[idx]
            if dist <= acceptable_distance and dist < min_distance:
                assigned_wall_idx = idx
                min_distance = dist

        if assigned_wall_idx is not None:
            wall_groups[assigned_wall_idx].append((x, y, z))

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


def plot_wall(wall_points, thickness, wall_number):
    """Visualize a wall using both 2D and 3D scatter plots as subfigures within a single figure."""

    # Create a single figure with two subplots
    fig = plt.figure(figsize=(15, 7))

    # 2D Plot (subplot 1)
    ax2D = fig.add_subplot(121)  # 1 row, 2 columns, plot 1
    xs, ys, zs = zip(*wall_points)
    ax2D.scatter(xs, zs, c='b', marker='o', s=1)
    ax2D.set_aspect('equal', 'box')  # Equal aspect ratio
    ax2D.text(0.05, 0.95, f'Thickness: {thickness:.3f} m', transform=ax2D.transAxes,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2D.set_xlabel('x-coordinate (m)')
    ax2D.set_ylabel('z-coordiante (m)')

    # 3D Plot (subplot 2)
    ax3D = fig.add_subplot(122, projection='3d')  # 1 row, 2 columns, plot 2
    ax3D.scatter(xs, ys, zs, c='b', marker='o', s=1)
    ax3D.set_xlabel('x-coordinate (m)')
    ax3D.set_ylabel('y-coordinate (m)')
    ax3D.set_zlabel('z-coordinate (m)')

    plt.tight_layout()
    plt.savefig('images/wall_outputs_images/wall_%d_2D_3D.jpg' % wall_number, dpi=300)
    plt.savefig('images/wall_outputs_images/wall_%d_2D_3D.pdf' % wall_number)
    plt.show()


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


def detect_rectangular_openings(wall_number, wall_points, resolution, grid_roughness,
                                histogram_threshold=0.7, thickness_for_extraction=0.03, min_opening_width=0.3,
                                min_opening_height=0.3, max_opening_aspect_ratio=4, door_z_min=0.1):
    """Detect rectangular openings (windows and doors) in the wall."""

    # Project points within the region of interest onto the x-z plane
    y1, y2 = identify_wall_faces(wall_number, wall_points, resolution)
    inner_threshold = y1 - thickness_for_extraction
    outer_threshold = y1 + thickness_for_extraction

    projected_points = [(x, z) for x, y, z in wall_points if inner_threshold <= y <= outer_threshold]

    # Project all points onto the x-coordinate
    x_coords, z_coords = zip(*projected_points)

    # Create a histogram with bins of size equal to point_cloud_resolution
    x_min, x_max = min(x_coords), max(x_coords)
    bins = int((x_max - x_min) / (resolution * grid_roughness))
    hist, edges = np.histogram(x_coords, bins=bins, range=(x_min, x_max))

    z_min, z_max = min(z_coords), max(z_coords)
    z_bins = int((z_max - z_min) / (resolution * grid_roughness))

    # Define a threshold to decide if a bin contains an opening or not
    max10 = sorted(hist, reverse=True)[10]
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
            if abs(end - start) > min_opening_width:
                openings.append((start, end))

    # For each valid opening, determine more precise height using z-histogram
    valid_opening_widths, valid_opening_heights, valid_opening_types = [], [], []
    for x_start, x_end in openings:
        middle_x = (x_start + x_end) / 2
        tolerance = min_opening_width * 0.45
        points_at_middle = [z for x, z in projected_points if (middle_x - tolerance) <= x <= (middle_x + tolerance)]

        z_hist, z_edges = np.histogram(points_at_middle, bins=z_bins, range=(z_min, z_max))
        max2 = sorted(z_hist, reverse=True)[2]
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

            if height > min_opening_height and (height / width) < max_opening_aspect_ratio:
                valid_opening_widths.append((x_start, x_end))
                valid_opening_heights.append((refined_z_min, refined_z_max))
                if min([refined_z_min, refined_z_max]) < door_z_min:
                    valid_opening_heights[-1] = (0.0, refined_z_max)
                    valid_opening_types.append('door')
                else:
                    valid_opening_types.append('window')

    # Plotting
    fig = plt.figure(figsize=(18, 10))
    bin_width_x = (x_max - x_min) / bins
    bin_width_z = (z_max - z_min) / z_bins

    # Plot the projected points and the openings
    axs0 = fig.add_subplot(221)
    xs, zs = zip(*projected_points)
    axs0.scatter(xs, zs, s=1, c='g')
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
    axs0.set_xlabel("x-coordinate (m)")
    axs0.set_ylabel("z-coordinate (m)")

    # Plot x-histogram
    axs1 = fig.add_subplot(223)
    axs1.bar(edges[:-1], hist, width=bin_width_x)
    axs1.axhline(y=x_threshold, color='r', linestyle='dashed', label='x-threshold')
    axs1.legend(loc='upper right')
    axs1.set_xlabel("x-coordinate (m)")
    axs0.set_ylabel("z-coordinate (m)")

    # Plot z-histogram for the opening refinement
    axs2 = fig.add_subplot(222)
    axs2.bar(z_edges[:-1], z_hist, width=bin_width_z)
    axs2.axhline(y=z_threshold, color='g', linestyle='dashed', label='z-threshold')
    axs2.legend()
    axs2.set_xlabel("z-coordinate (m)")
    axs2.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig('images/wall_outputs_images/wall_%d_openings.jpg' % wall_number, dpi=300)
    plt.savefig('images/wall_outputs_images/wall_%d_openings.pdf' % wall_number)
    plt.show()

    return valid_opening_widths, valid_opening_heights, valid_opening_types