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


'''
def e57_data_to_xyz(e57_data_array, output_file_name):
    x, y, z, red, green, blue, intensity = [], [], [], [], [], [], []
    for e57_data in e57_data_array:
        x.extend(e57_data['cartesianX'])
        y.extend(e57_data['cartesianY'])
        z.extend(e57_data['cartesianZ'])
        red.extend(e57_data['colorRed'])
        green.extend(e57_data['colorGreen'])
        blue.extend(e57_data['colorBlue'])
        intensity.extend(e57_data['intensity'])

    df = pd.DataFrame({'X': x, 'Y': y, 'Z': z, 'R': red, 'G': green, 'B': blue, 'Intensity': intensity})
    df.to_csv(output_file_name, sep='\t', index=False)
'''


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
    # df = pd.read_csv(file_name, delim_whitespace=True, header=0)
    # print(df.dtypes)
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


def identify_slabs_from_point_cloud(points_xyz, points_rgb, z_step, plot_segmented_plane=False):
    # ransac_threshold = 0.2  # threshold for ransac identification of points corresponding to a plane (inliers)
    # n_iterations = 1000  # number of iterations when identifying inliers

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
    # eq, idx_inliers = ransac_plane(single_horiz_surface_selected, threshold=ransac_threshold, iterations=n_iterations)
    # inliers = points_xyz[idx_inliers]

    # merge lower and upper surface of each horiz_surface and create a hull
    slabs = []
    for i in range(len(horiz_surface_candidates)):
        if (i % 2) == 1:
            print('Creating hull for slab no. %d of %d.' % ((i + 1) / 2, len(horiz_surface_candidates) / 2))
            slab_bottom_z_coord = np.median(horiz_surface_planes[i - 1][:, 2])
            slab_top_z_coord = np.median(horiz_surface_planes[i][:, 2])
            slab_thickness = slab_top_z_coord - slab_bottom_z_coord
            slab_points = np.concatenate((horiz_surface_planes[i - 1], horiz_surface_planes[i]), axis=0)
            x_coords, y_coords, polygon = create_hull_alphashape(slab_points, concavity_level=0.0)  # 0.0 -> convex
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


def visualize_points_in_xy_plane(vertical_surfaces):
    plt.figure(figsize=(10, 6))  # You can adjust the figure size as needed

    for storey, points_list in vertical_surfaces.items():
        for points in points_list:
            points = np.array(points)
            x = points[:, 0]
            y = points[:, 1]
            plt.scatter(x, y, label=storey, s=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Visualization of Points in the x-y Plane')
    plt.legend()
    plt.grid(True)
    plt.show()


def merge_horizontal_pointclouds_in_storey(horiz_surface_planes):
    for i in range(1, len(horiz_surface_planes)-1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(np.concatenate((horiz_surface_planes[i][:, 0], horiz_surface_planes[i+1][:, 0])),
                   np.concatenate((horiz_surface_planes[i][:, 1], horiz_surface_planes[i+1][:, 1])), s=0.02)

        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        plt.savefig('images/joint_surface_%d.jpg' % (i + 1), dpi=200)
        plt.close(fig)


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
    return (random.random(), random.random(), random.random())


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


def identify_walls(pointcloud, pointcloud_resolution, minimum_wall_length, minimum_wall_thickness, maximum_wall_thickness, storey):
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
    return start_points, end_points, wall_thicknesses, wall_materials
