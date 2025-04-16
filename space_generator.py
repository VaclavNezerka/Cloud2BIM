import math
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union
import copy
import re
import os


def plot_wall_center_lines(walls_dictionary, title):
    # Set text and font style to match
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=11)

    # Create a 2D plot for the center lines with adjusted figure size
    fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2))

    # Extract start and end points for plotting
    start_points = [wall['start_point'] for wall in walls_dictionary]
    end_points = [wall['end_point'] for wall in walls_dictionary]

    # Plot each wall's center line
    for start, end in zip(start_points, end_points):
        x_values = [start[0], end[0]]
        y_values = [start[1], end[1]]
        ax.plot(x_values, y_values, marker='o')  # Specify color for consistency

    # Set plot labels and title with LaTeX formatting
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    # ax.set_title(r'{}'.format(title), fontsize=11)  # Ensure the title font size matches

    # Set equal scaling for x and y axes
    ax.set_aspect('equal', adjustable='box')

    # Enable grid and apply tight layout
    ax.grid(True)
    fig.tight_layout()

    # Sanitize title for a safe filename
    sanitized_title = re.sub(r'[^a-zA-Z0-9]', '_', title)

    # Ensure the directory exists
    save_path = f'images/pdf/space/{sanitized_title}.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the plot
    plt.savefig(save_path)
    plt.show()


# Connection of elements disconnected very closely
def find_disconnected_walls(walls_local):
    # Step 1: Count occurrences of each start and end point
    point_count = {}
    for wall in walls_local:
        for point in (tuple(wall['start_point']), tuple(wall['end_point'])):
            point_count[point] = point_count.get(point, 0) + 1

    # Step 2: Identify walls where either start or end point has only one connection
    not_connected_walls = [
        wall for wall in walls_local
        if point_count[tuple(wall['start_point'])] == 1 or point_count[tuple(wall['end_point'])] == 1
    ]

    return not_connected_walls


# Function to check if a point lies on the centerline of a wall
def is_point_on_centerline(point, wall):
    (x1, y1), (x2, y2) = wall['start_point'], wall['end_point']
    px, py = point

    # Check if the point is on the line using the cross product
    if abs((py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)) > 1e-6:
        return False

    # Calculate dot product to check if the point lies within the segment bounds
    dotproduct = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
    if dotproduct < 0:
        return False

    # Check if the point is beyond the wall's segment
    squared_length = (x2 - x1) ** 2 + (y2 - y1) ** 2
    return dotproduct <= squared_length


# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Function to divide a wall at a point
def divide_wall(wall, point, min_wall_length):
    # Ensure division only occurs if the point is different from both start and end
    if wall['start_point'] == point or wall['end_point'] == point:
        return None, None

    # Copy the original wall and create two new walls divided at the point
    new_wall_1 = wall.copy()
    new_wall_1['end_point'] = point

    new_wall_2 = wall.copy()
    new_wall_2['start_point'] = point

    # Avoid creating walls with the same start and end point
    if new_wall_1['start_point'] == new_wall_1['end_point'] or new_wall_2['start_point'] == new_wall_2['end_point']:
        return None, None

    # Calculate lengths of the new walls
    length_new_wall_1 = calculate_distance(new_wall_1['start_point'], new_wall_1['end_point'])
    length_new_wall_2 = calculate_distance(new_wall_2['start_point'], new_wall_2['end_point'])

    # Avoid creating short walls in a single condition
    if length_new_wall_1 < min_wall_length or length_new_wall_2 < min_wall_length:
        return (None if length_new_wall_1 < min_wall_length else new_wall_1,
                None if length_new_wall_2 < min_wall_length else new_wall_2)

    return new_wall_1, new_wall_2


# Function to process the walls list of dictionaries
def process_disconnected_walls(walls_for_processing, not_connected_walls, min_wall_length):
    i = 0
    while i < len(walls_for_processing):
        wall = walls_for_processing[i]
        divided = False

        # Prioritize disconnected wall points first
        for not_connected_wall in not_connected_walls:
            if wall == not_connected_wall:
                continue

            # Flag to track if we need to restart the loop after handling both points
            need_restart = False

            # Check if the start point of this disconnected wall lies on the centerline of another wall
            if is_point_on_centerline(not_connected_wall['start_point'], wall):
                new_wall_1, new_wall_2 = divide_wall(wall, not_connected_wall['start_point'], min_wall_length)
                if new_wall_1 and new_wall_2:
                    walls_for_processing.pop(i)  # Remove the current wall
                    walls_for_processing.insert(i, new_wall_2)  # Insert new divided walls
                    walls_for_processing.insert(i, new_wall_1)
                    divided = True
                    need_restart = True

            # Check if the end point of this disconnected wall lies on the centerline of another wall
            if is_point_on_centerline(not_connected_wall['end_point'], wall):
                # Adjust index to account for new inserted walls
                if divided:
                    i += 1  # Move to the next wall segment
                    wall = walls_for_processing[i]  # Update the wall reference
                new_wall_1, new_wall_2 = divide_wall(wall, not_connected_wall['end_point'], min_wall_length)
                if new_wall_1 and new_wall_2:
                    walls_for_processing.pop(i)  # Remove the updated current wall
                    walls_for_processing.insert(i, new_wall_2)  # Insert new divided walls
                    walls_for_processing.insert(i, new_wall_1)
                    divided = True
                    need_restart = True

            # Restart if any division was made at either start or end points
            if need_restart:
                i = 0
                break

        # If no division was made, process intersections between walls
        if not divided:
            for j, other_wall in enumerate(walls_for_processing):
                if i != j:
                    if is_point_on_centerline(wall['start_point'], other_wall):
                        new_wall_1, new_wall_2 = divide_wall(other_wall, wall['start_point'], min_wall_length)
                        if new_wall_1 and new_wall_2:
                            walls_for_processing.pop(j)  # Remove the other wall
                            walls_for_processing.insert(j, new_wall_2)  # Insert new divided walls
                            walls_for_processing.insert(j, new_wall_1)
                            i = 0  # Restart the loop
                            divided = True
                        break
                    elif is_point_on_centerline(wall['end_point'], other_wall):
                        new_wall_1, new_wall_2 = divide_wall(other_wall, wall['end_point'], min_wall_length)
                        if new_wall_1 and new_wall_2:
                            walls_for_processing.pop(j)  # Remove the other wall
                            walls_for_processing.insert(j, new_wall_2)  # Insert new divided walls
                            walls_for_processing.insert(j, new_wall_1)
                            i = 0  # Restart the loop
                            divided = True
                        break

        # Move to the next wall only if no division occurred
        if not divided:
            i += 1

    return walls_for_processing


def extend_to_centerline(not_connected_walls, walls_local, distance_threshold_extension):
    # Function to extend the wall until the point lies on a centerline of another wall
    for i, not_connected_wall in enumerate(not_connected_walls):
        # Check start and end points
        for point_type in ['start_point', 'end_point']:
            point = not_connected_wall[point_type]

            # Check if the point is already on any centerline
            point_on_centerline = False
            for wall in walls_local:
                # Skip if the wall is the same as the not_connected_wall
                if wall == not_connected_wall:
                    continue

                # Check if the point is on the centerline of this wall
                if is_point_on_centerline(point, wall):
                    point_on_centerline = True
                    break

            # If the point is not on any centerline, extend the wall
            if not point_on_centerline:
                # Extend the wall using the previously defined function
                updated_wall = extend_point_on_centerline(point, not_connected_wall, walls_local,
                                                          distance_threshold_extension)

                # If the wall was updated, save it back and remove the original
                if updated_wall != not_connected_wall:
                    not_connected_walls[i] = updated_wall

    return not_connected_walls


def find_intersection(moving_point, extended_point, wall):
    # Coordinates of the first line segment (moving_point to extended_point)
    x1, y1 = moving_point
    x2, y2 = extended_point

    # Coordinates of the second line segment (wall's start to end points)
    x3, y3 = wall['start_point']
    x4, y4 = wall['end_point']

    # Calculate the determinants
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:  # Lines are parallel if the denominator is close to zero
        return None

    # Calculate the intersection point
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

    # Check if the intersection is within the bounds of both segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_point = (
            x1 + t * (x2 - x1),
            y1 + t * (y2 - y1)
        )
        return intersection_point

    # No valid intersection found within the bounds of the segments
    return None


def extend_point_on_centerline(point, not_connected_wall, walls_local, distance_threshold_extension):
    # Determine if 'point' matches the start or end point of 'not_connected_wall'
    if point == not_connected_wall['start_point']:
        moving_point_coords = not_connected_wall['start_point']
        fixed_point_coords = not_connected_wall['end_point']
    elif point == not_connected_wall['end_point']:
        moving_point_coords = not_connected_wall['end_point']
        fixed_point_coords = not_connected_wall['start_point']
    else:
        # If the point does not match, return the original wall unmodified
        return not_connected_wall

    # Calculate the direction vector for extension
    direction = (
        moving_point_coords[0] - fixed_point_coords[0],
        moving_point_coords[1] - fixed_point_coords[1]
    )

    # Normalize the direction vector
    length = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
    direction = (direction[0] / length, direction[1] / length)

    # Extend the point by the maximum distance allowed (distance_threshold_extension)
    extended_point = (
        moving_point_coords[0] + distance_threshold_extension * direction[0],
        moving_point_coords[1] + distance_threshold_extension * direction[1]
    )

    shorten_point = (moving_point_coords[0] - direction[0], moving_point_coords[1] - direction[1])

    # Check for intersection with any wall in the list
    wall_partime = {}
    for wall in walls_local:
        extended_points = extend_segment(wall, distance_threshold_extension)
        wall_partime['start_point'] = extended_points[0]
        wall_partime['end_point'] = extended_points[1]

        if wall == not_connected_wall:
            continue

        # Calculate the intersection point using actual coordinates
        intersection_point = find_intersection(moving_point_coords, shorten_point, wall_partime)

        if not intersection_point:
            intersection_point = find_intersection(moving_point_coords, extended_point, wall_partime)

        if intersection_point:
            # Calculate the distance between the original point and the intersection
            distance = ((moving_point_coords[0] - intersection_point[0]) ** 2 +
                        (moving_point_coords[1] - intersection_point[1]) ** 2) ** 0.5

            # If within the threshold, update the wall and return it
            if distance <= distance_threshold_extension:
                if point == not_connected_wall['start_point']:
                    not_connected_wall['start_point'] = intersection_point
                else:
                    not_connected_wall['end_point'] = intersection_point
                return not_connected_wall

    # If no valid intersection was found, return the wall unmodified
    return not_connected_wall


# Main function for wall axes processing
def process_centerlines(walls_local, distance_threshold_extension, min_wall_length, plot=True):
    """Splits walls (center lines) at intersections and returns the updated list of walls."""
    not_connected_walls = find_disconnected_walls(walls_local)
    if plot:
        plot_wall_center_lines(not_connected_walls, 'Walls without start-end point in intersection')
    extended_walls = extend_to_centerline(not_connected_walls, walls_local, distance_threshold_extension)
    if plot:
        plot_wall_center_lines(extended_walls, 'Walls extended to centerline')
    processed_walls = process_disconnected_walls(copy.deepcopy(walls_local), extended_walls, min_wall_length)
    if plot:
        plot_wall_center_lines(processed_walls, 'Divided in intersection')

    return processed_walls


def calculate_parallel_segments(start, end, thickness):
    """Calculate parallel segments offset by half the wall thickness on both sides."""
    d = thickness / 2  # Distance to offset

    # Vector along the wall
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    # Normalize the direction vector
    length = math.sqrt(dx ** 2 + dy ** 2)

    # Check for zero-length walls
    if length == 0:
        return None, None

    # Normalized direction vector
    unit_dx = dx / length
    unit_dy = dy / length

    # Perpendicular vector (rotated 90 degrees)
    perp_dx = -unit_dy
    perp_dy = unit_dx

    # Offset the start and end points by the perpendicular vector scaled by d in both directions
    offset_start_1 = (start[0] + perp_dx * d, start[1] + perp_dy * d)
    offset_end_1 = (end[0] + perp_dx * d, end[1] + perp_dy * d)

    offset_start_2 = (start[0] - perp_dx * d, start[1] - perp_dy * d)
    offset_end_2 = (end[0] - perp_dx * d, end[1] - perp_dy * d)

    return (offset_start_1, offset_end_1), (offset_start_2, offset_end_2)


def generate_space_boundaries(walls_local, snapping_distance):
    """Generate parallel segments for each wall and store them in space_segments."""
    space_segments = []

    for wall in walls_local:
        start = wall['start_point']
        end = wall['end_point']
        thickness = wall['thickness']

        # Calculate the two parallel segments
        parallel_segment_1, parallel_segment_2 = calculate_parallel_segments(start, end, thickness)

        # If the segments are valid (not None), add them to space_segments
        if parallel_segment_1:
            space_segments.append({
                'start_point': parallel_segment_1[0],
                'end_point': parallel_segment_1[1],
                'thickness': thickness,
                'material': wall['material'],
                'z_placement': wall['z_placement'],
                'height': wall['height'],
                'storey': wall['storey']
            })

        if parallel_segment_2:
            space_segments.append({
                'start_point': parallel_segment_2[0],
                'end_point': parallel_segment_2[1],
                'thickness': thickness,
                'material': wall['material'],
                'z_placement': wall['z_placement'],
                'height': wall['height'],
                'storey': wall['storey']
            })

    zone_segments_extended = []
    for segment in space_segments:
        segment_extended = extend_segment(segment, snapping_distance / 2)
        zone_segments_extended.append({
            'start_point': segment_extended[0],
            'end_point': segment_extended[1],
            'thickness': segment['thickness'],
            'material': segment['material'],
            'z_placement': segment['z_placement'],
            'height': segment['height'],
            'storey': segment['storey']
        })

    return space_segments, zone_segments_extended


def plot_zone_segments(zone_segments, label):
    """Plots the zone segments with a consistent style."""

    # Set text and font style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=11)

    # Create a figure with the adjusted size
    fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2))

    # Plot each segment
    for segment in zone_segments:
        start_point = segment['start_point']
        end_point = segment['end_point']
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]])

    # Set labels and title with LaTeX formatting
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    # ax.set_title(r'{}'.format(label), fontsize=11)

    # Set equal aspect ratio and add grid
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # Tight layout for better spacing
    fig.tight_layout()

    # Sanitize label for use in the filename
    sanitized_label = re.sub(r'[^a-zA-Z0-9]', '_', label)

    # Ensure the directory exists
    save_path = f'images/pdf/space/{sanitized_label}.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save and display the plot
    plt.savefig(save_path)
    plt.show()


def extract_space_dimensions(walls_local, plot_zones):
    # Step 1: Get wall axes
    lines_extended = []
    for wall in walls_local:
        point_1, point_2 = wall['start_point'], wall['end_point']
        extended_points = (point_1, point_2)
        extended_line = LineString(extended_points)
        lines_extended.append(extended_line)

    # Step 2: Create closed polygons from wall axes
    merged_lines = unary_union(lines_extended)  # Merge lines into a MultiLineString
    polygons = list(polygonize(merged_lines))  # Detect closed polygons

    # Step 3: Filter the smallest polygon(s) and create space dimension dictionary
    space_dimensions_dict = {}
    polygon_id = ord('A')  # Start with label 'A'
    for poly in polygons:
        if poly.is_valid:
            space_dimensions_dict[chr(polygon_id)] = list(poly.exterior.coords)
            polygon_id += 1

    # Step 4: Print out the result
    for key, value in space_dimensions_dict.items():
        print(f"Space {key}: {value}")

    # Step 5: Plot each polygon
    if plot_zones:
        # Set text and font style
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=11)

        # Create a figure with adjusted size
        fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2))

        for key, poly_coords in space_dimensions_dict.items():
            # Extract x and y coordinates
            x, y = zip(*poly_coords)

            # Plot polygon with fill color and label
            ax.fill(x, y, alpha=0.5, label=f'Space {key}')

            # Annotate the polygon with its label at the centroid
            centroid = Polygon(poly_coords).centroid
            ax.text(centroid.x, centroid.y, key, ha='center', va='center', fontsize=12, fontweight='bold')

        # Configure plot
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$y$ (m)')
        ax.set_aspect('equal', adjustable='box')
        # ax.set_title(r'Detected Polygons', fontsize=11)
        ax.grid(True)

        # Apply tight layout for spacing
        fig.tight_layout()

        # Ensure the directory exists for saving the plot
        save_path = 'images/pdf/space/detected_polygons_plot.pdf'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save and display the plot
        plt.savefig(save_path)
        plt.show()

    return space_dimensions_dict


def is_point_in_polygon(point, polygon):
    shapely_point = Point(point)
    shapely_polygon = Polygon(polygon)
    # Use the 'intersects' method to match the custom function's behavior
    return shapely_polygon.exterior.contains(shapely_point) or shapely_polygon.contains(shapely_point)


def get_segment_inside_space(segment, space, relative_min_portion_in_polygon):
    """Calculate the part of the segment that lies inside the space (polygon)."""

    def line_segment_intersection(p1_local, p2_local, q1_local, q2_local):
        """Calculate the intersection point of two line segments."""
        # Unpack the points
        x1, y1 = p1_local
        x2, y2 = p2_local
        x3, y3 = q1_local
        x4, y4 = q2_local

        # Calculate the denominators
        denom: object = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if abs(denom) < 1e-12:
            return None  # Lines are parallel or coincident

        ua_num = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
        ua = ua_num / denom

        if 0 <= ua <= 1:
            # Intersection point is within the first segment
            ub_num = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            ub = ub_num / denom
            if 0 <= ub <= 1:
                # Intersection point is within both segments
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                return x, y
        return None  # No valid intersection

    def distance(a, b):
        """Calculate the Euclidean distance between two points."""
        return math.hypot(b[0] - a[0], b[1] - a[1])

    start, end = segment
    segment_length = distance(start, end)

    # Collect all intersection points between the segment and the polygon edges
    intersections = []

    for i in range(len(space)):
        space_start = space[i]
        space_end = space[(i + 1) % len(space)]  # Wrap around
        intersection = line_segment_intersection(start, end, space_start, space_end)
        if intersection:
            intersections.append(intersection)

    # Add the segment's endpoints if they are inside the polygon
    if is_point_in_polygon(start, space):
        intersections.append(start)
    if is_point_in_polygon(end, space):
        intersections.append(end)

    # Remove duplicates and sort the points along the segment
    intersections = list(set(intersections))
    intersections.sort(key=lambda point: distance(start, point))

    # If we have at least two points, determine the part inside the polygon
    if len(intersections) >= 2:
        total_inside_length = 0

        # List to store segments inside the polygon
        inside_segments = []

        for i in range(len(intersections) - 1):
            p1 = intersections[i]
            p2 = intersections[i + 1]
            mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            if is_point_in_polygon(mid_point, space):
                segment_length_i = distance(p1, p2)
                total_inside_length += segment_length_i
                inside_segments.append((p1, p2))

        # Check if both endpoints are inside the polygon
        if is_point_in_polygon(start, space) and is_point_in_polygon(end, space):
            return start, end  # Entire segment is inside

        # Check if at least 70% of the segment is inside the polygon
        if total_inside_length / segment_length >= relative_min_portion_in_polygon:
            # Return the first and last points inside the polygon
            return inside_segments[0][0], inside_segments[-1][1]

    return False  # The segment does not lie sufficiently inside the polygon


def extend_segment(segment_in_space, extension_length):
    # Unpack the two points
    point_1, point_2 = segment_in_space['start_point'], segment_in_space['end_point']

    # Calculate the direction vector
    dx = point_2[0] - point_1[0]
    dy = point_2[1] - point_1[1]

    # Calculate the length of the direction vector
    length = math.sqrt(dx ** 2 + dy ** 2)

    # Avoid division by zero if points are the same
    if length == 0:
        return segment_in_space  # No extension possible if points are the same

    # Normalize the direction vector
    dx /= length
    dy /= length

    # Extend the start and end points in both directions
    new_x1 = point_1[0] - dx * extension_length
    new_y1 = point_1[1] - dy * extension_length
    new_x2 = point_2[0] + dx * extension_length
    new_y2 = point_2[1] + dy * extension_length

    # Return the extended segment as a tuple
    return (new_x1, new_y1), (new_x2, new_y2)


def find_segments_in_space(space, segment_extended):
    """Check if at least 'relative_min_portion_in_polygon' of a given segment's length is inside the space."""

    # Extract start and end points of the segment
    segment_start_extended = segment_extended['start_point']
    segment_end_extended = segment_extended['end_point']

    # Check if the segment center point is inside the space

    axis_centerpoint = tuple(np.mean([segment_start_extended, segment_end_extended], axis=0))
    if is_point_in_polygon(axis_centerpoint, space):
        return segment_extended  # Return the segment if condition is met
    return None  # Return None if no significant part of the segment is inside the space


def plot_space_segments(new_space_dimensions):
    """Plots segments for each space in different colors with a consistent style."""

    # Set text and font style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=11)

    # Create a figure with adjusted size
    fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2))

    # Define colors for spaces
    colors = ['blue', 'green', 'orange', 'purple', 'red', 'cyan', 'yellow', 'black', 'brown', 'violet']

    # Iterate through each space and plot segments with a unique color
    for idx, (space_name, segments) in enumerate(new_space_dimensions.items()):
        color = colors[idx % len(colors)]  # Cycle through colors if needed
        for segment in segments:
            start = segment['start_point']
            end = segment['end_point']
            # Plot segments in the assigned color for each space
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color)

    # Set labels and title with LaTeX formatting
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    # ax.set_title(r"Segments for Each Space (Color by Space)", fontsize=11)

    # Set equal aspect ratio and add grid
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # Tight layout for better spacing
    fig.tight_layout()

    # Ensure the directory exists for saving the plot
    save_path = 'images/pdf/space/space_segments_plot.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save and display the plot
    plt.savefig(save_path)
    plt.show()


def adjust_segments(segments):
    def update_segment_endpoints(intersection_local, segment):
        # Find the start or end point of the segment that is closest to the intersection point
        start_point = Point(segment['start_point'])
        end_point = Point(segment['end_point'])

        # Determine which endpoint is closer to the intersection
        if intersection_local.distance(start_point) < intersection_local.distance(end_point):
            # Update the start point to the intersection point
            segment['start_point'] = (intersection_local.x, intersection_local.y)
        else:
            # Update the end point to the intersection point
            segment['end_point'] = (intersection_local.x, intersection_local.y)

        return segment

    # Iterate through rooms and their segments
    for room, room_segments in segments.items():
        i = 0
        # First pass: Combine connected or overlapping segments
        while i < len(room_segments):
            j = i + 1
            while j < len(room_segments):
                seg1 = room_segments[i]
                seg2 = room_segments[j]

                # Create LineString objects from the segments
                line1 = LineString([
                    tuple(round(coord, 5) for coord in seg1['start_point']),
                    tuple(round(coord, 5) for coord in seg1['end_point'])
                ])

                line2 = LineString([
                    tuple(round(coord, 5) for coord in seg2['start_point']),
                    tuple(round(coord, 5) for coord in seg2['end_point'])
                ])

                # Check if segments are overlapping or connected at the ends
                if line1.intersects(line2) and isinstance(line1.intersection(line2), LineString):
                    # Determine the longest combined segment without assuming start or end
                    points = [seg1['start_point'], seg1['end_point'], seg2['start_point'], seg2['end_point']]
                    # Find the two points with the maximum distance between them
                    combined_start, combined_end = max(
                        ((p1, p2) for p1 in points for p2 in points if p1 != p2),
                        key=lambda pair: LineString([pair[0], pair[1]]).length
                    )
                    combined_segment = {
                        'start_point': combined_start,
                        'end_point': combined_end,
                        'thickness': seg1['thickness'],
                        'material': seg1['material'],
                        'z_placement': seg1['z_placement'],
                        'height': seg1['height'],
                        'storey': seg1['storey']
                    }
                    # Replace seg1 with combined_segment and remove seg2
                    room_segments[i] = combined_segment
                    room_segments.pop(j)
                    # Decrement j to adjust for the removed segment
                    j -= 1
                j += 1
            i += 1

        # Second pass: Find intersections in the updated list of segments
        for i in range(len(room_segments)):
            for j in range(i + 1, len(room_segments)):
                seg1 = room_segments[i]
                seg2 = room_segments[j]

                # Create LineString objects from the segments
                line1 = LineString([seg1['start_point'], seg1['end_point']])
                line2 = LineString([seg2['start_point'], seg2['end_point']])

                # Find intersection if it exists
                if line1.intersects(line2):
                    intersection = line1.intersection(line2)

                    # Check if the intersection is a point (not overlapping lines)
                    if isinstance(intersection, Point):
                        # Update both segments with the intersection point
                        room_segments[i] = update_segment_endpoints(intersection, seg1)
                        room_segments[j] = update_segment_endpoints(intersection, seg2)

        # Update the room's segments in the dictionary
        segments[room] = room_segments

    return segments


def convert_to_dictionary(final_spaces):
    zone_dict = {}

    for space, walls_local in final_spaces.items():
        ordered_points = []
        remaining_walls = walls_local[:]

        if not remaining_walls:
            print(f"Warning: No walls_local found in space {space}. Skipping this space.")
            continue  # skip to next space if there are no walls_local

        first_wall = remaining_walls.pop(0)
        first_wall['start_point'] = tuple(map(float, first_wall['start_point']))
        first_wall['end_point'] = tuple(map(float, first_wall['end_point']))
        ordered_points.append(first_wall['start_point'])
        ordered_points.append(first_wall['end_point'])

        while remaining_walls:
            connection_found = False

            for i, wall in enumerate(remaining_walls):
                start = tuple(map(float, wall['start_point']))
                end = tuple(map(float, wall['end_point']))

                if ordered_points[-1] == start:
                    ordered_points.append(end)
                    remaining_walls.pop(i)
                    connection_found = True
                    break
                elif ordered_points[-1] == end:
                    ordered_points.append(start)
                    remaining_walls.pop(i)
                    connection_found = True
                    break

            if not connection_found:
                # Calculate the closest point to the last point in ordered_points
                last_point = ordered_points[-1]
                closest_point = None
                closest_distance = float('inf')
                closest_wall_index = -1

                for i, wall in enumerate(remaining_walls):
                    start = tuple(map(float, wall['start_point']))
                    end = tuple(map(float, wall['end_point']))
                    distance_to_start = np.linalg.norm(np.array(last_point) - np.array(start))
                    distance_to_end = np.linalg.norm(np.array(last_point) - np.array(end))

                    # Find the closest of start or end points
                    if distance_to_start < closest_distance:
                        closest_distance = distance_to_start
                        closest_point = start
                        closest_wall_index = i
                    if distance_to_end < closest_distance:
                        closest_distance = distance_to_end
                        closest_point = end
                        closest_wall_index = i

                # Append the closest point and remove the corresponding wall
                ordered_points.append(closest_point)
                remaining_walls.pop(closest_wall_index)
                print(f"No direct connection found. Adding closest point: {closest_point}")

        # Store the ordered points in the desired format
        zone_dict[space] = {
            'vertices': ordered_points,
            'height': walls_local[-1]['height'],
            'storey': walls_local[-1]['storey']
        }
    return zone_dict


def get_sample_walls():
    artificial_walls = [
        {'start_point': (0.1, 0), 'end_point': (10, 0), 'thickness': 0.2, 'material': 'Brick', 'z_placement': 0,
         'height': 3, 'storey': 1},
        {'start_point': (10, 0), 'end_point': (12, 3), 'thickness': 0.2, 'material': 'Brick', 'z_placement': 0,
         'height': 3, 'storey': 1},
        {'start_point': (12, 3), 'end_point': (0, 3), 'thickness': 0.2, 'material': 'Brick', 'z_placement': 0,
         'height': 3, 'storey': 1},
        {'start_point': (0, 3), 'end_point': (0, 0.1), 'thickness': 0.2, 'material': 'Brick', 'z_placement': 0,
         'height': 3, 'storey': 1},
        {'start_point': (0, 3), 'end_point': (3, 0), 'thickness': 0.2, 'material': 'Brick', 'z_placement': 0,
         'height': 3, 'storey': 1},
        {'start_point': (4, 0), 'end_point': (4, 2.6), 'thickness': 0.2, 'material': 'Brick', 'z_placement': 0,
         'height': 3, 'storey': 1},
        {'start_point': (7, 0.1), 'end_point': (7, 3), 'thickness': 0.2, 'material': 'Brick', 'z_placement': 0,
         'height': 3, 'storey': 1},
        {'start_point': (10, 0), 'end_point': (10, 3), 'thickness': 0.2, 'material': 'Brick', 'z_placement': 0,
         'height': 3, 'storey': 1},
    ]

    return artificial_walls


def identify_zones(walls_local, snapping_distance=0.5, plot_zones=True):
    # Split wall center lines if they are in intersection with other center lines
    updated_walls = process_centerlines(walls_local, snapping_distance, min_wall_length=0.02, plot=plot_zones)

    # Generate wall surfaces (zone_segments) and extend them with half of clipping distance
    zone_segments, zone_segments_extended = generate_space_boundaries(updated_walls, snapping_distance)

    # Plot visuals
    if plot_zones:
        plot_zone_segments(zone_segments, 'Wall surfaces')
        plot_zone_segments(zone_segments_extended, 'wall surfaces extended')
        plot_wall_center_lines(updated_walls, 'Extended and connected center lines')

    # Get space area from wall axes
    space_dimensions = extract_space_dimensions(updated_walls, plot_zones)

    # Get segments inside area
    new_space_dimensions = {}
    for space_name, space_coords in space_dimensions.items():
        segments_in_space = []

        for idx, segment in enumerate(zone_segments):
            result = find_segments_in_space(space_coords, zone_segments_extended[idx])

            if result:
                segments_in_space.append(result)

        new_space_dimensions[space_name] = segments_in_space

    if plot_zones:
        plot_space_segments(new_space_dimensions)

    # Adjust geometry and crop the lines to intersection
    final_spaces = adjust_segments(new_space_dimensions)
    pass
    if plot_zones:
        plot_space_segments(final_spaces)

    # Convert into desired output
    space_dimensions_dict = convert_to_dictionary(final_spaces)

    return space_dimensions_dict


if __name__ == '__main__':
    walls = get_sample_walls()
    zones = identify_zones(walls, snapping_distance=0.5, plot_zones=True)
