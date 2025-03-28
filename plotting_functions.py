import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Polygon


def set_plot_style():
    """
    Set the common plotting style for all plots.
    """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=11)


def plot_contours(contours):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8 / 1.2, 5 / 1.2))
    for contour in contours:
        contour = contour.squeeze(axis=1)  # Remove unnecessary axis
        plt.plot(contour[:, 0], contour[:, 1], linewidth=2)  # Plot each contour
    ax.set_xlabel(r'$x$ (px)')
    ax.set_ylabel(r'$y$ (px)')
    ax.set_aspect('equal')  # Ensure equal scaling on both axes
    fig.tight_layout()
    plt.savefig('images/pdf/wall_contours.pdf')
    plt.show()


# Corrected for Article
def plot_binary_image(binary_image):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8 / 1.2, 5 / 1.2))
    ax.imshow(binary_image, cmap='gray', origin='lower')
    ax.set_xlabel(r'$x$ (px)')
    ax.set_ylabel(r'$y$ (px)')
    fig.tight_layout()
    plt.savefig('images/pdf/wall_mask.pdf')
    plt.show()


def plot_histogram(grid_full, x_values_full, y_values_full):
    plt.imshow(grid_full, origin='lower',
               extent=[x_values_full[0], x_values_full[-1], y_values_full[0], y_values_full[-1]], cmap='viridis')
    plt.colorbar(label='Density')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()


def plot_segments(segments, color='blue', label=None):
    for segment in segments:
        x_values = [segment[0][0], segment[1][0]]
        y_values = [segment[0][1], segment[1][1]]
        plt.plot(x_values, y_values, color=color, label=label)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Plot of Segments')
    plt.grid(True)
    plt.show()


def plot_segments_with_random_colors(segments, name=None):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8 / 1.2, 5 / 1.2))

    num_segments = len(segments)
    random_colors = [np.random.rand(3, ) for _ in range(num_segments)]

    for segment, color in zip(segments, random_colors):
        x_values = [segment[0][0], segment[1][0]]
        y_values = [segment[0][1], segment[1][1]]
        ax.plot(x_values, y_values, color=color)

    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')

    fig.tight_layout()
    plt.savefig(f'images/pdf/{name}.pdf')
    plt.show()


def plot_2d_wall_groups(wall_groups, rotated_wall_groups, rotated_wall_axes, original_wall_axes):
    plt.figure()

    # Plotting original wall groups
    for i, wall_group in enumerate(wall_groups):
        wall_group_x, wall_group_y = zip(*[(x, y) for x, y, _ in wall_group])
        plt.plot(wall_group_x, wall_group_y, label=f'Wall Group {i + 1} (Original)')

    # Plotting rotated wall groups
    for i, rotated_wall_group in enumerate(rotated_wall_groups):
        rotated_wall_group_x, rotated_wall_group_y = zip(*[(x, y) for x, y, _ in rotated_wall_group])
        plt.plot(rotated_wall_group_x, rotated_wall_group_y, '--', label=f'Wall Group {i + 1} (Rotated)')

        # Plotting rotated wall axes
        for i, rotated_wall_axis in enumerate(rotated_wall_axes):
            # Projecting the 3D points onto the XY plane
            rotated_wall_axis_xy = [(x, y) for x, y, _ in rotated_wall_axis]

            # Extracting X and Y coordinates for plotting
            rotated_wall_axis_x, rotated_wall_axis_y = zip(*rotated_wall_axis_xy)

            # Plotting rotated wall axes in 2D
            plt.plot(rotated_wall_axis_x, rotated_wall_axis_y, 'b-', label=f'Rotated Axis {i + 1}')

        # Plotting original wall axes
        for j, original_wall_axis in enumerate(original_wall_axes):
            # If Z coordinate is available, use it, otherwise ignore it
            if len(original_wall_axis) == 3:
                original_wall_axis_xy = [(x, y) for x, y, _ in original_wall_axis]
            else:
                original_wall_axis_xy = original_wall_axis  # Just use X and Y

            # Extracting X and Y coordinates for plotting
            original_wall_axis_x, original_wall_axis_y = zip(*original_wall_axis_xy)

            # Plotting original wall axes in 2D
            plt.plot(original_wall_axis_x, original_wall_axis_y, 'r--', label=f'Original Axis {j + 1}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.title('2D Plot of Wall Groups')
    plt.show()


def plot_threshold_and_filtered_points(threshold, wall_group, filtered_wall_points):
    # Extract y-coordinates from wall_group
    y_coordinates = [point[1] for point in wall_group]

    # Plot the threshold
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')

    # Plot the wall points and the filtered wall points
    plt.scatter(y_coordinates, [0] * len(y_coordinates), color='b', label='Wall Points')
    plt.scatter([point[1] for point in filtered_wall_points], [0] * len(filtered_wall_points), color='g',
                label='Filtered Wall Points')

    # Add labels and legend
    plt.xlabel('Y-coordinate')
    plt.ylabel('Z-coordinate')
    plt.title('Threshold and Filtered Wall Points')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()


def plot_histogram_with_threshold(hist, height_threshold):
    # Vykreslení histogramu
    plt.figure(figsize=(10, 6))
    bin_edges = np.arange(len(hist) + 1)  # Hrany binů
    plt.bar(bin_edges[:-1], hist, width=1, align='edge', color='blue', alpha=0.7, label='Histogram')

    # Vykreslení thresholdu
    plt.axhline(y=height_threshold, color='red', linestyle='--', label='Threshold')

    # Nastavení popisků a legendy
    plt.xlabel('Bins')
    plt.ylabel('Counts')
    plt.title('Histogram s Thresholdem')
    plt.legend()

    # Zobrazení grafu
    plt.show()


def plot_smoothed_contour(original_polygon, smoothed_polygon):
    set_plot_style()

    fig, ax = plt.subplots(figsize=(8/1.2, 5/1.2))

    tick_interval = 4  # Interval for major ticks on the axes

    ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
    ax.xaxis.set_major_locator(MultipleLocator(tick_interval))

    # Original contour
    ax.add_patch(original_polygon)

    # Smoothed contour
    ax.add_patch(smoothed_polygon)

    # Plot points
    x_contour, y_contour = original_polygon.get_xy().T
    x_smoothed, y_smoothed = smoothed_polygon.get_xy().T
    ax.scatter(x_contour, y_contour, s=2, color='blue')
    ax.scatter(x_smoothed, y_smoothed, s=2, color='red')

    # Main plot settings
    # ax.legend(['Original Contour', 'Smoothed Contour'], loc='lower left')
    ax.set_aspect('equal', 'box')
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$y$ (m)')
    fig.tight_layout()

    # Define the zoom region (let's focus on a quarter of the smoothed polygon)
    zoom_x1, zoom_x2 = -13.70, -13.35
    zoom_y1, zoom_y2 = 3.8, 4.1

    # Create new Polygon objects for the inset
    original_polygon_inset = Polygon(original_polygon.get_xy(), closed=True, fill=None, edgecolor='blue')
    smoothed_polygon_inset = Polygon(smoothed_polygon.get_xy(), closed=True, fill=None, edgecolor='red')

    # Create inset of the zoomed region
    axins = inset_axes(ax, width="50%", height="50%", loc='center')  # Adjust the size and location of the inset
    axins.add_patch(original_polygon_inset)
    axins.add_patch(smoothed_polygon_inset)
    axins.scatter(x_contour, y_contour, s=2, color='blue')
    axins.scatter(x_smoothed, y_smoothed, s=2, color='red')

    # Set the limits for the zoomed region
    axins.set_xlim(zoom_x1, zoom_x2)
    axins.set_ylim(zoom_y1, zoom_y2)
    axins.set_aspect('equal', 'box')

    # Remove axis text annotations from the zoom plot
    # axins.set_xticklabels([r'$x$ (m)'])
    # axins.set_yticklabels([r'$y$ (m)'])

    # Indicate the zoomed region on the main plot
    ax.indicate_inset_zoom(axins, edgecolor="black")
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

    plt.savefig('images/pdf/slab_contour_smoothing.pdf')
    plt.show()


def plot_parallel_wall_groups(parallel_groups):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8 / 1.2, 5 / 1.2))
    num_segments = len(parallel_groups)
    random_colors = [np.random.rand(3, ) for _ in range(num_segments)]

    for idx, (group, color) in enumerate(zip(parallel_groups, random_colors)):
        for seg in group:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color=color, linewidth=2,
                    label=f'Group {idx + 1}' if seg is group[0] else "")

    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    fig.tight_layout()
    plt.savefig('images/pdf/parallel_wall_groups.pdf')

    plt.show()


def plot_segments_with_candidates(facade_wall_candidates):
    set_plot_style()
    """Plot facade wall candidates as distinct segments."""
    fig, ax = plt.subplots(figsize=(8/1.2, 6/1.2))

    # Define a distinct color for facade wall candidates
    color = 'grey'

    # Plot each segment in facade wall candidates
    for seg in facade_wall_candidates:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color=color, linewidth=2, label='Facade Wall Candidate')

    ax.set_title('Visualization of Facade Wall Candidates')
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$z$ (m)')
    ax.grid(False)
    plt.show()


def plot_point_cloud_data(points_xyz, n_points_array, z_array, max_n_points_array, z_step):
    set_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12/1.5, 6/1.5))  # 2 plots horizontally

    # Convert n_points_array to a numpy array for safe operations
    n_points_array = np.array(n_points_array)

    # First plot: Histogram of point counts vs z-coordinate
    ax1.plot(n_points_array / 1000, z_array, '-r', linewidth=0.8)
    ax1.plot([max_n_points_array / 1000, max_n_points_array / 1000], [min(z_array), max(z_array)], '--b', linewidth=1.0)
    ax1.set_ylabel(r'$z$ (m)')
    ax1.set_xlabel('Number of points ($\\times 10^3$)')

    # Determine horizontal surface candidates based on point counts
    h_surf_candidates = [(z, z + z_step) for z, n in zip(z_array, n_points_array) if n > max_n_points_array]

    # Second plot: Scatter plot for X-Z plane view of all point cloud data
    downsampled_indices = np.arange(0, len(points_xyz), 100)
    downsampled_points = points_xyz[downsampled_indices]

    colors = ['orange' if any(lower - z_step <= z <= upper + z_step for (lower, upper) in h_surf_candidates) else 'blue'
              for z in downsampled_points[:, 2]]

    ax2.scatter(downsampled_points[:, 1], downsampled_points[:, 2], c=colors, marker='.', alpha=1, s=0.2)
    ax2.set_xlabel(r'$x$ (m)')
    ax2.set_ylabel(r'$z$ (m)')
    ax2.grid(False)

    # Set the same y-axis limits for both subplots based on the min and max of z_array
    common_z_limit = [min(z_array), max(z_array)]
    ax1.set_ylim(common_z_limit)
    ax2.set_ylim(common_z_limit)

    plt.tight_layout()  # Adjust the layout to make sure there is no overlap
    plt.savefig('images/pdf/slab_histogram.pdf')
    plt.show()


def plot_horizontal_surfaces(horiz_surface_planes):
    set_plot_style()

    tick_interval = 5 # Interval for major ticks on the axes

    # Create a figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(12/1.5, 6/1.5))
    axs[0].xaxis.set_major_locator(MultipleLocator(tick_interval))
    axs[0].yaxis.set_major_locator(MultipleLocator(tick_interval))
    axs[1].xaxis.set_major_locator(MultipleLocator(tick_interval))
    axs[1].yaxis.set_major_locator(MultipleLocator(tick_interval))

    # Define colors for each surface
    colors = ['orange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

    # Plot each horizontal surface
    for i, surface in enumerate(horiz_surface_planes):
        # Random subsampling - taking every 20th point
        subsampled_surface = surface[::70]
        axs[i].scatter(subsampled_surface[:, 0], subsampled_surface[:, 1], s=0.1, color=colors[i])
        axs[i].set_xlabel(r'$x$ (m)')
        axs[i].set_ylabel(r'$y$ (m)')

    # Show the plot
    plt.tight_layout()
    plt.savefig('images/pdf/horiz_surf_difference.pdf')
    plt.show()


def plot_2d_histogram(mask, x_edges, y_edges):
    set_plot_style()
    fig = plt.figure(figsize=(8/1.2, 6/1.2))
    plt.imshow(mask, extent=[x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()], cmap='jet',
               origin='lower')
    plt.colorbar(label='Relative point cloud density', shrink=0.63)
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$y$ (m)')
    plt.grid(False)
    fig.tight_layout()
    fig.savefig('images/pdf/2d_histogram.pdf', bbox_inches='tight')
    fig.show()


def plot_shifted_mask(shifted_mask, x_edges, y_edges):
    set_plot_style()
    plt.figure(figsize=(8/1.2, 6/1.2))
    plt.imshow(shifted_mask, extent=[x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()], cmap='binary',
               origin='lower')
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$y$ (m)')
    plt.title('Binarized mask - shifted')
    plt.grid(False)
    plt.savefig('images/pdf/slab_mask_binarized.pdf')
    plt.show()


def plot_wall(wall_points, thickness, wall_number):
    """Visualize a wall using both 2D and 3D scatter plots as sub-figures within a single figure."""

    # Create a single figure with two subplots
    fig = plt.figure(figsize=(15, 7))

    # 2D Plot (subplot 1)
    ax2d = fig.add_subplot(121)  # 1 row, 2 columns, plot 1
    xs, ys, zs = zip(*wall_points)
    ax2d.scatter(xs, zs, c='b', marker='o', s=1)
    ax2d.set_aspect('equal', 'box')  # Equal aspect ratio
    ax2d.text(0.05, 0.95, f'Thickness: {thickness:.3f} m', transform=ax2d.transAxes,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2d.set_xlabel('x-coordinate (m)')
    ax2d.set_ylabel('z-coordiante (m)')

    # 3D Plot (subplot 2)
    ax3d = fig.add_subplot(122, projection='3d')  # 1 row, 2 columns, plot 2
    ax3d.scatter(xs, ys, zs, c='b', marker='o', s=1)
    ax3d.set_xlabel('x-coordinate (m)')
    ax3d.set_ylabel('y-coordinate (m)')
    ax3d.set_zlabel('z-coordinate (m)')

    plt.tight_layout()
    plt.savefig('images/wall_outputs_images/wall_%d_2D_3D.jpg' % wall_number, dpi=300)
    plt.savefig('images/wall_outputs_images/wall_%d_2D_3D.pdf' % wall_number)
    plt.show()