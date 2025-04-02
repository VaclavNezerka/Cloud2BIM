import argparse
from aux_functions import *
from generate_ifc import IFCmodel
from space_generator import *


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process point clouds and generate IFC models.")

    # Input files
    parser.add_argument("--e57_input", action="store_true", help="Use E57 files as input")
    parser.add_argument("--e57_files", nargs="+", default=["input_e57/multiple_floor.e57"],
                        help="List of E57 input files")
    parser.add_argument("--xyz_files", nargs="+", default=["input_xyz/new_data/Kladno_station_floor_no_exterior.xyz"],
                        help="List of XYZ input files")

    # Processing options
    parser.add_argument("--dilute", action="store_true", help="Dilute the point cloud")
    parser.add_argument("--exterior_scan", action="store_true", help="Scan exterior walls")
    parser.add_argument("--dilution_factor", type=int, default=10, help="Dilution factor (skip every ith line)")
    parser.add_argument("--pc_resolution", type=float, default=0.002, help="Minimum point distance after dilution (m)")
    parser.add_argument("--grid_coefficient", type=int, default=5, help="Computational grid size [px/mm]")

    # Slab parameters
    parser.add_argument("--bfs_thickness", type=float, default=0.3, help="Bottom floor slab thickness (m)")
    parser.add_argument("--tfs_thickness", type=float, default=0.4, help="Top floor slab thickness (m)")

    # Wall parameters
    parser.add_argument("--min_wall_length", type=float, default=0.08, help="Minimum wall length (m)")
    parser.add_argument("--min_wall_thickness", type=float, default=0.05, help="Minimum wall thickness (m)")
    parser.add_argument("--max_wall_thickness", type=float, default=0.75, help="Maximum wall thickness (m)")
    parser.add_argument("--exterior_walls_thickness", type=float, default=0.3, help="Exterior wall thickness (m)")

    # IFC Output parameters
    parser.add_argument("--output_ifc", type=str, default="output_IFC/output-2.ifc", help="Output IFC file path")
    parser.add_argument("--ifc_project_name", type=str, default="Sample project", help="IFC Project Name")
    parser.add_argument("--ifc_project_long_name", type=str, default="Deconstruction of non-load-bearing elements",
                        help="IFC Project Long Name")
    parser.add_argument("--ifc_project_version", type=str, default="version 1.0", help="IFC Project Version")

    # IFC Author Information
    parser.add_argument("--ifc_author_name", type=str, default="Slavek", help="IFC Author Name")
    parser.add_argument("--ifc_author_surname", type=str, default="Zbirovsky", help="IFC Author Surname")
    parser.add_argument("--ifc_author_organization", type=str, default="CTU in Prague", help="IFC Author Organization")

    # Building information
    parser.add_argument("--ifc_building_name", type=str, default="Hotel Opatov", help="IFC Building Name")
    parser.add_argument("--ifc_building_type", type=str, default="Hotel", help="IFC Building Type")
    parser.add_argument("--ifc_building_phase", type=str, default="Reconstruction", help="IFC Building Phase")

    # Site Information
    parser.add_argument("--ifc_site_latitude", nargs=3, type=int, default=(50, 5, 0),
                        help="IFC Site Latitude (degrees, minutes, seconds)")
    parser.add_argument("--ifc_site_longitude", nargs=3, type=int, default=(4, 22, 0),
                        help="IFC Site Longitude (degrees, minutes, seconds)")
    parser.add_argument("--ifc_site_elevation", type=float, default=356.0,
                        help="Elevation of the site above sea level (m)")

    # Material settings
    parser.add_argument("--material_for_objects", type=str, default="Concrete", help="Material for objects")

    parsed_args = parser.parse_args()
    return parsed_args


# Parse arguments
args = parse_arguments()

# Assign parsed arguments to variables
e57_input = args.e57_input
e57_file_names = args.e57_files
xyz_filenames = args.xyz_files
dilute_pointcloud = args.dilute
exterior_scan = args.exterior_scan
dilution_factor = args.dilution_factor
pc_resolution = args.pc_resolution
grid_coefficient = args.grid_coefficient

bfs_thickness = args.bfs_thickness
tfs_thickness = args.tfs_thickness

min_wall_length = args.min_wall_length
min_wall_thickness = args.min_wall_thickness
max_wall_thickness = args.max_wall_thickness
exterior_walls_thickness = args.exterior_walls_thickness

ifc_output_file = args.output_ifc
ifc_project_name = args.ifc_project_name
ifc_project_long_name = args.ifc_project_long_name
ifc_project_version = args.ifc_project_version

ifc_author_name = args.ifc_author_name
ifc_author_surname = args.ifc_author_surname
ifc_author_organization = args.ifc_author_organization

ifc_building_name = args.ifc_building_name
ifc_building_type = args.ifc_building_type
ifc_building_phase = args.ifc_building_phase

ifc_site_latitude = tuple(args.ifc_site_latitude)
ifc_site_longitude = tuple(args.ifc_site_longitude)
ifc_site_elevation = args.ifc_site_elevation

material_for_objects = args.material_for_objects
# colours for the openings
door_colour_rgb = (0.541, 0.525, 0.486)
window_colour_rgb = (0.761, 0.933, 1.0)
column_colour_rgb = (0.596,0.576,1.0)

# Initiate the logger
last_time = time.time()
log_filename = "log.txt"

# SECTION: Import Point Clouds

# read e57 files and create xyz
if e57_input:
    for (idx, e57_file_name) in enumerate(e57_file_names):
        last_time = log('Reading %s.' % e57_file_name, last_time, log_filename)
        imported_e57_data = read_e57(e57_file_name)
        e57_data_to_xyz(imported_e57_data, xyz_filenames[idx], chunk_size=1e10)
        last_time = log('File %s converted to ASCII format, saved as %s.' % (e57_file_name, xyz_filenames[idx]),
                        last_time, log_filename)

# read xyz file
points_xyz, points_rgb = np.empty((0, 3)), np.empty((0, 3))
for xyz_filename in xyz_filenames:
    last_time = log('Extracting data from %s...' % xyz_filename, last_time, log_filename)
    points_xyz_temp, points_rgb_temp = load_xyz_file(xyz_filename, plot_xyz=False, select_ith_lines=dilute_pointcloud,
                                                     ith_lines=dilution_factor)
    points_xyz = np.vstack((points_xyz, np.array(points_xyz_temp)))
    # points_rgb = np.vstack((points_rgb, np.array(points_rgb_temp)))
points_xyz = np.round(points_xyz, 3)  # round the xyz coordinates to 3 decimals
last_time = log('All point cloud data imported.', last_time, log_filename)

# SECTION: Segment Slabs and Split the Point Cloud to Storeys
print("-" * 50)
print("Slab segmentation")
print("-" * 50)
# scan the model along the z-coordinate and search for planes parallel to xy-plane
slabs, horizontal_surface_planes = identify_slabs(points_xyz, points_rgb, bfs_thickness,
                                                  tfs_thickness, z_step=0.15,
                                                  pc_resolution=pc_resolution,
                                                  plot_segmented_plane=False)  # plot with open 3D

# SECTION: Segment Walls and Classify Openings
print("-" * 50)
print("Wall segmentation")
print("-" * 50)

# merge_horizontal_pointclouds_in_storey(horizontal_surface_planes)
point_cloud_storeys = split_pointcloud_to_storeys(points_xyz, slabs)
# display_cross_section_plot(point_cloud_storeys, slabs)
walls, all_openings, zones = [], [], []
wall_id = 0
for i, storey_pointcloud in enumerate(point_cloud_storeys):

    if exterior_scan:
        z_placement = slabs[i]['slab_bottom_z_coord'] + slabs[i]['thickness']
        wall_height = slabs[i + 1]['slab_bottom_z_coord'] - z_placement
    else:
        if i == 0:
            z_placement = slabs[i]['slab_bottom_z_coord']
            if i == len(point_cloud_storeys) - 1:
                wall_height = slabs[i + 1]['slab_bottom_z_coord'] - z_placement + tfs_thickness
            else:
                wall_height = slabs[i + 1]['slab_bottom_z_coord'] - z_placement
        elif i == len(point_cloud_storeys) - 1:
            z_placement = slabs[i]['slab_bottom_z_coord'] + slabs[i]['thickness']
            wall_height = slabs[i + 1]['slab_bottom_z_coord'] - z_placement + tfs_thickness
        else:
            z_placement = slabs[i]['slab_bottom_z_coord'] + slabs[i]['thickness']
            wall_height = slabs[i + 1]['slab_bottom_z_coord'] - z_placement + slabs[i + 1]['thickness']

    top_z_placement = slabs[i + 1]['slab_bottom_z_coord']

    (start_points, end_points, wall_thicknesses, wall_materials,
     translated_filtered_rotated_wall_groups, wall_labels) = (
        identify_walls(storey_pointcloud, pc_resolution, min_wall_length, min_wall_thickness, max_wall_thickness,
                       z_placement, top_z_placement, grid_coefficient, slabs[i + 1]['polygon'], exterior_scan,
                       exterior_walls_thickness=0.3))

    print("-" * 50)
    print("Rectangular openings detection")
    print("-" * 50)
    for j in range(len(start_points)):
        wall_id += 1
        walls.append({'wall_id': wall_id, 'storey': i + 1, 'start_point': start_points[j], 'end_point': end_points[j],
                      'thickness': wall_thicknesses[j], 'material': wall_materials[j], 'z_placement': z_placement,
                      'height': wall_height, 'label': wall_labels[j]})

        (opening_widths, opening_heights,
         opening_types) = identify_openings(j + 1, translated_filtered_rotated_wall_groups[j],
                                            wall_labels[j], pc_resolution, grid_coefficient,
                                            min_opening_width=0.4, min_opening_height=0.6,
                                            max_opening_aspect_ratio=4, door_z_max=0.1,
                                            door_min_height=1.7, opening_min_z_top=1.6,
                                            plot_histograms_for_openings=False)

        # Temporary list to store openings for the current wall
        wall_openings = []

        # Iterate through the detected openings and store the information
        for (x_start, x_end), (z_min, z_max), opening_type in zip(opening_widths, opening_heights, opening_types):
            opening_info = {
                "opening_wall_id": wall_id,
                "opening_type": opening_type,
                "x_range_start": x_start,
                "x_range_end": x_end,
                "z_range_min": z_min,
                "z_range_max": z_max
            }
            # Append the current opening's information to the wall's openings list
            wall_openings.append(opening_info)

        # After processing all openings for the current wall, append them to the all_openings list
        all_openings.extend(wall_openings)

        # Print or further process the results
        print(f"Wall {j + 1}:")
        for (x_start, x_end), (z_min, z_max), opening_type in zip(opening_widths, opening_heights, opening_types):
            print(
                f"Opening ({opening_type:s}): X-Range: {x_start:.2f} to {x_end:.2f}, Z-Range: {z_min:.2f} to {z_max:.2f}")
        print("-" * 50)

    # SECTION: Split the Storeys to Zones (Spaces in the IFC)
    print('Segmenting the storey to zones (spaces)...')
    print("-" * 50)
    zones_in_storey = identify_zones(walls, snapping_distance=0.8, plot_zones=False)
    zones.append(zones_in_storey)

# SECTION: Generate IFC
print("-" * 50)
print("Generating IFC model")
print("-" * 50)
ifc_model = IFCmodel(ifc_project_name, ifc_output_file)
ifc_model.define_author_information(ifc_author_name + ' ' + ifc_author_surname, ifc_author_organization)
ifc_model.define_project_data(ifc_building_name, ifc_building_type, ifc_building_phase,
                              ifc_project_long_name, ifc_project_version, ifc_author_organization,
                              ifc_author_name, ifc_author_surname, ifc_site_latitude, ifc_site_longitude,
                              ifc_site_elevation)

# Add building storeys and zones
storeys_ifc, slabs_ifc = [], []
for idx, slab in enumerate(slabs):
    # define a storey
    slab_position = slab['slab_bottom_z_coord'] + slab['thickness']
    storeys_ifc.append(ifc_model.create_building_storey('Floor %.1f m' % slab_position, slab_position))

    # define a slab
    # Convert separate x and y coordinate lists into a list of coordinate pairs
    points = [[float(x), float(y)] for x, y in zip(slab['polygon_x_coords'], slab['polygon_y_coords'])]

    # Optionally remove duplicate points to avoid redundancy in the polygon
    # This example uses a simple method by converting each pair into a tuple and then back into a list.
    points_no_duplicates = list(dict.fromkeys(tuple(pt) for pt in points))
    points_no_duplicates = [list(pt) for pt in points_no_duplicates]

    # Use the refactored function to create the slab IFC entity
    # The create_slab function internally creates the slab placement, extrusion, and shape representation.
    slab_entity = ifc_model.create_slab(
        slab_name='Slab %d' % (idx + 1),
        points=points_no_duplicates,
        slab_z_position=round(slab['slab_bottom_z_coord'], 3),
        slab_height=round(slab['thickness'], 3),
        material_name=material_for_objects
    )

    # Optionally assign the slab to a building storey later on:
    ifc_model.assign_product_to_storey(slab_entity, storeys_ifc[-1])

    # order is important (without last (initial point)
    # IFCSpace initialization
    ifc_space_placement = ifc_model.space_placement(slab_position)
    if idx != len(slabs) - 1:  # avoid creating zones on the uppermost slab
        zone_number = 1
        for space_name, space_data in zones[idx].items():  # Iterate over each space dictionary inside the zone
            # Create the space using the data from the space dictionary
            ifc_space = ifc_model.create_space(space_data, ifc_space_placement, (idx + 1), zone_number, storeys_ifc[-1],
                                               space_data["height"])
            zone_number += 1

'''# Column definition for IFC
columns_example = [
    {
        "name": "Column A1",
        "storey": 1,
        "start_point": (0.0, 0.0),  # Only X, Y coordinates
        "direction": (0.2, 0.5),  # Direction only in X, Y plane
        "profile_points": [[-0.1, -0.1], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]],  # Square profile
        "height": 3.0
    }
]

column_material, column_material_def_rep= ifc_model.create_material_with_color("Column material",
                                                                               column_colour_rgb, transparency=0)
column_id=1
for column in columns_example:
    ifc_column = ifc_model.create_column(f"C{column_id:02d}", column['name'], storeys_ifc[column['storey'] - 1], column['start_point'],
                                         column['direction'], column['profile_points'], column['height'])
    ifc_model.assign_material(ifc_column, column_material)
    column_id +=1'''

# Wall definition for IFC
for wall in walls:
    start_point = tuple(float(num) for num in wall['start_point'])
    end_point = tuple(float(num) for num in wall['end_point'])
    if start_point == end_point:
        continue
    wall_thickness = wall['thickness']
    wall_material = wall['material']
    wall_z_placement = wall['z_placement']
    wall_heights = wall['height']
    wall_label = wall['label']

    wall_openings = [opening for opening in all_openings if opening['opening_wall_id'] == wall['wall_id']]

    # Create a material layer
    material_layer = ifc_model.create_material_layer(wall_thickness, wall_material)
    # Create an IfcMaterialLayerSet using the material layer (in a list)
    material_layer_set = ifc_model.create_material_layer_set([material_layer])
    # Create an IfcMaterialLayerSetUsage and associate it with the element or product
    material_layer_set_usage = ifc_model.create_material_layer_set_usage(material_layer_set, wall_thickness)
    # Local placement
    wall_placement = ifc_model.wall_placement(wall['z_placement'])
    wall_axis_placement = ifc_model.wall_axis_placement(start_point, end_point)
    wall_axis_representation = ifc_model.wall_axis_representation(wall_axis_placement)
    wall_swept_solid_representation = ifc_model.wall_swept_solid_representation(start_point, end_point, wall_heights,
                                                                                wall_thickness)
    product_definition_shape = ifc_model.product_definition_shape(wall_axis_representation,
                                                                  wall_swept_solid_representation)
    current_story = wall['storey']
    wall = ifc_model.create_wall(wall_placement, product_definition_shape)
    assign_material = ifc_model.assign_material(wall, material_layer_set_usage)
    wall_type = ifc_model.create_wall_type(wall, wall_thickness)
    assign_material_2 = ifc_model.assign_material(wall_type[0], material_layer_set)
    # assign_object = ifc_model.assign_product_to_storey(wall, storeys_ifc[0])
    assign_object = ifc_model.assign_product_to_storey(wall, storeys_ifc[current_story - 1])
    wall_ext_int_parameter = ifc_model.create_property_single_value("IsExternal",wall_label == 'exterior')
    ifc_model.create_property_set(wall, wall_ext_int_parameter)

    # Create materials
    window_material, window_material_def_rep = ifc_model.create_material_with_color(
        'Window material',
        window_colour_rgb,
        transparency=0.7
    )

    door_material, door_material_def_rep = ifc_model.create_material_with_color(
        'Door material',
        door_colour_rgb,
        transparency=0
    )

    # Initialize ID counters
    window_id = 1
    door_id = 1

    for opening in wall_openings:
        # Each 'opening' is a dictionary with the opening data
        opening_type = opening['opening_type']
        x_range_start = opening['x_range_start']
        x_range_end = opening['x_range_end']
        z_range_min = opening['z_range_min']
        z_range_max = opening['z_range_max']

        # Assign unique ID based on opening type
        if opening_type == "window":
            opening_id = f"W{window_id:02d}"  # Format as W01, W02, ...
            window_id += 1
        elif opening_type == "door":
            opening_id = f"D{door_id:02d}"  # Format as D01, D02, ...
            door_id += 1
        else:
            print(f"Warning: Unknown opening type: {opening_type}, skipping this opening")
            continue

        # Store the ID in the opening dictionary
        opening['wall_id'] = opening_id

        opening_width = x_range_end - x_range_start
        opening_height = z_range_max - z_range_min
        window_sill_height = z_range_min
        offset_from_start = x_range_start

        opening_closed_profile = ifc_model.opening_closed_profile_def(float(opening_width), wall_thickness)
        opening_placement = ifc_model.opening_placement(start_point, wall_placement)
        opening_extrusion = ifc_model.opening_extrusion(opening_closed_profile, float(opening_height), start_point,
                                                        end_point, float(window_sill_height), float(offset_from_start))
        opening_representation = ifc_model.opening_representation(opening_extrusion)
        opening_product_definition = ifc_model.product_definition_shape_opening(opening_representation)
        wall_opening = ifc_model.create_wall_opening(opening_placement[1], opening_product_definition)
        rel_voids_element = ifc_model.create_rel_voids_element(wall, wall_opening)
        if opening_type == "window":
            window_closed_profile = ifc_model.opening_closed_profile_def(float(opening_width), 0.01)
            window_extrusion = ifc_model.opening_extrusion(window_closed_profile, float(opening_height), start_point,
                                                           end_point, float(window_sill_height), float(offset_from_start))
            window_representation = ifc_model.opening_representation(window_extrusion)
            window_product_definition = ifc_model.product_definition_shape_opening(window_representation)
            window = ifc_model.create_window(opening_placement[1], window_product_definition, opening_id)
            window_type = ifc_model.create_window_type()
            rel_defined_by_type = ifc_model.rel_defined_by_type(window, window_type)
            rel_fills_element = ifc_model.create_rel_fills_element(wall_opening, window)
            ifc_model.assign_product_to_storey(window, storeys_ifc[current_story - 1])
            ifc_model.assign_material(window, window_material)
        elif opening_type == "door":
            door_closed_profile = ifc_model.opening_closed_profile_def(float(opening_width), 0.01)
            door_extrusion = ifc_model.opening_extrusion(door_closed_profile, float(opening_height), start_point,
                                                         end_point, float(window_sill_height), float(offset_from_start))
            door_representation = ifc_model.opening_representation(door_extrusion)
            door_product_definition = ifc_model.product_definition_shape_opening(door_representation)
            door = ifc_model.create_door(opening_placement[1], door_product_definition, opening_id)
            rel_fills_element = ifc_model.create_rel_fills_element(wall_opening, door)
            ifc_model.assign_product_to_storey(door, storeys_ifc[current_story - 1])
            ifc_model.assign_material(door, door_material)

# Write the IFC model to a file
ifc_model.write()
last_time = log('\nIFC model saved to %s.' % ifc_output_file, last_time, log_filename)