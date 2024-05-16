from aux_functions import *
from generate_ifc import IFCmodel

# input point clouds
e57_input = False
# e57_file_names = ["input_e57/test_room.e57"]
# e57_file_names = ['input_e57/05th.e57', "input_e57/06th.e57", "input_e57/07th.e57"]
# e57_file_names = ["input_e57/06th.e57", "input_e57/07th.e57"]
e57_file_names = ["input_e57/multiple_floor.e57"]

# xyz_filenames = ["input_xyz/06th.xyz", "input_xyz/07th.xyz"]
# xyz_filenames = ["input_xyz/new_data/multiple_floor.xyz"]
# xyz_filenames = ["input_xyz/new_data/Test_room_initial_dataset_002.xyz"]
# xyz_filenames = ["input_xyz/new_data/Zurich_dataset_synth3_segmnet_merge_bug_01.xyz"]
xyz_filenames = ["input_xyz/new_data/Zurich_dataset_synth3_01.xyz"]
# xyz_filenames = ["input_xyz/new_data/Vienna_rummelhartgasse_corner_005.xyz"]
# xyz_filenames = ["input_xyz/new_data/Zurich_dataset_synth2_002.xyz"]
# xyz_filenames = ["input_xyz/new_data/Opatov_19th_half_01.xyz"]
# xyz_filenames = ["input_xyz/new_data/Alserstrase_2nd_floor_01.xyz"]
# xyz_filenames = ["input_xyz/new_data/Kladno_station_floor.xyz"]
# xyz_filenames = ["input_xyz/new_data/Kladno_station_floor_no_exterior.xyz"]
# xyz_filenames = ["input_xyz/new_data/Kladno_station_bug_segments.xyz"]
# xyz_filenames = ["input_xyz/new_data/Opatov_2_rooms.xyz"]
dilute_pointcloud = False
dilution_factor = 10

# input parameters for identification of elements
pc_resolution = 0.002
grid_coefficient = 5  # computational grid size (multiplies the point_cloud_resolution)

# used if there is no slab in the point cloud for the bottom and uppermost floor
bfs_thickness = 0.2  # bottom floor slab thickness
tfc_thickness = 0.05  #top floor ceiling thickness

min_wall_length = 0.1
min_wall_thickness = 0.05
max_wall_thickness = 0.65

# IFC model parameters
ifc_output_file = 'output_IFC/output-2.ifc'
ifc_project_name = 'Sample project'
ifc_project_long_name = 'Deconstruction of non-load-bearing elements'
ifc_project_version = 'version 1.0'

ifc_author_name = 'Slavek'
ifc_author_surname = 'Zbirovsky'
ifc_author_organization = 'CTU in Prague'

ifc_building_name = 'Hotel Opatov'
ifc_building_type = 'Hotel'
ifc_building_phase = 'Reconstruction'

ifc_site_latitude = (50, 5, 0)
ifc_site_longitude = (4, 22, 0)
ifc_site_elevation = 356.0  # elevation of the site above the sea level
material_for_slabs = 'Concrete'

# initiate the logger
last_time = time.time()
log_filename = "log.txt"

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
    points_rgb = np.vstack((points_rgb, np.array(points_rgb_temp)))
points_xyz = np.round(points_xyz, 3)  # round the xyz coordinates to 3 decimals
last_time = log('All point cloud data imported.', last_time, log_filename)

# scan the model along the z-coordinate and search for planes parallel to xy-plane
slabs, horizontal_surface_planes = identify_slabs(points_xyz, points_rgb, bfs_thickness,
                                                  tfc_thickness, z_step=0.05,
                                                  pc_resolution=pc_resolution,
                                                  plot_segmented_plane=False)  # plot with open 3D

# merge_horizontal_pointclouds_in_storey(horizontal_surface_planes)
point_cloud_storeys = split_pointcloud_to_storeys(points_xyz, slabs)
# display_cross_section_plot(point_cloud_storeys, slabs)
walls = []
all_openings = []
id = 0
for i, storey_pointcloud in enumerate(point_cloud_storeys):
    (start_points, end_points, wall_thicknesses, wall_materials,
     translated_filtered_rotated_wall_groups) = identify_walls(storey_pointcloud, pc_resolution, min_wall_length,
                                                               min_wall_thickness, max_wall_thickness, grid_coefficient)
    z_placement = slabs[i]['slab_bottom_z_coord'] + slabs[i]['thickness']
    wall_height = slabs[i + 1]['slab_bottom_z_coord'] - z_placement
    for j in range(len(start_points)):
        id += 1
        walls.append({'id': id, 'storey': i + 1, 'start_point': start_points[j], 'end_point': end_points[j],
                      'thickness': wall_thicknesses[j], 'material': wall_materials[j], 'z_placement': z_placement,
                      'height': wall_height})

        opening_widths, opening_heights, opening_types = identify_openings(j + 1, translated_filtered_rotated_wall_groups[j],
                                                                           pc_resolution, grid_coefficient,
                                                                           min_opening_width=0.4, min_opening_height=0.3,
                                                                           max_opening_aspect_ratio=4, door_z_max=0.1)

        # Temporary list to store openings for the current wall
        wall_openings = []

        # Iterate through the detected openings and store the information
        for (x_start, x_end), (z_min, z_max), opening_type in zip(opening_widths, opening_heights, opening_types):
            opening_info = {
                "opening_wall_id": id,
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



# generate IFC model
ifc_model = IFCmodel(ifc_project_name, ifc_output_file)
ifc_model.define_author_information(ifc_author_name + ' ' + ifc_author_surname, ifc_author_organization)
ifc_model.define_project_data(ifc_building_name, ifc_building_type, ifc_building_phase,
                              ifc_project_long_name, ifc_project_version, ifc_author_organization,
                              ifc_author_name, ifc_author_surname, ifc_site_latitude, ifc_site_longitude,
                              ifc_site_elevation)

# Add building storeys
storeys_ifc, slabs_ifc = [], []
for idx, slab in enumerate(slabs):
    # define a storey
    slab_position = slab['slab_bottom_z_coord'] + slab['thickness']
    storeys_ifc.append(ifc_model.create_building_storey('Floor %.1f m' % slab_position, slab_position))

    # define a slab
    points = [list(point) for point in zip(slab['polygon_x_coords'], slab['polygon_y_coords'])]
    points_no_duplicates = list(dict.fromkeys(map(tuple, points)))
    points_no_duplicates = [list(point) for point in points_no_duplicates]
    points_no_duplicates = [[float(coord) for coord in point] for point in points_no_duplicates]
    slabs_ifc.append(ifc_model.create_slab('Slab %d' % (idx + 1), points_no_duplicates,
                                           round(slab['slab_bottom_z_coord'], 3),
                                           round(slab['thickness'], 3), material_for_slabs))

    # assign the slab to a storey and save them to the IFC model
    ifc_model.assign_product_to_storey(slabs_ifc[-1], storeys_ifc[-1])

# Wall definition for IFC
for wall in walls:
    start_point = tuple(float(num) for num in wall['start_point'])
    end_point = tuple(float(num) for num in wall['end_point'])
    wall_thickness = wall['thickness']
    wall_material = wall['material']
    wall_z_placement = wall['z_placement']
    wall_heights = wall['height']

    wall_openings = [opening for opening in all_openings if opening['opening_wall_id'] == wall['id']]

    # Create a material layer
    material_layer = ifc_model.create_material_layer(wall_thickness, wall_material)
    # Create an IfcMaterialLayerSet using the material layer (in a list)
    material_layer_set = ifc_model.create_material_layer_set([material_layer])
    # Create an IfcMaterialLayerSetUsage and associate it with the element or product
    material_layer_set_usage = ifc_model.create_material_layer_set_usage(material_layer_set, wall_thickness)
    # Local placement
    wall_placement = ifc_model.wall_placement(float(wall_z_placement))
    wall_axis_placement = ifc_model.wall_axis_placement(start_point, end_point)
    wall_axis_representation = ifc_model.wall_axis_representation(wall_axis_placement)
    wall_swept_solid_representation = ifc_model.wall_swept_solid_representation(start_point, end_point, wall_heights, wall_thickness)
    product_definition_shape = ifc_model.product_definition_shape(wall_axis_representation, wall_swept_solid_representation)
    wall = ifc_model.create_wall(wall_placement, product_definition_shape)
    assign_material = ifc_model.assign_material(wall, material_layer_set_usage)
    wall_type = ifc_model.create_wall_type(wall, wall_thickness)
    assign_material_2 = ifc_model.assign_material(wall_type[0], material_layer_set)
    assign_object = ifc_model.assign_product_to_storey(wall, storeys_ifc[0])

    for opening in wall_openings:
        # Each 'opening' is a dictionary with the opening data
        opening_type = opening['opening_type']
        x_range_start = opening['x_range_start']
        x_range_end = opening['x_range_end']
        z_range_min = opening['z_range_min']
        z_range_max = opening['z_range_max']

        opening_width = x_range_end - x_range_start
        opening_height = z_range_max - z_range_min
        window_sill_height = z_range_min
        offset_from_start = x_range_start

        opening_closed_profile = ifc_model.opening_closed_profile_def(float(opening_width), wall_thickness)
        opening_placement = ifc_model.opening_placement(start_point, wall_placement)
        opening_extrusion = ifc_model.opening_extrusion(opening_closed_profile, float(opening_height), start_point, end_point, float(window_sill_height), float(offset_from_start))
        opening_representation = ifc_model.opening_representation(opening_extrusion)
        opening_product_definition = ifc_model.product_definition_shape_opening(opening_representation)
        wall_opening = ifc_model.create_wall_opening(opening_placement[1], opening_product_definition)
        rel_voids_element = ifc_model.create_rel_voids_element(wall, wall_opening)

# Write the IFC model to a file
ifc_model.write()
last_time = log('\nIFC model saved to %s.' % ifc_output_file, last_time, log_filename)
