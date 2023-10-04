from aux_functions import *
from generate_ifc import IFCmodel


# input point clouds
e57_input = False  # if True, only one xyz_file is created (e57 files are merged)
# e57_file_names = ["input_e57/test_room.e57"]
# e57_file_names = ['input_e57/05th.e57', "input_e57/06th.e57", "input_e57/07th.e57"]
e57_file_names = ["input_e57/06th.e57", "input_e57/07th.e57"]

xyz_filenames = ['input_xyz/Test_room_merged_subsampled.xyz']

# input parameters for identification of elements
pointcloud_resolution = 0.002
minimum_wall_length = 0.2
minimum_wall_thickness = 0.05
maximum_wall_thickness = 0.6

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

# read e57 file
if e57_input:
    imported_e57_data = []
    for e57_file_name in e57_file_names:
        last_time = log('Reading %s.' % e57_file_name, last_time, log_filename)
        imported_e57_data.append(read_e57(e57_file_name))
    last_time = log('Saving the data to %s...' % xyz_filenames[0], last_time, log_filename)
    e57_data_to_xyz(imported_e57_data, xyz_filenames[0], chunk_size=1000)
    last_time = log('e57 file(s) converted to ASCII format, saved as %s.' % xyz_filenames[0], last_time, log_filename)

# read xyz file
points_xyz, points_rgb = np.empty((0, 3)), np.empty((0, 3))
for xyz_filename in xyz_filenames:
    last_time = log('Extracting data from %s...' % xyz_filename, last_time, log_filename)
    points_xyz_temp, points_rgb_temp = load_xyz_file(xyz_filename, plot_xyz=False)
    points_xyz = np.vstack((points_xyz, np.array(points_xyz_temp)))
    points_rgb = np.vstack((points_rgb, np.array(points_rgb_temp)))
points_xyz = np.round(points_xyz, 3)  # round the xyz coordinates to 3 decimals
last_time = log('All point cloud data imported.', last_time, log_filename)

# scan the model along the z-coordinate and search for planes parallel to xy-plane
slabs, horizontal_surface_planes = identify_slabs_from_point_cloud(points_xyz, points_rgb, z_step=0.1, plot_segmented_plane=False)

# merge_horizontal_pointclouds_in_storey(horizontal_surface_planes)
pointcloud_storeys = split_pointcloud_to_storeys(points_xyz, slabs)
walls = []
id = 0
for i, storey_pointcloud in enumerate(pointcloud_storeys):
    start_points, end_points, wall_thicknesses, wall_materials = identify_walls(storey_pointcloud, pointcloud_resolution, minimum_wall_length,
                                                                                minimum_wall_thickness, maximum_wall_thickness, i)
    z_placement = slabs[i]['slab_bottom_z_coord'] + slabs[i]['thickness']
    wall_height = slabs[i+1]['slab_bottom_z_coord'] - z_placement
    for j in range(len(start_points)):
        id += 1
        walls.append({'id': id, 'storey': i+1, 'start_point': start_points[j], 'end_point': end_points[j],
                      'thickness': wall_thicknesses[j], 'material': wall_materials[j], 'z_placement': z_placement,
                      'height': wall_height})

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
    slabs_ifc.append(ifc_model.create_slab('Slab %d' % (idx + 1), points_no_duplicates,
                                           round(slab['slab_bottom_z_coord'], 3),
                                           round(slab['thickness'], 3), material_for_slabs))

    # assign the slab to a storey and save them to the IFC model
    ifc_model.assign_product_to_storey(slabs_ifc[-1], storeys_ifc[-1])

# Wall definitition for IFC
for wall in walls:
    start_point = tuple(float(num) for num in wall['start_point'])
    end_point = tuple(float(num) for num in wall['end_point'])
    wall_thickness = wall['thickness']
    wall_material = wall['material']
    wall_z_placement = wall['z_placement']
    wall_heights = wall['height']

    # Create a material layer
    material_layer = ifc_model.create_material_layer(wall_thickness, wall_material)
    # Create an IfcMaterialLayerSet using the material layer (in a list)
    material_layer_set = ifc_model.create_material_layer_set([material_layer])
    # Create an IfcMaterialLayerSetUsage and associate it with the element or product
    material_layer_set_usage = ifc_model.create_material_layer_set_usage(material_layer_set, wall_thickness)
    # Local placement
    wall_placement = ifc_model.wall_placement(start_point, float(wall_z_placement))
    wall_axis_placement = ifc_model.wall_axis_placement(start_point, end_point)
    wall_axis_representation = ifc_model.wall_axis_representation(wall_axis_placement)
    wall_swept_solid_representation = ifc_model.wall_swept_solid_representation(start_point, end_point, wall_heights, wall_thickness)
    product_definition_shape = ifc_model.product_definition_shape(wall_axis_representation, wall_swept_solid_representation)
    wall = ifc_model.create_wall(wall_placement, product_definition_shape)
    assign_material = ifc_model.assign_material(wall, material_layer_set_usage)
    wall_type = ifc_model.create_wall_type(wall, wall_thickness)
    assign_material_2 = ifc_model.assign_material(wall_type[0], material_layer_set)
    assign_object = ifc_model.assign_product_to_storey(wall, storeys_ifc[0])

"""start_point = (-92.64, -34.203)
end_point = (7.5, 5.5)
wall_thickness = 0.3
wall_material = "Concrete"
wall_z_placement = 50.3
wall_heights = 2.8

# Create a material layer
material_layer = ifc_model.create_material_layer(wall_thickness, wall_material)
# Create an IfcMaterialLayerSet using the material layer (in a list)
material_layer_set = ifc_model.create_material_layer_set([material_layer],wall_thickness)
# Create an IfcMaterialLayerSetUsage and associate it with the element or product
material_layer_set_usage = ifc_model.create_material_layer_set_usage(material_layer_set, wall_thickness)
# Local placement
wall_placement = ifc_model.wall_placement(start_point, float(wall_z_placement))
wall_axis_placement = ifc_model.wall_axis_placement(start_point, end_point)
wall_axis_representation = ifc_model.wall_axis_representation(wall_axis_placement)
wall_swept_solid_representation = ifc_model.wall_swept_solid_representation(start_point, end_point, wall_heights, wall_thickness)
product_definition_shape = ifc_model.product_definition_shape(wall_axis_representation, wall_swept_solid_representation)
wall = ifc_model.create_wall(wall_placement, product_definition_shape)
assign_material = ifc_model.assign_material(wall, material_layer_set_usage)
wall_type = ifc_model.create_wall_type(wall, wall_thickness)
assign_material_2 = ifc_model.assign_material(wall_type[0], material_layer_set)
assign_object = ifc_model.assign_product_to_storey(wall, storeys_ifc[0])"""

# Write the IFC model to a file
ifc_model.write()
last_time = log('\nIFC model saved to %s.' % ifc_output_file, last_time, log_filename)
