from aux_functions import *
from generate_ifc import IFCmodel

# user inputs
e57_input = False
e57_file_name = 'input_e57/test_room.e57'
xyz_filename = 'input_xyz/Test_room_merged.xyz'

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

# read e57 file
if e57_input:
    imported_e57_data = read_e57(e57_file_name)
    e57_data_to_xyz(imported_e57_data, xyz_filename)

# read xyz file
points_xyz, points_rgb = load_xyz_file(xyz_filename)
points_xyz = np.round(np.array(points_xyz), 3)  # round the xyz coordinates to 3 decimals

# scan the model along the z-coordinate and search for planes parallel to xy-plane
z_step = 0.1
slabs = identify_slabs_from_point_cloud(points_xyz, points_rgb, z_step, plot_segmented_plane=False)

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

# Write the IFC model to a file
ifc_model.write()
print('\nIFC model saved to %s.' % ifc_output_file)
