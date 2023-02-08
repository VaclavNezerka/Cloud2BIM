import numpy as np
import pye57
import pandas as pd


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
    df = pd.DataFrame({'X': x, 'Y': y, 'Z': z, 'R': red, 'G': green, 'B': blue, 'Intensity': intensity})
    df.to_csv(output_file_name, sep='\t', index=False)
    print('e57 converted to ASCII format, saved as %s' % output_file_name)


if __name__ == '__main__':

    # inputs
    e57_input = True
    e57_file_name = 'input_e57/test_room.e57'
    create_xyz = True
    xyz_filename = 'input_xyz/test_room.xyz'

    # read e57
    if e57_input:
        imported_e57_data = read_e57(e57_file_name)
        if create_xyz:
            e57_data_to_xyz(imported_e57_data, xyz_filename)
    else:
        pass




