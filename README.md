# Cloud2BIM 
![Workflow](title.png)
Cloud2BIM automates the Scan-to-BIM process by converting point clouds into 3D parametric entities.  It employs a segmentation algorithm that utilizes point cloud density analysis, augmented by image and morphological operations. This allows the software to precisely extract the geometry of building elements such as slabs, walls, windows, and doors. The output is generated in IFC format, ensuring compatibility with other OpenBIM tools. The primary motivation for this software is to streamline and enhance decision-making at the end of a building's lifecycle, leading to more efficient material use during demolition or deconstruction.
# Installation
To install Cloud2BIM, follow these steps:
git clone 
```
https://github.com/yourusername/Cloud2BIM.git
```

Install dependencies:

First, ensure you have Python and pip installed.
Then, install the required dependencies listed in the requirements.txt file:
```
pip install -r requirements.txt
```
# Running the Script
You can run the `cloud2entities.py` script directly from the terminal. When no command-line arguments are provided, the script uses its default parameters. You can override these defaults by passing options as shown below.

### Command Syntax

```
python cloud2entities.py [OPTIONS]
```
### Available Options

#### Input Files
- **`--e57_input`**  
  Use E57 files as input.
- **`--e57_files`**  
  List of E57 input files.  
  *Default: `["input_e57/multiple_floor.e57"]`*
- **`--xyz_files`**  
  List of XYZ input files.  
  *Default: `["input_xyz/new_data/Zurich_dataset_synth3_01.xyz"]`*

#### Processing Options
- **`--dilute`**  
  Dilute the point cloud.
- **`--exterior_scan`**  
  Enable scanning of exterior walls.
- **`--dilution_factor`**  
  Dilution factor (skip every nth line).  
  *Default: `10`*
- **`--pc_resolution`**  
  Minimum point distance after dilution in meters.  
  *Default: `0.002`*
- **`--grid_coefficient`**  
  Computational grid size in [px/mm].  
  *Default: `5`*

#### Slab Parameters
- **`--bfs_thickness`**  
  Bottom floor slab thickness in meters.  
  *Default: `0.3`*
- **`--tfs_thickness`**  
  Top floor slab thickness in meters.  
  *Default: `0.4`*

#### Wall Parameters
- **`--min_wall_length`**  
  Minimum wall length in meters.  
  *Default: `0.08`*
- **`--min_wall_thickness`**  
  Minimum wall thickness in meters.  
  *Default: `0.05`*
- **`--max_wall_thickness`**  
  Maximum wall thickness in meters.  
  *Default: `0.75`*
- **`--exterior_walls_thickness`**  
  Exterior wall thickness in meters.  
  *Default: `0.3`*

#### IFC Output & Project Settings
- **`--output_ifc`**  
  Output IFC file path.  
  *Default: `"output_IFC/output-2.ifc"`*
- **`--ifc_project_name`**  
  IFC Project Name.  
  *Default: `"Sample project"`*
- **`--ifc_project_long_name`**  
  IFC Project Long Name.  
  *Default: `"Deconstruction of non-load-bearing elements"`*
- **`--ifc_project_version`**  
  IFC Project Version.  
  *Default: `"version 1.0"`*

#### IFC Author & Building Information
- **`--ifc_author_name`**  
  IFC Author Name.  
  *Default: `"John"`*
- **`--ifc_author_surname`**  
  IFC Author Surname.  
  *Default: `"Doe"`*
- **`--ifc_author_organization`**  
  IFC Author Organization.  
  *Default: `"CTU in Prague"`*
- **`--ifc_building_name`**  
  IFC Building Name.  
  *Default: `"Hotel Opatov"`*
- **`--ifc_building_type`**  
  IFC Building Type.  
  *Default: `"Hotel"`*
- **`--ifc_building_phase`**  
  IFC Building Phase.  
  *Default: `"Reconstruction"`*

#### Site Information
- **`--ifc_site_latitude`**  
  IFC Site Latitude (degrees, minutes, seconds).  
  *Default: `(50, 5, 0)`*
- **`--ifc_site_longitude`**  
  IFC Site Longitude (degrees, minutes, seconds).  
  *Default: `(4, 22, 0)`*
- **`--ifc_site_elevation`**  
  Elevation above sea level in meters.  
  *Default: `356.0`*

#### Material Settings
- **`--material_for_objects`**  
  Material for objects.  
  *Default: `"Concrete"`*

### Example command
```
python cloud2entities.py --xyz_files input_xyz/custom.xyz --output_ifc output_IFC/custom.ifc --dilute --dilution_factor 5
```
# License
GPL (General Public License)
https://www.gnu.org/licenses/gpl-3.0.html
