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
The `cloud2entities.py` script requires a YAML configuration file to run. You can provide the path to the 
configuration file as a command-line argument. If no argument is provided, the script will 
automatically use the default file `config.yaml`.

### Example command
```
python cloud2entities.py config.yaml
```
### Dataset
The complete original point cloud for Kladno station is available at Zenodo platform.
```
https://zenodo.org/records/14221915
```
## Citation

If you find this project or any part of it useful in your research or work, please consider citing the following preprint:
```
@misc{Cloud2BIM2025, 
title={Cloud2BIM: An open-source automatic pipeline for efficient conversion of large-scale point clouds into IFC format}, 
    author={Slávek Zbirovský and Václav Nežerka}, 
    year={2025}, 
    eprint={2503.11498}, 
    archivePrefix={arXiv}, 
    primaryClass={cs.CV}, 
    url={https://arxiv.org/abs/2503.11498}, 
    note={Accessed: 28.3.2025} 
    }
```
# License
GPL (General Public License)
https://www.gnu.org/licenses/gpl-3.0.html
