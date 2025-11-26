# Spinal Canal Segmentation Tool

A Python utility for automated spinal canal segmentation from MRI images using the Spinal Cord Toolbox (SCT) deep learning model, with visualization and HPC batch processing capabilities.

## Overview

This tool provides three main functions for processing spinal MRI data:

1. **Segmentation**: Automated spinal canal detection using a pre-trained deep learning model (SCT's `sc_canal_t2`)
2. **Visualization**: Overlay segmentation masks on original images with quality control (QC) output
3. **Batch Processing**: SLURM-based distributed processing on HPC clusters

## Requirements

- Python 3.7+
- **Core dependencies**:
  - `spinalcordtoolbox` (SCT) – for deep learning segmentation
  - `nibabel` – for reading/writing NIfTI medical images
  - `numpy` – for array operations
  - `matplotlib` – for image visualization

### Installation

Install SCT and dependencies:

```bash
# Install Spinal Cord Toolbox (recommended: conda)
conda install -c conda-forge spinalcordtoolbox

# Or install individual dependencies
pip install spinalcordtoolbox nibabel numpy matplotlib
```

Ensure SCT is properly installed and accessible in your Python environment.

## Functions

### `spinal_cord_segmentation(img_path, output_im)`

Runs automated spinal canal segmentation using SCT's deep learning model.

**Parameters:**
- `img_path` (str): Path to input T2-weighted MRI image (NIfTI format, `.nii.gz`)
- `output_im` (str): Path where the segmentation mask will be saved

**Returns:** None

**Example:**
```python
spinal_cord_segmentation('input_image.nii.gz', 'segmentation_mask.nii.gz')
```

**Details:**
- Uses the `sc_canal_t2` model (optimized for T2-weighted spinal cord images)
- Output is a binary segmentation mask in NIfTI format
- Processing time depends on image size and GPU availability

---

### `vis_seg(img_path, seg_path)`

Creates a quality-control visualization overlaying the segmentation on the original image.

**Parameters:**
- `img_path` (str): Path to the original MRI image
- `seg_path` (str): Path to the segmentation mask

**Returns:** None

**Output:** Saves `seg_qc.png` in the same directory as `seg_path`

**Example:**
```python
vis_seg('input_image.nii.gz', 'segmentation_mask.nii.gz')
```

**Details:**
- Original image: grayscale colormap (inverted grayscale)
- Segmentation overlay: red with 50% opacity
- Images are rotated 90° for axial slice visualization
- Intensity clipping: image [0-150], mask [0.9-1.0]

---

### `canal_seg_slurm(input_img, seg_path, job_name, python_env, node_list, partition)`

Submits a SLURM batch job for segmentation and visualization on an HPC cluster.

**Parameters:**
- `input_img` (str): Path to input MRI image
- `seg_path` (str): Path for output segmentation mask
- `job_name` (str): SLURM job name
- `python_env` (str): Conda environment name (must be available on compute nodes)
- `node_list` (list): List of node names to restrict execution (e.g., `['node1', 'node2']`)
- `partition` (str): SLURM partition/queue name

**Returns:** None (prints SLURM job submission output)

**Example:**
```python
canal_seg_slurm(
    'input.nii.gz',
    'output_seg.nii.gz',
    'canal_seg_job_001',
    'myenv',
    ['gpu-node-1', 'gpu-node-2'],
    'gpu_partition'
)
```

**SLURM Configuration:**
- Job time limit: 20 minutes
- Nodes: 1
- CPUs per task: 1
- Memory: 11 GB
- Restricted to specified nodes and partition
- Automatically runs segmentation and then visualization
- Output/error logs saved as `{job_name}.out` and `{job_name}.err` in the output directory

---

## Usage

### Command-line Interface

The script supports command-line execution via the `__main__` block:

```bash
# Run segmentation
python run_canal_seg.py spawn input_image.nii.gz output_segmentation.nii.gz

# Generate QC visualization
python run_canal_seg.py vis input_image.nii.gz segmentation_mask.nii.gz
```

### Python Module

Import and use functions directly:

```python
from run_canal_seg import spinal_cord_segmentation, vis_seg, canal_seg_slurm

# Segmentation
spinal_cord_segmentation('mri.nii.gz', 'mask.nii.gz')

# Visualization
vis_seg('mri.nii.gz', 'mask.nii.gz')

# HPC batch job
canal_seg_slurm('mri.nii.gz', 'mask.nii.gz', 'job1', 'myenv', 
                 ['node1'], 'gpu')
```

## Workflow Example

```bash
# Step 1: Segment an image
python run_canal_seg.py spawn patient_001_t2.nii.gz patient_001_seg.nii.gz

# Step 2: Visualize the result
python run_canal_seg.py vis patient_001_t2.nii.gz patient_001_seg.nii.gz

# Step 3: Check QC image
open patient_001_seg_qc.png
```

## Input/Output Formats

- **Input images**: NIfTI format (`.nii` or `.nii.gz`)
  - Recommended: T2-weighted MRI with axial orientation
- **Output segmentation**: Binary NIfTI mask (same voxel dimensions as input)
- **QC visualization**: PNG image (8x8 inches, 100 dpi default)

## HPC Cluster Setup

Before using `canal_seg_slurm`:

1. Ensure the conda environment is available on compute nodes
2. Verify node names and partition are correct
3. Check SLURM configuration matches your cluster policies
4. Test with a small image first

**Example SLURM job submission:**
```python
canal_seg_slurm(
    '/data/patients/001_t2.nii.gz',
    '/output/001_seg.nii.gz',
    'patient_001_seg',
    'sct_env',          # conda environment
    ['gpu-01', 'gpu-02'],  # available GPU nodes
    'gpu'               # GPU partition
)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: spinalcordtoolbox` | Install SCT: `conda install -c conda-forge spinalcordtoolbox` |
| `nibabel` import error | `pip install nibabel` |
| SLURM job fails | Check node availability, partition name, and conda environment accessibility |
| Segmentation is poor quality | Ensure input is T2-weighted and properly preprocessed; check image intensity range |
| QC image is blank | Verify segmentation mask has non-zero values; check colormap/clipping ranges |

## Notes

- The script uses **T2-weighted imaging** by default (`sc_canal_t2` model)
- For other modalities (T1, FLAIR, etc.), modify the model string in `spinal_cord_segmentation()`
- QC visualization assumes **axial slice orientation** after 90° rotation
- GPU acceleration is recommended for faster segmentation (SCT will auto-detect)
- SLURM integration is tailored for cluster environments; modify resource allocations as needed

## License

Refer to the Spinal Cord Toolbox license: https://spinalcordtoolbox.com/

## References

- **Spinal Cord Toolbox**: https://spinalcordtoolbox.com/
- **SCT Deep Learning**: https://github.com/sct-pipeline/deepseg-spinalcord
- **NIfTI Format**: https://nifti.nimh.nih.gov/

---

**Last Updated**: November 2025
