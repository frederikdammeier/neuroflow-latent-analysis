import nibabel as nib
import numpy as np
import os

def transform_to_voxel_space(fmri_data: np.ndarray, nsd_general_mask_path: str):
    """
    Transforms fMRI data back to voxel space using the NSD general mask.
    The NSD general mask is a subject specifc Nifti file that knows the subject geometry.

    Args:
        fmri_data (np.ndarray): The fMRI data in masked space.
        nsd_general_mask_path (str): Path to the NSD general mask.

    Returns:
        nib.Nifti1Image: The fMRI data transformed back to voxel space.
    """
    mask_img = nib.load(nsd_general_mask_path)
    mask_data = mask_img.get_fdata()

    voxel_space_data = np.zeros(mask_data.shape)
    voxel_space_data[mask_data > 0] = fmri_data

    return nib.Nifti1Image(voxel_space_data, mask_img.affine)

def get_roi_names(floc_name: str, label_path: str) -> tuple[dict, dict]:
    """
    Retrieves the names of ROIs for a given fLoc experiment.

    Args:
        floc_name (str): Name of the fLoc experiment.
        label_path (str): Path to the label file containing ROI names.

    Returns:
        tuple[dict, dict]: Two dictionaries, one mapping ROI IDs to ROI names 
        and the other mapping ROI names to ROI IDs.
    """
    id_to_name = {}
    name_to_id = {}
    
    filepath = os.path.join(label_path, f"{floc_name}.mgz.ctab")
    with open(filepath, 'r') as f:
        # Each line contains the integer followed by the ROI name, separated by a space
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                roi_id, roi_name = parts
                id_to_name[int(roi_id)] = roi_name
                name_to_id[roi_name] = int(roi_id)
        f.close()
    return id_to_name, name_to_id