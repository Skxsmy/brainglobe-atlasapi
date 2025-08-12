from pathlib import Path

import numpy as np
from skimage import filters, exposure
from tifffile import imread, imwrite


# 1. Load and process multi-channel image
def create_mask_from_channel(image_path, sigma=2.0, manual_threshold=None):
    """Create binary mask from specified channel"""
    # Load multi-channel image (assumes channels are first dimension)
    image = imread(image_path)

    # Extract target channel
    # Preprocessing
    if manual_threshold is not None:
        # Use manual threshold if provided
        binary = image > manual_threshold
    else:
        p2, p98 = np.percentile(image, (2, 98))
        enhanced = exposure.rescale_intensity(image, in_range=(p2, p98))
        denoised = filters.gaussian(enhanced, sigma=sigma)
        # Thresholding
        thresh = filters.threshold_otsu(denoised)
        binary = denoised > thresh

    return binary.astype(bool)


# 2. Apply mask to structural channel
def apply_mask(structural_image, mask):
    """Apply binary mask to structural channel"""
    masked = structural_image.copy()
    masked[~mask] = 0
    return masked


# 3. Calculate Dice score
def dice_score_mask(mask_true, mask_pred):
    """
    Calculate Dice coefficient from boolean masks using TP/FP/FN definition

    Args:
        mask_true: Ground truth boolean mask (2D/3D)
        mask_pred: Predicted boolean mask (same shape as mask_true)

    Returns:
        Dice score (float), and tuple of (TP, FP, FN)
    """
    # Ensure masks are boolean
    mask_true = mask_true > 0
    mask_pred = mask_pred > 0

    # Calculate confusion matrix elements
    TP = np.sum(mask_true & mask_pred)  # True positives
    FP = np.sum(~mask_true & mask_pred)  # False positives
    FN = np.sum(mask_true & ~mask_pred)  # False negatives

    # Calculate Dice coefficient (with epsilon to avoid division by zero)
    epsilon = 1e-7
    dice = (2 * TP) / (2 * TP + FP + FN + epsilon)

    return dice, (TP, FP, FN)


def dice_score_image(raw_img1, raw_img2):
    """
    Compute Dice score between two 3D non-binary images.

    Parameters:
    - img1, img2: 3D numpy arrays (same shape)
    Returns:
    - Dice score (float) or array of per-slice scores
    """
    assert raw_img1.shape == raw_img2.shape, "Images must have same dimensions"

    img1 = (raw_img1 - raw_img1.min()) / (raw_img1.max() - raw_img1.min())
    img2 = (raw_img2 - raw_img2.min()) / (raw_img2.max() - raw_img2.min())

    intersection = np.sum(img1 * img2)
    sum_images = np.sum(img1) + np.sum(img2)
    return (2. * intersection) / (sum_images + 1e-8)  # Avoid division by zero


# Main workflow
if __name__ == "__main__":
    # Load your multi-channel image
    workspace_path = Path("D:/UCL/Postgraduate_programme/Validation/DICE/0508Nubbin")
    mode = 'original_atlas'  # or 'original_atlas'
    for folder in workspace_path.iterdir():
        sample_id = str(folder.name).split("_")[0].lower()
        atlas_version = str(folder.name).split("_")[2].lower()
        exp_id = str(folder.name).split("_")[3].lower()
        gender = str(folder.name).split("_")[1].lower()

        if mode == 'transformed_atlas':
            pouch_marker_label_path = workspace_path / f"{sample_id}_{gender}_{atlas_version}_{exp_id}/downsampled_EGFP-{sample_id}_{gender}.tiff"
            anatomical_label_path = workspace_path / f"{sample_id}_{gender}_{atlas_version}_{exp_id}/downsampled_C2MDRP-{sample_id}_{gender}.tiff"
            atlas_label_path = workspace_path / f"{sample_id}_{gender}_{atlas_version}_{exp_id}/registered_atlas.tiff"
        elif mode == 'original_atlas':
            pouch_marker_label_path = workspace_path / f"{sample_id}_{gender}_{atlas_version}_{exp_id}/downsampled_standard_EGFP-{sample_id}_{gender}.tiff"
            anatomical_label_path = workspace_path / f"{sample_id}_{gender}_{atlas_version}_{exp_id}/downsampled_standard_C2MDRP-{sample_id}_{gender}.tiff"
            atlas_label_path = Path(f'C:/Users/skxmy/.brainglobe/annotation_{atlas_version}.tiff')

        # Create masks from different channels
        pouch_marker_mask = create_mask_from_channel(pouch_marker_label_path, sigma=1.5, manual_threshold=None)
        imwrite(workspace_path / f'{sample_id}_{gender}_{atlas_version}_{exp_id}/pouch_marker_label.tiff',
                pouch_marker_mask)

        atlas_label = imread(atlas_label_path)
        pouch_filtered_atlas = atlas_label.copy()
        pouch_filtered_atlas[atlas_label != 1.0] = 0
        pouch_filtered_atlas_mask = pouch_filtered_atlas.astype(bool)

        # Calculate Dice score
        dice_mask, values = dice_score_mask(pouch_marker_mask, pouch_filtered_atlas_mask)
        print(f"Dice Score: {dice_mask:.4f}"
              f" (TP: {values[0]}, FP: {values[1]}, FN: {values[2]})")

        structure_channel = imread(anatomical_label_path)
        pouch_label_masked_structure = apply_mask(structure_channel, pouch_marker_mask)
        atlas_masked_structure = apply_mask(structure_channel, pouch_filtered_atlas_mask)
        dice_image = dice_score_image(pouch_label_masked_structure, atlas_masked_structure)
        print(f"Dice Score for masked structures: {dice_image:.4f}")

        # Save Dice score results
        results_path = workspace_path / f'{sample_id}_{gender}_{atlas_version}_{exp_id}/dice_results_{mode}.txt'
        with open(results_path, 'w') as f:
            f.write(f"Dice Score: {dice_mask:.4f}\n"
                    f"TP: {values[0]}, FP: {values[1]}, FN: {values[2]}\n"
                    f"Dice Score for masked structures: {dice_image:.4f}\n")
