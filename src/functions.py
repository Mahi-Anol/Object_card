import os
import numpy as np
import cv2
from pathlib import Path


def crop_segmented_region(image, mask, contour, output_path, min_area=100):
    """
    Crop and rotate the segmented region to correct the contour's angle, and save it.
    Args:
        image: Original image (numpy array, HxWx3, RGB).
        mask: Binary mask (numpy array, HxW).
        contour: Contour of the mask (numpy array, Nx1x2).
        output_path: Path to save the cropped and rotated image.
        min_area: Minimum area of the mask to consider (to filter noise).
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    # Ensure mask is binary (0 or 255)
    mask = mask.astype(np.uint8)
    if np.sum(mask > 0) < min_area:
        print(f"Mask too small (area={np.sum(mask > 0)}). Skipping.")
        return False

    # Create RGBA image for transparency
    rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgba_image[:, :, :3] = image  # Copy RGB channels
    rgba_image[:, :, 3] = mask  # Alpha channel: 255 where mask is 255, 0 elsewhere

    # Find the bounding rectangle of non-zero mask pixels (initial crop)
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        print("Empty mask. Skipping.")
        return False
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the region
    cropped_image = rgba_image[y_min:y_max+1, x_min:x_max+1]

    # Get the contour's angle
    rect = cv2.minAreaRect(contour)
    mask_angle = rect[2]
    print(f"Mask angle: {mask_angle}")

    if mask_angle > 45:
        mask_angle = -1 * (90 - mask_angle)

    # if mask_angle < -45:
    #     mask_angle += 90  # Adjust for upright orientation

    # Rotate to correct the contour's angle (target: 0°)
    rotation_angle = -mask_angle  # Negate to align with 0°
    print(f"Contour angle: {mask_angle}, Rotation applied: {rotation_angle}")

    # Get the center of the cropped image
    (h, w) = cropped_image.shape[:2]
    center = (w // 2, h // 2)

    # Compute the rotation matrix
    # M = cv2.getRotationMatrix2D(center, rotation_angle, scale=1.0)
    M = cv2.getRotationMatrix2D(center, mask_angle, scale=1.0)

    # Rotate the cropped image with a larger canvas to avoid clipping
    rotated_image = cv2.warpAffine(
        cropped_image,
        M,
        (w * 2, h * 2),  # Larger canvas
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)  # Transparent border
    )

    # Recrop to remove extra padding
    alpha_channel = rotated_image[:, :, 3]
    coords = np.column_stack(np.where(alpha_channel > 0))
    if coords.size == 0:
        print("Empty rotated mask. Skipping.")
        return False
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    final_image = rotated_image[y_min:y_max+1, x_min:x_max+1]

    # Save as PNG to preserve transparency
    cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGBA2BGRA))
    print(f"Saved cropped and rotated segmented region to {output_path}")
    return True