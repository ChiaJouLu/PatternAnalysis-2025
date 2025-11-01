"""
Prediction, testing, and visualization for UNet prostate segmentation.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import nibabel as nib
import cv2
from tqdm import tqdm

from modules import UNet


def load_data_with_resize(image_paths, target_size=(256, 128), normImage=True):
    """
    Load the image and resize it to a uniform size.
    """
    n = len(image_paths)
    images = np.zeros((n, target_size[0], target_size[1]), dtype=np.float32)

    for i, path in enumerate(tqdm(image_paths, desc='Loading images')):
        img = nib.load(path).get_fdata(caching='unchanged')

        if len(img.shape) == 3:
            img = img[:, :, 0]

        # Resize if needed
        if img.shape != target_size:
            img = cv2.resize(img, (target_size[1], target_size[0]),
                           interpolation=cv2.INTER_LINEAR)

        # Normalize
        if normImage:
            img = (img - img.mean()) / (img.std() + 1e-8)

        images[i] = img

    return images


def load_labels_with_resize(seg_paths, target_size=(256, 128), n_classes=4):
    """
    Load tags, resize, clean up categories, and perform one-hot encoding.
    """
    n = len(seg_paths)
    labels = np.zeros((n, target_size[0], target_size[1], n_classes), dtype=np.float32)

    for i, path in enumerate(tqdm(seg_paths, desc='Loading labels')):
        label = nib.load(path).get_fdata(caching='unchanged')

        if len(label.shape) == 3:
            label = label[:, :, 0]

        # Resize (Use INTER_NEAREST to keep the category unchanged)
        if label.shape != target_size:
            label = cv2.resize(label, (target_size[1], target_size[0]),
                             interpolation=cv2.INTER_NEAREST)

        # Clean up redundant categories (should address the issues with categories 4 and 5)
        label[label >= n_classes] = 0

        # One-hot encoding
        for c in range(n_classes):
            labels[i, :, :, c] = (label == c)

    return labels


def dice_coefficient_per_class(predicted, target, n_classes=4):
    """
    Calculate Dice coefficient for each class separately.

    Args:
        predicted: Predicted probabilities (N, C, H, W)
        target: One-hot encoded ground truth (N, C, H, W)
        n_classes: Number of classes

    Returns:
        dict: Dice scores for each class (class_0, class_1, class_2, class_3)
    """
    dice_scores = {}

    for c in range(n_classes):
        # Extract specific class
        predicted_c = predicted[:, c, :, :]  # (N, H, W)
        target_c = target[:, c, :, :]        # (N, H, W)

        # Calculate intersection and union
        intersection = (predicted_c * target_c).sum()
        denominator = predicted_c.sum() + target_c.sum()

        # Handle empty cases
        if denominator == 0:
            dice_scores[f'class_{c}'] = 1.0  # Both empty = perfect match
        else:
            dice_scores[f'class_{c}'] = (2. * intersection / denominator).item()

    return dice_scores


def test(model, data_loader, device, save_path='results', visualize=False):
    """
    Test model on dataset and generate visualizations.

    Args:
        model: Trained UNet model.
        data_loader: DataLoader for test data.
        device: The computation device to run the test on (e.g., 'cuda' or 'cpu').
        save_path: Directory to save results.
        visualize: Whether to generate visualization images.

    Returns:
        dict: average Dice coefficient across all batches
    """
    model.eval()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    running_dice = {
        'class_0': 0.0,  # Background
        'class_1': 0.0,  # Peripheral Zone
        'class_2': 0.0,  # Transition Zone
        'class_3': 0.0   # Prostate (MAIN TARGET)
    }
    batch_count = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Testing')

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            dice = dice_coefficient_per_class(probs, labels, n_classes=4)

            for key in running_dice.keys():
                running_dice[key] += dice[key]

            batch_count += 1

            pbar.set_postfix({'Prostate_Dice': f'{dice["class_3"]:.4f}'})

            # Visualize first few batches
            if visualize and batch_idx < 5:
                visualize_prediction(images, labels, probs, batch_idx, save_path)

    avg_dice = {key: value / batch_count for key, value in running_dice.items()}

    return avg_dice




    def visualiza_prediction(images, labels, predicts, batch_idx, save_path):
        """
        Save a figure showing the input, ground truth, and prediction.
        """



    
    def visualize_sample_prediction(model, data_loader, device, save_path='results', num_samples=5):
        """
        """
        pass

  


    if __name__ == "__main__":
        main()


