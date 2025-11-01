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
        dict: average Dice coefficient across all batches.
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


def visualize_prediction(images, labels, predicts, batch_idx, save_path):
    """
    Save a figure showing the input, ground truth, and prediction.
    """
    # Get first image from batch
    img = images[0, 0].cpu().numpy()
    label = torch.argmax(labels[0], dim=0).cpu().numpy()
    predict = torch.argmax(predicts[0], dim=0).cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('MRI Image', fontsize=14)
    axes[0].axis('off')

    # Ground truth
    axes[1].imshow(label, cmap='tab10', vmin=0, vmax=3)
    axes[1].set_title('Ground Truth', fontsize=14)
    axes[1].axis('off')

    # Prediction
    axes[2].imshow(predict, cmap='tab10', vmin=0, vmax=3)
    axes[2].set_title('Prediction', fontsize=14)
    axes[2].axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:blue', label='Background'),
        Patch(facecolor='tab:orange', label='Peripheral Zone'),
        Patch(facecolor='tab:green', label='Transition Zone'),
        Patch(facecolor='tab:red', label='Prostate')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()

    # Save figure
    save_file = os.path.join(save_path, f'prediction_batch_{batch_idx}.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"  Saved visualization: {save_file}")


def plot_curves(history, save_path='results'):
    """
    Plot training curves.

    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_dice', 'val_dice'.
        save_path: Directory to save the plot.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(history['train_loss'], label='Training')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over epochs')
    ax1.legend()
    ax1.grid(True)
      
    # Dice plot
    ax2.plot(history['train_dice'], label='Training')
    if 'val_dice' in history:
        ax2.plot(history['val_dice'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Coefficient (Prostate)')
    ax2.set_title('Dice over epochs')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_file = os.path.join(save_path, 'training_curves.png')
    plt.savefig(save_file, dpi=150)
    plt.close()

    print(f"Saved training curves to {save_file}")


def main():
    # Configuration
    data_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    checkpoint_path = "unet_final.pth"
    test_split = "test"
    batch_size = 16
    visualize = True

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    print("Loading model...")
    model = UNet(n_channels=1, n_classes=4).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load test data
    print(f"Loading {test_split} dataset...")
    from torch.utils.data import DataLoader, TensorDataset
    
    test_images = sorted(glob.glob(f'{data_path}/keras_slices_{test_split}/*.nii.gz'))
    test_labels = sorted(glob.glob(f'{data_path}/keras_slices_seg_{test_split}/*.nii.gz'))

    X_test = load_data_with_resize(test_images)
    y_test = load_labels_with_resize(test_labels)

    test_dataset = TensorDataset(
        torch.from_numpy(X_test).unsqueeze(1).float(),
        torch.from_numpy(y_test).permute(0, 3, 1, 2).float()
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate on test set
    print(f"\nEvaluating on {test_split} set...")
    results = test(model, test_loader, device, save_path='results', visualize=visualize)

    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    class_names = ['Background', 'Peripheral Zone', 'Transition Zone', 'Prostate']
    for i, name in enumerate(class_names):
        print(f"{name:20s}: Dice = {results[f'class_{i}']:.4f}")
    print("="*60)
    print(f"\n{'Prostate (Target)':20s}: {results['class_3']:.4f} (Requirement: ≥ 0.75)")
    
    if results['class_3'] >= 0.75:
        print("PASSED - Prostate segmentation meets requirement!")
    else:
        print("FAILED - Below requirement")
    print("="*60)
    
    if visualize:
        print("\nVisualizations saved to 'results/' directory")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

