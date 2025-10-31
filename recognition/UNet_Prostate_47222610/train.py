"""
Train Unet for prostate segmentation.

This is the training script for the HipMRI dataset.
"""
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import cv2

from modules import UNet
from dataset import load_data_2D

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


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    """
    This function trains the model for one epoch on the given data loader.

    It goes through all the training data once and updates the model, includes
    loss calculation, and backpropagation.

    Args:
        model: The neural network model to be trained.
        data_loader: Iterable that provides batches of training data. Each batch
                     should be a tuple of input tensors and corresponding
                     segmentation masks or class labels.
        loss_fn: The loss function used to measure prediction error.
        optimizer: Updating model parameters (weights) based on computed gradients.
        device: The computation device to run the training on (e.g., 'cuda' or 'cpu').
                Both the model and data batches will be moved to this device.

    Returns:
        A tuple (avg_loss, avg_dice), where avg_loss (float) is the average loss across
        all batches, and avg_dice (float) is the average Dice coefficient across all batches,
        used for segmentation performance monitoring.
    """
    model.train()

    running_loss = 0.0
    running_dice = {
        'class_0': 0.0,  # Background
        'class_1': 0.0,  # Peripheral Zone
        'class_2': 0.0,  # Transition Zone
        'class_3': 0.0   # Prostate (MAIN TARGET)
    }
    batch_count = 0

    pbar = tqdm(data_loader, desc='Training')

    for images, labels in pbar:
        # Move data to device (GPU/CPU)
        images = images.to(device) # (N, 1, H, W)
        labels = labels.to(device) # (N, 4, H, W)

        # Forward
        outputs = model(images)

        # Loss Function
        class_indices = torch.argmax(labels, dim=1).long() # (N, H, W)

        loss = loss_fn(outputs, class_indices)

        # Backpropagation
        optimizer.zero_grad() # Clean
        loss.backward() # Calculate
        optimizer.step() # Update

        # Use Dice to do monitor (no need gradient here)
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            dice = dice_coefficient_per_class(probs, labels, n_classes=4)

        # Accumulate dice scores
        for key in running_dice.keys():
            running_dice[key] += dice[key]

        running_loss += loss.item()
        batch_count += 1

        # Update progress bar with prostate (class_3) Dice
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Prostate_Dice': f'{dice["class_3"]:.4f}'
        })

    avg_loss = running_loss / batch_count
    avg_dice = {key: value / batch_count for key, value in running_dice.items()}

    return avg_loss, avg_dice


def validate(model, data_loader, loss_fn, device):
    """
    Validate on validation set.
    Similar to training but no backpropagation.

    Args:
        model: The neural network model being evaluated.
        data_loader: Iterable that provides bathes of validation data.
        loss_fn: The loss function used to compute prediction error during validation.
        device: The computation device to run the validation on (e.g., 'cuda' or 'cpu').

    Returns:
        A tuple (avg_loss, avg_dice), where avg_loss (float) is the average loss across
        all batches, and avg_dice (float) is the average Dice coefficient across all batches,
        used for segmentation performance monitoring.
    """
    model.eval()

    running_loss = 0.0
    running_dice = {
        'class_0': 0.0,  # Background
        'class_1': 0.0,  # Peripheral Zone
        'class_2': 0.0,  # Transition Zone
        'class_3': 0.0   # Prostate (MAIN TARGET)
    }
    batch_count = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Validation')

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            class_indices = torch.argmax(labels, dim=1)

            loss = loss_fn(outputs, class_indices)
            probs = torch.softmax(outputs, dim=1)
            dice = dice_coefficient_per_class(probs, labels, n_classes=4)

             # Accumulate dice scores
            for key in running_dice.keys():
                running_dice[key] += dice[key]

            running_loss += loss.item()
            batch_count += 1

            # Update progress bar with prostate (class_3) Dice
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Prostate_Dice': f'{dice["class_3"]:.4f}'
            })

    avg_loss = running_loss / batch_count
    avg_dice = {key: value / batch_count for key, value in running_dice.items()}  

    return avg_loss, avg_dice






def main():
  # Configuration
    DATA_PATH = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data'
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    TARGET_SIZE = (256, 128)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("2D UNet Prostate Segmentation Training")
    print("="*70)
    print(f"Configuration:")
    print(f"  Target Size: {TARGET_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Device: {DEVICE}")
    print(f"\nClass Labels:")
    print(f"  Class 0: Background")
    print(f"  Class 1: Peripheral Zone")
    print(f"  Class 2: Transition Zone")
    print(f"  Class 3: Prostate (TARGET - need Dice >= 0.75)")

    # Load data
    print("\n[1] Loading training data...")
    train_image_paths = sorted(glob.glob(f'{DATA_PATH}/keras_slices_train/*.nii.gz'))
    train_seg_paths = sorted(glob.glob(f'{DATA_PATH}/keras_slices_seg_train/*.nii.gz'))

    X_train = load_data_with_resize(train_image_paths, target_size=TARGET_SIZE, normImage=True)
    y_train = load_labels_with_resize(train_seg_paths, target_size=TARGET_SIZE, n_classes=4)

    print(f"   Loaded {len(X_train)} training images")
    print(f"   Image shape: {X_train.shape}")
    print(f"   Label shape: {y_train.shape}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).unsqueeze(1).float()  # (N,1,H,W)
    y_train_tensor = torch.from_numpy(y_train).permute(0, 3, 1, 2).float()  # (N,4,H,W)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("\n[DEBUG] Checking batch dimensions...")
    for images, labels in train_loader:
        print(f"   Images batch shape: {images.shape}")
        print(f"   Labels batch shape: {labels.shape}")
        print(f"   Labels argmax shape: {torch.argmax(labels, dim=1).shape}")
        break

    # Initialize model
    print(f"\n[2] Initializing model on {DEVICE}...")
    model = UNet(n_channels=1, n_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    print(f"\n[3] Training for {NUM_EPOCHS} epochs...")
    print("-"*70)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        # Print detailed results
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {train_loss:.4f}")
        print(f"  Dice Scores:")
        print(f"    Class 0 (Background):     {train_dice['class_0']:.4f}")
        print(f"    Class 1 (Peripheral):     {train_dice['class_1']:.4f}")
        print(f"    Class 2 (Transition):     {train_dice['class_2']:.4f}")
        print(f"    Class 3 (Prostate):       {train_dice['class_3']:.4f} ★")
        print("-"*70)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_dice': train_dice,
            }
            torch.save(checkpoint, f'unet_epoch_{epoch+1}.pth')
            print(f"   Checkpoint saved: unet_epoch_{epoch+1}.pth")
            print("-"*70)

    # Save final model
    final_checkpoint = {
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_dice': train_dice,
    }
    torch.save(final_checkpoint, 'unet_final.pth')

    print("\n[4] Training complete!")
    print(f"Final Prostate Dice Score: {train_dice['class_3']:.4f}")
    if train_dice['class_3'] >= 0.75:
        print("Yeah! Target achieved! (Dice >= 0.75)")
    else:
        print("No! Target not reached. Consider training longer or adjusting hyperparameters.")
    print(f"Model saved to: unet_final.pth")



if __name__ == "__main__":
    """
    Main function to run the training script.
    """
    main()
