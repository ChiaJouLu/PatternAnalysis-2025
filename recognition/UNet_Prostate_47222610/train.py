"""
Train Improved Unet for prostate segmentation.

This is the training script for the HipMRI dataset.
"""
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import glob
from tqdm import tqdm
import cv2

from modules import ImprovedUNet

def load_data_with_resize(image_paths, target_size=(256, 128), normImage=True):
    """
    Load the image and resize it to a uniform size.

    Args:
        image_paths: Paths to the image files to be loaded.
        target_size: Desired spatial dimensions for the output images.
        normImage: If true, each image is normalized. Default is true.

    Returns:
        A NumPy array of shape (N, H, W) containing the processed images.
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

    Args:
        seg_paths: Paths to the segmentation label files to be loaded.
        target_size: Desired output dimensions (height, width) after resizing.
        n_classes: Number of valid classes for one-hot encoding.

    Returns:
        A NumPy array of shape (N, H, W, n_classes) containing the processed
        one-hot encoded labels.
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
    This function trains the model for one epoch on the given data loader 
    with deep supervision.

    Deep supervision weights decrease for deeper layers:
        - Main output: weight = 1.0
        - DSV1: weight = 0.8
        - DSV2: weight = 0.6
        - DSV3: weight = 0.4
        - DSV4: weight = 0.2

    Args:
        model: The neural network model to be trained. Its forward may return a tensor
               or a dict of tensors for deep supervision.
        data_loader: Iterable providing batches of (images, targets). Images are tensors
                     of shape (N, C_in, H, W). Targets are class indices (N, H, W) or
                     one-hot masks depending on loss_fn requirements.
        loss_fn: The criterion used to measure prediction error (e.g., CrossEntropyLoss
                 or Dice-based losses). Must accept per-head logits and matched-size targets.
        optimizer: Optimizer used to update model parameters based on computed gradients.
        device: Computation device (e.g., 'cuda' or 'cpu'). Model, images, and targets
                are moved to this device.
    Returns:
        A tuple (avg_loss, avg_dice), where avg_loss (float) is the average loss across
        all batches, and avg_dice (float) is the average Dice coefficient across all batches,
        used for segmentation performance monitoring.
    """
    model.train()

    running_loss = 0.0
    running_dice = {
        'class_0': 0.0,  # Background
        'class_1': 0.0,  # Body
        'class_2': 0.0,  # Bone
        'class_3': 0.0   # Prostate (MAIN TARGET)
    }
    batch_count = 0

    # Deep supervision weights
    ds_weights = [1.0, 0.8, 0.6, 0.4, 0.2]

    pbar = tqdm(data_loader, desc='Training')

    for images, labels in pbar:
        # Move data to device (GPU/CPU)
        images = images.to(device) 
        labels = labels.to(device) 

        # Forward
        outputs = model(images)

        # Loss Function
        class_indices = torch.argmax(labels, dim=1).long() # (N, H, W)

        if isinstance(outputs, tuple):
            # Deep supervision is active
            main_output, dsv1, dsv2, dsv3, dsv4 = outputs
            
            loss = (ds_weights[0] * loss_fn(main_output, class_indices) +
                   ds_weights[1] * loss_fn(dsv1, class_indices) +
                   ds_weights[2] * loss_fn(dsv2, class_indices) +
                   ds_weights[3] * loss_fn(dsv3, class_indices) +
                   ds_weights[4] * loss_fn(dsv4, class_indices))
        else:
            # Inference mode, no deep supervision
            loss = loss_fn(outputs, class_indices)
            main_output = outputs

        # Backpropagation
        optimizer.zero_grad() 
        loss.backward() 

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Use Dice to do monitor (no need gradient here)
        with torch.no_grad():
            probs = torch.softmax(main_output, dim=1)
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
        data_loader: Iterable that provides batches of validation data.
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


if __name__ == "__main__":
    # Configuration
    data_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    num_epochs = 30
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-5  
    target_size = (256, 128)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print("Loading training data...")
    train_images = sorted(glob.glob(f'{data_path}/keras_slices_train/*.nii.gz'))
    train_labels = sorted(glob.glob(f'{data_path}/keras_slices_seg_train/*.nii.gz'))
    
    X_train = load_data_with_resize(train_images, target_size=target_size)
    y_train = load_labels_with_resize(train_labels, target_size=target_size)
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).unsqueeze(1).float(),
        torch.from_numpy(y_train).permute(0, 3, 1, 2).float()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    print("Initializing model...")
    model = ImprovedUNet(n_channels=1, n_classes=4, deep_supervision=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    best_dice = 0.0

    for epoch in range(num_epochs):
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)

        scheduler.step(train_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Prostate Dice: {train_dice['class_3']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if train_dice['class_3'] > best_dice:
            best_dice = train_dice['class_3']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_dice': train_dice,
            }, 'improved_unet_best.pth')
            print(f"Saved best model (Dice: {best_dice:.4f})")

        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_dice': train_dice,
            }, f'improved_unet_epoch_{epoch+1}.pth')
            print(f"Saved checkpoint: improved_unet_epoch_{epoch+1}.pth")
        
        print()

    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'train_dice': train_dice,
    }, 'improved_unet_final.pth')
        
    print(f"\nTraining complete!")
    print(f"Best Prostate Dice: {best_dice:.4f}")
    print(f"Final Prostate Dice: {train_dice['class_3']:.4f}")
    print(f"Model saved to: improved_unet_final.pth")


