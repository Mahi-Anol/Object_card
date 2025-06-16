import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import argparse
import sys

def parse_arguments():
    """
    Parse command line arguemnts for directories and split sizes.
    """
    #python src/data_split.py --image-dir D:\card_detection_proj_tashfiq\labeled_data\images --label-dir D:\card_detection_proj_tashfiq\labeled_data\labels --base-path D:\card_detection_proj_tashfiq
    parser = argparse.ArgumentParser(description="Split image and labels into train/val/test directories")
    parser.add_argument('--image-dir', type=str, required=True, help="Path to the directory containing images")
    parser.add_argument('--label-dir', type=str, required=True, help="Path to the directory containing labels")
    parser.add_argument('--base-path', type=str, default=None, help="Base path for data directory creation")
    parser.add_argument('--current', action='store_true', help="Create data directory in current working directory")
    parser.add_argument('--train-size', type=float, default=0.7, help="Proportion of data for training (0.0 to 1.0)")
    parser.add_argument('--val-size', type=float, default=0.2, help="Proportion of data for validation (0.0 to 1.0)")
    parser.add_argument('--test-size', type=float, default=0.1, help="Proportion of data for testing (0.0 to 1.0)")
    
    return parser.parse_args()

def validate_split_sizes(train_size, val_size, test_size):
    """
    Validate the split sizes sum to approx. 1.0
    """
    total = train_size + val_size + test_size
    if not (0.99 <= total <= 1.01):
        print(f"Error: train_size ({train_size}), val_size ({val_size}), and test_size ({test_size}) must sum to 1.0, got {total}")
        sys.exit(1)
    if train_size <= 0 or val_size < 0 or test_size < 0:
        print("Error: Split sizes must be non-negative, with train_size > 0")
        sys.exit(1)


def split_data(image_dir,
               label_dir,
               train_image_dir,
               train_label_dir,
               val_image_dir,
               val_label_dir,
               test_image_dir,
               test_label_dir,
               train_size=0.7,
               val_size=0.2,
               test_size=0.1,
               random_state=11, 
               ):
    """
    Split images and labels into train, validation, and test sets, copying them to the respective directories.
    
    Args:
        image_dir (str or Path): Directory containing images
        label_dir (str or Path): Directory containing labels
        train_image_dir (Path): Destination for train images
        train_label_dir (Path): Destination for train labels
        val_image_dir (Path): Destination for validation images
        val_label_dir (Path): Destination for validation labels
        test_image_dir (Path): Destination for test images
        test_label_dir (Path): Destination for test labels
        train_size (float): Proportion of data for training
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    images = [f for f in os.listdir(image_dir) if f.endswith(('jpg', '.jpeg', 'png'))]
    # print(images)

    if not images:
        print("No images found in the provided image directory. Exiting.")
        sys.exit(1)
    
    # Calculate relative sizes for sklearn's train_test_split
    val_relative = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0
    test_relative = test_size / (val_size + test_size) if (val_size + test_size) > 0 else 0

    # Split into train and temp (val + test)
    train_images, temp_images = train_test_split(images, train_size=train_size/(train_size + val_size + test_size), random_state=random_state)

    # Split temp into val and test
    if temp_images and (val_size + test_size) > 0:
        val_images, test_images = train_test_split(temp_images, train_size=val_relative, random_state=random_state)
    else:
        val_images, test_images = [], []
    
    # Copy train images and labels
    for image in train_images:
        shutil.copy(image_dir / image, train_image_dir)
        label = image.rsplit('.', 1)[0] + '.txt'
        if (label_dir / label).exists():
            shutil.copy(label_dir / label, train_label_dir)
        else:
            print(f"Warning: Label file {label} not found for image {image}")
    
    # Copy validation images and labels
    for image in val_images:
        shutil.copy(image_dir / image, val_image_dir)
        label = image.rsplit('.', 1)[0] + '.txt'
        if (label_dir / label).exists():
            shutil.copy(label_dir / label, val_label_dir)
        else:
            print(f"Warning: Label file {label} not found for image {image}")
    
    # Copy test images and labels
    for image in test_images:
        shutil.copy(image_dir / image, test_image_dir)
        label = image.rsplit('.', 1)[0] + '.txt'
        if (label_dir / label).exists():
            shutil.copy(label_dir / label, test_label_dir)
        else:
            print(f"Warning: Label file {label} not found for image {image}")
    
    print(f"Data split completed: {len(train_images)} train, {len(val_images)} validation, {len(test_images)} test")

    # train_images, val_images = train_test_split(images, test_size=test_size, random_state=random_state)

    # for image in train_images:
    #     shutil.copy(os.path.join(image_dir, image), train_image_dir)
    #     label = image.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
    #     shutil.copy(os.path.join(label_dir, label), train_label_dir)
    
    # for image in val_images:
    #     shutil.copy(os.path.join(image_dir, image), val_image_dir)
    #     label = image.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
    #     shutil.copy(os.path.join(label_dir, label), val_label_dir)

def create_data_dirs(base_path=None, current=False):
    """
    Create a data directory with images and labels subdirectories,
    each containing train, val, and test subdirectories.
    
    Args:
        base_path (str, optional): Path to existing directory. Defaults to None.
        current (bool): If True, create in current directory; if False, create one directory up.
    Returns:
        dict: Dictionary containing paths to train/images, train/labels, val/images, val/labels, test/images, test/labels
    """
    # Determine base directory
    if base_path:
        root_dir = Path(base_path)
    else:
        root_dir = Path.cwd() if current else Path.cwd().parent
    
    # Define directory structure
    data_dir = root_dir / 'data'
    subdirs = ['images', 'labels']
    splits = ['train', 'val', 'test']

    result_paths = {
        'train/images': None, 
        'train/labels': None,
        'val/images': None,
        'val/labels': None,
        'test/images': None,
        'test/labels': None
    }
    
    try:
        # Create data directory
        data_dir.mkdir(exist_ok=True)
        
        # Create images and labels directories with their subdirectories
        for subdir in subdirs:
            subdir_path = data_dir / subdir
            subdir_path.mkdir(exist_ok=True)
            
            for split in splits:
                split_path = subdir_path / split
                split_path.mkdir(exist_ok=True)

                if split in splits:
                    split_path = subdir_path / split
                    split_path.mkdir(exist_ok=True)
        
                    key = f'{split}/{subdir}'
                    result_paths[key] = split_path.resolve()
                
        print(f"Directory structure created successfully at: {data_dir}")
        return result_paths
        
    except Exception as e:
        print(f"Error creating directories: {e}")
        return None

# Example usage
def main():
    # Create in current directory
    args = parse_arguments()
    validate_split_sizes(args.train_size, args.val_size, args.test_size)

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)

    if not image_dir.exists():
        print(f"Image directory {image_dir} does not exist. Exiting.")
        sys.exit(1)
    if not label_dir.exists():
        print(f"Label directory {label_dir} does not exist. Exiting.")
        sys.exit(1)

    # paths = create_data_dirs(base_path='E:/license_detection_proj/field_segment_data')
    paths = create_data_dirs(base_path=args.base_path, current=args.current)
    if not paths:
        print("Failed to create directory sturcture. Exiting")
        sys.exit(1)

    # split data
    split_data(image_dir=image_dir,
               label_dir=label_dir,
               train_image_dir=paths['train/images'],
               train_label_dir=paths['train/labels'],
               val_image_dir=paths['val/images'],
               val_label_dir=paths['val/labels'],
               test_image_dir=paths['test/images'],
               test_label_dir=paths['test/labels'],
               train_size=args.train_size,
               val_size=args.val_size,
               test_size=args.test_size,
               )
    
if __name__ == "__main__":
    main()
    
    
    # Create in specific path
    # create_data_dirs(base_path="/path/to/existing/directory")
    
    # Create one directory up
    # create_data_dirs(current=False)

# image_dir = "E:/license_detection_proj/registration_card_detection/data/registration_card_labeled_data/images"
# label_dir = E:/license_detection_proj/registration_card_detection/data/registration_card_labeled_data/labels
# base_path = E:/license_detection_proj/registration_card_detection/
