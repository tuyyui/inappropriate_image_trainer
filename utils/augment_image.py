import os
import random
from urllib.request import urlretrieve
from PIL import Image, ImageEnhance
from tqdm import tqdm


def create_yolo_label_for_entire_image(image_path, label_path, class_id=0):
    """
    Create a YOLO label file treating the entire image as a single bounding box.
    YOLO format: class_id x_center y_center width height (all normalized)
    """
    try:
        # Open the image to get actual dimensions
        with Image.open(image_path) as img:
            w, h = img.size

        # Entire image bounding box
        x_center = 0.5
        y_center = 0.5
        width = 1.0
        height = 1.0

        # Write the label
        with open(label_path, "w") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    except Exception as e:
        print(f"Error creating label for {image_path}: {e}")


def augment_image(original_image_path, output_image_dir, output_label_dir, base_name):
    """
    Applies various augmentations to the image, saves them,
    and creates YOLO labels for each augmented copy.
    """
    try:
        image = Image.open(original_image_path)

        augmentations = [
            lambda x: x,
            lambda x: x.rotate(15),
            lambda x: x.rotate(-15),
            lambda x: ImageEnhance.Brightness(x).enhance(1.5),
            lambda x: ImageEnhance.Brightness(x).enhance(0.5),
            lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        ]

        for i, augment in enumerate(augmentations):
            augmented_image = augment(image)
            aug_image_name = f"{base_name}_aug_{i}.jpg"
            aug_image_path = os.path.join(output_image_dir, aug_image_name)

            # Save augmented image
            augmented_image.save(aug_image_path)

            # Corresponding label filename
            label_name = f"{base_name}_aug_{i}.txt"
            aug_label_path = os.path.join(output_label_dir, label_name)

            # Create a YOLO label for each augmented image
            create_yolo_label_for_entire_image(
                aug_image_path, aug_label_path, class_id=0
            )

    except Exception as e:
        print(f"Error processing image {original_image_path}: {e}")


def download_and_augment_images(image_url, root_output_dir, idx, train_ratio=0.8):
    """
    Downloads an image, splits into train/val based on train_ratio,
    creates YOLO labels, then augments and labels those as well.
    """
    # YOLO directory structure
    images_train_dir = os.path.join(root_output_dir, "images", "train")
    images_val_dir = os.path.join(root_output_dir, "images", "val")
    labels_train_dir = os.path.join(root_output_dir, "labels", "train")
    labels_val_dir = os.path.join(root_output_dir, "labels", "val")

    # Make sure all folders exist
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    # Decide train or val by random
    is_train = random.random() < train_ratio
    if is_train:
        out_img_dir = images_train_dir
        out_label_dir = labels_train_dir
    else:
        out_img_dir = images_val_dir
        out_label_dir = labels_val_dir

    # Unique image name (logo_{idx}.jpg)
    img_name = f"logo_{idx}.jpg"
    local_path = os.path.join(out_img_dir, img_name)

    try:
        # Download the image
        urlretrieve(image_url, local_path)

        # Create a YOLO label for the original downloaded image
        base_name = os.path.splitext(img_name)[0]  # "logo_{idx}"
        label_path = os.path.join(out_label_dir, f"{base_name}.txt")
        create_yolo_label_for_entire_image(local_path, label_path, class_id=0)

        # Augment the image
        augment_image(local_path, out_img_dir, out_label_dir, base_name)

    except Exception as e:
        print(f"Failed to download or augment {image_url}: {e}")
