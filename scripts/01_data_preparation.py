import os
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image

def load_and_verify_images(image_directory):
    images = []
    labels = []
    for class_folder in os.listdir(image_directory):
        class_folder_path = os.path.join(image_directory, class_folder)
        if os.path.isdir(class_folder_path):
            for img_file in os.listdir(class_folder_path):
                try:
                    img_path = os.path.join(class_folder_path, img_file)
                    img = load_img(img_path)  # This loads an image and checks if it's corrupted
                    images.append(img_path)
                    labels.append(int(class_folder))  # Assumes folder names are '0', '1'
                except (IOError, SyntaxError) as e:
                    print('Bad file:', img_file)  # Report corrupted file
    return images, labels

# Define your dataset directories
train_dir = 'C:\\Users\\X1 CARBON\\Desktop\\GlaucomaDetectionProject\\data\\train'
val_dir = 'C:\\Users\\X1 CARBON\\Desktop\\GlaucomaDetectionProject\\data\\val'
test_dir = 'C:\\Users\\X1 CARBON\\Desktop\\GlaucomaDetectionProject\\data\\test'

# Load and verify datasets
train_images, train_labels = load_and_verify_images(train_dir)
val_images, val_labels = load_and_verify_images(val_dir)
test_images, test_labels = load_and_verify_images(test_dir)
