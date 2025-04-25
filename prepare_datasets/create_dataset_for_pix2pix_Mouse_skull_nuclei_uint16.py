import numpy as np
import tifffile

# Download data
import os
import urllib
import zipfile

root_folder = './data_FUPN2V'
path = root_folder + '/Mouse_skull_nuclei/'
zip_file = 'Mouse_skull_nuclei.zip'
url_download_from = 'https://owncloud.mpi-cbg.de/index.php/s/31ZiGfcLmJXZk3X/download'
filename = 'example2_digital_offset300.tif'
save_to_folder = 'fupn2v_pix2pix_uint16'
extension = '.tif'
save_folder_B_train = os.path.join(path, f'{save_to_folder}/B/train')
save_folder_A_train = os.path.join(path, f'{save_to_folder}/A/train')
save_folder_B_test = os.path.join(path, f'{save_to_folder}/B/test')
save_folder_A_test = os.path.join(path, f'{save_to_folder}/A/test')
os.makedirs(save_folder_A_train, exist_ok=True)
os.makedirs(save_folder_B_train, exist_ok=True)
os.makedirs(save_folder_A_test, exist_ok=True)
os.makedirs(save_folder_B_test, exist_ok=True)

# Get data
if not os.path.isdir(root_folder):
    os.mkdir(root_folder)

    zipPath = root_folder + os.sep + zip_file
    if not os.path.exists(zipPath):
        data = urllib.request.urlretrieve(url_download_from, zipPath)
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall(root_folder)

# Noisy Data(Input to network)
image_stack_images = tifffile.imread(path + filename)

# # Normalize between 0-1 and dtype=uint8
# image_stack_images = (image_stack_images - image_stack_images.min()) / \
#                      (image_stack_images.max() - image_stack_images.min())
# image_stack_images = (255 * image_stack_images).astype(np.uint8)

# Ground truth Data(Target of Network)
image_stack_annotation = np.mean(image_stack_images, axis=0)[np.newaxis, ...].astype(np.uint16)
image_stack_annotation = np.repeat(image_stack_annotation, 200, axis=0)
print("Shape of Raw Noisy Image is ", image_stack_images.shape, "; Shape of Target Image is ",
      image_stack_annotation.shape)

# Crop away the top-left portion of the data since this portion will be used later for testing
image_stack_images_train = image_stack_images[:, :, 256:]
image_stack_annotation_train = image_stack_annotation[:, :, 256:]

image_stack_images_test = image_stack_images[:, :, :256]
image_stack_annotation_test = image_stack_annotation[:, :, :256]

# save training images
for i, (img, label) in enumerate(zip(image_stack_images_train, image_stack_annotation_train)):
    img_name = os.path.basename(filename).replace(extension, f'_{i}.tif')
    tifffile.imwrite(os.path.join(save_folder_A_train, img_name), img)
    tifffile.imwrite(os.path.join(save_folder_B_train, img_name), label)
    print(f'[{i}] TRAIN: Saved {os.path.join(save_folder_A_train, img_name)}!')
    print(f'[{i}] TRAIN: Saved {os.path.join(save_folder_B_train, img_name)}!')

    # plt.imshow(img), plt.show()
    # plt.imshow(label), plt.show()

print('Save folder A:', save_folder_A_train)
print('Save folder B:', save_folder_B_train)

# save testing images
for i, (img, label) in enumerate(zip(image_stack_images_test, image_stack_annotation_test)):
    img_name = os.path.basename(filename).replace(extension, f'_{i}.tif')
    tifffile.imwrite(os.path.join(save_folder_A_test, img_name), img)
    tifffile.imwrite(os.path.join(save_folder_B_test, img_name), label)
    print(f'[{i}] TEST: Saved {os.path.join(save_folder_A_test, img_name)}!')
    print(f'[{i}] TEST: Saved {os.path.join(save_folder_B_test, img_name)}!')

    # plt.imshow(img), plt.show()
    # plt.imshow(label), plt.show()

print('Save folder A:', save_folder_A_test)
print('Save folder B:', save_folder_B_test)
