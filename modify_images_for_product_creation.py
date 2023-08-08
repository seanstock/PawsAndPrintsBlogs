import os
import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve, label, find_objects
import argparse
from PIL import Image

def reduce_colors(img, r_bit, g_bit, b_bit):
    """Reduce the number of colors in the image using bit manipulation."""
    r = np.right_shift(img[..., 2], 8 - r_bit)
    r = np.left_shift(r, 8 - r_bit)
    g = np.right_shift(img[..., 1], 8 - g_bit)
    g = np.left_shift(g, 8 - g_bit)
    b = np.right_shift(img[..., 0], 8 - b_bit)
    b = np.left_shift(b, 8 - b_bit)
    return np.dstack((b, g, r))

def remove_color(img):
    """Remove contiguous areas of a given color, specified by the top left pixel."""
    top_left_color = img[0, 0, :3]
    mask = np.all(img[:, :, :3] == top_left_color, axis=-1)
    labeled, num_labels = ndimage.label(mask)
    label_sizes = np.bincount(labeled.ravel())
    remove_label = labeled[0, 0]
    large_labels = np.where(label_sizes[1:] > 100000)[0] + 1
    large_mask = np.isin(labeled, large_labels)
    img_rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    img_rgba[:, :, :3] = img[:, :, :3]
    img_rgba[:, :, 3] = 255
    img_rgba[..., 3][large_mask] = 0
    return img_rgba

def is_border_pixel(image):
    """Identify pixels on the border of transparent regions."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    transparent_mask = (image[..., 3] == 0).astype(int)
    transparent_neighbors = convolve(transparent_mask, kernel, mode='constant', cval=0)
    return (transparent_neighbors >= 2)

def remove_fragments(image):
    """Remove small fragments near transparent regions."""
    mask = is_border_pixel(image)
    new_image = image.copy()
    new_image[mask] = (0, 0, 0, 0)  # RGBA channels
    return new_image

def remove_small_contiguous_areas(image, min_size=1, max_size=10000):
    """Remove small contiguous non-transparent regions."""
    non_transparent_mask = image[..., 3] > 0
    labeled, num_labels = label(non_transparent_mask)
    slices = find_objects(labeled)
    for i, slice_ in enumerate(slices):
        region_size = (labeled[slice_] == i + 1).sum()
        region_mask = (labeled[slice_] == i + 1)
        if min_size <= region_size <= max_size:
            image[slice_][region_mask] = (0, 0, 0, 0)
    return image

def process_image(input_path, output_dir):
    """Process an image by reducing colors and removing specific regions."""
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    reduced_img = reduce_colors(img, 4, 4, 4)
    transparent_img = remove_color(reduced_img)
    frag_img = transparent_img.copy()
    for _ in range(4):
        frag_img = remove_fragments(frag_img)
    processed_img = remove_small_contiguous_areas(frag_img)
    # Save images with meaningful filenames
    filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_reduced.png")
    output_path_trans = os.path.join(output_dir, f"{filename}_trans.png")
    output_path_frag = os.path.join(output_dir, f"{filename}_frag.png")
    Image.fromarray(reduced_img).save(output_path)
    Image.fromarray(transparent_img).save(output_path_trans, format='PNG')
    Image.fromarray(processed_img).save(output_path_frag, format='PNG')  

def main():
    """Main function to handle command-line arguments and apply processing."""
    parser = argparse.ArgumentParser(description="Image transparency tool")
    parser.add_argument("input_dir", type=str, help="Path to the input directory")
    args = parser.parse_args()
    if not os.path.isdir(args.input_dir):
        print("Error: Input directory does not exist.")
        return
    output_dir = os.path.join(args.input_dir, "testing1")
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".png"):
            input_path = os.path.join(args.input_dir, filename)
            process_image(input_path, output_dir)
    print("Image processing complete.")

if __name__ == "__main__":
    main()
