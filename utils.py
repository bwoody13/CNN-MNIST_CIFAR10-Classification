import numpy as np
from scipy import ndimage
from PIL import Image


def find_bounding_box(img):
    non_zero_pixels = np.argwhere(img)
    upper_left = np.min(non_zero_pixels, axis=0)
    lower_right = np.max(non_zero_pixels, axis=0)
    return upper_left[0], lower_right[0], upper_left[1], lower_right[1]


# Helper function to scale and center the digit
def scale_and_center_digit(img):
    # Find the bounding box of the digit
    top, bottom, left, right = find_bounding_box(np.array(img))

    # Extract the digit
    digit_img = img.crop((left, top, right, bottom))

    # Calculate the new size, preserving the aspect ratio
    width, height = digit_img.size
    max_dim = max(width, height)
    new_size = (int(width * 20 / max_dim), int(height * 20 / max_dim))  # We scale the image to fit in a 20x20 box

    # Resize the digit to fit in a 20x20 box and pad the rest to make it 28x28
    digit_img = digit_img.resize(new_size, Image.Resampling.LANCZOS)

    # Create a new 28x28 black image
    new_img = Image.new('L', (28, 28), color=0)

    # Paste the resized digit into the center of the 28x28 image
    new_img.paste(digit_img, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))
    return new_img


def center_image(img):
    # Convert the image to a binary array (assuming the image is a single channel grayscale image)
    img_array = np.array(img)
    # Calculate the center of mass of the digit
    cy, cx = ndimage.measurements.center_of_mass(img_array)
    # Calculate the shift required to center the digit
    shiftx = np.round(img_array.shape[1]/2.0-cx).astype(int)
    shifty = np.round(img_array.shape[0]/2.0-cy).astype(int)
    # Create a shifted image using the determined shift
    shifted_image = ndimage.shift(img_array, (shifty, shiftx), mode='constant', cval=0)
    # Convert back to PIL Image
    return Image.fromarray(shifted_image)