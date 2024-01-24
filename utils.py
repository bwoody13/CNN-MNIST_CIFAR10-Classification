import numpy as np
from scipy import ndimage
from PIL import Image


def find_bounding_box(img):
    non_zero_pixels = np.argwhere(img)
    upper_left = np.min(non_zero_pixels, axis=0)
    lower_right = np.max(non_zero_pixels, axis=0)
    return upper_left[0], lower_right[0], upper_left[1], lower_right[1]


# Helper function to scale and center the digit
def scale_and_center_digit(img, canvas_size=(280, 280), final_size=(28, 28), padding=2):

    img = img.point(lambda p: p > 128 and 255)

    # Find the bounding box of the digit
    top, bottom, left, right = find_bounding_box(np.array(img))

    # Extract the digit
    digit_img = img.crop((left, top, right, bottom))

    # Calculate the new size, preserving the aspect ratio
    width, height = digit_img.size
    max_dim = max(width, height)
       # Calculate new size, keeping the aspect ratio and add padding
    new_size = (max_dim + padding, max_dim + padding)

    # Create a new square image with white background and paste the digit in the center
    square_img = Image.new('L', new_size, color=255)
    square_img.paste(digit_img, ((new_size[0] - width) // 2, (new_size[1] - height) // 2))

    # Resize the image to 20x20, maintaining aspect ratio and resample with antialiasing
    resized_img = square_img.resize((20, 20), Image.LANCZOS)

    # Place the 20x20 image within a 28x28 black canvas to match MNIST format
    final_img = Image.new('L', final_size, color=0)
    final_img.paste(resized_img, ((final_size[0] - 20) // 2, (final_size[1] - 20) // 2))

    return final_img


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
