import cv2
import numpy as np

import pathlib
from enum import Enum

def shift_image(image, direction):
    """
    Shifts an image in the specified direction.
    
    Args:
        image: numpy array representing the image
        direction: TransformDirection enum value
    
    Returns:
        Shifted image as numpy array
    """
    # Get the translation matrix
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, direction.x],
                    [0, 1, direction.y]])
    
    # Apply translation
    shifted = cv2.warpAffine(image, M, (cols, rows))
    return shifted


def shift_image_numpy(image, direction):
    """
    Shifts an image in the specified direction using numpy roll.
    This method wraps pixels around to the other side.
    
    Args:
        image: numpy array representing the image
        direction: TransformDirection enum value
    
    Returns:
        Shifted image as numpy array
    """
    shifted = np.roll(image, direction.y, axis=0)  # Shift rows (y direction)
    shifted = np.roll(shifted, direction.x, axis=1)  # Shift columns (x direction)
    return shifted


class TransformDirection(Enum):
    LEFT = (0, -1)
    RIGHT = (0, 1)
    UP = (-1, 0)
    DOWN = (1, 0)
    UP_LEFT = (-1, -1)
    UP_RIGHT = (-1, 1)
    DOWN_LEFT = (1, -1)
    DOWN_RIGHT = (1, 1)
    UP_UP_LEFT = (-2, -1)
    UP_UP_RIGHT = (-2, 1)
    DOWN_DOWN_LEFT = (2, -1)
    DOWN_DOWN_RIGHT = (2, 1)
    UP_LEFT_LEFT = (-1, -2)
    UP_RIGHT_RIGHT = (-1, 2)
    DOWN_LEFT_LEFT = (1, -2)
    DOWN_RIGHT_RIGHT = (1, 2)

    def __init__(self, y, x):
        self.y = y
        self.x = x

    @property
    def offset(self):
        return (self.y, self.x)




if __name__ == "__main__":
    img_path = pathlib.Path("images/hr.png")  # 226, 226, 3
    save_folder = pathlib.Path("data/")
    img = cv2.imread(img_path)
    
    # Remove the black border
    height, width = img.shape[:2]

    img = cv2.resize(img, (height * 2, width * 2))

    for direction in TransformDirection:
        print(direction.name, direction.offset)
        shifted = shift_image(img, direction)

        # Visualize the image
        cv2.imshow("Shifted", shifted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Remove the black border
        height, width = shifted.shape[:2]
        if direction.y < 0:
            shifted = shifted[-direction.y:, :]
        elif direction.y > 0:
            shifted = shifted[:height-direction.y, :]
            
        if direction.x < 0:
            shifted = shifted[:, -direction.x:]
        elif direction.x > 0:
            shifted = shifted[:, :width-direction.x]

        # Downsample image by 2
        shifted = cv2.resize(shifted, (113, 113))

        # Save image with direction name
        save_path = save_folder / f"{direction.name}.png"
        cv2.imwrite(save_path, shifted)

