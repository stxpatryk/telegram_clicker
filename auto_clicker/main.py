import cv2
from config import BLUM
import pyautogui
import time
import numpy as np
from datetime import datetime
from mss import mss

from typing import Tuple

ROI = Tuple[int, int, int, int]  # e.g. (2063, 4, 1377, 1314)


def take_full_screenshot() -> np.array:
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)


def take_selection_screenshot(roi: tuple = None) -> np.array:
    with mss() as sct:
        monitor = {
            "top": roi[1],  # y-coordinate
            "left": roi[0],  # x-coordinate
            "width": roi[2],  # width of the region
            "height": roi[3],  # height of the region
        }

        screenshot = sct.grab(monitor)
        img = np.array(screenshot)

        return img[:, :, :3]


def select_roi(image: np.array) -> tuple:
    roi = cv2.selectROI("Select ROI", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    print(f"Selected ROI: {roi}")
    return roi


def crop(image: np.array, roi: ROI) -> np.array:
    return image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]


def label_image(image: np.array, text: str) -> np.array:
    img = np.array(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # Green
    thickness = 2
    position = (10, 30)
    cv2.putText(img, text, position, font, font_scale, font_color, thickness)
    return img


def main():
    image = take_full_screenshot()
    roi = select_roi(image)

    while True:
        start_time = time.time()
        image_roi = take_selection_screenshot(roi)
        elapsed_time = time.time() - start_time

        text = f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}, FPS: {1/elapsed_time:.2f}"
        print(text)
        img = label_image(image_roi, text)

        cv2.imshow("Cropped ROI", img)

        # Check for 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close all OpenCV windows when done
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    print("Done.")
