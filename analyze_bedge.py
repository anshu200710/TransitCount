import cv2
import numpy as np

def analyze_badge(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image")
        return
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # We want to find the orange and green parts
    # Saffron/Orange range (approx)
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    
    # Green range (approx)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    orange_pct = (np.sum(mask_orange > 0) / mask_orange.size) * 100
    green_pct = (np.sum(mask_green > 0) / mask_green.size) * 100
    
    print(f"Orange pixels: {orange_pct:.2f}%")
    print(f"Green pixels: {green_pct:.2f}%")
    
    # Let's find the mean HSV of the masked areas to refine
    if orange_pct > 0:
        mean_orange = cv2.mean(hsv, mask=mask_orange)
        print(f"Mean Orange HSV: {mean_orange[:3]}")
        
    if green_pct > 0:
        mean_green = cv2.mean(hsv, mask=mask_green)
        print(f"Mean Green HSV: {mean_green[:3]}")

if __name__ == "__main__":
    analyze_badge("bedge img.jpeg")
