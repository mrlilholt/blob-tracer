#!/usr/bin/env python3
"""
Test script to create sample images with blobs for testing blob detection.
"""

import cv2
import numpy as np

def create_test_image_with_circles():
    """Create a test image with various circular blobs."""
    # Create a blank white image
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw some circles of different sizes and colors
    circles = [
        ((150, 150), 40, (0, 0, 0)),      # Black circle
        ((400, 200), 60, (100, 100, 100)), # Gray circle
        ((600, 150), 30, (50, 50, 50)),    # Dark gray circle
        ((200, 400), 80, (0, 0, 0)),       # Large black circle
        ((500, 450), 25, (80, 80, 80)),    # Small gray circle
        ((350, 350), 50, (40, 40, 40)),    # Medium circle
    ]
    
    for center, radius, color in circles:
        cv2.circle(img, center, radius, color, -1)
    
    return img

def create_test_image_with_noise():
    """Create a test image with circles and some noise."""
    img = create_test_image_with_circles()
    
    # Add some random noise
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Add some random rectangles (should not be detected as blobs)
    cv2.rectangle(img, (50, 50), (100, 120), (60, 60, 60), -1)
    cv2.rectangle(img, (650, 400), (750, 500), (70, 70, 70), -1)
    
    return img

def main():
    """Create test images for blob detection."""
    # Create test images
    test_img1 = create_test_image_with_circles()
    test_img2 = create_test_image_with_noise()
    
    # Save the images
    cv2.imwrite('test_circles.jpg', test_img1)
    cv2.imwrite('test_circles_noisy.jpg', test_img2)
    
    print("Created test images:")
    print("  - test_circles.jpg (clean circles)")
    print("  - test_circles_noisy.jpg (circles with noise and rectangles)")
    print("\nYou can now test blob detection with:")
    print("  python blob_detector.py test_circles.jpg")
    print("  python blob_detector.py test_circles_noisy.jpg")

if __name__ == "__main__":
    main()