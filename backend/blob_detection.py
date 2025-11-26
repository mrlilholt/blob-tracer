#!/usr/bin/env python3
"""
Blob Detection Script using OpenCV
This script demonstrates how to detect blobs (circular objects) in images and videos.
"""

import cv2
import numpy as np
import sys
import os
import argparse

def create_blob_detector(min_area=50, max_area=5000):
    """Create and configure a SimpleBlobDetector with custom parameters."""
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by Area
    params.filterByArea = True
    params.minArea = min_area      # Minimum blob area
    params.maxArea = max_area    # Maximum blob area
    
    # Filter by Circularity (how circular the blob is)
    params.filterByCircularity = True
    params.minCircularity = 0.3
    
    # Filter by Convexity (how convex the blob is)
    params.filterByConvexity = True
    params.minConvexity = 0.5
    
    # Filter by Inertia (how elongated the blob is)
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    
    # Create detector with parameters
    detector = cv2.SimpleBlobDetector_create(params)
    return detector

def detect_blobs_in_image(image_path):
    """Detect blobs in a given image file."""
    # Read the image
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'.")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create detector
    detector = create_blob_detector()
    
    # Detect blobs
    keypoints = detector.detect(gray)
    
    # Draw detected blobs as red circles
    img_with_keypoints = cv2.drawKeypoints(
        img, keypoints, np.array([]), (0, 0, 255), 
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Print results
    print(f"Found {len(keypoints)} blobs in '{image_path}'")
    for i, kp in enumerate(keypoints):
        print(f"Blob {i+1}: Position ({kp.pt[0]:.1f}, {kp.pt[1]:.1f}), Size: {kp.size:.1f}")
    
    return img_with_keypoints, keypoints

def detect_blobs_in_video(video_path, output_path=None, max_boxes=None, min_area=50):
    """Detect blobs in a video file and optionally save the result."""
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return None
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'.")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path}")
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create detector with custom min area
    detector = create_blob_detector(min_area=min_area)
    
    # Setup video writer if output path is specified
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output video will be saved to: {output_path}")
    
    frame_count = 0
    total_blobs = 0
    previous_keypoints = []
    
    # Initialize trail layer
    trail_layer = np.zeros((height, width, 3), dtype=np.float32)
    trail_decay = 0.9  # How much trail persists (0.9 = keeps 90% each frame)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect blobs
            keypoints = detector.detect(gray)
            
            # Limit number of boxes if specified
            if max_boxes and len(keypoints) > max_boxes:
                # Sort by size and take the largest ones
                keypoints = sorted(keypoints, key=lambda kp: kp.size, reverse=True)[:max_boxes]
            
            total_blobs += len(keypoints)
            
            # Calculate movement percentages
            movement_percentages = []
            for i, kp in enumerate(keypoints):
                movement_pct = 0.0
                if previous_keypoints and i < len(previous_keypoints):
                    # Calculate distance moved from previous frame
                    prev_pt = previous_keypoints[i].pt
                    curr_pt = kp.pt
                    distance = np.sqrt((curr_pt[0] - prev_pt[0])**2 + (curr_pt[1] - prev_pt[1])**2)
                    # Convert distance to percentage (normalize by blob size)
                    movement_pct = min(100.0, (distance / kp.size) * 100) if kp.size > 0 else 0.0
                movement_percentages.append(movement_pct)
            
            # Fade the trail layer
            trail_layer *= trail_decay
            
            # Draw current frame's boxes onto trail layer (thinner lines)
            for i, kp in enumerate(keypoints):
                half_size = int(kp.size)
                x, y = int(kp.pt[0]), int(kp.pt[1])
                
                # Draw white square on trail layer
                top_left = (x - half_size, y - half_size)
                bottom_right = (x + half_size, y + half_size)
                cv2.rectangle(trail_layer, top_left, bottom_right, (255, 255, 255), 1)
            
            # Convert trail layer to uint8 for blending
            trail_layer_uint8 = np.clip(trail_layer, 0, 255).astype(np.uint8)
            
            # Blend trail layer with original frame
            frame_with_trails = cv2.addWeighted(frame, 0.7, trail_layer_uint8, 0.3, 0)
            
            # Draw crisp current boxes and labels on top
            for i, kp in enumerate(keypoints):
                # Calculate square dimensions based on blob size
                half_size = int(kp.size)
                x, y = int(kp.pt[0]), int(kp.pt[1])
                
                # Draw white square around blob (crisp, current frame - thinner lines)
                top_left = (x - half_size, y - half_size)
                bottom_right = (x + half_size, y + half_size)
                cv2.rectangle(frame_with_trails, top_left, bottom_right, (255, 255, 255), 1)
            
            # Draw connecting lines between all blobs (web effect - thinner lines)
            for i, kp1 in enumerate(keypoints):
                for j, kp2 in enumerate(keypoints):
                    if i < j:  # Only draw each line once
                        pt1 = (int(kp1.pt[0]), int(kp1.pt[1]))
                        pt2 = (int(kp2.pt[0]), int(kp2.pt[1]))
                        # Draw white connecting line with some transparency effect
                        cv2.line(frame_with_trails, pt1, pt2, (255, 255, 255), 1)
            
            # Draw labels on top of everything
            for i, kp in enumerate(keypoints):
                half_size = int(kp.size)
                x, y = int(kp.pt[0]), int(kp.pt[1])
                
                # Add blob number and movement percentage
                movement_pct = movement_percentages[i] if i < len(movement_percentages) else 0.0
                label = f"{i+1}: {movement_pct:.1f}%"
                
                # Calculate text position (above the square)
                text_x = x - half_size
                text_y = y - half_size - 10
                
                # Ensure text doesn't go off screen
                if text_y < 20:
                    text_y = y + half_size + 20
                if text_x < 0:
                    text_x = 0
                
                # Draw text background for better visibility (smaller text)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.375, 1)[0]
                cv2.rectangle(frame_with_trails, (text_x, text_y - text_size[1] - 5), 
                             (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
                
                # Draw the text (75% of original size: 0.5 * 0.75 = 0.375)
                cv2.putText(frame_with_trails, label, (text_x + 2, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.375, (0, 0, 0), 1)
            
            # Store current keypoints for next frame movement calculation
            previous_keypoints = list(keypoints)
            
            # Write frame to output video if specified
            if out:
                out.write(frame_with_trails)
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                print(f"Processed frame {frame_count}/{total_frames} - Found {len(keypoints)} blobs")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        # Clean up
        cap.release()
        if out:
            out.release()
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Average blobs per frame: {total_blobs/frame_count if frame_count > 0 else 0:.2f}")
        if output_path:
            print(f"Output saved to: {output_path}")

def detect_blobs_from_webcam():
    """Detect blobs in real-time from webcam feed."""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create detector
    detector = create_blob_detector()
    
    print("Press 'q' to quit, 's' to save current frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam.")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect blobs
        keypoints = detector.detect(gray)
        
        # Draw detected blobs
        img_with_keypoints = cv2.drawKeypoints(
            frame, keypoints, np.array([]), (0, 0, 255), 
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        # Add text overlay
        cv2.putText(img_with_keypoints, f"Blobs detected: {len(keypoints)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Blob Detection', img_with_keypoints)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"blob_detection_frame_{cv2.getTickCount()}.jpg"
            cv2.imwrite(filename, img_with_keypoints)
            print(f"Frame saved as {filename}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to run blob detection."""
    parser = argparse.ArgumentParser(description='Detect blobs in images or videos')
    parser.add_argument('input_file', nargs='?', help='Input image or video file')
    parser.add_argument('-o', '--output', help='Output video file (for video input)')
    parser.add_argument('--max_boxes', type=int, help='Maximum number of blob boxes to draw')
    parser.add_argument('--min_area', type=int, default=50, help='Minimum blob area')
    
    args = parser.parse_args()
    
    if args.input_file:
        # Check if it's a video file
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        file_ext = os.path.splitext(args.input_file)[1].lower()
        
        if file_ext in video_extensions:
            # Process video file
            detect_blobs_in_video(args.input_file, args.output, args.max_boxes, args.min_area)
        else:
            # Process image file
            result = detect_blobs_in_image(args.input_file)
            
            if result is not None:
                img_with_blobs, keypoints = result
                
                # Display the image
                cv2.imshow('Blob Detection Result', img_with_blobs)
                print("Press any key to close the image window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # Save result
                output_filename = f"blob_result_{os.path.basename(args.input_file)}"
                cv2.imwrite(output_filename, img_with_blobs)
                print(f"Result saved as {output_filename}")
    else:
        # No input file provided, use webcam
        print("No input file provided. Starting webcam blob detection...")
        detect_blobs_from_webcam()

if __name__ == "__main__":
    main()