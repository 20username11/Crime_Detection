import cv2

# Use the absolute path to your video file here
video_path = r'C:\Users\choll\Desktop\BTP\BTP\static\AdobeStock_984294293_Video_HD_Preview.mp4'

# Attempt to open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    print("Success: Video opened correctly.")
cap.release()
