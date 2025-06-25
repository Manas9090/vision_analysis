#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import face_recognition
import boto3

# Paths
reference_image_path = "target_person.jpg"
input_video_path = "cctv_test_video.mp4"
output_video_path = "processed_output.mp4"
s3_bucket = "your-bucket"
output_s3_path = "outputs/processed_output.mp4"

# Load reference image
print("[INFO] Loading reference image...")
reference_image = face_recognition.load_image_file(reference_image_path)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Open CCTV footage
print("[INFO] Processing video...")
video_capture = cv2.VideoCapture(input_video_path)

# Get video details
fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0
detected_frames = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        if matches[0]:
            detected_frames += 1
            print(f"[ALERT] Target person detected in frame {frame_count}!")
            # Scale back to original frame size
            top, right, bottom, left = top*2, right*2, bottom*2, left*2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "TARGET", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)
    frame_count += 1

video_capture.release()
out.release()

print(f"[INFO] Processing complete. Total frames: {frame_count}, Target detected in {detected_frames} frames.")
print("[INFO] Uploading result to S3...")

# Upload to S3
s3 = boto3.client('s3')
s3.upload_file(output_video_path, s3_bucket, output_s3_path)

print(f"[INFO] Output video uploaded to s3://{s3_bucket}/{output_s3_path}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




