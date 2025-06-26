import os
import boto3
import traceback
import face_recognition
import cv2

reference_image_path = "Manas-the-small.jpeg"
input_video_path = "Manas-video.mp4"

s3_bucket = "manas-bucket100"
output_s3_path = "outputs/processed_output.mp4"
error_s3_path = "outputs/error_log.txt"

output_video_path = "processed_output.mp4"
error_log_path = "error_log.txt"

def upload_to_s3(local_path, bucket, s3_path):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, bucket, s3_path)
        print(f"[INFO] Uploaded {local_path} to s3://{bucket}/{s3_path}")
    except Exception as e:
        print(f"[ERROR] Failed to upload {local_path} to S3: {e}")

try:
    print("[INFO] Loading reference image...")
    reference_image = face_recognition.load_image_file(reference_image_path)
    reference_encoding = face_recognition.face_encodings(reference_image)[0]

    print("[INFO] Opening video...")
    video_capture = cv2.VideoCapture(input_video_path)

    if not video_capture.isOpened():
        raise Exception("Failed to open input video file.")

    fps = video_capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video properties - FPS: {fps}, Width: {width}, Height: {height}, Total Frames: {total_frames}")

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
                print(f"[ALERT] Target detected in frame {frame_count}")
                top, right, bottom, left = top*2, right*2, bottom*2, left*2
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "TARGET", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

    video_capture.release()
    out.release()

    print(f"[INFO] Processing complete. Total frames: {frame_count}, Target detected in {detected_frames} frames.")
    upload_to_s3(output_video_path, s3_bucket, output_s3_path)

except Exception as e:
    print(f"[ERROR] Processing failed: {e}")
    with open(error_log_path, 'w') as f:
        f.write(traceback.format_exc())
    upload_to_s3(error_log_path, s3_bucket, error_s3_path)
