import os
import boto3
import traceback
import cv2
from deepface import DeepFace

# File paths
reference_image_path = "Manas-the-small.jpeg"
input_video_path = "Manas-video.mp4"

# S3 config
s3_bucket = "manas-bucket100"
output_s3_path = "outputs/processed_output.mp4"
error_s3_path = "outputs/error_log.txt"

# Local temp files
output_video_path = "processed_output.mp4"
error_log_path = "error_log.txt"

# S3 Upload helper
def upload_to_s3(local_path, bucket, s3_path):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, bucket, s3_path)
        print(f"[INFO] Uploaded {local_path} to s3://{bucket}/{s3_path}")
    except Exception as e:
        print(f"[ERROR] S3 Upload failed: {e}")

try:
    print("[INFO] Loading reference image...")
    reference_img = cv2.imread(reference_image_path)

    if reference_img is None:
        raise Exception("Reference image not found!")

    print("[INFO] Starting video processing...")
    cap = cv2.VideoCapture(input_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    detected_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.verify(frame, reference_img, enforce_detection=False)
            if result["verified"]:
                detected_frames += 1
                print(f"[ALERT] Target detected in frame {frame_count}")
                cv2.putText(frame, "TARGET", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        except Exception as inner_e:
            print(f"[WARNING] DeepFace failed on frame {frame_count}: {inner_e}")

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    print(f"[INFO] Completed. Frames: {frame_count}, Target detected in {detected_frames} frames.")
    upload_to_s3(output_video_path, s3_bucket, output_s3_path)

except Exception as e:
    print(f"[ERROR] {e}")
    with open(error_log_path, "w") as f:
        f.write(traceback.format_exc())
    upload_to_s3(error_log_path, s3_bucket, error_s3_path)
