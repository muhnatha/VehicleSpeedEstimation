from ultralytics import YOLO
import cv2 as cv
import torch
import numpy as np

# 1. Load YOLO tracker model
model = YOLO("/Users/atika/VehicleSpeedEstimation-main/Model/best.pt") 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Using device: {device}")

# Get class names dictionary {class_id: class_name}
CLASS_NAMES = model.names
print(f"Loaded classes: {CLASS_NAMES}")

# 2. Load MiDaS monocular depth estimation model
try:
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    midas.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    print("MiDaS model loaded successfully.")
except Exception as e:
    print(f"Error loading MiDaS model: {e}")
    print("Please ensure you have an internet connection and the model name is correct.")
    exit()

# 3. Camera intrinsic parameters (focal lengths FX, FY and principal point CX, CY)
FX, FY = 1300.0, 1300.0  # Contoh focal length piksel, sesuaikan dari kalibrasi
TARGET_WIDTH, TARGET_HEIGHT = 960, 540  # Resolusi proses video
DEPTH_SCALE_FACTOR = 0.01  # Skala kedalaman, perlu tuning agar hasil kecepatan realistis

def compute_depth(frame):
    """Run MiDaS model on input frame and return scaled depth map."""
    if frame is None:
        return None
    try:
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # MiDaS expects RGB
        input_batch = midas_transforms(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],  # Resize depth to frame shape (H, W)
                mode="bicubic",
                align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        scaled_depth_map = depth_map * DEPTH_SCALE_FACTOR
        return scaled_depth_map
    except Exception as e:
        print(f"Error in compute_depth: {e}")
        return None

def backproject(u, v, Z, fx, fy, cx, cy):
    """Backproject pixel (u,v) with depth Z to 3D point."""
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)

def speedEst(video_path):
    """Main processing function: track, estimate speed, write output video."""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Could not get FPS from video. Assuming 30.0 FPS.")
        fps = 30.0
    dt = 1.0 / fps
    print(f"Video FPS: {fps:.2f}, dt per frame: {dt:.4f} s")
    print(f"Processing frames at resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out_video = cv.VideoWriter("test4.mp4", fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))
    if not out_video.isOpened():
        print("Error: Could not open video writer.")
        cap.release()
        return

    prev_points_3d = {}  # Track previous 3D points per track id

    try:
        results_generator = model.track(
            source=video_path,
            stream=True,
            persist=True,
            device=device,
            conf=0.3,
            iou=0.5,
        )
    except Exception as e:
        print(f"Error starting YOLO tracking: {e}")
        cap.release()
        out_video.release()
        return

    for frame_idx, result in enumerate(results_generator):
        original_frame = result.orig_img
        frame = cv.resize(original_frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame_height, frame_width = frame.shape[:2]

        cx, cy = frame_width / 2, frame_height / 2
        fx, fy = FX, FY

        depth_map = compute_depth(frame)
        if depth_map is None:
            print(f"Skipping frame {frame_idx} due to depth error.")
            out_video.write(frame)
            continue

        annotated_frame = frame.copy()

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().numpy()

            scale_x = frame_width / result.orig_shape[1]
            scale_y = frame_height / result.orig_shape[0]

            current_ids = set()

            for i, tid in enumerate(track_ids):
                current_ids.add(tid)

                x1, y1, x2, y2 = boxes[i]
                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y

                cls_id = class_ids[i]
                vehicle_name = CLASS_NAMES.get(cls_id, "Unknown")

                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                    Z = depth_map[v, u]
                    if Z <= 0:
                        continue
                else:
                    continue

                P_3d = backproject(u, v, Z, fx, fy, cx, cy)

                speed_text = "N/A"
                if tid in prev_points_3d:
                    dist = np.linalg.norm(P_3d - prev_points_3d[tid])
                    speed_mps = dist / dt if dt > 0 else 0
                    speed_kmh = speed_mps * 3.6
                    speed_text = f"{speed_kmh:.1f} km/h"

                prev_points_3d[tid] = P_3d

                cv.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv.putText(annotated_frame, f"ID:{tid} {vehicle_name}", (int(x1), max(int(y1) - 10, 15)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv.putText(annotated_frame, f"Speed: {speed_text}", (int(x1), max(int(y1) - 30, 35)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Clean up IDs no longer tracked
            old_ids = set(prev_points_3d.keys()) - current_ids
            for oid in old_ids:
                del prev_points_3d[oid]

        out_video.write(annotated_frame)

        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx}...")

    cap.release()
    out_video.release()
    cv.destroyAllWindows()
    print(f"Processing finished. Output saved to test4.mp4")

if __name__ == "__main__":
    video_path = "/Users/atika/VehicleSpeedEstimation-main/AMPLAZ.mp4"  # Ganti path video
    print("-" * 50)
    print("IMPORTANT CALIBRATION NOTE:")
    print(f"Frames resized to: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Using focal lengths: FX={FX}, FY={FY}")
    print(f"Principal point (CX, CY) set to frame center.")
    print(f"Depth scale factor: {DEPTH_SCALE_FACTOR}")
    print("-" * 50)

    speedEst(video_path)
