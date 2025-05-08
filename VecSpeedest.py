from ultralytics import YOLO
import cv2 as cv
import torch
import numpy as np
import math
import timm
import torchvision.transforms as T

# 1. Load YOLO tracker
model = YOLO("Model/best.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 2. Load MiDaS for monocular depth
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas.to(device).eval()

# 3. Transformation for MiDaS inputs
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# 4. Define camera intrinsics here (pixels)
fx, fy = 1300.0, 1300.0  # focal lengths in pixels
cx, cy = 1920/2, 1080/2  # principal point (image center)

def compute_depth(frame):
    """Run MiDaS and return a depth map (in meters, up to scale)."""
    input_batch = midas_transforms(frame).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    scale_factor = 0.01 # scale based on camera calibration
    depth_map *= scale_factor 
    return depth_map

def backproject(u, v, Z):
    # Backproject 2D pixel coordinates to 3D world coordinates
    # Assuming Z is the depth in meters
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)

def speedEst(path):
    fps = 30.0
    prev_points = {}  

    cv.namedWindow("Speed Estimation", cv.WINDOW_NORMAL)
    cv.resizeWindow("Speed Estimation", 800, 600)

    results = model.track(path, stream=True, persist=True, device=device)

    for result in results:
        frame = result.orig_img
        depth = compute_depth(frame) # Get depth map each frame

        boxes = result.boxes.xyxy.cpu().numpy()
        ids   = result.boxes.id.cpu().numpy()

        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            Z = depth[int(v), int(u)]
            P = backproject(u, v, Z)  

            if tid in prev_points:
                P_prev = prev_points[tid]
                dP = P - P_prev
                dist_m = np.linalg.norm(dP)
                dt = 1.0 / fps
                speed_kmh = (dist_m / dt) * 3.6

                cv.putText(
                    frame,
                    f"ID{tid}: {speed_kmh:.1f} km/h",
                    (int(u), int(v)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    3,
                )

            prev_points[tid] = P

        cv.imshow("Speed Estimation", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    video_path = "2103099-uhd-3840-2160-30fps_PMP29KnQ.mp4"
    speedEst(video_path)
