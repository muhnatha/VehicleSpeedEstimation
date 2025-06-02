from ultralytics import YOLO
import cv2 as cv
import torch
import numpy as np
import json
from collections import deque 

# 1. Load YOLO tracker
model = YOLO("Model/yolo11n_best.pt") 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Using device: {device}")

# Get class names from the model
CLASS_NAMES = model.names
print(f"Loaded classes: {CLASS_NAMES}")

# 2. Load MiDaS for monocular depth
try:
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    midas.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    print("MiDaS model loaded successfully.")
except Exception as e:
    print(f"Error loading MiDaS model: {e}")
    print("Please ensure you have an internet connection and the model name is correct.")
    exit()

# 3. Define camera intrinsics here (pixels)
# CX, CY will be dynamically set to the center of the processed (resized) frame.
FX, FY = 1300.0, 1300.0  # Assumption

# This scale factor is highly dependent on scene and MiDaS model.
DEPTH_SCALE_FACTOR = 0.01
MIN_METRIC_DEPTH = 0.1 

# Define target processing resolution
TARGET_WIDTH = 960
TARGET_HEIGHT = 540
VIDEO_OUTPUT_FILENAME = "FKH01a_part_5.mp4"
JSON_OUTPUT_FILENAME = "FKH01a_part_5.json"
DEPTH_PATCH_HALF_SIZE = 3

def compute_depth(frame_to_process):
    if frame_to_process is None:
        return None
    try:
        img_rgb = cv.cvtColor(frame_to_process, cv.COLOR_BGR2RGB) 
        input_batch = midas_transforms(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_to_process.shape[:2], 
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        
        # If `prediction` from MiDaS is disparity (higher value = closer):
        # depth_map_metric = DEPTH_SCALE_FACTOR / (depth_map + 1e-6) # Add epsilon to avoid division by zero
        # If `prediction` is already depth-like (higher value = further):
        depth_map_metric = depth_map * DEPTH_SCALE_FACTOR
        
        return depth_map_metric
    except Exception as e:
        print(f"Error in compute_depth: {e}")
        return None

# Convert 2D pixel coordinates to 3D world coordinates
def backproject(u, v, Z, fx_cam, fy_cam, cx_cam, cy_cam):
    X = (u - cx_cam) * Z / fx_cam
    Y = (v - cy_cam) * Z / fy_cam 
    return np.array([X, Y, Z], dtype=np.float32)

# Track and estimate speed
def speedEst(video_path):
    try:
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        original_fps = cap.get(cv.CAP_PROP_FPS)
        if original_fps == 0:
            print("Warning: Could not get FPS from video. Assuming 30.0 FPS.")
            fps = 30.0
        else:
            fps = original_fps
        
        dt = 1.0 / fps
        print(f"Video FPS: {fps:.2f}, Time per frame (dt): {dt:.4f} s")
        print(f"Processing frames resized to: {TARGET_WIDTH}x{TARGET_HEIGHT}")

    except Exception as e:
        print(f"Error initializing video capture: {e}")
        return

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out_video = cv.VideoWriter(VIDEO_OUTPUT_FILENAME, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))
    if not out_video.isOpened():
        print(f"Error: Could not open video writer for {VIDEO_OUTPUT_FILENAME}")
        cap.release()
        return
    print(f"Output video will be saved to: {VIDEO_OUTPUT_FILENAME}")

    prev_points_3d = {} # Stores previous 3D positions {track_id: P_3d}
    track_statistics_data = {} # Stores {track_id: {"speeds_kmh": [], "class_name": str}}

    try:
        results_generator = model.track(source=video_path, stream=True, persist=True, device=device, conf=0.25, iou=0.5, tracker="botsort.yaml")
    except Exception as e:
        print(f"Error starting YOLO tracking: {e}")
        cap.release()
        out_video.release()
        return
    
    depth_map_metric = None
    FRAMES_TO_SKIP_DEPTH_COMPUTATION = 2

    processed_frame_count = 0
    for frame_idx, result in enumerate(results_generator):
        original_frame = result.orig_img
        
        frame = cv.resize(original_frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
        frame_height, frame_width = frame.shape[:2]
        current_cx = frame_width / 2.0
        current_cy = frame_height / 2.0
        current_fx, current_fy = FX, FY

        depth_map_metric = compute_depth(frame)
        if frame_idx % (FRAMES_TO_SKIP_DEPTH_COMPUTATION + 1) == 0 or depth_map_metric is None:
            current_depth_map_metric = compute_depth(frame)
            if current_depth_map_metric is not None:
                depth_map_metric = current_depth_map_metric 
            elif depth_map_metric is None: 
                print(f"Warning: Skipping frame {frame_idx} due to initial depth computation error.")
                out_video.write(frame) 
                continue
        if depth_map_metric is None:
            print(f"Warning: Skipping frame {frame_idx} as no valid depth map is available.")
            out_video.write(frame)
            continue

        annotated_frame = frame.copy() 

        if result.boxes.id is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().numpy()

            scale_x_orig_to_resized = frame_width / result.orig_shape[1]
            scale_y_orig_to_resized = frame_height / result.orig_shape[0]

            current_tracked_ids_in_frame = set()

            for i, tid in enumerate(track_ids):
                current_tracked_ids_in_frame.add(tid)
                
                x1_orig, y1_orig, x2_orig, y2_orig = boxes_xyxy[i]

                # Scale bounding box to the resized frame dimensions
                x1 = x1_orig * scale_x_orig_to_resized
                y1 = y1_orig * scale_y_orig_to_resized
                x2 = x2_orig * scale_x_orig_to_resized
                y2 = y2_orig * scale_y_orig_to_resized
                
                cls_id = class_ids[i]
                vehicle_name = CLASS_NAMES.get(cls_id, "Unknown")

                # Center of the bounding box in the resized frame
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                Z = 0.0
                if 0 <= v < frame_height and 0 <= u < frame_width:
                    v_start = max(0, v - DEPTH_PATCH_HALF_SIZE)
                    v_end = min(frame_height, v + DEPTH_PATCH_HALF_SIZE + 1)
                    u_start = max(0, u - DEPTH_PATCH_HALF_SIZE)
                    u_end = min(frame_width, u + DEPTH_PATCH_HALF_SIZE + 1)

                    if v_start < v_end and u_start < u_end: 
                        depth_patch = depth_map_metric[v_start:v_end, u_start:u_end]
                        # Consider only physically plausible depths from the scaled map
                        valid_depths = depth_patch[depth_patch > MIN_METRIC_DEPTH] 
                        if valid_depths.size > 0:
                            Z = np.median(valid_depths) # Use median for robustness
                        else:
                            # print(f"ID {tid}: No valid depth in patch. Center ({u},{v}) raw depth: {depth_map_metric[v,u] if 0 <= v < frame_height and 0 <= u < frame_width else 'OOB'}")
                            continue
                    else:
                        # print(f"ID {tid}: Invalid patch coords. Center ({u},{v})")
                        continue 
                else:
                    # print(f"ID {tid}: Bbox center ({u},{v}) out of bounds {frame_width}x{frame_height}")
                    continue 

                if Z <= MIN_METRIC_DEPTH: 
                    # print(f"ID {tid}: Z value {Z:.2f} too small or non-positive. Skipping.")
                    continue
                
                P_3d_current = backproject(u, v, Z, current_fx, current_fy, current_cx, current_cy)
                speed_kmh_str = "N/A"

                if tid in prev_points_3d:
                    P_3d_prev = prev_points_3d[tid]
                    delta_P_3d = P_3d_current - P_3d_prev
                    dist_meters = np.linalg.norm(delta_P_3d)

                    max_dist_per_frame = 300 * dt # Max speed of 300 m/s
                    if dist_meters > max_dist_per_frame and frame_idx > 0 : # frame_idx > 0 to avoid first frame issues
                        print(f"ID {tid}: Implausible distance {dist_meters:.2f}m in dt={dt:.3f}s. Prev Z: {P_3d_prev[2]:.2f}, Curr Z: {P_3d_current[2]:.2f}. Skipping speed calc.")
                    elif dt > 0:
                        speed_mps = dist_meters / dt
                        speed_kmh = speed_mps * 3.6
                        speed_kmh_str = f"{speed_kmh:.1f} km/h"

                        # Store speed for statistics
                        if tid not in track_statistics_data:
                            track_statistics_data[tid] = {"speeds_kmh": [], "class_name": vehicle_name}
                        elif not track_statistics_data[tid]["speeds_kmh"]: 
                            track_statistics_data[tid]["class_name"] = vehicle_name 

                        track_statistics_data[tid]["speeds_kmh"].append(speed_kmh)
                        prev_points_3d[tid] = P_3d_current 
                    else:
                        speed_kmh_str = "FPS error"
                else: 
                    prev_points_3d[tid] = P_3d_current

                # Drawing annotations
                cv.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label_text = f"ID:{tid} {vehicle_name}"
                speed_text = f"Speed: {speed_kmh_str}"
                
                text_x = int(x1)
                base_y = int(y1)
                
                # Position text carefully to avoid overlap if possible
                y_offset_label = -10
                y_offset_speed = -30
                
                # Adjust if text goes off-screen (top)
                label_pos_y = base_y + y_offset_label if base_y + y_offset_label > 15 else base_y + 15
                speed_pos_y = base_y + y_offset_speed if base_y + y_offset_speed > 35 else base_y + 35
                if label_pos_y > speed_pos_y - 10 and label_pos_y < speed_pos_y +10 : 
                    speed_pos_y = label_pos_y + 20 


                cv.putText(annotated_frame, label_text, (text_x, label_pos_y),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if speed_kmh_str != "N/A": 
                    cv.putText(annotated_frame, speed_text, (text_x, speed_pos_y),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Clean up old track IDs from prev_points_3d
            ids_to_remove = set(prev_points_3d.keys()) - current_tracked_ids_in_frame
            for old_id in ids_to_remove:
                del prev_points_3d[old_id]
        
        out_video.write(annotated_frame)
        processed_frame_count +=1
        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx}...")

    cap.release()
    out_video.release()
    cv.destroyAllWindows()
    print(f"Video processing finished. {processed_frame_count} frames saved to {VIDEO_OUTPUT_FILENAME}")

    # --- Calculate and Save Statistics ---
    final_stats = []
    all_average_speeds = []

    for tid, data in track_statistics_data.items():
        if data["speeds_kmh"]:
            min_speed = min(data["speeds_kmh"])
            max_speed = max(data["speeds_kmh"])
            avg_speed = sum(data["speeds_kmh"]) / len(data["speeds_kmh"])
            all_average_speeds.append(avg_speed)
            
            final_stats.append({
                "id": int(tid), 
                "class_name": data["class_name"],
                "min_speed_kmh": round(min_speed, 2),
                "max_speed_kmh": round(max_speed, 2),
                "average_speed_kmh": round(avg_speed, 2),
                "speed_readings_count": len(data["speeds_kmh"])
            })
        else: 
             final_stats.append({
                "id": int(tid),
                "class_name": data["class_name"],
                "min_speed_kmh": None,
                "max_speed_kmh": None,
                "average_speed_kmh": None,
                "speed_readings_count": 0
            })


    overall_avg_of_avgs = None
    if all_average_speeds:
        overall_avg_of_avgs = round(sum(all_average_speeds) / len(all_average_speeds), 2)

    output_json_data = {
        "video_file_processed": video_path,
        "processing_resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        "frames_processed": processed_frame_count,
        "camera_parameters_used": {"FX": FX, "FY": FY, "depth_scale_factor": DEPTH_SCALE_FACTOR},
        "depth_sampling": {"method": "median_of_patch", "patch_half_size": DEPTH_PATCH_HALF_SIZE, "min_metric_depth_threshold": MIN_METRIC_DEPTH},
        "vehicle_summary_statistics": final_stats,
        "overall_average_speed_of_averages_kmh": overall_avg_of_avgs
    }

    try:
        with open(JSON_OUTPUT_FILENAME, 'w') as f:
            json.dump(output_json_data, f, indent=4)
        print(f"Speed statistics saved to {JSON_OUTPUT_FILENAME}")
    except IOError as e:
        print(f"Error saving JSON file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during JSON serialization: {e}")


if __name__ == "__main__":
    video_path = "dataset_cv/FKH01/FKH01a_part_5.mp4"
    
    print("-" * 50)
    print("IMPORTANT CALIBRATION AND USAGE NOTES:")
    print(f"Video frames will be resized to: {TARGET_WIDTH}x{TARGET_HEIGHT} for processing.")
    print(f"Using focal lengths: FX={FX}, FY={FY}.")
    print(f"Principal point (CX, CY) will be set to the center of the {TARGET_WIDTH}x{TARGET_HEIGHT} frame.")
    print("Ensure FX, FY are correctly set for this TARGET resolution.")
    print("If FX, FY were calibrated for a different resolution (e.g., 1920x1080),")
    print("they MUST be scaled. Example for 1920x1080 original calibration:")
    print(f"  FX_scaled = FX_original * ({TARGET_WIDTH}/1920)")
    print(f"  FY_scaled = FY_original * ({TARGET_HEIGHT}/1080)")
    print(f"Using depth scale factor: {DEPTH_SCALE_FACTOR}. This is CRITICAL and scene-dependent.")
    print(f"MiDaS (DPT_Hybrid) typically outputs relative inverse depth. The scaling here assumes a conversion to metric depth.")
    print(f"Minimum plausible metric depth for patch sampling: {MIN_METRIC_DEPTH}m.")
    print(f"Depth for 3D points will be the MEDIAN from a {2*DEPTH_PATCH_HALF_SIZE+1}x{2*DEPTH_PATCH_HALF_SIZE+1} patch.")
    print(f"An implausible distance jump check (max ~300m/s) is added to stabilize speed calculation.")
    print("-" * 50)
    
    speedEst(video_path)