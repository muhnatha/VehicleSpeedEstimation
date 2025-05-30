from ultralytics import YOLO
import cv2 as cv
import torch
import numpy as np
import json
import os
# import math # Not strictly needed with np.linalg.norm
# import timm # Not used in the provided snippet
# import torchvision.transforms as T # MiDaS handles its own transforms

# 1. Load YOLO tracker
model = YOLO("Model/best.pt") # Ensure this path is correct
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

# 3. Define camera intrinsics here (pixels) - **Crucial for accurate 3D backprojection**
# These values are placeholders. FX, FY might need scaling if original calibration was for a different resolution.
# CX, CY will be dynamically set to the center of the processed (resized) frame.

# Assumption
FOCAL_LENGTH = 16.0 
SENSOR_WIDTH_MM = 23.5
SENSOR_HEIGHT_MM = 15.6

# Input Size
IN_WIDTH = 1920
IN_HEIGHT = 1080

FX = FOCAL_LENGTH * IN_WIDTH / SENSOR_WIDTH_MM
FY = FOCAL_LENGTH * IN_HEIGHT / SENSOR_HEIGHT_MM

# FX, FY = 1300.0, 1300.0  # Focal lengths in pixels (potentially needs scaling if video is resized)
# Initial CX, CY for reference (e.g. if original video was 1920x1080)
INITIAL_CX, INITIAL_CY = 1920/2, 1080/2

# This scale factor is highly dependent on your scene and MiDaS model.
DEPTH_SCALE_FACTOR = 0.01 # Adjust this based on calibration!

files = []
for file in os.listdir("dataset_cv/amplaz01/"):
    if file.endswith(".mp4"):
        files.append(file)

NAME_FILE = files[4].split(".")[0]

# Define target processing resolution
TARGET_WIDTH = 960
TARGET_HEIGHT = 540
OUTPUT_FILENAME = "out_"+NAME_FILE+".mp4"
JSON_OUTPUT_FILENAME = NAME_FILE+".json"

def compute_depth(frame_to_process): # Renamed input variable for clarity
    """Run MiDaS and return a depth map (scaled)."""
    if frame_to_process is None:
        return None
    try:
        img_rgb = cv.cvtColor(frame_to_process, cv.COLOR_BGR2RGB) # MiDaS expects RGB
        input_batch = midas_transforms(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_to_process.shape[:2], # (H, W) of the resized frame
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        scaled_depth_map = depth_map * DEPTH_SCALE_FACTOR
        return scaled_depth_map
    except Exception as e:
        print(f"Error in compute_depth: {e}")
        return None

def backproject(u, v, Z, fx_cam, fy_cam, cx_cam, cy_cam): # Renamed parameters for clarity
    """
    Backproject 2D pixel coordinates to 3D world coordinates.
    Assumes Z is the depth in metric units (e.g., meters).
    """
    X = (u - cx_cam) * Z / fx_cam
    Y = (v - cy_cam) * Z / fy_cam
    return np.array([X, Y, Z], dtype=np.float32)

def speedEst(video_path):
    """
    Tracks vehicles, estimates their speed, displays bounding boxes with names
    on frames resized to TARGET_WIDTH x TARGET_HEIGHT, saves the output to a video file,
    and exports tracking data to a JSON file.
    """
    try:
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        original_fps = cap.get(cv.CAP_PROP_FPS)

        if original_fps == 0:
            print("Warning: Could not get FPS from video. Assuming 30.0 FPS.")
            fps = 30.0 # Default FPS if not available
        else:
            fps = original_fps

        dt = 1.0 / fps
        print(f"Video FPS: {fps:.2f}, Time per frame (dt): {dt:.4f} s")
        print(f"Processing frames resized to: {TARGET_WIDTH}x{TARGET_HEIGHT}")

    except Exception as e:
        print(f"Error initializing video capture: {e}")
        return

    # Initialize VideoWriter
    fourcc = cv.VideoWriter_fourcc(*'mp4v') # Codec for MP4
    out_video = cv.VideoWriter(OUTPUT_FILENAME, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))
    if not out_video.isOpened():
        print(f"Error: Could not open video writer for {OUTPUT_FILENAME}")
        cap.release()
        return
    print(f"Output video will be saved to: {OUTPUT_FILENAME}")

    prev_points_3d = {} # Stores previous 3D positions {track_id: P_3d}

    # No live display window needed if saving to file
    # cv.namedWindow("Speed Estimation", cv.WINDOW_NORMAL)
    # cv.resizeWindow("Speed Estimation", TARGET_WIDTH, TARGET_HEIGHT)

    # Inisialisasi struktur data untuk menyimpan informasi JSON
    # Strukturnya akan: {track_id: {"class_name": str,
    #                              "speed_readings_kmh": [float],
    #                              "frame_details": [dict]}}
    all_tracks_data_for_json = {}

    try:
        results_generator = model.track(source=video_path, stream=True, persist=True, device=device, conf=0.3, iou=0.5)
    except Exception as e:
        print(f"Error starting YOLO tracking: {e}")
        cap.release()
        out_video.release() # Release video writer on error
        return

    processed_frame_count = 0
    for frame_idx, result in enumerate(results_generator):
        original_frame = result.orig_img

        frame = cv.resize(original_frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame = frame.copy()

        frame_height, frame_width = frame.shape[:2]

        current_cx = frame_width / 2.0
        current_cy = frame_height / 2.0
        current_fx, current_fy = FX, FY

        depth_map = compute_depth(frame)
        if depth_map is None:
            print(f"Warning: Skipping frame {frame_idx} due to depth computation error.")
            # Write the unprocessed (but resized) frame to keep video length consistent
            out_video.write(frame)
            continue

        # Initialize frame with detections even if no IDs for consistent output
        annotated_frame = frame.copy()

        if result.boxes.id is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().numpy()

            scale_x = frame_width / result.orig_shape[1]
            scale_y = frame_height / result.orig_shape[0]

            current_tracked_ids_in_frame = set() # Untuk membersihkan prev_points_3d

            for i, tid in enumerate(track_ids):
                current_tracked_ids_in_frame.add(tid)

                x1_orig, y1_orig, x2_orig, y2_orig = boxes_xyxy[i]

                x1 = x1_orig * scale_x
                y1 = y1_orig * scale_y
                x2 = x2_orig * scale_x
                y2 = y2_orig * scale_y

                cls_id = class_ids[i]
                vehicle_name = CLASS_NAMES.get(cls_id, "Unknown")

                # Inisialisasi atau update data untuk JSON jika track ID baru atau nama kelas berubah
                if tid not in all_tracks_data_for_json:
                    all_tracks_data_for_json[tid] = {
                        "class_name": vehicle_name,
                        "speed_readings_kmh": [],
                        "frame_details": []
                    }
                all_tracks_data_for_json[tid]["class_name"] = vehicle_name # Update jika ada perubahan

                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                # Variabel untuk menyimpan data JSON frame ini
                current_Z_meters = None
                current_P_3d_world = None
                current_speed_kmh = None
                speed_kmh_str_display = "N/A"

                if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                    Z = depth_map[v, u]
                    if Z > 0:
                        current_Z_meters = float(Z)
                        P_3d_current = backproject(u, v, current_Z_meters, current_fx, current_fy, current_cx, current_cy)
                        current_P_3d_world = P_3d_current.tolist()

                        if tid in prev_points_3d:
                            P_3d_prev = prev_points_3d[tid]
                            delta_P_3d = P_3d_current - P_3d_prev
                            dist_meters= np.linalg.norm(delta_P_3d)
                            dist_meters = float(dist_meters)

                            if dt > 0:
                                speed_mps = dist_meters / dt
                                speed_kmh_calc = speed_mps * 3.6
                                speed_kmh_str_display = f"{speed_kmh_calc:.1f} km/h"
                                current_speed_kmh = float(round(speed_kmh_calc, 2))
                                all_tracks_data_for_json[tid]["speed_readings_kmh"].append(current_speed_kmh)
                            else:
                                speed_kmh_str_display = "FPS error"

                        prev_points_3d[tid] = P_3d_current

                cv.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label_text = f"ID:{tid} {vehicle_name}"
                text_x = int(x1)
                text_y_id_name = int(y1) - 10
                text_y_speed = int(y1) - 30
                text_y_id_name = max(text_y_id_name, 15)
                text_y_speed = max(text_y_speed, 35)

                cv.putText(annotated_frame, label_text, (text_x, text_y_id_name),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv.putText(annotated_frame, speed_kmh_str_display, (text_x, text_y_speed),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            ids_to_remove = set(prev_points_3d.keys()) - current_tracked_ids_in_frame
            for old_id in ids_to_remove:
                del prev_points_3d[old_id]

        out_video.write(annotated_frame) # Write the anotated frame
        processed_frame_count +=1
        if frame_idx % 30 == 0: #Print progress every 30 frames
            print(f"Processed frame {frame_idx}...")

    # Release resources
    cap.release()
    out_video.release()
    cv.destroyAllWindows() # Clean up any OpenCV windows if they were created
    print(f"Processing finished. {processed_frame_count} frames saved to {OUTPUT_FILENAME}")

    # --- Bagian Pembuatan dan Penyimpanan Data JSON ---
    final_vehicle_stats_for_json = []
    all_average_speeds_collected = []

    for tid, data in all_tracks_data_for_json.items():
        min_s, max_s, avg_s = None, None, None

        if data["speed_readings_kmh"]:
            min_s_calc = min(data["speed_readings_kmh"])
            max_s_calc = max(data["speed_readings_kmh"])
            avg_s_calc = sum(data["speed_readings_kmh"]) / len(data["speed_readings_kmh"])

            min_s = float(round(min_s_calc, 2))
            max_s = float(round(max_s_calc, 2))
            avg_s = float(round(avg_s_calc, 2))
            all_average_speeds_collected.append(avg_s)

        final_vehicle_stats_for_json.append({
            "id": int(tid),
            "class_name": data["class_name"],
            "min_speed_kmh": min_s,
            "max_speed_kmh": max_s,
            "average_speed_kmh": avg_s,
            "speed_readings_count": len(data["speed_readings_kmh"]),
            "total_frames_detected": len(data["frame_details"])
        })

    overall_avg_of_all_vehicle_averages = None
    if all_average_speeds_collected:
        overall_calc = sum(all_average_speeds_collected) / len(all_average_speeds_collected)
        overall_avg_of_all_vehicle_averages = float(round(overall_calc, 2))

    # Struktur data JSON akhir
    output_json_content = {
        "video_file_processed": video_path,
        "processing_resolution_wh": [int(TARGET_WIDTH), int(TARGET_HEIGHT)],
        "frames_processed_count": int(processed_frame_count),
        "video_fps_used_for_calc": float(round(fps, 2)),
        "time_delta_per_frame_seconds": float(round(dt, 4)),
        "camera_parameters_used": {
            "FX_pixels": float(FX),
            "FY_pixels": float(FY),
            "CX_pixels_approx": f"center of {int(TARGET_WIDTH)}px width",
            "CY_pixels_approx": f"center of {int(TARGET_HEIGHT)}px height",
            "midas_depth_scale_factor": float(DEPTH_SCALE_FACTOR)
        },
        "vehicle_summary_statistics": final_vehicle_stats_for_json,
        "overall_average_speed_of_vehicle_averages_kmh": overall_avg_of_all_vehicle_averages,
    }

    try:
        with open(JSON_OUTPUT_FILENAME, 'w') as f:
            json.dump(output_json_content, f, indent=4)
        print(f"Speed statistics and detailed tracking data saved to {JSON_OUTPUT_FILENAME}")
    except IOError as e:
        print(f"Error saving JSON file: {e}")
    except TypeError as e: # Menangkap TypeError secara spesifik jika masih ada
        print(f"Still a TypeError during JSON serialization: {e}")
        print("Problematic data structure might be (first 500 chars):", repr(output_json_content)[:500])
    except Exception as e:
        print(f"An unexpected error occurred during JSON serialization: {e}")


if __name__ == "__main__":
    video_path ="/content/gdrive/MyDrive/dataset/dataset_cv/amplaz01/"+NAME_FILE+".mp4" # Replace with your video file path
    # video_path = 0 # For webcam

    print("-" * 50)
    print("IMPORTANT CALIBRATION NOTE:")
    print(f"Video frames will be resized to: {TARGET_WIDTH}x{TARGET_HEIGHT} for processing.")
    print(f"Using UNMODIFIED focal lengths: FX={FX}, FY={FY}.")
    print(f"Principal point (CX, CY) will be set to the center of the {TARGET_WIDTH}x{TARGET_HEIGHT} frame.")
    print("If FX, FY were calibrated for a different resolution (e.g., 1920x1080),")
    print("they should ideally be scaled for the new resolution to maintain geometric accuracy.")
    print(f"E.g., if calibrated for 1920x1080: FX_scaled = FX * ({TARGET_WIDTH}/1920), FY_scaled = FY * ({TARGET_HEIGHT}/1080).")
    print(f"Using depth scale factor: {DEPTH_SCALE_FACTOR}")
    print("These values are CRITICAL for accurate 3D reconstruction and speed estimation.")
    print("-" * 50)

    speedEst(video_path)