import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=15):
    # Check if the video path exists
    if not os.path.exists(video_path):
        print(f"Error: Video path '{video_path}' not found.")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    frame_count = 0
    saved_frame_count = 0

    while True:
        # Read a frame
        ret, frame = cap.read()

        # If frame is not read correctly (end of video), break the loop
        if not ret:
            break

        # Save frame based on the interval
        if frame_count % frame_interval == 0:
            video_filename = os.path.splitext(os.path.basename(video_path))[0]
            frame_filename = os.path.join(output_folder, f"{video_filename}_frame_{saved_frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Finished extracting frames from '{video_path}'.")
    print(f"Saved {saved_frame_count} frames to '{output_folder}'.")

if __name__ == "__main__":
    base_dataset_folder = "output fix"
    frames_parent_folder_name = "extracted_frames"
    save_every_n_frames = 15 

    main_output_base = os.path.join(os.path.dirname(base_dataset_folder), frames_parent_folder_name)

    if not os.path.exists(main_output_base):
        os.makedirs(main_output_base)
        print(f"Created main output base directory: {main_output_base}")

    # # --- Iterate through your dataset structure (can multifolder or just 1 folder) ---
    # for folder_name in os.listdir(base_dataset_folder):
    #     folder_path = os.path.join(base_dataset_folder, folder_name)
    #     if os.path.isdir(folder_path): 
    print(f"\nProcessing folder: {base_dataset_folder}")
    for item_name in os.listdir(base_dataset_folder):
        if item_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')): 
            video_file_path = os.path.join(base_dataset_folder, item_name)
            video_file_basename = os.path.splitext(item_name)[0]
            output_frames_dir_for_video = os.path.join(main_output_base, frames_parent_folder_name, f"frames_{video_file_basename}")

            print(f"  Input video: {video_file_path}")
            print(f"  Output frames to: {output_frames_dir_for_video}")

            extract_frames(video_file_path, output_frames_dir_for_video, save_every_n_frames)
    print("\n--- All videos processed ---")