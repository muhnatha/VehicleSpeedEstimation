from ultralytics import YOLO
import cv2 as cv
import torch
import math

# load your trained weights
model = YOLO("Model/best.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


def speedEst(path):
    fps = 30
    prev_centroids = {}
    cv.namedWindow("Speed Estimation", cv.WINDOW_NORMAL)
    cv.resizeWindow("Speed Estimation", 800, 600)

    results = model.track(path, stream=True, persist=True, device = device)
    
    for result in results:
        frame = result.orig_img.copy()
        boxes = result.boxes.xyxy.cpu().tolist()
        track_ids = result.boxes.id.cpu().tolist()

        for (x1, y1, x2, y2),tid in zip(boxes, track_ids):
            cx, cy = (x1+x2)/2, (y1+y2)/2
            if tid in prev_centroids:
                dx = cx - prev_centroids[tid][0]
                dy = cy - prev_centroids[tid][1]
                pixel_dist = math.hypot(dx, dy)

                speed_pix_s = pixel_dist*fps

                cv.putText(frame, f"ID{tid}:{speed_pix_s:.2f} pix/s",
                           (int(cx), int(cy)),
                           cv.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 4)

                prev_centroids[tid] = (cx, cy)
            else:
                prev_centroids[tid] = (cx, cy)
        # show the result
        cv.imshow("Speed Estimation", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    path = "2103099-uhd_3840_2160_30fps.mp4"

    speedEst(path)