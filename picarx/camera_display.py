import cv2
import time

def open_camera():
    # Try V4L2 first (works for USB cams; sometimes fails for CSI)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap.isOpened():
        return cap, "V4L2 /dev/video0"

    # Fallback: libcamera via GStreamer (works for CSI cameras)
    gst = (
        "libcamerasrc ! "
        "video/x-raw,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! appsink drop=true"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap, "GStreamer libcamerasrc"

    return None, "none"

cap, mode = open_camera()
print("Camera mode:", mode)

if cap is None:
    raise RuntimeError("Could not open camera with V4L2 or GStreamer/libcamera")

last_print = time.time()

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        if time.time() - last_print > 1.0:
            print("No frame read...")
            last_print = time.time()
        continue

    cv2.imshow("Camera Test (press q)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
