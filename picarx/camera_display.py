import time
import cv2
from picamera2 import Picamera2

def main():
    picam2 = Picamera2()

    # Simple preview config (fast enough)
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    time.sleep(0.2)  # give camera time to warm up

    try:
        while True:
            # Returns RGB image
            frame = picam2.capture_array()
            if frame is None:
                print("No frame from Picamera2")
                continue

            # OpenCV expects BGR for correct colors
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Picamera2 Feed (press q)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
