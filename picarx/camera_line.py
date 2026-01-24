import time
import cv2
import numpy as np
from vilib import Vilib
from picarx_improved import Picarx

# -------------------------
# 1) SENSOR: get a frame
# -------------------------
def sensor(cap):
    """
    Returns a BGR frame (numpy array) or None if capture fails.
    """
    ok, frame = cap.read()
    return frame if ok else None


def interpreter(frame, polarity="dark"):
    """
    Returns:
      (offset, seen)
    offset in [-1,1], positive means line is LEFT of robot.
    seen is True if line detected.
    """
    if frame is None:
        return 0.0, False

    h, w = frame.shape[:2]
    roi = frame[int(h * 0.5):h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if polarity == "dark":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, False

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return 0.0, False

    M = cv2.moments(c)
    if M["m00"] == 0:
        return 0.0, False

    cx = float(M["m10"] / M["m00"])  # 0..w

    center = w / 2.0
    # Map x to [-1,1], positive = line left => (center - cx)/center
    offset = (center - cx) / center
    offset = max(-1.0, min(1.0, offset))
    return offset, True


def controller(px, offset, seen, speed=12, scale=25.0):
    """
    scale converts offset [-1,1] -> steering angle in degrees.
    returns commanded angle.
    """
    if not seen:
        px.stop()
        return 0.0

    angle = scale * float(offset)
    px.set_dir_servo_angle(angle)
    px.forward(speed)
    return angle


def main():
    px = Picarx()

    # Vilib camera + web display (matches your example)
    Vilib.camera_start(vflip=False, hflip=False)
    Vilib.display(local=True, web=True)  # :contentReference[oaicite:1]{index=1}

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    polarity = "dark"        # change to "light" for white line on dark floor
    speed = 12               # slow
    turn_angle = 18          # gentle turns
    dt = 0.05

    lost_timeout = 1.2       # seconds to tolerate "lost"
    lost_t = 0.0

    try:
        while True:
            frame = sensor(cap)
            state = interpreter(frame, polarity=polarity)

            angle = controller(px, state, speed=speed, turn_angle=turn_angle)

            if state == "lost":
                lost_t += dt
                if lost_t >= lost_timeout:
                    px.stop()
                    px.set_dir_servo_angle(0)
                    print("Lost line too long -> stopping.")
                    break
            else:
                lost_t = 0.0

            print(f"state={state:7s} angle={angle:+3d} lost={lost_t:.2f}s")
            time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        px.stop()
        px.set_dir_servo_angle(0)
        try:
            cap.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
