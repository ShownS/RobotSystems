import time
import cv2
import numpy as np
from picamera2 import Picamera2

from picarx_improved import Picarx


def sensor(picam2):
    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return frame_bgr



def interpreter(frame, polarity="dark", roi_frac=0.35, min_area=200):
    h, w = frame.shape[:2]

    y0 = int(h * (1.0 - roi_frac))
    roi = frame[y0:h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if polarity == "dark":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, False, {"y0": y0, "w": w, "h": h}

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return 0.0, False, {"y0": y0, "w": w, "h": h, "area": area}

    M = cv2.moments(c)
    if M["m00"] == 0:
        return 0.0, False, {"y0": y0, "w": w, "h": h, "area": area}

    cx = float(M["m10"] / M["m00"])  
    center = w / 2.0

    offset = (center - cx) / center
    offset = max(-1.0, min(1.0, offset))

    dbg = {"y0": y0, "cx": int(cx), "area": area, "w": w, "h": h}
    return offset, True, dbg


def controller(px, offset, seen, speed=35, scale=25.0):
    if not seen:
        px.stop()
        return 0.0

    angle = scale * float(-offset)
    px.set_dir_servo_angle(angle)
    px.forward(speed)
    return angle


def draw_debug(frame, dbg, offset, seen, angle):
    if frame is None:
        return frame

    h, w = frame.shape[:2]
    y0 = dbg.get("y0", int(h * 0.65))

    cv2.rectangle(frame, (0, y0), (w - 1, h - 1), (0, 255, 0), 2)
    cv2.line(frame, (w // 2, y0), (w // 2, h - 1), (255, 255, 0), 2)

    if seen and "cx" in dbg:
        cx = dbg["cx"]
        cy = y0 + (h - y0) // 2
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    txt = f"seen={seen} offset={offset:+.2f} angle={angle:+.1f}"
    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main():
    px = Picarx()

    px.set_cam_tilt_angle(-30)
    time.sleep(0.4)
    input("Press Enter")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.2)

    polarity = "dark" 
    speed = 30          
    scale = 25.0        
    dt = 0.05

    lost_timeout = 1.2
    lost_t = 0.0

    try:
        while True:
            frame = sensor(picam2)
            offset, seen, dbg = interpreter(frame, polarity=polarity, roi_frac=0.35)

            angle = controller(px, offset, seen, speed=speed, scale=scale)

            # Lost-line stop logic
            if not seen:
                lost_t += dt
                if lost_t >= lost_timeout:
                    px.stop()
                    px.set_dir_servo_angle(0)
                    print("Lost line too long -> stopping.")
                    break
            else:
                lost_t = 0.0

            # Display
            disp = draw_debug(frame, dbg, offset, seen, angle)
            cv2.imshow("PiCar-X Line Follow (press q)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.05)

    finally:
        px.stop()
        px.set_dir_servo_angle(0)
        try:
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
