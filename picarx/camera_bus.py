import time
import cv2
import numpy as np
from picamera2 import Picamera2

from picarx_improved import Picarx


class Bus():
    def __init__(self, initial=None):
        self.message = initial
    
    def write(self, message):
        self.message = message

    def read(self):
        return self.message


def sensor(frame_bus, picam2):
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_bus.write(frame)
        time.sleep(0.05)

def interpreter(frame_bus, offset_bus, polarity="dark", roi_frac=0.35, min_area=200):
    while True:
        frame = frame_bus.read()
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

        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)

        cx = float(M["m10"] / M["m00"])  
        center = w / 2.0

        offset = (center - cx) / center
        offset = max(-1.0, min(1.0, offset))

        offset_bus.write(offset)

        time.sleep(0.05)


def controller(px, offset_bus, speed=35, scale=25.0):
    while True:    
        offset = offset_bus.read()
        angle = scale * float(-offset)
        px.set_dir_servo_angle(angle)
        px.forward(speed)
        time.sleep(0.05)


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
    speed = 35          
    scale = 30.0        
    dt = 0.05

    lost_timeout = 1.2
    lost_t = 0.0

    try:
        while True:
            frame = sensor(picam2)
            offset, seen, dbg = interpreter(frame, polarity=polarity, roi_frac=0.65)

            angle = controller(px, offset, seen, speed=speed, scale=scale)

            if not seen:
                lost_t += dt
                if lost_t >= lost_timeout:
                    px.stop()
                    px.set_dir_servo_angle(0)
                    print("Lost line too long")
                    break
            else:
                lost_t = 0.0

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
