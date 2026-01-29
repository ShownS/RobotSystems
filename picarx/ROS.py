import time
import cv2
import numpy as np
from picamera2 import Picamera2
from threading import Event
from concurrent.futures import ThreadPoolExecutor
from readerwriterlock import rwlock


from picarx_improved import Picarx


class Bus:
    def __init__(self, initial=None):
        self.message = initial
        self.lock = rwlock.RWLockWriteD() 

    def write(self, message):
        with self.lock.gen_wlock():
            self.message = message

    def read(self):
        with self.lock.gen_rlock():
            return self.message



def handle_exception(future):
    exc = future.exception()
    if exc is not None:
        print(f"Exception in worker thread: {exc}")

def ultrasonic_loop(px: Picarx, sonic_bus: Bus, delay: float, shutdown_event: Event):
    while not shutdown_event.is_set():
        dist = round(px.ultrasonic.read(), 2)
        sonic_bus.write(dist)
        time.sleep(delay)


def sensor_loop(picam2: Picamera2, frame_bus: Bus, delay: float, shutdown_event: Event):
    while not shutdown_event.is_set():
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_bus.write(frame)
        time.sleep(delay)


def interpreter_loop(frame_bus: Bus, interp_bus: Bus, delay: float, shutdown_event: Event,
                     polarity="dark", roi_frac=0.35, min_area=200):
    while not shutdown_event.is_set():
        frame = frame_bus.read()
        if frame is None:
            time.sleep(delay)
            continue

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

        seen = False
        offset = 0.0

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area >= min_area:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    center = w / 2.0
                    offset = (center - cx) / center
                    offset = max(-1.0, min(1.0, offset))
                    seen = True

        interp_bus.write((offset, seen))
        time.sleep(delay)


def control_loop(px: Picarx, interp_bus: Bus, sonic_bus: Bus, delay: float, shutdown_event: Event,
                 speed=30, scale=25.0, steer_sign=+1):
    while not shutdown_event.is_set():
        msg = interp_bus.read()
        if msg is None:
            time.sleep(delay)
            continue

        dist = sonic_bus.read()

        offset, seen = msg

        if not seen:
            px.stop()
            px.set_dir_servo_angle(0)
            time.sleep(delay)
            continue

        angle = steer_sign * scale * (-offset)
        px.set_dir_servo_angle(angle)
        if dist > 8.0:
            px.forward(speed)
        time.sleep(delay)



def main():
    px = Picarx()

    px.set_cam_tilt_angle(-30)
    time.sleep(0.4)
    input("Camera tilted down ~30°. Press Enter to begin.")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.2)

    polarity = "dark"
    speed = 40
    scale = 25.0
    steer_sign = +1
    trig, echo = 'D2', 'D3'

    sensor_dt = 0.03
    sonic_dt = 0.03
    interp_dt = 0.03
    control_dt = 0.05

    frame_bus = Bus(None)
    interp_bus = Bus(None)
    sonic_bus = Bus(None)

    shutdown_event = Event()

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    sensor_loop, picam2, frame_bus, sensor_dt, shutdown_event
                ),
                executor.submit(
                    interpreter_loop, frame_bus, interp_bus, interp_dt,
                    shutdown_event, polarity, 0.35, 200
                ),
                executor.submit(
                    ultrasonic_loop, px, sonic_bus, sonic_dt, shutdown_event
                ),
                executor.submit(
                    control_loop, px, interp_bus, sonic_bus, control_dt,
                    shutdown_event, speed, scale, steer_sign
                ),
            ]

            for f in futures:
                f.add_done_callback(handle_exception)

            while not shutdown_event.is_set():
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("Shutting down")
        shutdown_event.set()

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
