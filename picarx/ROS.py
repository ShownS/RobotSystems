#!/usr/bin/env python3
import time
import cv2
import numpy as np
from picamera2 import Picamera2

from picarx_improved import Picarx
from rossros import Bus, Producer, ConsumerProducer, Consumer, Timer, runConcurrently


def setup(cam_tilt_deg=-30, size=(640, 480), fmt="RGB888"):
    px = Picarx()
    px.set_cam_tilt_angle(cam_tilt_deg)
    time.sleep(0.4)

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"size": size, "format": fmt})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.2)

    return px, picam2


def camera_sense(picam2: Picamera2):
    frame = picam2.capture_array()
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def camera_interp(frame, polarity="dark", roi_frac=0.35, min_area=200):
    if frame is None:
        return 0.0, False

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
        return 0.0, False

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return 0.0, False

    M = cv2.moments(c)
    if M["m00"] == 0:
        return 0.0, False

    cx = float(M["m10"] / M["m00"])
    center = w / 2.0

    offset = (center - cx) / center
    offset = max(-1.0, min(1.0, offset))
    return offset, True


def ultrasonic_sense(px: Picarx):
    return px.ultrasonic.read()


def ultrasonic_interp(dist_cm, stop_threshold_cm=8.0):
    if dist_cm is None:
        return False
    if dist_cm < 0:
        return False
    return dist_cm > stop_threshold_cm


def control(px: Picarx, line_msg, clear_ahead, speed=40, scale=25.0, steer_sign=+1):
    offset, seen = line_msg

    if not clear_ahead or not seen:
        px.stop()
        px.set_dir_servo_angle(0)
        return None

    angle = steer_sign * scale * (-offset)
    px.set_dir_servo_angle(angle)
    px.forward(speed)
    return None


def make_services(px, picam2, frame_bus, line_bus, dist_bus, clear_bus,
                  term_bus, dt_camera=0.03, dt_line=0.03, dt_ultra=0.05, dt_control=0.07,
                  runtime_s=60, polarity="dark", roi_frac=0.35, min_area=200,
                  stop_threshold_cm=9.0, speed=40, scale=25.0, steer_sign=+1):

    timer = Timer(
        output_buses=term_bus,duration=runtime_s, delay=0.1, name="Runtime Timer")

    cam_service = Producer(
        producer_function=lambda: camera_sense(picam2), output_buses=frame_bus, delay=dt_camera, termination_buses=term_bus,
        name="Camera Sense")

    line_service = ConsumerProducer(
        consumer_producer_function=lambda frame: camera_interp(frame, polarity=polarity, roi_frac=roi_frac, min_area=min_area),
        input_buses=frame_bus, output_buses=line_bus, delay=dt_line, termination_buses=term_bus,
        name="Camera Interpreter")

    ultra_service = Producer(
        producer_function=lambda: ultrasonic_sense(px),
        output_buses=dist_bus, delay=dt_ultra, termination_buses=term_bus, name="Ultrasonic Sense")

    clear_service = ConsumerProducer(
        consumer_producer_function=lambda dist: ultrasonic_interp(dist, stop_threshold_cm=stop_threshold_cm),
        input_buses=dist_bus, output_buses=clear_bus, delay=dt_ultra, termination_buses=term_bus,
        name="Ultrasonic Interpreter")

    drive_service = Consumer(
        consumer_function=lambda line_msg, clear: control(px, line_msg, clear, speed=speed, scale=scale, steer_sign=steer_sign), 
        input_buses=(line_bus, clear_bus), delay=dt_control, termination_buses=term_bus,
        name="Drive Controller")

    return [timer, cam_service, line_service, ultra_service, clear_service, drive_service]


def main():
    px, picam2 = setup()
    input("Press Enter to begin...")

    frame_bus = Bus(None, "Frame Bus")
    line_bus = Bus((0.0, False), "Line Bus")
    dist_bus = Bus(-1, "Distance Bus")
    clear_bus = Bus(False, "Clear Bus")
    term_bus = Bus(False, "Termination Bus")

    services = make_services(
        px, picam2,
        frame_bus, line_bus,
        dist_bus, clear_bus,
        term_bus,
        runtime_s=60,
        speed=40,
        stop_threshold_cm=8.0
    )

    try:
        runConcurrently(services)
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
