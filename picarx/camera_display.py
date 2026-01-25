import cv2
from flask import Flask, Response

app = Flask(__name__)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def gen():
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

@app.route("/")
def index():
    return "<h3>PiCar-X Camera Stream</h3><img src='/mjpg'>"

@app.route("/mjpg")
def mjpg():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # Use 0.0.0.0 so other computers can open it
    app.run(host="0.0.0.0", port=9000, threaded=True)
