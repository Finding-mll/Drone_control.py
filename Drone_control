import numpy as np
import cv2
import time
from djitellopy import Tello

# Set points (center of the frame coordinates in pixels)
rifX = 960 / 2
rifY = 720 / 2

# PI constant
Kp_X = 0.1
Ki_X = 0.0
Kp_Y = 0.2
Ki_Y = 0.0

# Loop time
Tc = 0.05

# PI terms initialized
integral_X = 0
error_X = 0
previous_error_X = 0
integral_Y = 0
error_Y = 0
previous_error_Y = 0

centroX_pre = rifX
centroY_pre = rifY

# Neural network
net = cv2.dnn.readNetFromCaffe("models/MobileNetSSD_deploy.prototxt.txt", "models/MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

drone = Tello()  # Declaring drone object
time.sleep(2.0)  # Waiting 2 seconds
print("Connecting...")
drone.connect()
print("BATTERY: ")
print(drone.get_battery())
time.sleep(1.0)
print("Loading...")
drone.streamon()  # Start camera streaming
print("Takeoff...")
drone.takeoff()  # Drone takeoff

while True:
    start = time.time()
    frame = drone.get_frame_read().frame

    cv2.circle(frame, (int(rifX), int(rifY)), 1, (0, 0, 255), 10)

    h, w, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame,
                                 0.007843, (180, 180), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):

        idx = int(detections[0, 0, i, 1])
        confidence = detections[0, 0, i, 2]

        if CLASSES[idx] == "person" and confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          colors[idx], 2)
            # Draw the center of the person detected
            centroX = (startX + endX) / 2
            centroY = (2 * startY + endY) / 3

            centroX_pre = centroX
            centroY_pre = centroY

            cv2.circle(frame, (int(centroX), int(centroY)), 1, (0, 0, 255), 10)

            error_X = -(rifX - centroX)
            error_Y = rifY - centroY

            cv2.line(frame, (int(rifX), int(rifY)), (int(centroX), int(centroY)), (0, 255, 255), 5)

            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

            # PI controller
            integral_X = integral_X + error_X * Tc  # Updating integral PID term
            uX = Kp_X * error_X + Ki_X * integral_X  # Updating control variable uX
            previous_error_X = error_X  # Update previous error variable

            integral_Y = integral_Y + error_Y * Tc  # Updating integral PID term
            uY = Kp_Y * error_Y + Ki_Y * integral_Y
            previous_error_Y = error_Y

            drone.send_rc_control(0, 0, round(uY), round(uX))
            # Break when a person is recognized
            break

        else:  # If nobody is recognized take as reference centerX and centerY of the previous frame
            centroX = centroX_pre
            centroY = centroY_pre
            cv2.circle(frame, (int(centroX), int(centroY)), 1, (0, 0, 255), 10)

            error_X = -(rifX - centroX)
            error_Y = rifY - centroY

            cv2.line(frame, (int(rifX), int(rifY)), (int(centroX), int(centroY)), (0, 255, 255), 5)

            integral_X = integral_X + error_X * Tc  # Updating integral PID term
            uX = Kp_X * error_X + Ki_X * integral_X  # Updating control variable uX
            previous_error_X = error_X  # Update previous error variable

            integral_Y = integral_Y + error_Y * Tc  # Updating integral PID term
            uY = Kp_Y * error_Y + Ki_Y * integral_Y
            previous_error_Y = error_Y

            drone.send_rc_control(0, 0, round(uY), round(uX))

            continue

    cv2.imshow("Frame", frame)

    end = time.time()
    elapsed = end - start
    if Tc - elapsed > 0:
        time.sleep(Tc - elapsed)
    end_ = time.time()
    elapsed_ = end_ - start
    fps = 1 / elapsed_
    print("FPS: ", fps)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

drone.streamoff()
cv2.destroyAllWindows()
drone.land()
print("Landing...")
print("BATTERY: ")
print(drone.get_battery())
drone.end()
