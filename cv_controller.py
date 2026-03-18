# /// script
# dependencies = ["websockets", "opencv-python", "numpy"]
# ///

import asyncio
import time
import cv2
import numpy as np
import websockets

WS_URI = "ws://localhost:8765"


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 120, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = clean_mask(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    command = None
    h, w, _ = frame.shape

    left_limit = int(w * 0.35)
    right_limit = int(w * 0.65)
    fire_limit = int(h * 0.25)

    cv2.line(frame, (left_limit, 0), (left_limit, h), (255, 255, 0), 2)
    cv2.line(frame, (right_limit, 0), (right_limit, h), (255, 255, 0), 2)
    cv2.line(frame, (0, fire_limit), (w, fire_limit), (0, 255, 255), 2)

    cv2.putText(frame, "LEFT", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "RIGHT", (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "FIRE", (w // 2 - 40, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 800:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv2.circle(frame, (cX, cY), 10, (0, 255, 0), -1)

                if cY < fire_limit:
                    command = "FIRE"
                elif cX < left_limit:
                    command = "LEFT"
                elif cX > right_limit:
                    command = "RIGHT"

                cv2.putText(
                    frame,
                    f"X={cX} Y={cY}",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

    cv2.putText(
        frame,
        f"CMD: {command if command else 'NONE'}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if command else (0, 0, 255),
        2,
    )

    return command, frame, mask


async def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la caméra.")
        return

    last_command = None
    last_sent_time = 0.0
    cooldown = 0.25

    async with websockets.connect(WS_URI) as ws:
        print("Contrôle caméra actif.")

        await asyncio.sleep(1)
        await ws.send("ENTER")
        print("Envoyé : ENTER")
        await asyncio.sleep(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            command, debug_frame, mask = process_frame(frame)

            now = time.time()
            if command:
                if command != last_command or (now - last_sent_time) >= cooldown:
                    await ws.send(command)
                    print(f"Envoyé : {command}")
                    last_command = command
                    last_sent_time = now
            else:
                last_command = None

            cv2.imshow("CV Controller", debug_frame)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("e"):
                await ws.send("ENTER")
                print("Envoyé : ENTER")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())