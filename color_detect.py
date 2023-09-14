import cv2
import numpy as np

# Initialize the video capture from the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Define colors in BGR format
yellow = [0, 255, 255]


# Function to get lower and upper HSV limits for a given color
def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvC[0][0][0]
    lower = np.array([hue - 10, 100, 100], dtype=np.uint8)
    upper = np.array([hue + 10, 255, 255], dtype=np.uint8)
    return lower, upper


while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the BGR frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the lower and upper HSV limits for the yellow color
    lower, upper = get_limits(yellow)

    # Create a mask to isolate the yellow color
    mask = cv2.inRange(hsv, lower, upper)

    # Apply a threshold to the mask
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through detected contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw a green rectangle around the detected object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the original frame and the mask
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Mask", mask)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
