import cv2

# 1. Load the pre-trained face detection model (Haar Cascade)
# OpenCV includes these files by default
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Access the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the application")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert to grayscale for better detection performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )

    # Draw a rectangle around every detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection Project', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()