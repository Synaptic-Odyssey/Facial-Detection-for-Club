import cv2
from time import sleep
import numpy as np
from pygrabber.dshow_graph import FilterGraph

#Stuff you need to know: Press q to initialize facial recognition, and w to close program

graph = FilterGraph()
devices = graph.get_input_devices()

print("Available webcams:")
for idx, device in enumerate(devices):
    print(f"{idx}: {device}")

# Check accessibility of each webcam
# for i in range(len(devices)):
#     webcam = cv2.VideoCapture(i)
#     if not webcam.isOpened():
#         print(f"Failed to open webcam with index {i} ({devices[i]})")
#     else:
#         print(f"Webcam with index {i} ({devices[i]}) is accessible.")


webcam_index = int(input("Input webcam index \n"))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(webcam_index)
sleep(2)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

club = input("Input Club Name \n")

while True:
    try:
        check, frame = webcam.read()
        if not check:
            print("Error: Could not read frame from webcam.")
            break

        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                compatibility = 90 + 10 * np.random.random()
                text = f"Compatibility: {compatibility:.2f}%"
                join_text = f"JOIN {club}!"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.9
                font_thickness = 2
                color = (0, 255, 0)

                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = x + (w - text_width) // 2
                text_y = y - 10
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, font_thickness)

                (join_text_width, join_text_height), _ = cv2.getTextSize(join_text, font, font_scale, font_thickness)
                join_text_x = x + (w - join_text_width) // 2
                join_text_y = y + h + join_text_height + 20
                cv2.putText(frame, join_text, (join_text_x, join_text_y), font, font_scale, color, font_thickness)

            cv2.imshow("Face Detection", frame)
            cv2.waitKey(0)
            
            print("Scanning facial geometry and biometric markers...")
            print("Extracting multidimensional feature vectors...")
            print("Executing advanced eigenface decomposition...")
            print("Running convolutional neural networks for feature extraction...")
            print("Analyzing facial emotion recognition algorithms...")
            print("Performing high-dimensional data clustering...")
            print("Calculating social compatibility indices...")
            print("Integrating facial recognition with behavioral analytics...")
            print("Executing deep learning models for personality inference...")
            print("Simulating multi-agent compatibility scenarios...")
            print("Applying probabilistic graphical models for relationship mapping...")
            print("Optimizing compatibility scores using genetic algorithms...")
            print("Cross-referencing with sociometric network data...")
            print("Calibrating hyperparameters for facial compatibility model...")
            print("Generating visual similarity matrices...")
            print("Performing non-linear multidimensional scaling...")
            print("Executing real-time compatibility diagnostics...")
            print("Aggregating results from ensemble learning classifiers...")
            print("Finalizing compatibility analysis report...\n")

            print(f"Compatibility: {compatibility:.2f}%")
            print("EXTREMELY COMPATIBLE")
            print(f"JOIN {club} CLUB")

            # webcam.release()
            # cv2.destroyAllWindows()
            # break
        
        elif key == ord('w'):
            print("Ending Program...")
            webcam.release()
            cv2.destroyAllWindows()
            break
        
    except KeyboardInterrupt:
        # Handle program exit with Ctrl+C
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
