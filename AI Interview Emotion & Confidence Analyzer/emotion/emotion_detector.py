import cv2
from deepface import DeepFace
import time


def run_emotion_detection():

    cap = cv2.VideoCapture(0)

    last_analysis_time = 0
    emotion = "Detecting..."

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        current_time = time.time()

        # Analyze every 2 seconds
        if current_time - last_analysis_time > 2:

            try:

                analysis = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False
                )

                if isinstance(analysis, list):
                    analysis = analysis[0]

                emotion = analysis['dominant_emotion']
                emotion_scores = analysis['emotion']
                confidence = emotion_scores[emotion]

                region = analysis['region']

                x = region['x']
                y = region['y']
                w = region['w']
                h = region['h']

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # Emotion label
                cv2.putText(
                    frame,
                    f"Emotion: {emotion}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"Confidence: {confidence:.2f}%",
                    (x, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2
                )

                last_analysis_time = current_time

            except Exception as e:
                print("Error:", e)

        cv2.imshow("Emotion Detection", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
