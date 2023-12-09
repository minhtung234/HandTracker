import cv2
import mediapipe

mediapipe_drawing_utils = mediapipe.solutions.drawing_utils
mediapipe_drawing_styles = mediapipe.solutions.drawing_styles
mediapipe_hands = mediapipe.solutions.hands

# Set your desired window dimensions
window_width = 800
window_height = 600

# Create a named window
cv2.namedWindow('Handtracker', cv2.WINDOW_NORMAL)

# Set the window dimensions
cv2.resizeWindow('Handtracker', window_width, window_height)

capture_video = cv2.VideoCapture(0)
hands = mediapipe_hands.Hands()

while True:
    data, image = capture_video.read()
    # flip image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # store result
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for each_landmark in results.multi_hand_landmarks:
            # Draw all hand landmarks
            mediapipe_drawing_utils.draw_landmarks(
                image,
                each_landmark,
                mediapipe_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mediapipe_drawing_utils.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2,
                    circle_radius=4
                ),
                connection_drawing_spec=mediapipe_drawing_utils.DrawingSpec(
                    color=(0, 0, 255),
                    thickness=2,
                    circle_radius=2
                )
            )

            # Customize drawing style for specific palm landmarks
            palm_landmarks = [0, 5, 9, 13, 17]  # Modify as needed
            for landmark_id in palm_landmarks:
                landmark = each_landmark.landmark[landmark_id]
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 4, (255, 0, 0), -1)

    # click a / control-c to quit
    cv2.imshow('Handtracker', image)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

capture_video.release()
cv2.destroyAllWindows()


