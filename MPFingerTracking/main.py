
#imports
import time
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import cv2
import numpy as np
##################################

#CONSTANTS
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88,205,54)
##################################

MpImage = mp.Image

class landmarkerResult():
    def __init__(self):
       self.result = mp.tasks.vision.HandLandmarkerResult
       self.detector = mp.tasks.vision.HandLandmarker
       self.createLandmarker()
       
    def createLandmarker(self):
        def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result
            
        options = mp.tasks.vision.HandLandmarkerOptions(base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task'),
                                    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                                    num_hands = 2,
                                    min_hand_detection_confidence = 0.5,
                                    min_hand_presence_confidence = 0.5,
                                    min_tracking_confidence = 0.5,
                                    result_callback=print_result
                                    )
        
        self.detector = self.detector.create_from_options(options)
            
    def detectAsync(self, frame):
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
        
        (self.detector).detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))
        
    def close(self):
        self.detector.close()
        
    def drawLandmarksOnImage(self, rgb_image):
        #Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
        detection_result = self.result
        try:
            if detection_result.hand_landmarks == []:
                return rgb_image
            else:
                hand_landmarks_list = detection_result.hand_landmarks
                handedness_list = detection_result.handedness
                annotated_image = np.copy(rgb_image)
                # Loop through the detected hands to visualize.
                for idx in range(len(hand_landmarks_list)):
                    hand_landmarks = hand_landmarks_list[idx]
                    handedness = handedness_list[idx]

                    # Draw the hand landmarks.
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                    ])
                    mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

                    # Get the top left corner of the detected hand's bounding box.
                    height, width, _ = annotated_image.shape
                    x_coordinates = [landmark.x for landmark in hand_landmarks]
                    y_coordinates = [landmark.y for landmark in hand_landmarks]
                    text_x = int(min(x_coordinates) * width)
                    text_y = int(min(y_coordinates) * height) - MARGIN

                    # Draw handedness (left or right hand) on the image.
                    # cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

                return annotated_image 
        except Exception as e:
            print("Error drawing landmarks on image: ", str(e))
            return rgb_image
        
  


def main():
    cap = cv2.VideoCapture(0)
    hand_landmarker = landmarkerResult()


    while cap.isOpened():
        ret, frame = cap.read()
        
        frame = cv2.flip(frame, 1)
        
        hand_landmarker.detectAsync(frame)
        
        frame = hand_landmarker.drawLandmarksOnImage(rgb_image=frame)

        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
            

    cap.release()
    hand_landmarker.close()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
        