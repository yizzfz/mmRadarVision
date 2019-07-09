from imageai.Detection import ObjectDetection
import os
import cv2

class Detector_Human():
    def __init__(self, *args, **kwargs):
        model = os.path.join(
            os.getcwd(), 'detector', 'model', 'resnet50_coco_best_v2.0.1.h5')
        print(model)
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(model)
        detector.loadModel()
        custom_objects = detector.CustomObjects(person=True)
        self.detector = detector
        self.target = custom_objects

    def process(self, frame):
        detections = self.detector.detectCustomObjectsFromImage(custom_objects=self.target,
                                                                input_type='array',
                                                                input_image=frame,
                                                                minimum_percentage_probability=50)

        for detection in detections:
            box = detection['box_points']
            cv2.rectangle(frame, (box[0], box[1]),
                          (box[2], box[3]), (255, 0, 0))

        return frame, [detection['box_points'] for detection in detections]

