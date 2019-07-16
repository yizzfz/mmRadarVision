from imageai.Detection import ObjectDetection
import os
import cv2

class Detector_Human():
    def __init__(self, min_prob=50, out_type='array'):
        model = os.path.join(
            os.path.dirname(__file__), 'model', 'resnet50_coco_best_v2.0.1.h5')

        print(model)
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(model)
        detector.loadModel()
        custom_objects = detector.CustomObjects(person=True)
        self.detector = detector
        self.target = custom_objects
        self.min_prob = min_prob
        self.out_type = out_type
        

    def process(self, frame):
        if self.out_type == 'array':
            frame, detections = self.detector.detectCustomObjectsFromImage(custom_objects=self.target,
                                                                input_type='array',
                                                                input_image=frame,
                                                                output_type='array',
                                                                minimum_percentage_probability=self.min_prob)
        else:
            detections = self.detector.detectCustomObjectsFromImage(custom_objects=self.target,
                                                                    input_type='array',
                                                                    input_image=frame,
                                                                    minimum_percentage_probability=self.min_prob)

        for detection in detections:
            box = detection['box_points']
            cv2.rectangle(frame, (box[0], box[1]),
                          (box[2], box[3]), (255, 0, 0))

        return frame, [detection['box_points'] for detection in detections]

