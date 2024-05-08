import os
import cv2
import numpy as np

from typing import Optional


class YOLO:
    yolo_dir: str
    classes: list[str]
    net: cv2.dnn.Net

    def __init__(self, yolo_dir: Optional[str] = None):
        if yolo_dir is None:
            self.yolo_dir = os.path.join(os.getcwd(), '..', '..', 'data', 'yolo')
        else:
            self.yolo_dir = yolo_dir

        classes_file = os.path.join(self.yolo_dir, 'yolov3.txt')
        config_file = os.path.join(self.yolo_dir, 'yolov3.cfg')
        weights_file = os.path.join(self.yolo_dir, 'yolov3.weights')

        with open(classes_file, 'r') as file:
            self.classes = [line.strip() for line in file.readlines()]

        self.net = cv2.dnn.readNet(weights_file, config_file)

    def detect(self, image: np.ndarray, min_confidence: float = 0.5) -> list[tuple[list[int], str]]:
        """
        Detect objects in an image
        :param image:
        :param min_confidence:
        :return:
        """

        # Check if the image is gray
        if len(image.shape) == 2:
            # Copy the gray image to all 3 channels
            image = cv2.merge((image, image, image))

        # Set the input image
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        # Blob is the input image to the network. It's looks like this (1, 3, 416, 416)
        self.net.setInput(blob)

        # Set the input layer
        layer_names = self.net.getLayerNames()
        unconnected_layers = self.net.getUnconnectedOutLayers()
        output_layers = [layer_names[inx - 1] for inx in unconnected_layers]

        # Find the objects
        class_ids = []
        confidences = []
        boxes = []

        width, height = image.shape[1], image.shape[0]

        for out in self.net.forward(output_layers):
            for detection in out:
                # The first 4 elements are the bounding box dimensions
                box = detection[:5]

                # The rest are the class probabilities
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence >= min_confidence:
                    center_x = int(box[0] * width)
                    center_y = int(box[1] * height)
                    w = int(box[2] * width)
                    h = int(box[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # A minimum confidence score below which detections are discarded entirely.
        score_threshold = 0.5

        # A non-maxima suppression threshold to filter overlapping and low-confidence boxes.
        nms_threshold = 0.4

        # Apply non-maximum suppression (Remove overlapping boxes)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)

        # Return the detected objects
        return [(boxes[i], self.classes[class_ids[i]]) for i in indices]

    def detect_classes(self, image: np.ndarray, clazz: str, min_confidence: float = 0.5) -> list[list[int]]:
        """
        Detect objects of a specific class in an image
        :param image:
        :param clazz:
        :param min_confidence:
        :return:
        """

        detections = self.detect(image, min_confidence)
        return [box for box, detected_class in detections if detected_class == clazz]
