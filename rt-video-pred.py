import tensorflow as tf
from keras.models import Model
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import time

from PIL import ImageGrab
import cv2


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if (objectness <= obj_thresh).all():        #this mfkin' line, i swear i'll slap experencior
                continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def do_nms(boxes, scores, threshold):             #will slap experencior for this too
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, 10, threshold)
    selected_boxes = tf.gather(boxes, selected_indices)

    return selected_boxes.numpy().astype(int)


def load_image_pixels(screen, shape):
    width, height = screen.size
    image = img_to_array(screen)
    image = tf.keras.preprocessing.image.smart_resize(image, shape)
    image = image.astype('float32')/255.0
    image = expand_dims(image, 0)

    return image, width, height


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


def draw_boxes(screen, v_boxes, v_labels, v_scores):
    screen = np.array(screen)
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # print([box.ymin, box.xmin, box.ymax, box.xmax])
        # print(len(v_boxes[i]))
        y1, x1, y2, x2 = box[0], box[1], box[2], box[3]

        color = (0, 0, 0)
        thickness = 2

        cv2.rectangle(screen, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(screen, (str(v_labels[i])+": "+str(v_scores[i])), (x1,
                    y1-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 0)

    # cv2.imshow('window',screen)
        cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


model = tf.keras.models.load_model('model.h5')
input_w, input_h = 416, 416
while True:
    screen = ImageGrab.grab(bbox=(300, 300, 660, 660))
    image, image_w, image_h = load_image_pixels(screen, (input_w, input_h))
    yhat = model.predict(image)
    print([a.shape for a in yhat])

    anchors = [[116, 90, 156, 198, 373, 326], [
        30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    last_time = time.time()

    # define the probability threshold for detected objects
    class_threshold = 0.6
    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i],
                               class_threshold, input_h, input_w)

        # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

    # define the labels
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    coords = np.empty([len(v_boxes), 4])
    for i in range(len(v_boxes)):
        coords[i] = [v_boxes[i].ymin, v_boxes[i].xmin,
                     v_boxes[i].ymax, v_boxes[i].xmax]

    s_boxes = do_nms(coords, v_scores, 0.5)

    num_preds = print(len(s_boxes))

    # summarize what we found
    for i in range(len(s_boxes)):
        print(v_labels[i], v_scores[i])

    # draw what we found
    last_time = time.time()
    draw_boxes(screen, s_boxes, v_labels, v_scores)
    print(time.time()-last_time)
