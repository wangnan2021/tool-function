import cv2
import numpy as np
import random
import onnxruntime as ort

from scipy import optimize

random.seed(44)


def filter_small_component(mask, label_index, area_thre, predict):
    _, num_labels, labels = check_connect_component(mask, label_index)
    for j in range(num_labels):
        if j == 0:
            continue
        else:
            temp = np.array(labels == j, np.uint8)
            area = np.sum(temp > 0)
            if area > area_thre:
                predict += temp * mask
    return


def is_louci_by_value(component, image_gray, louci_mean_val):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.morphologyEx(component, cv2.MORPH_ERODE, kernel)
    c_x, c_y = np.where(dst != 0)
    if len(c_x) > 0:
        temp = cv2.resize(image_gray, (2048, 2048)) * dst
        mean_value = np.sum(temp) // len(c_x)
        # max_value = np.max(temp)
        print('mean value is:', mean_value)
        return mean_value > louci_mean_val
    return False


def drawMask(mask, points_tmp, value, pen_size=3, label_method='polygon'):
    if label_method not in ['line', 'linestrip', 'polygon']:
        raise ValueError('label method wrong')
    points = []
    if len(points_tmp) == 2:
        for i in range(len(points_tmp[1])):
            points.append([int(points_tmp[1][i]), int(points_tmp[0][i])])
    else:
        for each_point in points_tmp:
            points.append([int(each_point[0]), int(each_point[1])])
    for each_point in points_tmp:
        points.append([int(each_point[0]), int(each_point[1])])
    points = np.array(points)
    if label_method == 'line':
        pt0 = (points[0][0], points[0][1])
        pt1 = (points[1][0], points[1][1])
        cv2.line(mask, pt0, pt1, value, thickness=int(pen_size))
    if label_method == 'linestrip':
        cv2.polylines(mask, [points], False, value, thickness=int(pen_size))
    if label_method == 'polygon':
        cv2.fillPoly(mask, [points], value)
    return mask


def random_color(num_classes):
    color_map = []
    color_map.append([255, 255, 255])
    for _ in range(1, num_classes):
        color_map.append((random.randint(0, 255), random.randint(0, 255),
                          random.randint(0, 255)))
    return color_map


def gen_mask(segm, num_classes, color_map=None):
    mask = np.zeros((segm.shape[0], segm.shape[1], 3), dtype=np.uint8)
    if not color_map:
        color_map = random_color(num_classes)
    elif len(color_map) != num_classes:
        raise ValueError('color map size != num_classes!!!')
    for i in range(0, num_classes):
        mask[np.where(segm == i)[0], np.where(segm == i)[1], :] = color_map[i]
    return mask


def max_length(mask, rang):
    max_i = 0
    for i in range(rang[0], rang[1]):
        line = mask[i:i + 1, :]

        lo = np.where(line > 0)

        if len(lo[1]) > 0:
            max_x, min_x = max(lo[1]), min(lo[1])
            max_i = max(max_i, max_x - min_x)
    return max_i


def check_connect_component(img, label_index):
    mask = np.array(img == label_index, np.uint8)
    num, label = cv2.connectedComponents(mask, connectivity=8)
    return mask, num, label


class CircleCallipers(object):
    def __init__(self, x, y, r1, r2):
        self._cx = x
        self._cy = y
        self._r1 = r1
        self._r2 = r2
        self._callipers = self._createCallipers()

    @property
    def callipers(self):
        return self._callipers

    def _createCallipers(self):
        callipers = list()
        for angle in range(0, 360, 36):
            angle = angle / 180 * np.pi
            x1 = self._cx + np.cos(angle) * self._r1
            y1 = self._cy + np.sin(angle) * self._r1
            x2 = self._cx + np.cos(angle) * self._r2
            y2 = self._cy + np.sin(angle) * self._r2
            p1 = (x1, y1)
            p2 = (x2, y2)
            callipers.append(self._getPointsInCalliper(p1, p2))
        return callipers

    def _getPointsInCalliper(self, p1, p2):
        points = []
        for i in range(0, self._r2 - self._r1 + 1):
            frac = i / (self._r2 - self._r1)
            px = int(p1[0] * (1 - frac) + p2[0] * frac)
            py = int(p1[1] * (1 - frac) + p2[1] * frac)

            points.append((px, py))

        return points


def callipersFindEdge(img, callipers, gray_th=20):
    kernel = np.array([-1, 0, 1])
    candidate_pts = list()
    candidate_pts_reverse = list()
    for calliper in callipers.callipers:
        gray_value = list()
        for pt in calliper:
            x, y = pt[0], pt[1]
            gray_value.append(img[y, x])
        edge = cv2.filter2D(np.array(gray_value), cv2.CV_16S, kernel)
        for idx in range(len(edge)):
            if abs(edge[idx]) > gray_th:
                candidate_pts.append(calliper[idx])
                break

        reverse_edge = edge[::-1]
        for idx in range(len(reverse_edge)):
            if abs(reverse_edge[idx]) > gray_th:
                candidate_pts_reverse.append(calliper[::-1][idx])
                break
    return np.array(candidate_pts), np.array(candidate_pts_reverse)


def fitCircleByLeastSquare(points, error_threshold=2):
    center_ = points.mean(axis=0)

    def calDist(center):
        return np.sqrt(np.sum((points - center)**2, axis=1))

    def diff(center):
        radius = calDist(center)
        return radius - radius.mean()

    while True:
        center_, ier = optimize.leastsq(diff, center_)
        radius_ = calDist(center_).mean()
        errors = np.abs(calDist(center_) - radius_)
        error_ = errors.mean()
        error_std = errors.std()
        idx = np.where(np.abs((errors - error_) / error_std) < error_threshold)
        if len(idx[0]) == len(points):
            break
        points = points[idx]

    return center_, radius_, error_


class ONNX_Model():
    def __init__(self, onnx_path):
        self.onnx_session = ort.InferenceSession(onnx_path)

    def __call__(self, x):
        onnx_predict = self.onnx_session.run(None,
                                             {'input': x.astype(np.float32)})
        # post-process
        predict = onnx_predict[0][0]
        return predict
