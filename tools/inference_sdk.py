from re import S
import numpy as np
import cv2
import pdb
import os
import random

from os import path as osp
from numpy.lib.type_check import imag
from postprocesslib import *

random.seed(44)

cfg = {
    'common': {
        "img_path":
        '/data/home/wengu/data/boruisi/0816_port_feedback/0816_clip/P091837493_FitImage_Cam1ML_091845332(1).png',
        "onnx_path":
        '/data/home/wengu/project/boruisi/exp_gw/20210812/ex1/deploy/20000.onnx',
        "result_saving_path": './output/onnx_result.jpg',
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "num_classes": 4,
        "mask_weight": 0.03,
        "color_map": [[255, 255, 255], [0, 255, 0], [255, 0, 0], [0, 0, 255]],
        "label_map": ['BG', 'LieWen', 'WaiYiWu', 'NeiYiWu']
    },
    'ct': {
        "num_classes": 3,
        'aspect_ratio': [None, 2, None],
        'two_edge': [400, 1600],
        'length_gap': 20,
        "area_thre": [-1, 150, 800],
    },
    'inner': {
        "num_classes": 3,
        'search_offset': 50,  # 搜索范围
        "area_thre": [-1, 100, 250],
        'quekou_area_threshold': 1300,  # 缺口面积阈值
        'quekou_gray_threshold': 10,  # 缺口灰度阈值
        'extend_rate': 1.0  # 内圆收缩比例
    },
    'port': {
        "num_classes": 3,
        "area_thre": [-1, 100, 100],
        'roi': {
            'cx': 1224,
            'cy': 1024,
            'r1': 500,
            'r2': 1000
        },
        "aspect_ratio": 1 / 3,
        'gray_th': 10,
        'louci_mean_val': 120
    }
}


def sdk_post_vanilla(predict_score):
    return np.argmax(predict_score, 0)


# 暂时不需要
def sdk_post_aspect_ratio(predict_score, cfg):
    num_classes = cfg['num_classes']
    _mask = np.argmax(predict_score, 0)
    predict = np.zeros(_mask.shape)
    thre = cfg['thre']
    aspect_ratio_thre = cfg['aspect_ratio']
    assert num_classes == len(thre)
    for i in range(num_classes):
        if i == 0:
            continue
        else:
            labels = np.array(_mask == i, np.uint8)
            contours, _ = cv2.findContours(labels, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > thre[i]:
                    if aspect_ratio_thre[i]:
                        x, y, w, h = cv2.boundingRect(cnt)
                        aspect_ratio = h / w
                        print('area:', area, 'aspect_ratio:', aspect_ratio)
                        if aspect_ratio > aspect_ratio_thre[i]:
                            continue
                    drawMask(predict, cnt[:, 0, :], i)
    return predict


# 更改打光后的端口后处理：1）磁仅面积筛选后保留。2）掉角。3）计算平均像素值
def sdk_post_port_v2(predict_score, image, vis, cfg):
    config = cfg.get('port')
    num_classes = config.get('num_classes')
    roi = config.get('roi')
    gray_th = config.get('gray_th')
    area_thre = config.get('area_thre')
    aspect_ratio = config.get('aspect_ratio')
    louci_mean_val = config.get('louci_mean_val')
    callipers = CircleCallipers(roi['cx'], roi['cy'], roi['r1'], roi['r2'])
    _mask = np.argmax(predict_score, 0)
    predict = np.zeros(_mask.shape)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    pts_inner, pts_outter = callipersFindEdge(image_gray, callipers, gray_th)
    inner_center, inner_radius, _ = fitCircleByLeastSquare(pts_inner)
    outter_center, outter_radius, _ = fitCircleByLeastSquare(pts_outter)
    x, y = int(outter_center[0]), int(outter_center[1])
    ro = int(outter_radius)
    ri = int(inner_radius)
    for i in range(num_classes):
        if i == 0:
            continue
        elif i == 1:
            filter_small_component(_mask, i, area_thre[i], predict)
        elif i == 2:
            _, num_labels, labels = check_connect_component(_mask, i)
            print('!!!!', num_labels)
            for j in range(1, num_labels):
                component = np.array(labels == j, np.uint8)
                mask_p = cv2.resize(image_gray, (2448, 2048),
                                    cv2.INTER_NEAREST)[y - ro:y + ro,
                                                        x - ro:x + ro]
                mask_polar = cv2.warpPolar(mask_p,
                                            (ro, int(2 * np.pi * ro)),
                                            (ro, ro), ro,
                                            cv2.WARP_POLAR_LINEAR)
                # pdb.set_trace()
                cv2.imwrite( './output/temp.jpg', mask_polar)
                for i in range(0, 4800, 100):
                    mean_value = mask_polar[i:i+20, -50:-20].mean()
                    print('angle:', i, ',mean_value:', mean_value)
                if np.sum(component) < area_thre[i]:
                    continue
                elif is_louci_by_value(component, image_gray, louci_mean_val):
                    component[component == 1] = 1
                    predict += component
                    continue
                else:
                    mask_p = cv2.resize(component, (2448, 2048),
                                        cv2.INTER_NEAREST)[y - ro:y + ro,
                                                           x - ro:x + ro]
                    mask_polar = cv2.warpPolar(mask_p,
                                               (ro, int(2 * np.pi * ro)),
                                               (ro, ro), ro,
                                               cv2.WARP_POLAR_LINEAR)
                    _, _, stats, _ = cv2.connectedComponentsWithStats(
                        np.array(mask_polar == 1, np.uint8), connectivity=8)
                    max_l = stats[1][2]
                    if max_l > (ro - ri) * aspect_ratio:
                        print(max_l, 'and', (ro - ri) * aspect_ratio)
                        # drawMeanValue(component, image_gray, vis)
                        predict += component * _mask
                    else:
                        # drawMeanValue(component, image_gray, vis)
                        component[component == 1] = 3
                        predict += component
    return predict


# ct后处理：去除划痕
def sdk_post_ct(predict_score, cfg):
    config = cfg.get('ct')
    num_classes = config['num_classes']
    area_thre = config['area_thre']
    two_edge = config['two_edge']
    length_gap = config['length_gap']
    _mask = np.argmax(predict_score, 0)
    predict = np.zeros(_mask.shape)
    assert num_classes == len(area_thre)
    for i in range(num_classes):
        if i == 0:
            continue
        elif i == 1:
            _, num, labels = check_connect_component(_mask, i)
            for j in range(1, num):
                temp = np.array(labels == j, np.uint8)
                locate = np.where(temp > 0)
                x_max, x_min = max(locate[1]), min(locate[1])
                y_max, y_min = max(locate[0]), min(locate[0])
                area = len(locate[1])
                if (x_max > two_edge[1] or x_min < two_edge[0]
                    ) and area > area_thre[i] and (x_max - x_min) > length_gap:
                    print(y_min, 'and', y_max, ',', x_min, 'and', x_max)
                    predict += temp * _mask
        elif i == 2:
            filter_small_component(_mask, i, area_thre[i], predict)
    return predict


# 内窥后处理：内外圆判定，去除凹槽的异物，凹槽外的裂纹去除
def sdk_post_inner_v2(predict_score, image, cfg):
    config = cfg.get('inner')
    num_classes = config.get('num_classes')
    search_offset = config.get('search_offset')
    quekou_area_threshold = config.get('quekou_area_threshold')
    quekou_gray_threshold = config.get('quekou_gray_threshold')
    extend_rate = config.get('extend_rate')
    area_thre = config.get('area_thre')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, th_img = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)
    cns, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = cns[0]
    ellipse = cv2.fitEllipseAMS(contour)
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    radius = int(np.mean(ellipse[1]) / 2)

    detect_img = image.copy()
    cv2.circle(detect_img, center, radius - search_offset, 255, -1)
    _, th_img = cv2.threshold(detect_img, quekou_gray_threshold, 255,
                              cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    th_img = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, kernel)

    offset = 0
    cns, _ = cv2.findContours(255 - th_img, cv2.RETR_TREE,
                              cv2.CHAIN_APPROX_NONE)
    for cn in cns:
        moments = cv2.moments(cn)  # 二维矩阵的零阶矩就是图像的面积
        print(moments['m00'])
        if moments['m00'] > quekou_area_threshold:
            # _, wh, _ = cv2.minAreaRect(cn)
            # offset = int((1 + extend_rate) * min(wh))
            rect = cv2.minAreaRect(cn)
            box = np.int64(cv2.boxPoints(rect))
            offset = int((1 + extend_rate) * min(rect[1]))
            break
    print('outter circle: ', center, radius)
    print('offset:', offset)
    print('innter circle: ', center, radius - offset)

    if offset < radius and offset > 0:
        circle_mask = np.zeros(predict_score.shape[1:])
        cv2.drawContours(circle_mask, [box], -1, float('inf'), -1)
        # 筛除缺口内的异物类
        predict_score[0] += circle_mask
    else:
        offset = 35

    # 再筛除小面积缺陷
    _mask = np.argmax(predict_score, 0)
    predict = np.zeros(_mask.shape)
    assert num_classes == len(area_thre)
    for i in range(num_classes):
        if i == 0:
            continue
        else:
            _, num_labels, labels = check_connect_component(_mask, i)
            for j in range(1, num_labels):
                component = np.array(labels == j, np.uint8)
                if np.sum(component) < area_thre[i]:
                    continue
                else:
                    mask_polar = cv2.warpPolar(
                        component, (radius, int(2 * np.pi * radius)),
                        center, radius, cv2.WARP_POLAR_LINEAR)
                    _, _, stats, _ = cv2.connectedComponentsWithStats(
                        np.array(mask_polar == 1, np.uint8), connectivity=8)
                    x_min, x_max = stats[1][0], stats[1][0] + stats[1][2]
                    if i == 1:
                        if x_max > radius - offset:
                            predict += component * _mask
                    elif i == 2:
                        if x_max > radius - offset:
                            predict += component * _mask
                        else:
                            component[component == 1] = 3
                            predict += component
    return predict


if __name__ == '__main__':
    common_cfg = cfg.get('common')
    num_classes = common_cfg['num_classes']
    onnx_model = ONNX_Model(common_cfg['onnx_path'])
    # preprocess
    image_ori = cv2.imread(common_cfg['img_path'])

    TEST_DATASET = False
    if TEST_DATASET:
        # read img list
        data_list = []
        data_file = '/data/home/wengu/data/boruisi/data_list/0818_inner_test.txt'
        with open(data_file, 'r', encoding='utf-8') as r:
            for line in r.readlines():
                line = line.strip()
                img_path = line.split(',')[0]
                data_list.append('{}/{}'.format('/data/home', img_path))

        for img_path in data_list:
            image_ori = cv2.imread(img_path)
            _image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
            print('original image:', _image.shape)
            # if _image.shape[:2] == (5200, 2048):
            #     _image = cv2.resize(_image, dsize=(2048, 4096))
            #     IMG_VERSION = 'CT'
            # elif _image.shape[:2] == (1800, 1800):
            #     _image = cv2.resize(_image, dsize=(2048, 2048))
            #     IMG_VERSION = 'INNER'
            # elif _image.shape[:2] == (2048, 2448):
            #     _image = cv2.resize(_image, dsize=(2048, 2048))
            #     IMG_VERSION = 'PORT'
            # else:
            #     raise RuntimeError('Invalid Input Size')

            if _image.shape[:2] == (1800, 1800):
                _image = cv2.resize(_image, dsize=(2048, 2048))
                IMG_VERSION = 'INNER'
            else:
                continue

            print('IMG_VERSION:', IMG_VERSION, '& RESIZED_SHAPE:',
                  _image.shape)
            print(img_path)

            h, w, _ = _image.shape
            predict_score = []
            for i in range(h // 2048):
                image = _image[2048 * i:2048 * (i + 1), :, :]
                image = image.astype(np.float32)
                image -= np.float32(common_cfg['mean'])
                image /= np.float32(common_cfg['std'])
                image = image[np.newaxis, :, :, :]
                image = np.transpose(image, [0, 3, 1, 2])
                predict_score.append(onnx_model(image))

            predict_score = np.concatenate(predict_score, axis=1)

            if IMG_VERSION == 'CT':
                predict = sdk_post_ct(predict_score, cfg)
            elif IMG_VERSION == 'INNER':
                predict = sdk_post_inner_v2(predict_score, _image, cfg)
            elif IMG_VERSION == 'PORT':
                predict = sdk_post_port_v2(predict_score, image_ori, _image,
                                           cfg)

            onnx_mask = gen_mask(predict, num_classes, common_cfg['color_map'])

            vis = cv2.addWeighted(_image, 1 - common_cfg['mask_weight'],
                                  onnx_mask, common_cfg['mask_weight'], 0)

            result_saving_path = './output' + img_path + '.jpg'
            if not osp.exists(osp.split(result_saving_path)[0]):
                os.makedirs(osp.split(result_saving_path)[0])

            cv2.imwrite(result_saving_path, np.array(vis, dtype=np.int))

            # 左上角添加类别说明
            for i in range(num_classes):
                if i == 0:
                    continue
                else:
                    cv2.putText(vis, common_cfg['label_map'][i],
                                (10, 40 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                                1, common_cfg['color_map'][i], 1, cv2.LINE_AA)
            cv2.imwrite(result_saving_path, np.array(vis, dtype=np.int))

    else:
        _image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        print('original image:', _image.shape)
        if _image.shape[:2] == (5200, 2048):
            _image = cv2.resize(_image, dsize=(2048, 4096))
            IMG_VERSION = 'CT'
            specific_cfg = cfg.get('ct')
        elif _image.shape[:2] == (1800, 1800):
            _image = cv2.resize(_image, dsize=(2048, 2048))
            IMG_VERSION = 'INNER'
            specific_cfg = cfg.get('inner')
        elif _image.shape[:2] == (2048, 2448):
            _image = cv2.resize(_image, dsize=(2048, 2048))
            IMG_VERSION = 'PORT'
            specific_cfg = cfg.get('port')
        else:
            print('INPUT SIZE:', _image.shape)
            raise RuntimeError('Invalid Input Size')

        print('IMG_VERSION:', IMG_VERSION, '& RESIZED_SHAPE:', _image.shape)

        h, w, _ = _image.shape
        predict_score = []
        for i in range(h // 2048):
            image = _image[2048 * i:2048 * (i + 1), :, :]
            image = image.astype(np.float32)
            image -= np.float32(common_cfg['mean'])
            image /= np.float32(common_cfg['std'])
            image = image[np.newaxis, :, :, :]
            image = np.transpose(image, [0, 3, 1, 2])
            print('patch shape:', image.shape)
            predict_score.append(onnx_model(image))

        predict_score = np.concatenate(predict_score, axis=1)
        print('predict score shape:', predict_score.shape)

        if IMG_VERSION == 'CT':
            # predict = sdk_post_ct(predict_score, cfg)
            predict = sdk_post_vanilla(predict_score)
        elif IMG_VERSION == 'INNER':
            predict = sdk_post_inner_v2(predict_score, _image, cfg)
        elif IMG_VERSION == 'PORT':
            predict = sdk_post_port_v2(predict_score, image_ori, _image, cfg)

        onnx_mask = gen_mask(predict, num_classes, common_cfg['color_map'])

        cv2.line(_image, (400, 0), (390, 1000), (0, 255, 0))
        cv2.line(_image, (1600, 0), (1600, 1000), (0, 255, 0))
        # [340, 1660]

        vis = cv2.addWeighted(_image, 1 - common_cfg['mask_weight'], onnx_mask,
                              common_cfg['mask_weight'], 0)
        # 左上角添加类别说明
        for i in range(num_classes):
            # print(num_classes, len(common_cfg['label_map']))
            if i == 0:
                continue
            else:
                cv2.putText(vis, common_cfg['label_map'][i],
                            (10, 40 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            common_cfg['color_map'][i], 1, cv2.LINE_AA)
        cv2.imwrite(common_cfg['result_saving_path'],
                    np.array(vis, dtype=np.int))
    print('!!!DONE')
