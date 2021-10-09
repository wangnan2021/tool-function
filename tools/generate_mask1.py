import PIL.Image
import PIL.ImageDraw
import cv2
import json
import os
import numpy as np
import uuid

#LABEL_ID = [['ok'],['zhiwen','yiwu'],['ng','NG'],['qipao'],['huashang']]
LABEL_ID = [['ok'],['jiaoshui'],['xianquanyinxian']]


def label_count():
    # list_file = "/data/home/daguo/data/freshmenTask/json_list_train.txt"
    list_file = "/data/home/daguo/data/freshmenTask/datalist/train_list_all_someok.txt"

    with open(list_file,'r') as f:
        lines = f.readlines()

    # label_count = {'impurity': 0, 'edgeCrinkle': 0, 'graphite-lose': 0, 'nc-crakle': 0}
    label_count = {'ok': 0, 'impurity': 0, 'edge': 0, 'lose': 0, 'nc': 0}
    labels_list = ['ok', 'impurity', 'edge', 'lose', 'nc']

    for line in lines:
        filename = line.strip()
        label_name = filename.split('/')[10]
        print(filename)
        print(label_name)
        # label_set.append(label_name)

        for item in labels_list:
            if item in label_name:
                label_count[item] += 1

    print(label_count)

def shape_to_mask(img_shape, points, shape_type=None, 
                  line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins

def labelme_shapes_to_label(img_shape, shapes):
    # logger.warn(
    #     "labelme_shapes_to_label is deprecated, so please use "
    #     "shapes_to_label."
    # )

    label_name_to_value = {"_background_": 0}
    for shape in shapes:
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

    lbl, _ = shapes_to_label(img_shape, shapes, label_name_to_value)
    return lbl, label_name_to_value

def generate_mask():
    print("generate_mask...")
    # list_file = "/data/home/daguo/data/freshmenTask/json_list_train.txt"
    # list_file = "/data/home/daguo/data/freshmenTask/images_list_train.txt"
    list_file = "/data/home/zhendongzhou/projects/haojida/data/datalist/train_0809.txt"
    # list_file = "/data/home/daguo/data/freshmenTask/images_list_val.txt"
    # list_file = "/data/home/daguo/data/freshmenTask/images_list_val_2.txt"
    # list_file = "/data/home/daguo/data/freshmenTask/images_list_val_else.txt"

    # list_file_save = '/data/home/daguo/data/freshmenTask/datalist/train_list.txt'
    list_file_save = '/data/home/zhendongzhou/projects/haojida/data/datalist/train_list_use_0818.txt'
    # list_file_save = '/data/home/daguo/data/freshmenTask/datalist/train_list_ok.txt'
    # list_file_save = '/data/home/daguo/data/freshmenTask/datalist/val_list.txt'
    # list_file_save = '/data/home/daguo/data/freshmenTask/datalist/val_list_eles.txt'
    # list_file_save = '/data/home/daguo/data/freshmenTask/datalist/val_list_2.txt'
    with open(list_file,'r') as f:
        lines = f.readlines()

    data_lists = []
    for line in lines:
        image_file = line.strip()

        img_prefix = image_file.split('.')[-1]
        json_file = image_file.replace(img_prefix, 'json')
        print(json_file)
        print(image_file)



        # data_type = "ng"
        # # if not os.path.exists(json_file):
        # #     data_type = "ok"
        # # if data_type == "ng":
        # #     continue

        # image = cv2.imread(image_file)

        # if data_type == "ok":
        #     mask = np.zeros_like(image,dtype=np.uint8)
        # else:
        #     with open(json_file,'r',encoding='utf-8',errors='ignore') as f:
        #         json_data = json.load(f)

        #     label, label_name_value = labelme_shapes_to_label(image.shape, json_data['shapes'])
            
        #     mask = np.zeros_like(label,dtype=np.uint8)
        #     for label_name,label_value in label_name_value.items():
        #         for LABEL_, ID_ in LABEL_ID.items():
        #             if LABEL_ in label_name:
        #                 mask[ label==label_value ] = ID_


        with open(json_file, 'r') as r:
            anno = json.loads(r.read())
        inform = anno['shapes']
        height = anno['imageHeight']
        width = anno['imageWidth']
        mask = np.zeros((height, width), dtype=np.uint8)
        for j in range(len(LABEL_ID)):
            if j ==0:
                continue
            elif j==1:

                for i in inform:
                    label_id = i['label']
                    points = i['points']
                    for id_, id_category_list in enumerate(LABEL_ID):
                        #print(id_,'id_l',id_category_list)
                        if label_id in id_category_list and 'jiaoshui' in label_id:
                            #print('.......................?',label_id,id_category_list)
                            cv2.polylines(mask, np.array([points], dtype=np.int32),
                                        True, id_)
                            cv2.fillPoly(mask, np.array([points], dtype=np.int32), id_)
            else:
                for i in inform:
                    label_id = i['label']
                    points = i['points']
                    for id_, id_category_list in enumerate(LABEL_ID):
                        #print(id_,'id_l',id_category_list)
                        if label_id in id_category_list and 'yinxian' in label_id:
                            #print('.......................?',label_id,id_category_list)
                            cv2.polylines(mask, np.array([points], dtype=np.int32),
                                        True, id_)
                            cv2.fillPoly(mask, np.array([points], dtype=np.int32), id_)

        #print('............',np.unique(mask))
        image_save_path = image_file.replace("/data/home/sharedir/industrial/freshManTask", "/data/home/daguo/data/freshmenTask")
        # mask_save_path = image_save_path.replace("sourceData", "label").replace(img_prefix, 'png')
        out_path = "/data/home/zhendongzhou/projects/haojida/data/raw/mask/"
        mask_save_path = "/data/home/zhendongzhou/projects/haojida/data/raw/mask/"+  image_file.split("/")[-1].split(".")[0]+".bmp"
        #print(image_save_path)
        #print(mask_save_path)
        if not os.path.exists(out_path):
                os.mkdir(out_path)
        # if not os.path.exists(os.path.dirname(image_save_path)):
        #     os.makedirs(os.path.dirname(image_save_path))
        # if not os.path.exists(os.path.dirname(mask_save_path)):
        #     os.makedirs(os.path.dirname(mask_save_path))

        # cv2.imwrite(image_save_path, image)
        cv2.imwrite(mask_save_path, mask)
        # break
        # exit()
        #data_lists.append(image_save_path+'||'+mask_save_path+'\n')
        data_lists.append(image_file+'||'+mask_save_path+'\n')

        with open(list_file_save,'w') as f:
            f.writelines(data_lists)
    print('success!')



def get_mask_from_json(self, json_path):
        """
        解析LabelMe分割标注json,返回numpy图像
        Args:
            json_path (str): 标注json文件
        Returns:
            np.array: 分割类别标注图
        """
        with open(json_path, 'r') as r:
            anno = json.loads(r.read())
        inform = anno['shapes']
        height = anno['imageHeight']
        width = anno['imageWidth']
        mask = np.zeros((height, width), dtype=np.uint8)
        for i in inform:
            label_id = i['label']
            points = i['points']
            for id_, id_category_list in enumerate(self.category_map):
                if label_id in id_category_list:
                    cv2.polylines(mask, np.array([points], dtype=np.int32),
                                  True, id_)
                    cv2.fillPoly(mask, np.array([points], dtype=np.int32), id_)
                    break
        return mask 
if __name__ == '__main__':
    # label_count()
    # print("generate_mask...")
    generate_mask()
    # label_count()