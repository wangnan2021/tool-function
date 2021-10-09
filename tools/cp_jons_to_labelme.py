import json
import cv2
import os
## 产品标注转标准json ##
def transform_pro_to_labelme(pro_json,save_root=None):
    print('.......',pro_json)
    ar = pro_json.split('/')
    # 拿到image的name
    name = ar[-1][:3]
    # 拿到imagePath
    image_path = os.path.join("/data/home/nanwang/data/MEIS/20210526/18_images", name+".png")
    image = cv2.imread(image_path)
    init ={
        'version':"4.5.7",
        'flag':{},
        'shapes':[],
        'imagePath':image_path,
        "imageData": None,
        'imageHeight':image.shape[0],
        'imageWidth':image.shape[1]

    }
    with open(pro_json,'r') as r:
        anno = json.loads(r.read())
        stracture = anno['Labels']
        for idx in stracture:
            Shape = idx['Shape']
            Type_ = idx['Type']
            print('=============',idx)
            if Shape =='polygon':
                xy =[]
                points = idx['Points']
                for p in points:
                    xy.append([p['X'],p['Y']])
                
            init['shapes'].append({'label':Type_,
                                    'points':xy,
                                    'groud_id':None,
                                    'shape_type':"polygon",
                                    'flags':{}})
    os.makedirs("/data/home/nanwang/data/MEIS/20210526/trans", exist_ok=True)
    with open(os.path.join("/data/home/nanwang/data/MEIS/20210526/trans",name+'.json'),'w') as f:
        json.dump(init,f,indent=2)
        # f.close()
# transform_pro_to_labelme(pro_json='3ccdcd2e_30.jpg.json',save_root='save')
import glob
root = '/data/home/nanwang/data/MEIS/20210526/18_labels'
path = []
for i in os.listdir(root):
    # pa = glob.glob(os.path.join(root,i,'*.json'))
    # print(pa)
    # for j in pa:
    #     path.append(j)
    path.append(os.path.join(root, i))
for i in path:
    transform_pro_to_labelme(pro_json=i,save_root='save')