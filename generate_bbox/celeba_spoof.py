import json
import os

from PIL import Image


def read_bbox_from_file(bbox_file):
    if not os.path.exists(bbox_file):
        return None

    with open(bbox_file, 'r') as f:
        line = f.readline()
        data = line.split(' ')

    if len(line) == 0:
        return None

    return data[0:4]


def read_image_bbox(root_path, data_path, label):
    result = {}
    full_path = os.path.join(root_path, data_path)
    try:
        image_pathes = filter(lambda x: not '.txt' in x, os.listdir(full_path))
        for img_path in list(image_pathes):
            img_ext = os.path.splitext(img_path)[1]
            if not (img_ext == '.jpg' or img_ext == '.png'):
                print('not supported image type {}'.format(img_path))
                continue
            img_path = os.path.join(data_path, img_path)
            bbox_path = img_path.replace(img_ext, '_BB.txt')
            bbox = read_bbox_from_file(os.path.join(root_path, bbox_path))
            if bbox == None:
                print('bbox file not exists: {}'.format(img_path))
                continue
            img = Image.open(os.path.join(root_path, img_path))
            img_w, img_h = img.size
            x = int(bbox[0]) * (img_w / 224)
            y = int(bbox[1]) * (img_h / 224)
            w = int(bbox[2]) * (img_w / 224)
            h = int(bbox[3]) * (img_h / 224)
            result[img_path] = {
                'label': label,
                'x1': int(x),
                'y1': int(y),
                'x2': int(x+w),
                'y2': int(y+h)
            }
    except Exception as e:
        print(full_path)
        print(e)

    return result

def save_bbox_to_json(root_path, data_path, out_file):
    results = {}
    live_count = 0
    spoof_count = 0
    full_path = os.path.join(root_path, data_path)
    for n, identity in enumerate(os.listdir(full_path)):
        if not os.path.isdir(os.path.join(full_path, identity)):
            continue
        live_path = os.path.join(data_path, identity, 'live')
        if os.path.exists(os.path.join(root_path, live_path)):
            data_dict = read_image_bbox(root_path, live_path, 0)
            live_count += len(data_dict)
            results.update(data_dict)
        # else:
            # print('{} has not live files.'.format(identity))

        spoof_path = os.path.join(data_path, identity, 'spoof')
        if os.path.exists(os.path.join(root_path, spoof_path)):
            data_dict = read_image_bbox(root_path, spoof_path, 1)
            spoof_count += len(data_dict)
            results.update(data_dict)
        # else:
            # print('{} has not spoof files.'.format(identity))
        if n % 10 == 0:
            print('{} identity are processed'.format(n))
    
    total = live_count + spoof_count
    print('# of live files : {}, {:.2f}'.format(live_count, live_count / total))
    print('# of spoof files : {}, {:.2f}'.format(spoof_count, spoof_count / total))
    with open(out_file, 'w') as out_f:
        json.dump(results, out_f, indent=2)
