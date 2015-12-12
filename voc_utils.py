import pandas as pd
import os
from bs4 import BeautifulSoup
from more_itertools import unique_everseen
import numpy as np
import matplotlib.pyplot as plt
import skimage


root_dir = '/Users/mprat/personal/VOCdevkit/VOCdevkit/VOC2012/'
img_dir = os.path.join(root_dir, 'JPEGImages/')
ann_dir = os.path.join(root_dir, 'Annotations')
set_dir = os.path.join(root_dir, 'ImageSets', 'Main')


def list_image_sets():
    return [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']


# category name is from above, dataset is either "train" or
# "val" or "train_val"
def imgs_from_category(cat_name, dataset):
    filename = os.path.join(set_dir, cat_name + "_" + dataset + ".txt")
    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        header=None,
        names=['filename', 'true'])
    return df


def imgs_from_category_as_list(cat_name, dataset):
    df = imgs_from_category(cat_name, dataset)
    df = df[df['true'] == 1]
    return df['filename'].values


def annotation_file_from_img(img_name):
    return os.path.join(ann_dir, img_name) + '.xml'


# annotation operations
def load_annotation(img_filename):
    xml = ""
    with open(annotation_file_from_img(img_filename)) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml)


def get_all_obj_and_box(objname, img_set):
    img_list = imgs_from_category_as_list(objname, img_set)

    for img in img_list:
        annotation = load_annotation(img)


# image operations
def load_img(img_filename):
    """
    Default is color
    """
    if os.path.isfile(img_filename):
        img = skimage.img_as_float(skimage.io.imread(
            img_filename)).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    return load_img(os.path.join(img_dir, img_filename + '.jpg'))


def _load_data(category, data_type=None):
    if data_type is None:
        raise ValueError('Must provide data_type = train or val')
    to_find = category
    filename = '/Users/mprat/personal/VOCdevkit/VOCdevkit/VOC2012/csvs/' + \
        data_type + '_' + \
        category + '.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        train_img_list = imgs_from_category_as_list(to_find, data_type)
        data = []
        for item in train_img_list:
            anno = load_annotation(item)
            objs = anno.findAll('object')
            for obj in objs:
                obj_names = obj.findChildren('name')
                for name_tag in obj_names:
                    if str(name_tag.contents[0]) == category:
                        fname = anno.findChild('filename').contents[0]
                        bbox = obj.findChildren('bndbox')[0]
                        xmin = int(bbox.findChildren('xmin')[0].contents[0])
                        ymin = int(bbox.findChildren('ymin')[0].contents[0])
                        xmax = int(bbox.findChildren('xmax')[0].contents[0])
                        ymax = int(bbox.findChildren('ymax')[0].contents[0])
                        data.append([fname, xmin, ymin, xmax, ymax])
        df = pd.DataFrame(
            data, columns=['fname', 'xmin', 'ymin', 'xmax', 'ymax'])
        df.to_csv(filename)
        return df


def get_image_url_list(category, data_type=None):
    df = _load_data(category, data_type=data_type)
    image_url_list = list(
        unique_everseen(list(img_dir + df['fname'])))
    return image_url_list


def get_feature_filename(
        net_name, feature_name, cat_name, data_type, normalize=False):
    if normalize:
        fname = cat_name + "_" + net_name + "_" + feature_name + "_norm.msgpack"
    else:
        fname = cat_name + "_" + net_name + "_" + feature_name + ".msgpack"
    return os.path.join('data', 'VOC2012', data_type, fname)


def get_masks(cat_name, data_type, mask_type=None):
    # change this to searching through the df
    # for the bboxes instead of relying on the order
    # so far, should be OK since I'm always loading
    # the df from disk anyway
    # mask_type should be bbox1 or bbox
    if mask_type is None:
        raise ValueError('Must provide mask_type')
    df = _load_data(cat_name, data_type=data_type)
    # load each image, turn into a binary mask
    masks = []
    prev_url = ""
    blank_img = None
    for row_num, entry in df.iterrows():
        img_url = os.path.join(img_dir, entry['fname'])
        if img_url != prev_url:
            if blank_img is not None:
                # TODO: options for how to process the masks
                # make sure the mask is from 0 to 1
                max_val = blank_img.max()
                if max_val > 0:
                    min_val = blank_img.min()
                    # print "min val before normalizing: ", min_val
                    # start at zero
                    blank_img -= min_val
                    # print "max val before normalizing: ", max_val
                    # max val at 1
                    blank_img /= max_val
                masks.append(blank_img)
            prev_url = img_url
            img = load_img(img_url)
            blank_img = np.zeros((img.shape[0], img.shape[1], 1))
        bbox = [entry['xmin'], entry['ymin'], entry['xmax'], entry['ymax']]
        if mask_type == 'bbox1':
            blank_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.0
        elif mask_type == 'bbox2':
            blank_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] += 1.0
        else:
            raise ValueError('Not a valid mask type')
    # TODO: options for how to process the masks
    # make sure the mask is from 0 to 1
    max_val = blank_img.max()
    if max_val > 0:
        min_val = blank_img.min()
        # print "min val before normalizing: ", min_val
        # start at zero
        blank_img -= min_val
        # print "max val before normalizing: ", max_val
        # max val at 1
        blank_img /= max_val
    masks.append(blank_img)
    return np.array(masks)


def get_imgs(cat_name, data_type):
    image_url_list = get_image_url_list(cat_name, data_type=data_type)
    imgs = []
    for url in image_url_list:
        imgs.append(load_img(url))
    return np.array(imgs)


def display_image_and_mask(img, mask):
    plt.figure(1)
    plt.clf()
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.imshow(img)
    ax1.set_title('Original image')
    ax2.imshow(mask)
    ax2.set_title('Mask')
    plt.show(block=False)


def get_predictor_name(cat_name=None, net_name=None, feature_name=None,
                       predictor_type=None, neuron_size=None,
                       max_degree=None, mask_type=None):
    if mask_type is None:
        raise ValueError('Must provide a mask type')
    if max_degree is None:
        return os.path.join(
                'predictor', 'VOC2012', predictor_type, cat_name, mask_type,
                '_'.join([net_name, feature_name, str(neuron_size),
                          ]) + '.msgpack')
    else:
        return os.path.join(
                'predictor', 'VOC2012', predictor_type, cat_name, mask_type,
                '_'.join([net_name, feature_name, str(neuron_size),
                          str(int(max_degree))]) + '.msgpack')


def cat_name_to_cat_id(cat_name):
    cat_id_dict = {
        "aeroplane": 1,
        "bicycle": 2,
        "bird": 3,
        "boat": 4,
        "bottle": 5,
        "bus": 6,
        "car": 7,
        "cat": 8,
        "chair": 9,
        "cow": 10,
        "diningtable": 11,
        "dog": 12,
        "horse": 13,
        "motorbike": 14,
        "person": 15,
        "pottedplant": 16,
        "sheep": 17,
        "sofa": 18,
        "train": 19,
        "tvmonitor": 20
        }
    return cat_id_dict[cat_name]


# methods for COCO result generation
def get_coco_result_struct(fname, cat_name, bbox, score):
    fname = os.path.basename(fname)
    image_id = fname.replace('.jpg', '')
    image_id = image_id.replace('_', '')
    cat_id = cat_name_to_cat_id(cat_name)
    result = {}
    result['bbox'] = bbox
    result['score'] = score
    result['image_id'] = int(image_id)
    result['category_id'] = cat_id
    return result


def get_coco_results_name(
        cat_name, net_name, feature_name, predictor_type,
        neuron_size,
        max_degree=None):
    #   single_no_thresh
    if max_degree is None:
        return os.path.join(
            'results', 'VOC2012', predictor_type, cat_name,
            '_'.join([net_name, feature_name,
                      str(neuron_size)]) + '.json')
    else:
        return os.path.join(
            'results', 'VOC2012', predictor_type, cat_name,
            '_'.join([net_name, feature_name,
                      str(neuron_size),
                      str(int(max_degree))]) + '.json')


def display_img_and_masks(
        img, true_mask, predicted_mask, block):
    m_predicted_color = predicted_mask.reshape(
        predicted_mask.shape[0], predicted_mask.shape[1])
    m_true_color = true_mask.reshape(
        true_mask.shape[0], true_mask.shape[1])
    # m_predicted_color = predicted_mask
    # m_true_color = true_mask
    # plt.close(1)
    plt.figure(1)
    plt.clf()
    plt.axis('off')
    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, num=1)
    # f.clf()
    ax1.get_xaxis().set_ticks([])
    ax2.get_xaxis().set_ticks([])
    ax3.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    ax3.get_yaxis().set_ticks([])

    ax1.imshow(img)
    ax2.imshow(m_true_color)
    ax3.imshow(m_predicted_color)
    plt.draw()
    plt.show(block=block)
