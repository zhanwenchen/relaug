import random
import os
from os import environ as os_environ
from os.path import join as os_path_join
import json
from json import load as json_load
from collections import defaultdict
from PIL import Image
from PIL.Image import Transpose
import h5py
import numpy as np
from numpy import array as np_array, int32 as np_int32
import torch
from torch import (
    as_tensor as torch_as_tensor,
    int64 as torch_int64,
)
from tqdm import tqdm
from maskrcnn_benchmark.structures.bounding_box import BoxList
try:
    from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
except:
    pass


ONE = 1
FLIP_LEFT_RIGHT = Transpose.FLIP_LEFT_RIGHT
BOX_SCALE = 1024  # Scale at which we have the boxes
VG_DIRPATH = os_path_join(os_environ['DATASETS_DIR'], 'visual_genome')
DICT_FILE_FPATH = os_path_join(VG_DIRPATH, 'VG-SGG-dicts-with-attri-info.json')


class VGDataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, use_graft, transforms=None,
                filter_empty_rels=True, num_im=-1, num_val_im=5000,
                filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, custom_eval=False, custom_path='', with_clean_classifier=False, get_state=False):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # for debug
        # num_im = 10000
        # num_val_im = 4

        assert split in {'train', 'val', 'test'}
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.is_train = is_train = split == 'train'
        self.filter_non_overlap = filter_non_overlap and is_train
        self.filter_duplicate_rels = filter_duplicate_rels and is_train
        self.transforms = transforms

        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(dict_file) # contiguous 151, 51 containing __background__
        self.categories = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        self.custom_eval = custom_eval
        if self.custom_eval:
            self.get_custom_imgs(custom_path)
        else:
            self.filenames, self.img_info = load_image_filenames(img_dir, image_file) # length equals to split_mask
            self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
                self.roidb_file, self.split, num_im, num_val_im=num_val_im,
                filter_empty_rels=filter_empty_rels,
                filter_non_overlap=self.filter_non_overlap,
                ind_to_predicates=self.ind_to_predicates,
                img_info=self.img_info,
                with_clean_classifier=with_clean_classifier,
                get_state =get_state,
            )

            self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
            self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]

        print(f'VGDataset: use_graft={use_graft}')
        self.use_graft = use_graft
        self.flip_aug_and_train = flip_aug and is_train

    def __getitem__(self, index):
        transforms = self.transforms
        if self.custom_eval:
            img = Image.open(self.custom_files[index]).convert("RGB")
            target = torch.LongTensor([-1])
            if transforms is not None:
                img, target = transforms(img, target)
            return img, target, index

        img = Image.open(self.filenames[index]).convert("RGB")
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('='*20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']), ' ', str(self.img_info[index]['height']), ' ', '='*20)

        flip_img = (random.random() > 0.5) and self.flip_aug and self.is_train
        target = self.get_groundtruth(index, flip_img)

        if flip_img:
            img = img.transpose(method=FLIP_LEFT_RIGHT)

        if transforms is not None:
            img, target = transforms(img, target)

        return img, target, index


    def get_statistics(self):
        fg_matrix, bg_matrix, stats = self.get_VG_statistics(must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_attributes,
            'stats': stats,
        }
        return result

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        if os.path.isdir(path):
            for file_name in tqdm(os.listdir(path)):
                self.custom_files.append(os.path.join(path, file_name))
                img = Image.open(os.path.join(path, file_name)).convert("RGB")
                self.img_info.append({'width':int(img.width), 'height':int(img.height)})
        # Expecting a list of paths in a json file
        if os.path.isfile(path):
            file_list = json.load(open(path))
            for file in tqdm(file_list):
                self.custom_files.append(file)
                img = Image.open(file).convert("RGB")
                self.img_info.append({'width': int(img.width), 'height': int(img.height)})

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        if flip_img:
            new_xmin = w - box[:,2]
            new_xmax = w - box[:,0]
            box[:,0] = new_xmin
            box[:,2] = new_xmax
        target = BoxList(box, (w, h), 'xyxy') # xyxy

        target.add_field("labels", torch_as_tensor(self.gt_classes[index], dtype=torch_int64))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy() # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i,0]), int(relation[i,1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
            else:
                relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
        target.add_field("relation", relation_map, is_triplet=True)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation)) # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target

    def __len__(self):
        if self.custom_eval:
            return len(self.custom_files)
        return len(self.filenames)


    def get_VG_statistics(self, must_overlap=True):
        num_obj_classes = len(self.ind_to_classes)
        num_rel_classes = len(self.ind_to_predicates)
        fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
        bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

        gt_classes = self.gt_classes
        relationships = self.relationships
        use_graft = self.use_graft is True
        if use_graft:
            stats = []
            get_groundtruth = self.get_groundtruth
        else:
            gt_boxes = self.gt_boxes

        for ex_ind in tqdm(range(len(self))):
            # NOTE: the gt_boxes are right. The images haven't been transformed yet.
            gt_classes_i = gt_classes[ex_ind]
            gt_relations_i = relationships[ex_ind]
            if use_graft:
                target = get_groundtruth(ex_ind, evaluation=True, flip_img=False)
                bbox = target.bbox.numpy()
                del target
                keep = (bbox[:, 3] > bbox[:, 1]) & (bbox[:, 2] > bbox[:, 0])
                gt_boxes_i = bbox.astype(int)
                del bbox
            else:
                gt_boxes_i = gt_boxes[ex_ind]

            # For the foreground, we'll just look at everything
            o1o2_indices = gt_relations_i[:, :2]
            o1o2 = gt_classes_i[o1o2_indices] # Regardless
            # QUESTION: are indicies and o1o2 even? Yes.
            assert len(o1o2_indices) == len(o1o2)
            for idx, ((o1_idx, o2_idx), (o1, o2), gtr) in enumerate(zip(o1o2_indices, o1o2, gt_relations_i[:, 2])):
                fg_matrix[o1, o2, gtr] += 1 # Keep shouldn't affect simple stats
                if use_graft:
                    if keep[o1_idx] and keep[o2_idx]:
                        gt_box_o1 = gt_boxes_i[o1_idx]
                        gt_box_o2 = gt_boxes_i[o2_idx]
                        row = [ex_ind, o1_idx, o1] + list(gt_box_o1) + [o2_idx, o2] + list(gt_box_o2) + [idx, gtr]
                        stats.append(row)
            # For the background, get all of the things that overlap.
            o1o2_total = gt_classes_i[np.array(
                box_filter(gt_boxes_i, must_overlap=must_overlap), dtype=int)]
            for (o1, o2) in o1o2_total:
                bg_matrix[o1, o2] += 1
        if use_graft:
            return fg_matrix, bg_matrix, stats
        return fg_matrix, bg_matrix, None


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    boxes_float = boxes.astype(float)
    overlaps = bbox_overlaps(boxes_float, boxes_float, to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter

def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:
        json.dump(data, outfile)

def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

    return ind_to_classes, ind_to_predicates, ind_to_attributes


def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info



def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap,
                ind_to_predicates=None, img_info=None, with_clean_classifier=False, get_state=False):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[:num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    pred_topk = []
    pred_num = 15
    pred_count=0
    with open(DICT_FILE_FPATH,'rb') as f:
        vg_dict_info = json_load(f)
    predicates_tree = vg_dict_info['predicate_count']
    del vg_dict_info
    predicates_sort = sorted(predicates_tree.items(), key=lambda x:x[1], reverse=True)
    for pred_i in predicates_sort:
        if pred_count >= pred_num:
            break
        pred_topk.append(str(pred_i[0]))
        pred_count += 1

    if with_clean_classifier:
        root_classes = pred_topk
    else:
        root_classes = None
    if get_state:
        root_classes = None
    root_classes_count = {}
    leaf_classes_count = {}
    all_classes_count = {}
    for i, image_idx in enumerate(image_index):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start : i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start : i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start : i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start : i_rel_end + 1]
            obj_idx = _relations[i_rel_start : i_rel_end + 1] - i_obj_start # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates)) # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_idx] = 0
                continue
        if root_classes is not None and split == 'train':
            # print('old boxes: ', boxes_i)
            # print('old gt_classes_i: ', gt_classes_i)
            # print('old rels: ', rels)
            rel_temp = []
            # print('rels: ',rels)
            for rel_i in rels:
                rel_i_pred = ind_to_predicates[rel_i[2]]
                if rel_i_pred not in all_classes_count:
                    all_classes_count[rel_i_pred] = 0
                all_classes_count[rel_i_pred] = all_classes_count[rel_i_pred] + 1
                if rel_i_pred not in root_classes or rel_i[2] == 0:
                    rel_i_leaf = rel_i

                    if rel_i_pred not in leaf_classes_count:
                        leaf_classes_count[rel_i_pred] = 0
                    leaf_classes_count[rel_i_pred] = leaf_classes_count[rel_i_pred] + 1
                    rel_temp.append(rel_i_leaf)
                if rel_i_pred in root_classes:
                    rel_i_root = rel_i
                    if rel_i_pred not in root_classes_count:
                        root_classes_count[rel_i_pred] = 0
                    if root_classes_count[rel_i_pred] < 2000: #1000: #2000:
                        rel_temp.append(rel_i_root)
                        root_classes_count[rel_i_pred] = root_classes_count[rel_i_pred] + 1
            if len(rel_temp) == 0:
                split_mask[image_idx] = 0
                continue
            rels = np_array(rel_temp, dtype=np_int32)
        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)
    print('split: ',split)
    print('root_classes_count: ', root_classes_count)
    count_list = [0,] + list(root_classes_count.values())
    count_list_np = np_array(count_list)
    print('mean root class number: ', count_list_np.mean())
    print('sum root class number: ', count_list_np.sum())

    print('leaf_classes_count: ', leaf_classes_count)
    count_list = [0,] + list(leaf_classes_count.values())
    count_list_np = np_array(count_list)
    print('mean leaf class number: ', count_list_np.mean())
    print('sum leaf class number: ', count_list_np.sum())
    print('all_classes_count: ', all_classes_count)
    count_list = [0,] + list(all_classes_count.values())
    count_list_np = np_array(count_list)
    print('mean all class number: ', count_list_np.mean())
    print('sum all class number: ', count_list_np.sum())
    print('number images: ', split_mask.sum())
    return split_mask, boxes, gt_classes, gt_attributes, relationships
