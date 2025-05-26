import numpy as np


def calculate_shape_IoU(pred, seg_label, category_id, mapping):
    category_id_to_hash_code_mapping = {}
    for hash_code in list(mapping.keys()):
        category_id_to_hash_code_mapping[str(mapping[hash_code]['category_id'])] = hash_code
    shape_ious = []
    for shape_id in range(category_id.shape[0]):
        hash_code = category_id_to_hash_code_mapping[str(category_id[shape_id])]
        parts_id = mapping[hash_code]['parts_id']
        part_ious = []
        for part in parts_id:
            I = np.sum(np.logical_and(pred[shape_id] == part, seg_label[shape_id] == part))
            U = np.sum(np.logical_or(pred[shape_id] == part, seg_label[shape_id] == part))
            if U == 0:
                iou = 1  # If the union of ground truth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def calculate_category_IoU(shape_ious, category_id, mapping):

    # get category names
    category_id_to_hash_code_mapping = {}
    for hash_code in list(mapping.keys()):
        category_id_to_hash_code_mapping[str(mapping[hash_code]['category_id'])] = hash_code
    categories = []
    for cat_id in category_id:
        hash_code = category_id_to_hash_code_mapping[str(cat_id)]
        categories.append(mapping[hash_code]['category'])

    # calculate category IoU
    collections = {}
    category_IoU = {}
    for hash_code in list(mapping.keys()):
        collections[mapping[hash_code]['category']] = []
    for category, shape_iou in zip(categories, shape_ious):
        collections[category].append(shape_iou)
    for category in list(collections.keys()):
        category_IoU[category] = sum(collections[category]) / len(collections[category])
    return category_IoU


def calculate_accuracy(preds, cls_labels):
    acc = np.mean(preds == cls_labels)
    return acc


def calculate_category_accuracy(preds, cls_labels, mapping):
    category_accuracy = {}
    for category_id in range(40):
        idx = np.where(cls_labels == category_id)
        acc = np.mean(preds[idx] == category_id)
        category_name = mapping[category_id]
        category_accuracy[category_name] = acc
    return category_accuracy
