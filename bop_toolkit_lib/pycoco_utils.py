# Author: Martin Sundermeyer (martin.sundermeyer@dlr.de)
# Robotics Institute at DLR, Department of Perception and Cognition
# Code borrowed from https://rmc-github.robotic.dlr.de/common/BlenderProc/blob/develop/src/utility/CocoUtility.py

import datetime
import numpy as np
import datetime
from skimage import measure
from bop_toolkit_lib import misc


def create_image_info(image_id, file_name, image_size):
    """Creates image info section of coco annotation

    :param image_id: integer to uniquly identify image
    :param file_name: filename for image
    :param image_size: The size of the image, given as [W, H]
    """
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": datetime.datetime.utcnow().isoformat(" "),
        "license": 1,
        "coco_url": "",
        "flickr_url": "",
    }

    return image_info


def create_annotation_info(
    annotation_id,
    image_id,
    object_id,
    binary_mask,
    bounding_box,
    mask_encoding_format="rle",
    tolerance=2,
    ignore=None,
):
    """Creates info section of coco annotation

    :param annotation_id: integer to uniquly identify the annotation
    :param image_id: integer to uniquly identify image
    :param object_id: The object id, should match with the object's category id
    :param binary_mask: A binary image mask of the object with the shape [H, W].
    :param bounding_box: [x,y,w,h] in pixels
    :param mask_encoding_format: Encoding format of the mask. Type: string.
    :param tolerance: The tolerance for fitting polygons to the objects mask.
    :param ignore: whether to ignore this gt annotation during evaluation (also matched detections with IoU>thres)
    :return: Dict containing coco annotations infos of an instance.
    """

    area = binary_mask.sum()
    if area < 1:
        return None

    if mask_encoding_format == "rle":
        segmentation = binary_mask_to_rle(binary_mask)
    elif mask_encoding_format == "polygon":
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentation:
            return None
    else:
        raise RuntimeError("Unknown encoding format: {}".format(mask_encoding_format))

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": object_id,
        "iscrowd": 0,
        "area": int(area),
        "bbox": bounding_box,
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }
    if ignore is not None:
        annotation_info["ignore"] = ignore

    return annotation_info


def merge_coco_results(existing_coco_results, new_coco_results, image_id_offset):
    """Merges the two given coco result dicts into one.

    :param existing_coco_results: A dict describing the first coco results.
    :param new_coco_results: A dict describing the second coco results.
    :return: A dict containing the merged coco results.
    """

    for res in new_coco_results:
        res["image_id"] += image_id_offset
    existing_coco_results += new_coco_results

    return existing_coco_results


def merge_coco_annotations(existing_coco_annotations, new_coco_annotations):
    """Merges the two given coco annotation dicts into one.

    The "images" and "annotations" sections are concatenated and respective ids are adjusted.

    :param existing_coco_annotations: A dict describing the first coco annotations.
    :param new_coco_annotations: A dict describing the second coco annotations.
    :return: A dict containing the merged coco annotations.
    """

    # Concatenate category sections
    for cat_dict in new_coco_annotations["categories"]:
        if cat_dict not in existing_coco_annotations["categories"]:
            existing_coco_annotations["categories"].append(cat_dict)

    # Concatenate images sections
    image_id_offset = (
        max([image["id"] for image in existing_coco_annotations["images"]]) + 1
    )
    for image in new_coco_annotations["images"]:
        image["id"] += image_id_offset
    existing_coco_annotations["images"].extend(new_coco_annotations["images"])

    # Concatenate annotations sections
    if len(existing_coco_annotations["annotations"]) > 0:
        annotation_id_offset = (
            max(
                [
                    annotation["id"]
                    for annotation in existing_coco_annotations["annotations"]
                ]
            )
            + 1
        )
    else:
        annotation_id_offset = 0
    for annotation in new_coco_annotations["annotations"]:
        annotation["id"] += annotation_id_offset
        annotation["image_id"] += image_id_offset
    existing_coco_annotations["annotations"].extend(new_coco_annotations["annotations"])

    return existing_coco_annotations, image_id_offset


def bbox_from_binary_mask(binary_mask):
    """Returns the smallest bounding box containing all pixels marked "1" in the given image mask.

    :param binary_mask: A binary image mask with the shape [H, W].
    :return: The bounding box represented as [x, y, width, height]
    """
    # Find all columns and rows that contain 1s
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    # Find the min and max col/row index that contain 1s
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Calc height and width
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    return [int(cmin), int(rmin), int(w), int(h)]


def close_contour(contour):
    """Makes sure the given contour is closed.

    :param contour: The contour to close.
    :return: The closed contour.
    """
    # If first != last point => add first point to end of contour to close it
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    :param binary_mask: a 2D binary numpy array where '1's represent the object
    :param tolerance: Maximum distance from original points of polygon to approximated polygonal chain. If
                        tolerance is 0, the original coordinate array is returned.
    :return: Mask in polygon format
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = np.array(measure.find_contours(padded_binary_mask, 0.5))
    # Reverse padding
    contours = contours - 1
    for contour in contours:
        # Make sure contour is closed
        contour = close_contour(contour)
        # Approximate contour by polygon
        polygon = measure.approximate_polygon(contour, tolerance)
        # Skip invalid polygons
        if len(polygon) < 3:
            continue
        # Flip xy to yx point representation
        polygon = np.flip(polygon, axis=1)
        # Flatten
        polygon = polygon.ravel()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        polygon[polygon < 0] = 0
        polygons.append(polygon.tolist())

    return polygons


def binary_mask_to_rle(binary_mask):
    """Converts a binary mask to COCOs run-length encoding (RLE) format. Instead of outputting
    a mask image, you give a list of start pixels and how many pixels after each of those
    starts are included in the mask.

    :param binary_mask: a 2D binary numpy array where '1's represent the object
    :return: Mask in RLE format
    """
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")
    mask = binary_mask.ravel(order="F")
    if len(mask) > 0 and mask[0] == 1:
        counts.append(0)

    if len(mask) > 0:
        # Determine mask for pixels where values switch
        mask_changes = mask[:-1] != mask[1:]
        # Determine indices of changing pixels
        changes_indx = np.where(np.concatenate(([True], mask_changes, [True]), 0))[0]
        # Compute diff of consecutive changing indices => length of segments
        rle2 = np.diff(changes_indx)
        counts.extend(rle2.tolist())
    return rle


def rle_to_binary_mask(rle):
    """Converts a COCOs run-length encoding (RLE) to a binary mask.

    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get("size")), dtype=bool)
    counts = rle.get("counts")
    if isinstance(counts, str):
        misc.log("===========")
        misc.log("RLEs are compressed, using cocoAPI to uncompress them..")
        misc.log("Make sure, requirements.txt are installed.")
        misc.log("===========")
        from pycocotools import mask as maskUtils

        binary_mask = maskUtils.decode(rle)
    else:
        start = 0
        for i in range(len(counts) - 1):
            start += counts[i]
            end = start + counts[i + 1]
            binary_array[start:end] = (i + 1) % 2

        binary_mask = binary_array.reshape(*rle.get("size"), order="F")

    return binary_mask


def compute_ious(gt, dt, iou_type):
    """
    Compute the Intersection over Union between masks in RLE format
    :param gt: Masks in RLE format
    :param dt: Masks in RLE format
    :param iou_type: Can be 'segm' or 'bbox'
    :return: matrix of ious between all gt and dt masks
    """

    if iou_type == "segm":
        gt_bin = np.array([rle_to_binary_mask(g["segmentation"]) for g in gt])
        dt_bin = np.array([rle_to_binary_mask(d["segmentation"]) for d in dt])
        intersections = np.einsum("ijk,ljk->il", dt_bin, gt_bin)
        unions = np.sum(
            (np.expand_dims(dt_bin, 1) + np.expand_dims(gt_bin, 0)) > 0, axis=(2, 3)
        )
        ious = intersections / unions
    return ious
