# Author: Martin Sundermeyer (martin.sundermeyer@dlr.de)
# Robotics Institute at DLR, Department of Perception and Cognition
# Code borrowed from https://rmc-github.robotic.dlr.de/common/BlenderProc/blob/develop/src/utility/CocoUtility.py


import datetime
import numpy as np
from skimage import measure

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
        "date_captured": datetime.datetime.utcnow().isoformat(' '),
        "license": 1,
        "coco_url": "",
        "flickr_url": ""
    }

    return image_info

def create_annotation_info(annotation_id, image_id, object_id, binary_mask, tolerance=2):
    """Creates info section of coco annotation

    :param annotation_id: integer to uniquly identify the annotation
    :param image_id: integer to uniquly identify image
    :param object_id: The object id, should match with the object's category id
    :param binary_mask: A binary image mask of the object with the shape [H, W].
    :param tolerance: The tolerance for fitting polygons to the objects mask.
    """

    if np.any(binary_mask):
        bounding_box = bbox_from_binary_mask(binary_mask)   
        area = bounding_box[2] * bounding_box[3]
        if area < 1:
            return None
    else:
        return None
        
    segmentation = binary_mask_to_polygon(binary_mask, tolerance)

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": object_id,
        "iscrowd": 0,
        "area": [area],
        "bbox": bounding_box,
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }
    return annotation_info

def bbox_from_binary_mask(binary_mask):
    """ Returns the smallest bounding box containing all pixels marked "1" in the given image mask.

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
    h = rmax - rmin
    w = cmax - cmin
    return [int(cmin), int(rmin), int(w), int(h)]

def close_contour(contour):
    """ Makes sure the given contour is closed.

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
        :param tolerance: Maximum distance from original points of polygon to approximated polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
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