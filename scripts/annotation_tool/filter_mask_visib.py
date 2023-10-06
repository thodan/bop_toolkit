import glob
import os
import numpy as np
import cv2
from tqdm import tqdm


# PARAMETERS.
################################################################################
p = {
    # Folder containing the BOP datasets.
    'dataset_path': '/path/to/dataset',

    # Dataset split. Options: 'train', 'test'.
    'dataset_split': 'train',

    # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
    'dataset_split_type': None,
}
################################################################################

def main():
    if p['dataset_split_type'] is not None:
        p['dataset_split'] = p['dataset_split'] + '_' + p['dataset_split_type']
    samples = glob.glob(p['dataset_path'] + '/' + p['dataset_split'] + '/*')

    for sample in samples:
        print('sample: ', sample)
        # load mask_visib folder
        mask_visib = os.path.join(sample, 'mask_visib')
        # read all mask_visib images
        mask_visib_images = glob.glob(mask_visib + '/*.png')
        # make filtered mask_visib folder
        mask_visib_folder = sample + '/mask_visib_filtered'
        if not os.path.exists(mask_visib_folder):
            os.mkdir(mask_visib_folder)

        for mask_visib_image in tqdm(mask_visib_images):
            # remove pixel values < 128
            mask_visib = cv2.imread(mask_visib_image, cv2.IMREAD_GRAYSCALE)
            #cv2.kmeans(mask_visib, 2, mask_visib, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)

            # get contours in mask_visib image
            contours, hierarchy = cv2.findContours(mask_visib, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # convert mask_visib image to 3 channel image to visualize it
            #mask_visib_visual = cv2.cvtColor(mask_visib, cv2.COLOR_GRAY2BGR)
            # draw contours in red color
            #mask_visib_visual_contour = cv2.drawContours(mask_visib_visual, contours, -1, (0, 0, 255), 1)
            # show genrated mask_visib image
            #cv2.imshow("img", mask_visib_visual_contour)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # remove contours with area less than 600 pixel
            for contour in contours:
                if cv2.contourArea(contour) < 800:
                    cv2.fillPoly(mask_visib, pts=[contour], color=(0))

            # replaced the 'mask_visib' with 'mask_visib_filtered' in the mask_visib_image path
            mask_visib_file = mask_visib_image.replace('mask_visib', 'mask_visib_filtered')
            cv2.imwrite(mask_visib_file, mask_visib)
            # show genrated mask_visib image
            #cv2.imshow("img", mask_visib)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
