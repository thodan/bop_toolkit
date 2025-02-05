import os
import unittest

from bop_toolkit_lib import inout


class TestInout(unittest.TestCase):

    def setUp(self) -> None:
        self.csv_6d_path = 'bop_toolkit_lib/tests/data/cnos-fastsammegapose_icbin-test_7c9f443f-b900-41bb-af01-09b8eddfc2c4.csv'
        self.json_coco_path = 'bop_toolkit_lib/tests/data/zebraposesat-effnetb4_ycbv-test_5ed0eecc-96f8-498b-9438-d586d4d92528.json'

    def test_coco_json(self):
        """
        Check json loading/savings functions.
        """
        json_coco_bis_path = self.json_coco_path + '_bis'
        json_coco_gz_path = self.json_coco_path + '.gz'

        # load example json coco submission
        coco_res = inout.load_json(self.json_coco_path)
        # save an uncompressed copy
        inout.save_json(json_coco_bis_path, coco_res)
        # load the uncompressed copy
        coco_res_from_bis = inout.load_json(json_coco_bis_path)
        # saves a compressed copy
        inout.save_json(self.json_coco_path, coco_res, compress=True)
        # load the compressed copy
        coco_res_from_gz = inout.load_json(json_coco_gz_path)

        # A few checks to assess if format/values are kept the same
        self.assertTrue(len(coco_res) == len(coco_res_from_bis) == len(coco_res_from_gz))
        self.assertTrue(coco_res[0].keys() == coco_res_from_bis[0].keys() == coco_res_from_gz[0].keys())
        self.assertTrue(coco_res[0]['time'] == coco_res_from_bis[0]['time'] == coco_res_from_gz[0]['time'])
        self.assertTrue(coco_res[0]['score'] == coco_res_from_bis[0]['score'] == coco_res_from_gz[0]['score'])
        self.assertTrue(coco_res[0]['bbox'] == coco_res_from_bis[0]['bbox'] == coco_res_from_gz[0]['bbox'])
        inout.check_coco_results(self.json_coco_path)
        inout.check_coco_results(json_coco_gz_path)

        # ISSUE: save_coco_results and check_coco_results not compatible
        # # create fake coco results with save_coco_results format 
        # save_coco_keys = ["scene_id", "im_id", "obj_id", "score", "bbox", "run_time"]  
        # coco_res_keys = ["scene_id", "image_id", "category_id", "score", "bbox", "time"]
        # nmap = {kc: ks for kc, ks in zip(coco_res_keys, save_coco_keys)}

        # coco_res_remap = []
        # for res in coco_res:
        #     # Remap keys
        #     coco_res_remap.append({nmap[kc]: val for kc, val in res.items()})
        # inout.save_coco_results(json_coco_bis_path, coco_res_remap)
        # inout.save_coco_results(self.json_coco_path, coco_res_remap, compress=True)
        # inout.check_coco_results(json_coco_gz_path)

        # cleanup
        os.remove(json_coco_bis_path)
        os.remove(json_coco_gz_path)




if __name__ == "__main__":
    unittest.main()
