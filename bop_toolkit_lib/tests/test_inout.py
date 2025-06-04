import os
import unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_almost_equal


from bop_toolkit_lib import inout


class TestInout(unittest.TestCase):

    def setUp(self) -> None:
        results_dir = Path(__file__).parent / "data/results_sub"
        self.csv_6d_path = results_dir / "cnos-fastsammegapose_icbin-test_7c9f443f-b900-41bb-af01-09b8eddfc2c4.csv"
        self.json_coco_path = results_dir / "zebraposesat-effnetb4_ycbv-test_5ed0eecc-96f8-498b-9438-d586d4d92528.json"

    def test_parse_result_filename(self):
        # works with full path .csv
        result_name, method, dataset, split, split_type, ext = inout.parse_result_filename(self.csv_6d_path)
        self.assertEqual(result_name, "cnos-fastsammegapose_icbin-test_7c9f443f-b900-41bb-af01-09b8eddfc2c4")
        self.assertEqual(method, "cnos-fastsammegapose")
        self.assertEqual(dataset, "icbin")
        self.assertEqual(split, "test")
        self.assertIsNone(split_type)
        self.assertEqual(ext, "csv")

        # or filename only .csv
        result_name, method, dataset, split, split_type, ext = inout.parse_result_filename(self.csv_6d_path.name)
        self.assertEqual(result_name, "cnos-fastsammegapose_icbin-test_7c9f443f-b900-41bb-af01-09b8eddfc2c4")
        self.assertEqual(method, "cnos-fastsammegapose")
        self.assertEqual(dataset, "icbin")
        self.assertEqual(split, "test")
        self.assertIsNone(split_type)
        self.assertEqual(ext, "csv")

        # .json too
        result_name, method, dataset, split, split_type, ext = inout.parse_result_filename(self.json_coco_path)
        self.assertEqual(result_name, "zebraposesat-effnetb4_ycbv-test_5ed0eecc-96f8-498b-9438-d586d4d92528")
        self.assertEqual(method, "zebraposesat-effnetb4")
        self.assertEqual(dataset, "ycbv")
        self.assertEqual(split, "test")
        self.assertIsNone(split_type)
        self.assertEqual(ext, "json")

        # and .json.gz
        result_name, method, dataset, split, split_type, ext = inout.parse_result_filename(self.json_coco_path.as_posix() + ".gz")
        self.assertEqual(result_name, "zebraposesat-effnetb4_ycbv-test_5ed0eecc-96f8-498b-9438-d586d4d92528")
        self.assertEqual(method, "zebraposesat-effnetb4")
        self.assertEqual(dataset, "ycbv")
        self.assertEqual(split, "test")
        self.assertIsNone(split_type)
        self.assertEqual(ext, "json.gz")

    def test_create_result_filename(self):
        method = "zebraposesat-effnetb4"
        dataset = "ycbv"
        split = "test"
        split_type = None
        optional_id = "5ed0eecc-96f8-498b-9438-d586d4d92528"
        result_filename = inout.create_coco_result_filename(method, dataset, split, split_type, optional_id)
        self.assertEqual(result_filename, self.json_coco_path.name)

        method = "some-method"
        dataset = "tless"
        split = "val"
        split_type = "primesense"
        optional_id = None
        result_filename = inout.create_coco_result_filename(method, dataset, split, split_type, optional_id)
        self.assertEqual(result_filename, "some-method_tless-val-primesense.json")
        optional_id = "5ed0eecc-96f8-498b-9438-d586d4d92528"
        result_filename = inout.create_coco_result_filename(method, dataset, split, split_type, optional_id)
        self.assertEqual(result_filename, "some-method_tless-val-primesense_5ed0eecc-96f8-498b-9438-d586d4d92528.json")

        result_filename = inout.create_pose_result_filename(method, dataset, split, optional_id=optional_id)
        self.assertEqual(result_filename, "some-method_tless-val_5ed0eecc-96f8-498b-9438-d586d4d92528.csv")
        result_filename = inout.create_pose_result_filename(method, dataset, split)
        self.assertEqual(result_filename, "some-method_tless-val.csv")

    def test_load_save_operations(self):
        tmp = Path(__file__).parent / "tmp"
        tmp.mkdir(exist_ok=True)

        # im
        im_path_png = tmp / "test_im.png"
        im_path_jpg = tmp / "test_im.jpg"
        im_path_jpeg = tmp / "test_im.jpeg"
        im = np.ones((10,10), dtype=np.uint8)
        inout.save_im(im_path_png, im)
        inout.save_im(im_path_png.as_posix(), im)
        inout.save_im(im_path_jpg.as_posix(), im)
        inout.save_im(im_path_jpeg.as_posix(), im)
        im_path_png_loaded = inout.load_im(im_path_png)
        im_path_jpg_loaded = inout.load_im(im_path_jpg)
        im_path_jpeg_loaded = inout.load_im(im_path_jpeg)
        assert_almost_equal(im, im_path_png_loaded)
        assert_almost_equal(im, im_path_jpg_loaded)
        assert_almost_equal(im, im_path_jpeg_loaded)

        # depth
        depth_path = tmp / "test_depth.png"
        depth = np.ones((10,10), dtype=np.float64) + 0.1
        inout.save_depth(depth_path, depth)
        inout.save_depth(depth_path.as_posix(), depth)
        inout.load_depth(depth_path)
        depth_loaded = inout.load_depth(depth_path.as_posix())
        assert_almost_equal(np.round(depth).astype(np.uint16), depth_loaded)

        # json
        json_coco_bis_path = self.json_coco_path.parent / (self.json_coco_path.stem + "_bis.json")
        json_coco_gz_path = self.json_coco_path.parent / (self.json_coco_path.stem + ".json.gz")

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

        self.assertTrue(len(coco_res) == len(coco_res_from_bis) == len(coco_res_from_gz))
        self.assertTrue(coco_res[0].keys() == coco_res_from_bis[0].keys() == coco_res_from_gz[0].keys())
        self.assertTrue(coco_res[0]["time"] == coco_res_from_bis[0]["time"] == coco_res_from_gz[0]["time"])
        self.assertTrue(coco_res[0]["score"] == coco_res_from_bis[0]["score"] == coco_res_from_gz[0]["score"])
        self.assertTrue(coco_res[0]["bbox"] == coco_res_from_bis[0]["bbox"] == coco_res_from_gz[0]["bbox"])

        # cleanup
        os.remove(im_path_png)
        os.remove(im_path_jpg)
        os.remove(im_path_jpeg)
        os.remove(depth_path)
        os.remove(json_coco_bis_path)
        os.remove(json_coco_gz_path)

    def test_loading_and_checking_results(self):
        ###################
        # high level checks
        check_passed, check_msg = inout.check_coco_results(self.json_coco_path) 
        self.assertTrue(check_passed)
        check_passed, check_msg = inout.check_bop_results(self.csv_6d_path) 
        self.assertTrue(check_passed)

        ###################
        # time checks
        bop_results = inout.load_bop_results(self.csv_6d_path)
        check_passed, check_msg, times, times_available = inout.check_consistent_timings(bop_results, "im_id")
        self.assertTrue(check_passed)
        self.assertTrue(times_available)

        coco_results = inout.load_json(self.json_coco_path)
        check_passed, check_msg, times, times_available = inout.check_consistent_timings(coco_results, "image_id")
        self.assertTrue(check_passed)
        self.assertTrue(times_available)

        # add unconsistent timings
        bop_results[0]["time"] = 42.0
        check_passed, check_msg, times, times_available = inout.check_consistent_timings(bop_results, "im_id")
        self.assertFalse(check_passed)
        self.assertTrue(times_available)
        coco_results[100]["time"] = 42.0
        check_passed, check_msg, times, times_available = inout.check_consistent_timings(coco_results, "image_id")
        self.assertFalse(check_passed)
        self.assertTrue(times_available)

        # in one image, timings were not provide
        for i in range(15):
            bop_results[i]["time"] = -1.0
        check_passed, check_msg, times, times_available = inout.check_consistent_timings(bop_results, "im_id")
        self.assertTrue(check_passed)
        self.assertFalse(times_available)

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


if __name__ == "__main__":
    unittest.main()
