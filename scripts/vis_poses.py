"""
Visualizes object models in the GT/estimated poses.
The script visualize datasets in the classical BOP19 format as well as the HOT3D dataset in H3 BOP24 format.
"""

import copy
import functools
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from bop_toolkit_lib import config, dataset_params, inout, misc, visualization
from bop_toolkit_lib.rendering import renderer
from bop_toolkit_lib.vis_utils import (
    calc_mask_visib_percent,
    combine_depth_diffs,
    draw_pose_contour,
    draw_pose_on_img,
    get_depth_diff_img,
    get_depth_map_and_obj_masks_from_renderings,
    get_pose_mat_from_dict,
    merge_masks,
    plot_depth,
)
from tqdm import tqdm
from vis_poses_cli import postprocess_args, setup_parser

file_name = os.path.splitext(os.path.basename(__file__))[0]
logger = misc.get_logger(file_name)


def main(args):

    if args.mode == "gt":
        dataset, split, split_type = (
            args.dataset,
            args.dataset_split,
            args.dataset_split_type,
        )
        method = "GT"
    else:
        result_filename = args.result_filename

        misc.log("Processing: " + result_filename)

        # Parse info about the method and the dataset from the filename.
        result_name, method, dataset, split, split_type, _ = (
            inout.parse_result_filename(result_filename)
        )

    #######################
    # hot3d specific checks
    if dataset == "hot3d":
        try:
            from bop_toolkit_lib import pose_error_htt
        except ImportError:
            raise ImportError(
                """Missing hand_tracking_toolkit dependency,
                mandatory if you are running evaluation on HOT3D.
                Refer to the README.md for installation instructions.
                """
            )

    if dataset == "hot3d" and args.renderer_type != "htt":
        raise ValueError("'htt' renderer_type is mandatory for HOT3D dataset.")

    # hot3d does not contain depth modality, some visualizations are not available
    if dataset in ["hot3d"]:
        args.vis_rgb = True
        args.vis_rgb_resolve_visib = False
        args.vis_depth_diff = False
    #######################

    # Load dataset parameters.
    dp_split = dataset_params.get_split_params(
        args.datasets_path, dataset, split, split_type
    )

    model_type = "eval"
    dp_model = dataset_params.get_model_params(args.datasets_path, dataset, model_type)

    if args.mode == "gt":
        if args.modality is not None:
            dp_split["eval_modality"] = args.modality
        if args.sensor is not None:
            dp_split["eval_sensor"] = args.sensor

        # List of considered scenes.
        scene_ids_curr = dp_split["scene_ids"]
        if args.scene_ids:
            target_scene_ids = args.scene_ids
            scene_ids_curr = set(scene_ids_curr).intersection(target_scene_ids)
            if len(scene_ids_curr) == 0:
                misc.log(
                    f"Dataset scene ids {dp_split['scene_ids']} do not overlap with chosen scene ids {args.scene_ids}"
                )
        scene_ids = scene_ids_curr

        # Subset of images for which the GT poses will be rendered.
        if args.targets_filename is not None:
            targets = inout.load_json(
                os.path.join(dp_split["base_path"], args.targets_filename)
            )
            scene_im_ids = {}
            for target in targets:
                scene_im_ids.setdefault(target["scene_id"], set()).add(target["im_id"])
        else:
            scene_im_ids = None
    else:
        misc.log("Loading pose estimates...")
        ests = inout.load_bop_results(os.path.join(args.results_path, result_filename))

        # Organize the pose estimates by scene, image and object.
        misc.log("Organizing pose estimates...")
        ests_org = {}
        for est in ests:
            ests_org.setdefault(est["scene_id"], {}).setdefault(
                est["im_id"], {}
            ).setdefault(est["obj_id"], []).append(est)
        scene_ids = list(ests_org.keys())

    colors_path = os.path.join(os.path.dirname(visualization.__file__), "colors.json")
    colors = inout.load_json(colors_path)

    do_extra_vis = args.mode == "est" and len(args.extra_vis_types) > 0
    if do_extra_vis:
        models = {}
        for obj_id in dp_model["obj_ids"]:
            models[obj_id] = inout.load_ply(
                dp_model["model_tpath"].format(obj_id=obj_id)
            )
        models_info = inout.load_json(dp_model["models_info_path"], keys_to_int=True)

    renderer_modalities = []
    if args.vis_rgb:
        renderer_modalities.append("rgb")
    if (
        args.vis_depth_diff
        or (args.vis_rgb and args.vis_rgb_resolve_visib)
        or ("depth_heatmap" in args.extra_vis_types)
    ):
        renderer_modalities.append("depth")
    renderer_mode = "+".join(renderer_modalities)

    width, height = None, None
    ren = None

    for scene_id in tqdm(scene_ids, desc="Scenes"):

        tpath_keys = dataset_params.scene_tpaths_keys(
            dp_split["eval_modality"], dp_split["eval_sensor"], scene_id
        )
        scene_modality = dataset_params.get_scene_sensor_or_modality(
            dp_split["eval_modality"], scene_id
        )
        scene_sensor = dataset_params.get_scene_sensor_or_modality(
            dp_split["eval_sensor"], scene_id
        )

        # Create a new renderer if image size has changed
        scene_width, scene_height = dataset_params.get_im_size(
            dp_split, scene_modality, scene_sensor
        )
        if (width, height) != (scene_width, scene_height):
            width, height = scene_width, scene_height
            misc.log(f"Creating renderer of type {args.renderer_type}")
            ren = renderer.create_renderer(
                width, height, args.renderer_type, mode=renderer_mode, shading="flat"
            )
            # Load object models in the new renderer.
            for obj_id in dp_model["obj_ids"]:
                misc.log(f"Loading 3D model of object {obj_id}...")
                model_path = dp_model["model_tpath"].format(obj_id=obj_id)
                model_color = None
                if not args.vis_orig_color:
                    model_color = tuple(colors[(obj_id - 1) % len(colors)])
                ren.add_object(obj_id, model_path, surf_color=model_color)

        # Load info and GT poses for the current scene.
        scene_camera = inout.load_scene_camera(
            dp_split[tpath_keys["scene_camera_tpath"]].format(scene_id=scene_id)
        )
        scene_gt = inout.load_scene_gt(
            dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id)
        )

        split_type_str = " - " + split_type if split_type is not None else ""
        misc.log(
            f"Visualizing pose estimates - method: {method}, dataset: {dataset}{split_type_str}, scene: {scene_id}"
        )

        if args.mode == "gt":
            # List of considered images.
            if scene_im_ids is not None:
                im_ids = scene_im_ids[scene_id]
            else:
                im_ids = sorted(scene_gt.keys())
            if args.im_ids:
                im_ids = set(im_ids).intersection(args.im_ids)

            poses_scene_vis = {}
            for im_id in im_ids:
                # List of considered GT poses.
                gt_ids_curr = range(len(scene_gt[im_id]))
                if args.gt_ids:
                    gt_ids_curr = set(gt_ids_curr).intersection(args.gt_ids)

                poses = misc.parse_gt_poses_from_scene_im(
                    scene_gt[im_id], gt_ids=gt_ids_curr
                )
                poses_scene_vis[im_id] = poses
        else:
            poses_scene = ests_org[scene_id]
            poses_scene_vis = {}
            for im_id, poses_img in poses_scene.items():

                im_ests_vis = []
                im_ests_vis_obj_ids = []
                for obj_id, obj_ests in poses_img.items():
                    # Sort the estimates by score (in descending order).
                    obj_ests_sorted = sorted(
                        obj_ests, key=lambda est: est["score"], reverse=True
                    )

                    # Select the number of top estimated poses to visualize.
                    if args.n_top == 0:  # All estimates are considered.
                        n_top_curr = None
                    elif args.n_top == -1:  # Given by the number of GT poses.
                        n_gt = sum([gt["obj_id"] == obj_id for gt in scene_gt[im_id]])
                        n_top_curr = n_gt
                    else:  # Specified by the parameter n_top.
                        n_top_curr = args.n_top
                    obj_ests_sorted = obj_ests_sorted[slice(0, n_top_curr)]

                    # Get list of poses to visualize.
                    for est in obj_ests_sorted:
                        est["obj_id"] = obj_id

                        # Text info to write on the image at the pose estimate.
                        val = "{}:{:.2f}".format(obj_id, est["score"])
                        est["text_info"] = [{"name": "", "val": val, "fmt": ""}]

                    im_ests_vis.append(obj_ests_sorted)
                    im_ests_vis_obj_ids.append(obj_id)

                # Join the per-object estimates to make it a single visual.
                # if there are multiple estimates per object, they are treated as independent entries
                poses = list(itertools.chain.from_iterable(im_ests_vis))
                poses_scene_vis[im_id] = poses

        for im_id, poses_img in tqdm(poses_scene_vis.items(), desc=f"Scene {scene_id}"):

            # Retrieve camera intrinsics.
            if dataset == "hot3d":
                cam = pose_error_htt.create_camera_model(scene_camera[im_id])
            else:
                cam = scene_camera[im_id]["cam_K"]

            rgb = None
            if args.vis_rgb:
                # rgb_tpath is an alias refering to the sensor|modality image paths on which the poses are rendered
                im_tpath = tpath_keys["rgb_tpath"]
                # check for BOP classic (itodd)
                rgb_available = dataset_params.sensor_has_modality(
                    dp_split, scene_sensor, "rgb"
                )
                if im_tpath == "rgb_tpath" and not rgb_available:
                    im_tpath = "gray_tpath"

                rgb = inout.load_im(
                    dp_split[im_tpath].format(scene_id=scene_id, im_id=im_id)
                )
                # if image is grayscale (e.g. quest3), convert it to 3 channels
                if rgb.ndim == 2:
                    rgb = np.dstack([rgb, rgb, rgb])
                else:
                    rgb = rgb[:, :, :3]

            depth = None
            if args.vis_depth_diff or (args.vis_rgb and args.vis_rgb_resolve_visib):
                depth_available = dataset_params.sensor_has_modality(
                    dp_split, scene_sensor, "depth"
                )
                if not depth_available:
                    misc.log(
                        f"{scene_sensor} has no depth data, skipping depth visualization"
                    )
                    args.vis_depth_diff = False
                    args.vis_rgb_resolve_visib = False
                else:
                    depth = inout.load_depth(
                        dp_split[tpath_keys["depth_tpath"]].format(
                            scene_id=scene_id, im_id=im_id
                        )
                    )
                    depth *= scene_camera[im_id]["depth_scale"]  # Convert to [mm].

            vis_depth_diff_path = None
            vis_rgb_path = None
            if args.mode == "gt":
                split = (
                    "{}_{}".format(args.dataset_split, scene_sensor)
                    if scene_sensor
                    else args.dataset_split
                )
                vis_path_base = functools.partial(
                    args.vis_path_template.format,
                    vis_path=args.vis_path,
                    dataset=args.dataset,
                    split=split,
                    scene_id=scene_id,
                    im_id=im_id,
                )
            else:
                vis_path_base = functools.partial(
                    args.vis_path_template.format,
                    vis_path=args.vis_path,
                    result_name=result_name,
                    scene_id=scene_id,
                    im_id=im_id,
                )
            if args.vis_rgb:
                vis_rgb_path = vis_path_base(suffix="_overlay")
            if args.vis_depth_diff:
                vis_depth_diff_path = vis_path_base(suffix="_depth_diff")

            vis_res = visualization.vis_object_poses(
                poses=poses_img,
                K=cam,
                renderer=ren,
                rgb=rgb,
                depth=depth,
                vis_rgb_resolve_visib=args.vis_rgb_resolve_visib,
                vis_rgb=args.vis_rgb,
                vis_depth_diff=args.vis_depth_diff,
            )
            if args.vis_rgb or args.vis_depth_diff or do_extra_vis:
                misc.ensure_dir(os.path.dirname(vis_rgb_path))
            if args.vis_rgb:
                inout.save_im(vis_rgb_path, vis_res["vis_im_rgb"], jpg_quality=95)
            if args.vis_depth_diff:
                inout.save_im(vis_depth_diff_path, vis_res["depth_diff_vis"])

            if do_extra_vis:

                gt_poses = misc.parse_gt_poses_from_scene_im(scene_gt[im_id])
                gt_poses_matched = misc.match_gt_poses_to_est(
                    est_poses=poses_img,
                    gt_poses=gt_poses,
                    models=models,
                    models_info=models_info,
                )
                res_per_obj_est = vis_res["res_per_obj"]
                res_per_obj_gt = visualization.vis_object_poses(
                    poses=gt_poses_matched,
                    K=cam,
                    renderer=ren,
                    rgb=rgb,
                    depth=depth,
                    vis_rgb_resolve_visib=args.vis_rgb_resolve_visib,
                    vis_rgb=False,
                    vis_depth_diff=False,
                )["res_per_obj"]
                bres_gt = get_depth_map_and_obj_masks_from_renderings(res_per_obj_gt)
                mask_objs_gt = bres_gt["mask_objs"]
                bres_est = get_depth_map_and_obj_masks_from_renderings(res_per_obj_est)
                mask_objs = bres_est["mask_objs"]

                if "depth_heatmap" in args.extra_vis_types:

                    depth_diff_meshs = []
                    for obj_idx, obj_res_gt in res_per_obj_gt.items():
                        obj_res = res_per_obj_est[obj_idx]
                        gt_depth = obj_res_gt["depth"]
                        obj_id = obj_res_gt["pose"]["obj_id"]
                        sym_mat = misc.get_symmetry_transformations(
                            models_info[obj_id], 0.01
                        )
                        pose = obj_res["pose"]
                        depth_diff_mesh = get_depth_diff_img(
                            gt_depth=gt_depth,
                            est_pose=pose,
                            gt_pose=obj_res_gt["pose"],
                            cam=cam,
                            syms=sym_mat,
                        )
                        depth_diff_meshs.append(depth_diff_mesh)

                    dres = combine_depth_diffs(
                        mask_objs_gt,
                        depth_diff_meshs,
                    )
                    depth_diff_img = dres["combined"]
                    mask_objs_gt_merged = merge_masks(mask_objs_gt)
                    plot_depth(
                        depth_diff_img,
                        cbar_title="Error (mm)",
                        ax=None,
                        cmap="turbo",
                        use_horiz_cbar=True,
                        use_white_bg=True,
                        include_colorbar=True,
                        use_fixed_cbar=True,
                        mask=mask_objs_gt_merged,
                    )
                    save_path = vis_path_base(suffix="_depth_heatmap")
                    plt.savefig(save_path, dpi=100, bbox_inches='tight')
                if "contour" in args.extra_vis_types:
                    contour_img = copy.deepcopy(rgb)
                    for idx in range(len(res_per_obj_est)):
                        depth_obj_gt = res_per_obj_gt[idx]["depth"]
                        depth_obj = res_per_obj_est[idx]["depth"]
                        depth_obj_mask = depth_obj_gt > 0
                        mask_obj = mask_objs[idx]
                        mask_obj_gt = mask_objs_gt[idx]

                        percent = calc_mask_visib_percent(mask_obj, depth_obj_mask)
                        if percent < 20:
                            print(f"{idx=} {percent=}")
                            continue

                        contour_img = draw_pose_contour(
                            contour_img,
                            rendered_depth=depth_obj_gt,
                            mask_visib=mask_obj_gt,
                            contour_color=(0, 255, 0),
                        )
                        contour_img = draw_pose_contour(
                            contour_img,
                            rendered_depth=depth_obj,
                            contour_color=(255, 0, 0),
                            mask_visib=mask_obj,
                        )
                    save_path = vis_path_base(suffix="_contour")
                    inout.save_im(save_path, contour_img)
                if "bbox3d" in args.extra_vis_types:
                    bbox_img = copy.deepcopy(rgb)
                    for idx in range(len(res_per_obj_est)):

                        obj_id = res_per_obj_gt[idx]["pose"]["obj_id"]
                        pose_gt = res_per_obj_gt[idx]["pose"]
                        pose_est = res_per_obj_est[idx]["pose"]
                        model = models[obj_id]
                        pts = model["pts"]
                        mesh = trimesh.Trimesh(vertices=pts, faces=model["faces"])
                        bbox_3d = trimesh.bounds.corners(
                            mesh.bounding_box_oriented.bounds
                        )
                        depth_obj_gt = res_per_obj_gt[idx]["depth"]
                        depth_obj_mask = depth_obj_gt > 0
                        mask_obj = mask_objs[idx]
                        percent = calc_mask_visib_percent(mask_obj, depth_obj_mask)
                        if percent < 35:
                            print(f"{idx=} {percent=}")
                            continue

                        syms = misc.get_symmetry_transformations(
                            models_info[obj_id], max_sym_disc_step=0.01
                        )
                        pts_est = misc.transform_pts_Rt(
                            pts, pose_est["R"], pose_est["t"].squeeze()
                        )
                        gt_poses_syms = []
                        es = []
                        for sym in syms:
                            R_gt_sym = pose_gt["R"].dot(sym["R"])
                            t_gt_sym = (
                                pose_gt["R"].dot(sym["t"].squeeze())
                                + pose_gt["t"].squeeze()
                            )
                            pts_gt_sym = misc.transform_pts_Rt(pts, R_gt_sym, t_gt_sym)
                            gt_poses_syms.append(
                                {"R": R_gt_sym, "t": t_gt_sym, "obj_id": obj_id}
                            )
                            es.append(
                                np.linalg.norm(pts_est - pts_gt_sym, axis=1).max()
                            )
                        best_idx = np.argmin(es)
                        pose_gt_sym = gt_poses_syms[best_idx]
                        pose_gt_sym_mat = get_pose_mat_from_dict(pose_gt_sym)

                        est_pose_mat = get_pose_mat_from_dict(
                            res_per_obj_est[idx]["pose"]
                        )
                        bbox_img = draw_pose_on_img(
                            bbox_img,
                            pose_pred=est_pose_mat,
                            pose_gt=pose_gt_sym_mat,
                            K=cam,
                            axes_scale=50 // 2,
                            mesh_bbox=bbox_3d,
                            bbox_color=(255, 0, 0),
                            bbox_color_gt=(0, 255, 0),
                        )
                    save_path = vis_path_base(suffix="_bbox3d")
                    inout.save_im(save_path, bbox_img)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if args.vis_path is None:
        vis_type = "gt" if args.mode == "gt" else "est"
        args.vis_path = os.path.join(config.output_path, f"vis_{vis_type}_poses")
    args = postprocess_args(args)
    main(args)
