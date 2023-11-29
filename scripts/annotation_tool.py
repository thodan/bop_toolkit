# Author: Anas Gouda (anas.gouda@tu-dortmund.de)
# FLW, TU Dortmund, Germany

"""Manual annotation tool for datasets with BOP format

Using RGB, Depth and Models the tool will generate the "scene_gt.json" annotation file

Other annotations can be generated usign other scripts [calc_gt_info.py, calc_gt_masks.py, ....]

original repo: https://github.com/FLW-TUDO/3d_annotation_tool

"""

import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import json
import cv2
import warnings

# PARAMETERS.
################################################################################
p = {
    # Folder containing the BOP datasets.
    "dataset_path": "/path/to/dataset",
    # Dataset split. Options: 'train', 'test'.
    "dataset_split": "val",
    # Dataset split type. Options: 'synt', 'real', None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # scene number to open tool on
    "start_scene_num": 1,
    # image number inside scene to open tool on
    "start_image_num": 0,
}
################################################################################

dist = 0.002
deg = 1


class Dataset:
    def __init__(self, dataset_path, dataset_split):
        self.scenes_path = os.path.join(dataset_path, dataset_split)
        self.objects_path = os.path.join(dataset_path, "models")


class AnnotationScene:
    def __init__(self, scene_point_cloud, scene_num, image_num):
        self.annotation_scene = scene_point_cloud
        self.scene_num = scene_num
        self.image_num = image_num

        self.obj_list = list()

    def add_obj(self, obj_geometry, obj_name, obj_instance, transform=np.identity(4)):
        self.obj_list.append(
            self.SceneObject(obj_geometry, obj_name, obj_instance, transform)
        )

    def get_objects(self):
        return self.obj_list[:]

    def remove_obj(self, index):
        self.obj_list.pop(index)

    class SceneObject:
        def __init__(self, obj_geometry, obj_name, obj_instance, transform):
            self.obj_geometry = obj_geometry
            self.obj_name = obj_name
            self.obj_instance = obj_instance
            self.transform = transform


class Settings:
    UNLIT = "defaultUnlit"

    def __init__(self):
        self.bg_color = gui.Color(1, 1, 1)
        self.show_axes = False
        self.highlight_obj = True

        self.apply_material = True  # clear to False after processing

        self.scene_material = rendering.MaterialRecord()
        self.scene_material.base_color = [0.9, 0.9, 0.9, 1.0]
        self.scene_material.shader = Settings.UNLIT

        self.annotation_obj_material = rendering.MaterialRecord()
        self.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1.0]
        self.annotation_obj_material.shader = Settings.UNLIT


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    MATERIAL_NAMES = ["Unlit"]
    MATERIAL_SHADERS = [Settings.UNLIT]

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red,
            self.settings.bg_color.green,
            self.settings.bg_color.blue,
            self.settings.bg_color.alpha,
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_axes(self.settings.show_axes)

        if self.settings.apply_material:
            self._scene.scene.modify_geometry_material(
                "annotation_scene", self.settings.scene_material
            )
            self.settings.apply_material = False

        self._show_axes.checked = self.settings.show_axes
        self._highlight_obj.checked = self.settings.highlight_obj
        self._point_size.double_value = self.settings.scene_material.point_size

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def __init__(self, width, height, scenes):
        self.scenes = scenes
        self.settings = Settings()

        self.window = gui.Application.instance.create_window(
            "BOP manual annotation tool", width, height
        )
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        # ---- Settings panel ----
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        view_ctrls = gui.CollapsableVert("View control", 0, gui.Margins(em, 0, 0, 0))
        view_ctrls.set_is_open(True)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_child(self._show_axes)

        self._highlight_obj = gui.Checkbox("Highligh annotation objects")
        self._highlight_obj.set_on_checked(self._on_highlight_obj)
        view_ctrls.add_child(self._highlight_obj)

        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 5)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        view_ctrls.add_child(grid)

        self._settings_panel.add_child(view_ctrls)
        # ----

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # 3D Annotation tool options
        annotation_objects = gui.CollapsableVert(
            "Annotation Objects", 0.33 * em, gui.Margins(em, 0, 0, 0)
        )
        annotation_objects.set_is_open(True)
        self._meshes_available = gui.ListView()
        # mesh_available.set_items(["bottle", "can"])
        self._meshes_used = gui.ListView()
        # mesh_used.set_items(["can_0", "can_1", "can_1", "can_1"])
        add_mesh_button = gui.Button("Add Mesh")
        remove_mesh_button = gui.Button("Remove Mesh")
        add_mesh_button.set_on_clicked(self._add_mesh)
        remove_mesh_button.set_on_clicked(self._remove_mesh)
        annotation_objects.add_child(self._meshes_available)
        annotation_objects.add_child(add_mesh_button)
        annotation_objects.add_child(self._meshes_used)
        annotation_objects.add_child(remove_mesh_button)
        self._settings_panel.add_child(annotation_objects)

        self._scene_control = gui.CollapsableVert(
            "Scene Control", 0.33 * em, gui.Margins(em, 0, 0, 0)
        )
        self._scene_control.set_is_open(True)

        self._images_buttons_label = gui.Label("Images:")
        self._samples_buttons_label = gui.Label("Scene: ")

        self._pre_image_button = gui.Button("Previous")
        self._pre_image_button.horizontal_padding_em = 0.8
        self._pre_image_button.vertical_padding_em = 0
        self._pre_image_button.set_on_clicked(self._on_previous_image)
        self._next_image_button = gui.Button("Next")
        self._next_image_button.horizontal_padding_em = 0.8
        self._next_image_button.vertical_padding_em = 0
        self._next_image_button.set_on_clicked(self._on_next_image)
        self._pre_sample_button = gui.Button("Previous")
        self._pre_sample_button.horizontal_padding_em = 0.8
        self._pre_sample_button.vertical_padding_em = 0
        self._pre_sample_button.set_on_clicked(self._on_previous_scene)
        self._next_sample_button = gui.Button("Next")
        self._next_sample_button.horizontal_padding_em = 0.8
        self._next_sample_button.vertical_padding_em = 0
        self._next_sample_button.set_on_clicked(self._on_next_scene)
        # 2 rows for sample and scene control
        h = gui.Horiz(0.4 * em)  # row 1
        h.add_stretch()
        h.add_child(self._images_buttons_label)
        h.add_child(self._pre_image_button)
        h.add_child(self._next_image_button)
        h.add_stretch()
        self._scene_control.add_child(h)
        h = gui.Horiz(0.4 * em)  # row 2
        h.add_stretch()
        h.add_child(self._samples_buttons_label)
        h.add_child(self._pre_sample_button)
        h.add_child(self._next_sample_button)
        h.add_stretch()
        self._scene_control.add_child(h)

        self._view_numbers = gui.Horiz(0.4 * em)
        self._image_number = gui.Label("Image: " + f"{0:06}")
        self._scene_number = gui.Label("Scene: " + f"{0:06}")
        self._view_numbers.add_child(self._image_number)
        self._view_numbers.add_child(self._scene_number)
        self._scene_control.add_child(self._view_numbers)

        self._settings_panel.add_child(self._scene_control)
        refine_position = gui.Button("Refine position")
        refine_position.set_on_clicked(self._on_refine)
        generate_save_annotation = gui.Button("generate annotation - save/update")
        generate_save_annotation.set_on_clicked(self._on_generate)
        self._scene_control.add_child(refine_position)
        self._scene_control.add_child(generate_save_annotation)

        # ---- Menu ----
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_separator()
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        # ---- annotation tool settings ----
        self._on_point_size(1)  # set default size to 1

        self._apply_settings()

        self._annotation_scene = None

        # set callbacks for key control
        self._scene.set_on_key(self._transform)

        self._left_shift_modifier = False

    def _update_scene_numbers(self):
        self._scene_number.text = "Scene: " + f"{self._annotation_scene.scene_num:06}"
        self._image_number.text = "Image: " + f"{self._annotation_scene.image_num:06}"

    def _transform(self, event):
        if event.is_repeat:
            return gui.Widget.EventCallbackResult.HANDLED

        if event.key == gui.KeyName.LEFT_SHIFT:
            if event.type == gui.KeyEvent.DOWN:
                self._left_shift_modifier = True
            elif event.type == gui.KeyEvent.UP:
                self._left_shift_modifier = False
            return gui.Widget.EventCallbackResult.HANDLED

        # if ctrl is pressed then increase translation and angle values
        global dist, deg
        if event.key == gui.KeyName.LEFT_CONTROL:
            if event.type == gui.KeyEvent.DOWN:
                dist = 0.05
                deg = 90
            elif event.type == gui.KeyEvent.UP:
                dist = 0.005
                deg = 1
            return gui.Widget.EventCallbackResult.HANDLED

        # if no active_mesh selected print error
        if self._meshes_used.selected_index == -1:
            self._on_error("No objects are highlighted in scene meshes")
            return gui.Widget.EventCallbackResult.HANDLED

        def move(x, y, z, rx, ry, rz):
            self._annotation_changed = True

            objects = self._annotation_scene.get_objects()
            active_obj = objects[self._meshes_used.selected_index]
            # translation or rotation
            if x != 0 or y != 0 or z != 0:
                h_transform = np.array(
                    [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
                )
            else:  # elif rx!=0 or ry!=0 or rz!=0:
                center = active_obj.obj_geometry.get_center()
                rot_mat_obj_center = (
                    active_obj.obj_geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
                )
                T_neg = np.vstack(
                    (np.hstack((np.identity(3), -center.reshape(3, 1))), [0, 0, 0, 1])
                )
                R = np.vstack(
                    (np.hstack((rot_mat_obj_center, [[0], [0], [0]])), [0, 0, 0, 1])
                )
                T_pos = np.vstack(
                    (np.hstack((np.identity(3), center.reshape(3, 1))), [0, 0, 0, 1])
                )
                h_transform = np.matmul(T_pos, np.matmul(R, T_neg))
            active_obj.obj_geometry.transform(h_transform)
            center = active_obj.obj_geometry.get_center()
            self._scene.scene.remove_geometry(active_obj.obj_name)
            self._scene.scene.add_geometry(
                active_obj.obj_name,
                active_obj.obj_geometry,
                self.settings.annotation_obj_material,
                add_downsampled_copy_for_fast_rendering=True,
            )
            # update values stored of object
            active_obj.transform = np.matmul(h_transform, active_obj.transform)

        if event.type == gui.KeyEvent.DOWN:  # only move objects with down strokes
            # Refine
            if event.key == gui.KeyName.R:
                self._on_refine()
            # Translation
            if not self._left_shift_modifier:
                if event.key == gui.KeyName.L:
                    print("L pressed: translate in +ve X direction")
                    move(dist, 0, 0, 0, 0, 0)
                elif event.key == gui.KeyName.H:
                    print("H pressed: translate in -ve X direction")
                    move(-dist, 0, 0, 0, 0, 0)
                elif event.key == gui.KeyName.J:
                    print("Comma pressed: translate in +ve Y direction")
                    move(0, dist, 0, 0, 0, 0)
                elif event.key == gui.KeyName.K:
                    print("I pressed: translate in -ve Y direction")
                    move(0, -dist, 0, 0, 0, 0)
                elif event.key == gui.KeyName.COMMA:
                    print("K pressed: translate in +ve Z direction")
                    move(0, 0, dist, 0, 0, 0)
                elif event.key == gui.KeyName.I:
                    print("J pressed: translate in -ve Z direction")
                    move(0, 0, -dist, 0, 0, 0)
            # Rotation - keystrokes are not in same order as translation to make movement more human intuitive
            else:
                print("Left-Shift is clicked; rotation mode")
                if event.key == gui.KeyName.K:
                    print("L pressed: rotate around +ve X direction")
                    move(0, 0, 0, 0, 0, deg * np.pi / 180)
                elif event.key == gui.KeyName.J:
                    print("H pressed: rotate around -ve X direction")
                    move(0, 0, 0, 0, 0, -deg * np.pi / 180)
                elif event.key == gui.KeyName.H:
                    print("K pressed: rotate around +ve Y direction")
                    move(0, 0, 0, 0, deg * np.pi / 180, 0)
                elif event.key == gui.KeyName.L:
                    print("J pressed: rotate around -ve Y direction")
                    move(0, 0, 0, 0, -deg * np.pi / 180, 0)
                elif event.key == gui.KeyName.COMMA:
                    print("Comma pressed: rotate around +ve Z direction")
                    move(0, 0, 0, deg * np.pi / 180, 0, 0)
                elif event.key == gui.KeyName.I:
                    print("I pressed: rotate around -ve Z direction")
                    move(0, 0, 0, -deg * np.pi / 180, 0, 0)

        return gui.Widget.EventCallbackResult.HANDLED

    def _on_refine(self):
        self._annotation_changed = True

        # if no active_mesh selected print error
        if self._meshes_used.selected_index == -1:
            self._on_error("No objects are highlighted in scene meshes")
            return gui.Widget.EventCallbackResult.HANDLED

        target = self._annotation_scene.annotation_scene
        objects = self._annotation_scene.get_objects()
        active_obj = objects[self._meshes_used.selected_index]
        source = active_obj.obj_geometry

        trans_init = np.identity(4)
        threshold = 0.004
        radius = 0.002
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
        )
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
        )
        reg = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )

        active_obj.obj_geometry.transform(reg.transformation)
        # active_obj.obj_geometry.paint_uniform_color([0,1,0])  # Debug
        self._scene.scene.remove_geometry(active_obj.obj_name)
        self._scene.scene.add_geometry(
            active_obj.obj_name,
            active_obj.obj_geometry,
            self.settings.annotation_obj_material,
            add_downsampled_copy_for_fast_rendering=True,
        )
        active_obj.transform = np.matmul(reg.transformation, active_obj.transform)

    def _on_generate(self):
        image_num = self._annotation_scene.image_num
        model_names = self.load_model_names()

        json_6d_path = os.path.join(
            self.scenes.scenes_path,
            f"{self._annotation_scene.scene_num:06}",
            "scene_gt.json",
        )

        if os.path.exists(json_6d_path):
            with open(json_6d_path, "r") as gt_scene:
                gt_6d_pose_data = json.load(gt_scene)
        else:
            gt_6d_pose_data = {}

        # wrtie/update "scene_gt.json"
        with open(json_6d_path, "w+") as gt_scene:
            view_angle_data = list()
            for obj in self._annotation_scene.get_objects():
                transform_cam_to_object = obj.transform
                translation = list(
                    transform_cam_to_object[0:3, 3] * 1000
                )  # convert meter to mm
                model_names = self.load_model_names()
                obj_id = (
                    model_names.index(obj.obj_name[:-2]) + 1
                )  # assuming max number of object of same object 10
                obj_data = {
                    "cam_R_m2c": transform_cam_to_object[
                        0:3, 0:3
                    ].tolist(),  # rotation matrix
                    "cam_t_m2c": translation,  # translation
                    "obj_id": obj_id,
                }
                view_angle_data.append(obj_data)
            gt_6d_pose_data[str(image_num)] = view_angle_data
            json.dump(gt_6d_pose_data, gt_scene)

        self._annotation_changed = False

    def _on_error(self, err_msg):
        dlg = gui.Dialog("Error")

        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(err_msg))

        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_highlight_obj(self, light):
        self.settings.highlight_obj = light
        if light:
            self.settings.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1.0]
        elif not light:
            self.settings.annotation_obj_material.base_color = [0.9, 0.9, 0.9, 1.0]

        self._apply_settings()

        # update current object visualization
        meshes = self._annotation_scene.get_objects()
        for mesh in meshes:
            self._scene.scene.modify_geometry_material(
                mesh.obj_name, self.settings.annotation_obj_material
            )

    def _on_point_size(self, size):
        self.settings.scene_material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("BOP manual annotation tool"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def _obj_instance_count(self, mesh_to_add, meshes):
        types = [
            i[:-2] for i in meshes
        ]  # remove last 3 character as they present instance number (OBJ_INSTANCE)
        equal_values = [i for i in range(len(types)) if types[i] == mesh_to_add]
        count = 0
        if len(equal_values):
            indices = np.array(meshes)
            indices = indices[equal_values]
            indices = [int(x[-1]) for x in indices]
            count = max(indices) + 1
            # TODO change to fill the numbers missing in sequence
        return count

    def _add_mesh(self):
        meshes = self._annotation_scene.get_objects()
        meshes = [i.obj_name for i in meshes]

        object_geometry = o3d.io.read_point_cloud(
            self.scenes.objects_path
            + "/obj_"
            + f"{self._meshes_available.selected_index + 1:06}"
            + ".ply"
        )
        object_geometry.points = o3d.utility.Vector3dVector(
            np.array(object_geometry.points) / 1000
        )  # convert mm to meter
        init_trans = np.identity(4)
        center = self._annotation_scene.annotation_scene.get_center()
        center[2] -= 0.2
        init_trans[0:3, 3] = center
        object_geometry.transform(init_trans)
        new_mesh_instance = self._obj_instance_count(
            self._meshes_available.selected_value, meshes
        )
        new_mesh_name = (
            str(self._meshes_available.selected_value) + "_" + str(new_mesh_instance)
        )
        self._scene.scene.add_geometry(
            new_mesh_name,
            object_geometry,
            self.settings.annotation_obj_material,
            add_downsampled_copy_for_fast_rendering=True,
        )
        self._annotation_scene.add_obj(
            object_geometry, new_mesh_name, new_mesh_instance, transform=init_trans
        )
        meshes = (
            self._annotation_scene.get_objects()
        )  # update list after adding current object
        meshes = [i.obj_name for i in meshes]
        self._meshes_used.set_items(meshes)
        self._meshes_used.selected_index = len(meshes) - 1

    def _remove_mesh(self):
        if not self._annotation_scene.get_objects():
            print("There are no object to be deleted.")
            return
        meshes = self._annotation_scene.get_objects()
        active_obj = meshes[self._meshes_used.selected_index]
        self._scene.scene.remove_geometry(active_obj.obj_name)  # remove mesh from scene
        self._annotation_scene.remove_obj(
            self._meshes_used.selected_index
        )  # remove mesh from class list
        # update list after adding removing object
        meshes = self._annotation_scene.get_objects()  # get new list after deletion
        meshes = [i.obj_name for i in meshes]
        self._meshes_used.set_items(meshes)

    def _make_point_cloud(self, rgb_img, depth_img, cam_K):
        # convert images to open3d types
        rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        depth_img_o3d = o3d.geometry.Image(depth_img)

        # convert image to point cloud
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            rgb_img.shape[0],
            rgb_img.shape[1],
            cam_K[0, 0],
            cam_K[1, 1],
            cam_K[0, 2],
            cam_K[1, 2],
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_img_o3d, depth_img_o3d, depth_scale=1, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        return pcd

    def scene_load(self, scenes_path, scene_num, image_num):
        self._annotation_changed = False

        self._scene.scene.clear_geometry()
        geometry = None

        scene_path = os.path.join(scenes_path, f"{scene_num:06}")

        camera_params_path = os.path.join(scene_path, "scene_camera.json")
        with open(camera_params_path) as f:
            data = json.load(f)
            cam_K = data[str(image_num)]["cam_K"]
            cam_K = np.array(cam_K).reshape((3, 3))
            depth_scale = data[str(image_num)]["depth_scale"]

        rgb_path = os.path.join(scene_path, "rgb", f"{image_num:06}" + ".png")
        rgb_img = cv2.imread(rgb_path)
        depth_path = os.path.join(scene_path, "depth", f"{image_num:06}" + ".png")
        depth_img = cv2.imread(depth_path, -1)
        depth_img = np.float32(depth_img * depth_scale / 1000)

        try:
            geometry = self._make_point_cloud(rgb_img, depth_img, cam_K)
        except Exception:
            print("Failed to load scene.")

        if geometry is not None:
            print("[Info] Successfully read scene ", scene_num)
            if not geometry.has_normals():
                geometry.estimate_normals()
            geometry.normalize_normals()
        else:
            print("[WARNING] Failed to read points")

        try:
            self._scene.scene.add_geometry(
                "annotation_scene",
                geometry,
                self.settings.scene_material,
                add_downsampled_copy_for_fast_rendering=True,
            )
            bounds = geometry.get_axis_aligned_bounding_box()
            self._scene.setup_camera(60, bounds, bounds.get_center())
            center = np.array([0, 0, 0])
            eye = center + np.array([0, 0, -0.5])
            up = np.array([0, -1, 0])
            self._scene.look_at(center, eye, up)

            self._annotation_scene = AnnotationScene(geometry, scene_num, image_num)
            self._meshes_used.set_items([])  # clear list from last loaded scene

            # load values if an annotation already exists

            model_names = self.load_model_names()

            scene_gt_path = os.path.join(
                self.scenes.scenes_path,
                f"{self._annotation_scene.scene_num:06}",
                "scene_gt.json",
            )
            # if os.path.exists(json_path):
            with open(scene_gt_path) as scene_gt_file:
                data = json.load(scene_gt_file)
                scene_data = data[str(image_num)]
                active_meshes = list()
                for obj in scene_data:
                    # add object to annotation_scene object
                    obj_geometry = o3d.io.read_point_cloud(
                        os.path.join(
                            self.scenes.objects_path,
                            "obj_" + f"{int(obj['obj_id']):06}" + ".ply",
                        )
                    )
                    obj_geometry.points = o3d.utility.Vector3dVector(
                        np.array(obj_geometry.points) / 1000
                    )  # convert mm to meter
                    model_name = model_names[int(obj["obj_id"]) - 1]
                    obj_instance = self._obj_instance_count(model_name, active_meshes)
                    obj_name = model_name + "_" + str(obj_instance)
                    translation = (
                        np.array(np.array(obj["cam_t_m2c"]), dtype=np.float64) / 1000
                    )  # convert to meter
                    orientation = np.array(np.array(obj["cam_R_m2c"]), dtype=np.float64)
                    transform = np.concatenate(
                        (orientation.reshape((3, 3)), translation.reshape(3, 1)), axis=1
                    )
                    transform_cam_to_obj = np.concatenate(
                        (transform, np.array([0, 0, 0, 1]).reshape(1, 4))
                    )  # homogeneous transform

                    self._annotation_scene.add_obj(
                        obj_geometry, obj_name, obj_instance, transform_cam_to_obj
                    )
                    # adding object to the scene
                    obj_geometry.translate(transform_cam_to_obj[0:3, 3])
                    center = obj_geometry.get_center()
                    obj_geometry.rotate(transform_cam_to_obj[0:3, 0:3], center=center)
                    self._scene.scene.add_geometry(
                        obj_name,
                        obj_geometry,
                        self.settings.annotation_obj_material,
                        add_downsampled_copy_for_fast_rendering=True,
                    )
                    active_meshes.append(obj_name)
            self._meshes_used.set_items(active_meshes)

        except Exception as e:
            print(e)

        self._update_scene_numbers()

    def update_obj_list(self):
        model_names = self.load_model_names()
        self._meshes_available.set_items(model_names)

    def load_model_names(self):
        path = self.scenes.objects_path + "/models_names.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                model_names = [data[x]["name"] for x in data]
        else:  # model names file doesn't exist
            warnings.warn(
                "models_names.json doesn't exist. Objects will be loaded with their literal id (obj_000001, obj_000002, ...)"
            )
            no_of_models = len(
                [
                    os.path.basename(x)[:-4]
                    for x in glob.glob(self.scenes.objects_path + "/*.ply")
                ]
            )
            model_names = ["obj_" + f"{i + 1:06}" for i in range(no_of_models)]

        return model_names

    def _check_changes(self):
        if self._annotation_changed:
            self._on_error(
                "Annotation changed but not saved. If you want to ignore the changes click the navigation button again."
            )
            self._annotation_changed = False
            return True
        else:
            return False

    def _on_next_scene(self):
        if self._check_changes():
            return

        if self._annotation_scene.scene_num + 1 > len(
            next(os.walk(self.scenes.scenes_path))[1]
        ):  # 1 for how many folder (dataset scenes) inside the path
            self._on_error("There is no next scene.")
            return
        self.scene_load(
            self.scenes.scenes_path, self._annotation_scene.scene_num + 1, 0
        )  # open next scene on the first image

    def _on_previous_scene(self):
        if self._check_changes():
            return

        if self._annotation_scene.scene_num - 1 < 1:
            self._on_error("There is no scene number before scene 1.")
            return
        self.scene_load(
            self.scenes.scenes_path, self._annotation_scene.scene_num - 1, 0
        )  # open next scene on the first image

    def _on_next_image(self):
        if self._check_changes():
            return

        num = len(
            next(
                os.walk(
                    os.path.join(
                        self.scenes.scenes_path,
                        f"{self._annotation_scene.scene_num:06}",
                        "depth",
                    )
                )
            )[2]
        )
        if self._annotation_scene.image_num + 1 >= len(
            next(
                os.walk(
                    os.path.join(
                        self.scenes.scenes_path,
                        f"{self._annotation_scene.scene_num:06}",
                        "depth",
                    )
                )
            )[2]
        ):  # 2 for files which here are the how many depth images
            self._on_error("There is no next image.")
            return
        self.scene_load(
            self.scenes.scenes_path,
            self._annotation_scene.scene_num,
            self._annotation_scene.image_num + 1,
        )

    def _on_previous_image(self):
        if self._check_changes():
            return

        if self._annotation_scene.image_num - 1 < 0:
            self._on_error("There is no image number before image 0.")
            return
        self.scene_load(
            self.scenes.scenes_path,
            self._annotation_scene.scene_num,
            self._annotation_scene.image_num - 1,
        )


def main():
    if p["dataset_split_type"]:
        split_and_type = p["dataset_split"] + "_" + p["dataset_split_type"]
    else:
        split_and_type = p["dataset_split"]

    scenes = Dataset(p["dataset_path"], split_and_type)
    gui.Application.instance.initialize()
    w = AppWindow(2048, 1536, scenes)

    if os.path.exists(scenes.scenes_path) and os.path.exists(scenes.objects_path):
        w.scene_load(scenes.scenes_path, p["start_scene_num"], p["start_image_num"])
        w.update_obj_list()
    else:
        w.window.show_message_box(
            "Error",
            "Could not find scenes or object meshes folders "
            + scenes.scenes_path
            + "/"
            + scenes.objects_path,
        )
        print("Could not find scene or object meshes folder")
        exit()

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
