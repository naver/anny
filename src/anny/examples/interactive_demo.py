# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Interactive demo for Anny models using Gradio.
It allows users to manipulate the pose and shape of the model.
"""

import anny
import roma
import torch
import gradio as gr
import tempfile
import json
import trimesh
import anny.anthropometry
import anny.utils.collision


def main(server_name: str = None, server_port: int = None):
    dtype = torch.float32

    with (
        tempfile.NamedTemporaryFile(suffix=".glb") as temp_file,
        tempfile.NamedTemporaryFile(suffix=".json") as temp_params_file,
    ):
        mesh_filename = temp_file.name

        model = None
        measurements_class = None
        bones_rotvec = None
        phenotype_kwargs = None
        local_changes_kwargs = None
        facial_actions_kwargs = None
        show_bones = False
        show_bone_axes = False
        show_self_intersections = False
        self_intersection_module = None

        def export_mesh():
            bones_rotmat = roma.rotvec_to_rotmat(torch.deg2rad(bones_rotvec))
            pose_parameters = roma.Rigid(
                bones_rotmat, torch.zeros((len(bones_rotmat), 3), dtype=dtype)
            )[None].to_homogeneous()
            output = model(
                pose_parameters=pose_parameters,
                phenotype_kwargs=phenotype_kwargs,
                local_changes_kwargs=local_changes_kwargs,
                facial_actions=facial_actions_kwargs,
            )
            vertices = output["vertices"]
            faces = model.faces

            # Save parameters to file
            with open(temp_params_file.name, "w") as f:
                data = dict(
                    phenotype_kwargs={
                        key: value
                        for key, value in phenotype_kwargs.items()
                        if value != 0.5
                    },
                    local_changes_kwargs={
                        key: value
                        for key, value in local_changes_kwargs.items()
                        if value != 0.0
                    },
                    facial_actions={
                        key: value
                        for key, value in facial_actions_kwargs.items()
                        if value != 0.0
                    },
                    pose_parameterization=model.pose_parameterization,
                    pose_parameters={
                        key: matrix
                        for key, matrix in zip(
                            model.bone_labels,
                            pose_parameters.squeeze(dim=0).cpu().numpy().tolist(),
                        )
                    },
                )
                json.dump(data, f)

            scene = trimesh.Scene()
            axis = trimesh.creation.axis(
                origin_size=0.01, axis_radius=0.005, axis_length=1.0
            )
            scene.add_geometry(axis)
            mesh = trimesh.Trimesh(
                vertices=vertices.squeeze(dim=0).cpu().numpy(),
                faces=faces.cpu().numpy(),
            )
            show_overlays = show_bones or show_bone_axes
            alpha = 0.5 if show_overlays else 1.0
            material = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[0.4, 0.8, 0.8, alpha],
                metallicFactor=0.5,
                doubleSided=False if show_overlays else True,
                alphaMode="BLEND" if show_overlays else "OPAQUE",
            )
            mesh.visual = trimesh.visual.TextureVisuals(material=material)
            scene.add_geometry(mesh, node_name="body")

            if show_self_intersections:
                # Show self intersections
                colliding_triangle_ids = (
                    self_intersection_module.detect_self_intersections(
                        vertices.squeeze(dim=0)
                    )
                )
                mask = colliding_triangle_ids > 0
                if torch.any(mask):
                    tris = self_intersection_module.triangular_faces[mask]
                    colliding_mesh = trimesh.Trimesh(
                        vertices=vertices.squeeze(dim=0).cpu().numpy(),
                        faces=tris.cpu().numpy(),
                    )
                    colliding_mesh.visual = trimesh.visual.TextureVisuals(
                        material=trimesh.visual.material.PBRMaterial(
                            baseColorFactor=[1.0, 0.2, 0.2, 0.5],
                            metallicFactor=0.5,
                            doubleSided=False,
                            alphaMode="OPAQUE",
                        )
                    )
                    scene.add_geometry(colliding_mesh, node_name="self_intersections")

            if show_bones:
                # Add bones visualization
                bone_heads = (
                    output["bone_poses"][..., :3, 3].detach().cpu().squeeze(dim=0)
                )

                bone_colors = [[0.3, 0.3, 0.3, 1.0]]
                bone_visuals = [
                    trimesh.visual.TextureVisuals(
                        material=trimesh.visual.material.PBRMaterial(
                            baseColorFactor=color,
                            metallicFactor=0.0,
                            roughnessFactor=1.0,
                            doubleSided=True,
                            alphaMode="BLEND",
                        )
                    )
                    for color in bone_colors
                ]
                for i in range(len(bone_heads)):
                    bone_head = bone_heads[i]
                    # Connect to the paent bone if it exists
                    parent_id = model.bone_parents[i]
                    if parent_id >= 0:
                        bone_tail = bone_heads[parent_id]
                        bone_length = torch.norm(bone_tail - bone_head).item()
                        # Skip degenerate bones coincident with their parent (e.g. root aligned onto
                        # the pelvis): a zero-length segment makes special_gramschmidt return NaN,
                        # which corrupts the exported scene so nothing renders.
                        if bone_length < 1e-6:
                            continue
                        cylinder = trimesh.creation.cylinder(
                            radius=0.0025, height=bone_length, sections=16
                        )
                        t = (bone_head + bone_tail) / 2
                        M = roma.special_gramschmidt(
                            torch.stack(
                                [
                                    bone_tail - bone_head,
                                    torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
                                ],
                                dim=-1,
                            )
                        )
                        R = torch.stack([M[:, 2], M[:, 1], M[:, 0]], dim=-1)
                        cylinder.visual = bone_visuals[i % len(bone_colors)]
                        scene.add_geometry(
                            cylinder,
                            transform=roma.Rigid(R, t).to_homogeneous().numpy(),
                            node_name=f"bone_{model.bone_labels[i]}",
                        )

                # Add some spheres at the joints
                bone_poses = output["bone_poses"].squeeze(dim=0).cpu()
                joint_sphere = trimesh.creation.icosphere(radius=0.004, subdivisions=2)
                joint_sphere.visual = trimesh.visual.TextureVisuals(
                    material=trimesh.visual.material.PBRMaterial(
                        baseColorFactor=[0.1, 0.1, 0.1, 1.0],
                        metallicFactor=0.5,
                        roughnessFactor=1.0,
                        doubleSided=True,
                        alphaMode="OPAQUE",
                    )
                )
                for i in range(len(bone_poses)):
                    scene.add_geometry(
                        joint_sphere,
                        transform=bone_poses[i],
                        node_name=f"joint_{model.bone_labels[i]}",
                    )

            if show_bone_axes:
                # Show each bone pose as a coordinate frame (axes) to visualize orientation
                bone_poses = output["bone_poses"].squeeze(dim=0).cpu()
                for i in range(len(bone_poses)):
                    frame = trimesh.creation.axis(
                        origin_size=0.002, axis_radius=0.001, axis_length=0.025
                    )
                    scene.add_geometry(
                        frame,
                        transform=bone_poses[i].numpy(),
                        node_name=f"pose_{model.bone_labels[i]}",
                    )

            # The gradio Model3D component does not use a Z-up camera orientation by default. We apply a scene rotation to compensate.
            view_transform = (
                roma.Rigid(
                    roma.euler_to_rotmat("x", [-90.0], degrees=True), torch.zeros(3)
                )
                .to_homogeneous()
                .numpy()
            )
            scene.apply_transform(view_transform)
            scene.export(mesh_filename)

            # Compute measurements
            if measurements_class is not None:
                measurements = measurements_class(output["rest_vertices"])
                measurements_summary = gr.Markdown(
                    "\n".join(
                        [
                            f" - {key}: {value.squeeze().item():.2f}"
                            for key, value in measurements.items()
                        ]
                    )
                )
            else:
                measurements_summary = gr.Markdown("No measurements available.")
            return mesh_filename, temp_params_file.name, measurements_summary

        def initialize_model(topology, rig):
            """
            Initialize the model and return a dropdown with bone labels.
            """
            nonlocal \
                model, \
                measurements_class, \
                bones_rotvec, \
                phenotype_kwargs, \
                local_changes_kwargs, \
                facial_actions_kwargs, \
                self_intersection_module
            model = anny.Anny(
                rig=rig, topology=topology, local_changes="default", facial_actions=True
            )

            measurements_class = None

            model = model.to(dtype=dtype)
            bones_rotvec = torch.zeros((len(model.bone_labels), 3), dtype=dtype)
            phenotype_kwargs = {key: 0.5 for key in model.phenotype_labels}
            local_changes_kwargs = {key: 0.0 for key in model.local_change_labels}
            facial_actions_kwargs = {key: 0.0 for key in model.facial_action_labels}

            self_intersection_module = anny.utils.collision.SelfInterpenetrationModule(
                model
            )

            description = gr.Markdown(
                "\n".join(
                    [
                        f"- Vertices: {len(model.template_vertices)}",
                        f"- Faces: {len(model.faces)}",
                        f"- Bones: {len(model.bone_labels)}",
                        f"- Blendshapes: {model.blendshapes.shape[0]}",
                        f"- Max influencing bones: {model.vertex_bone_weights.shape[1]}",
                    ]
                )
            )
            phenotype_dropdown = gr.Dropdown(
                label="Phenotype",
                choices=model.phenotype_labels,
                value=model.phenotype_labels[0],
            )
            macrodetail_slider = gr.Slider(
                label="Value",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=phenotype_kwargs[model.phenotype_labels[0]],
            )
            if len(model.local_change_labels) > 0:
                local_change_dropdown = gr.Dropdown(
                    label="Local change",
                    choices=model.local_change_labels,
                    value=model.local_change_labels[0],
                    interactive=True,
                    visible=True,
                )
                local_changes_slider = gr.Slider(
                    label="Value",
                    minimum=-1.0,
                    maximum=1.0,
                    step=0.05,
                    value=local_changes_kwargs[model.local_change_labels[0]],
                    interactive=True,
                    visible=True,
                )
            else:
                local_change_dropdown = gr.Dropdown(
                    label="Local change",
                    choices=["None"],
                    value="None",
                    interactive=False,
                    visible=False,
                )
                local_changes_slider = gr.Slider(
                    label="Value",
                    minimum=-1.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.0,
                    interactive=False,
                    visible=False,
                )

            if len(model.facial_action_labels) > 0:
                facial_action_dropdown = gr.Dropdown(
                    label="Facial action",
                    choices=model.facial_action_labels,
                    value=model.facial_action_labels[0],
                    interactive=True,
                    visible=True,
                )
                facial_actions_slider = gr.Slider(
                    label="Value",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=facial_actions_kwargs[model.facial_action_labels[0]],
                    interactive=True,
                    visible=True,
                )
            else:
                facial_action_dropdown = gr.Dropdown(
                    label="Facial action",
                    choices=["None"],
                    value="None",
                    interactive=False,
                    visible=False,
                )
                facial_actions_slider = gr.Slider(
                    label="Value",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.0,
                    interactive=False,
                    visible=False,
                )

            reset_shape_button = gr.Button("Reset shape")
            bone_dropdown = gr.Dropdown(
                label="Bone orientation",
                choices=model.bone_labels,
                type="index",
                value=model.bone_labels[0],
            )
            x_slider = gr.Slider(label="X", minimum=-180, maximum=180, step=1, value=0)
            y_slider = gr.Slider(label="Y", minimum=-180, maximum=180, step=1, value=0)
            z_slider = gr.Slider(label="Z", minimum=-180, maximum=180, step=1, value=0)
            reset_pose_button = gr.Button("Reset pose")
            filename, params_filename, measurements_summary = export_mesh()
            model3d = gr.Model3D(value=filename, height="100vh")
            return (
                description,
                phenotype_dropdown,
                macrodetail_slider,
                local_change_dropdown,
                local_changes_slider,
                facial_action_dropdown,
                facial_actions_slider,
                reset_shape_button,
                bone_dropdown,
                x_slider,
                y_slider,
                z_slider,
                reset_pose_button,
                model3d,
                measurements_summary,
            )

        default_model_value = "anny"
        default_rig_value = "anny"
        show_bones_checkbox = gr.Checkbox(
            label="Show bones", value=show_bones, visible=True, interactive=True
        )
        show_bone_axes_checkbox = gr.Checkbox(
            label="Show bone axes", value=show_bone_axes, visible=True, interactive=True
        )
        show_self_intersections_checkbox = gr.Checkbox(
            label="Show self intersections", value=False, visible=True, interactive=True
        )
        model_gradio_outputs = initialize_model(default_model_value, default_rig_value)
        (
            description,
            phenotype_dropdown,
            macrodetail_slider,
            local_change_dropdown,
            local_changes_slider,
            facial_action_dropdown,
            facial_actions_slider,
            reset_shape_button,
            bone_dropdown,
            x_slider,
            y_slider,
            z_slider,
            reset_pose_button,
            model3d,
            measurements_summary,
        ) = model_gradio_outputs

        with gr.Blocks(
            title="Anny Model", css="#control-column { max-width: 60pt; }"
        ) as demo:
            with gr.Row():
                with gr.Column("compact", elem_id="control-column"):
                    model_dropdown = gr.Dropdown(
                        label="Topology",
                        choices=[
                            "anny",
                            "hand.L",
                            "hand.R",
                            "head",
                            "notoes_collapse10pc",
                            "notoes_collapse5pc",
                            "smplx",
                            "smpl",
                            "soma",
                        ],
                        value=default_model_value,
                    )
                    rig_dropdown = gr.Dropdown(
                        label="Rig",
                        choices=[
                            "anny",
                            "makehuman",
                            "mixamo",
                            "anny-notoes",
                            "anny-nohands",
                            "soma",
                        ],
                        value=default_rig_value,
                    )
                    show_bones_checkbox.render()
                    show_bone_axes_checkbox.render()
                    show_self_intersections_checkbox.render()
                    description.render()
                    measurements_summary.render()
                    phenotype_dropdown.render()
                    macrodetail_slider.render()
                    local_change_dropdown.render()
                    local_changes_slider.render()
                    facial_action_dropdown.render()
                    facial_actions_slider.render()
                    reset_shape_button.render()
                    bone_dropdown.render()
                    x_slider.render()
                    y_slider.render()
                    z_slider.render()
                    reset_pose_button.render()
                    download_params_button = gr.DownloadButton(
                        label="Download parameters", value=temp_params_file.name
                    )
                model3d.render()

            model_dropdown.change(
                initialize_model,
                inputs=[model_dropdown, rig_dropdown],
                outputs=model_gradio_outputs,
            )
            rig_dropdown.change(
                initialize_model,
                inputs=[model_dropdown, rig_dropdown],
                outputs=model_gradio_outputs,
            )

            def update_show_bones(show_bones_checkbox):
                """
                Called when the show bones checkbox is changed.
                """
                nonlocal show_bones
                show_bones = show_bones_checkbox
                return export_mesh()

            show_bones_checkbox.change(
                update_show_bones,
                inputs=[show_bones_checkbox],
                outputs=[model3d, download_params_button, measurements_summary],
            )

            def update_show_bone_axes(show_bone_axes_checkbox):
                """
                Called when the show bone axes checkbox is changed.
                """
                nonlocal show_bone_axes
                show_bone_axes = show_bone_axes_checkbox
                return export_mesh()

            show_bone_axes_checkbox.change(
                update_show_bone_axes,
                inputs=[show_bone_axes_checkbox],
                outputs=[model3d, download_params_button, measurements_summary],
            )

            def update_show_self_intersections(show_self_intersections_checkbox):
                """
                Called when the show self intersections checkbox is changed.
                """
                nonlocal show_self_intersections
                show_self_intersections = show_self_intersections_checkbox
                return export_mesh()

            show_self_intersections_checkbox.change(
                update_show_self_intersections,
                inputs=[show_self_intersections_checkbox],
                outputs=[model3d, download_params_button, measurements_summary],
            )

            def update_phenotype_label(macrodetail_label):
                """
                Called when the selected macrodetail changes.
                """
                return phenotype_kwargs[macrodetail_label]

            phenotype_dropdown.change(
                update_phenotype_label,
                inputs=phenotype_dropdown,
                outputs=macrodetail_slider,
            )

            def update_phenotype_slider(macrodetail_label, value):
                """
                Called when the macrodetail slider is changed.
                """
                phenotype_kwargs[macrodetail_label] = value
                return export_mesh()

            macrodetail_slider.change(
                update_phenotype_slider,
                inputs=[phenotype_dropdown, macrodetail_slider],
                outputs=[model3d, download_params_button, measurements_summary],
            )

            def update_local_changes_label(local_changes_label):
                """
                Called when the selected local changes changes.
                """
                if len(local_changes_kwargs) == 0:
                    return 0.0
                else:
                    return local_changes_kwargs[local_changes_label]

            local_change_dropdown.change(
                update_local_changes_label,
                inputs=local_change_dropdown,
                outputs=local_changes_slider,
            )

            def update_local_changes_slider(local_changes_label, value):
                """
                Called when the local changes slider is changed.
                """
                if local_changes_kwargs is not None:
                    local_changes_kwargs[local_changes_label] = value
                return export_mesh()

            local_changes_slider.change(
                update_local_changes_slider,
                inputs=[local_change_dropdown, local_changes_slider],
                outputs=[model3d, download_params_button, measurements_summary],
            )

            def update_facial_action_label(facial_action_label):
                """
                Called when the selected facial action changes.
                """
                if not facial_actions_kwargs:
                    return 0.0
                return facial_actions_kwargs[facial_action_label]

            facial_action_dropdown.change(
                update_facial_action_label,
                inputs=facial_action_dropdown,
                outputs=facial_actions_slider,
            )

            def update_facial_actions_slider(facial_action_label, value):
                """
                Called when the facial action slider is changed.
                """
                if (
                    facial_actions_kwargs
                    and facial_action_label in facial_actions_kwargs
                ):
                    facial_actions_kwargs[facial_action_label] = value
                return export_mesh()

            facial_actions_slider.change(
                update_facial_actions_slider,
                inputs=[facial_action_dropdown, facial_actions_slider],
                outputs=[model3d, download_params_button, measurements_summary],
            )

            def reset_shape(macrodetail_label, local_change_label, facial_action_label):
                """
                Called when the reset shape button is clicked.
                """
                for key in model.phenotype_labels:
                    phenotype_kwargs[key] = 0.5
                if local_changes_kwargs is not None:
                    for key in list(local_changes_kwargs.keys()):
                        local_changes_kwargs[key] = 0.0
                if facial_actions_kwargs is not None:
                    for key in list(facial_actions_kwargs.keys()):
                        facial_actions_kwargs[key] = 0.0
                local_change_output = (
                    local_changes_kwargs[local_change_label]
                    if local_changes_kwargs is not None
                    else 0.0
                )
                facial_action_output = (
                    facial_actions_kwargs.get(facial_action_label, 0.0)
                    if facial_actions_kwargs is not None
                    else 0.0
                )
                return (
                    *export_mesh(),
                    phenotype_kwargs[macrodetail_label],
                    local_change_output,
                    facial_action_output,
                )

            reset_shape_button.click(
                reset_shape,
                inputs=[
                    phenotype_dropdown,
                    local_change_dropdown,
                    facial_action_dropdown,
                ],
                outputs=[
                    model3d,
                    download_params_button,
                    measurements_summary,
                    macrodetail_slider,
                    local_changes_slider,
                    facial_actions_slider,
                ],
            )

            def update_bone_label(bone_index):
                """
                Called when the selected bone changes.
                """
                index = bone_index
                return [
                    bones_rotvec[index, 0].item(),
                    bones_rotvec[index, 1].item(),
                    bones_rotvec[index, 2].item(),
                ]

            bone_dropdown.change(
                update_bone_label,
                inputs=bone_dropdown,
                outputs=[x_slider, y_slider, z_slider],
            )

            def update_bone_rotvec(bone_index, x, y, z):
                """
                Called when the sliders are changed.
                """
                index = bone_index
                bones_rotvec[index, 0] = x
                bones_rotvec[index, 1] = y
                bones_rotvec[index, 2] = z
                return export_mesh()

            x_slider.change(
                update_bone_rotvec,
                inputs=[bone_dropdown, x_slider, y_slider, z_slider],
                outputs=[model3d, download_params_button, measurements_summary],
            )
            y_slider.change(
                update_bone_rotvec,
                inputs=[bone_dropdown, x_slider, y_slider, z_slider],
                outputs=[model3d, download_params_button, measurements_summary],
            )
            z_slider.change(
                update_bone_rotvec,
                inputs=[bone_dropdown, x_slider, y_slider, z_slider],
                outputs=[model3d, download_params_button, measurements_summary],
            )

            def reset_bone_rotvec(bone_index):
                """
                Called when the reset all button is clicked."""
                bones_rotvec.zero_()
                return (
                    *export_mesh(),
                    bones_rotvec[bone_index, 0].item(),
                    bones_rotvec[bone_index, 1].item(),
                    bones_rotvec[bone_index, 2].item(),
                )

            reset_pose_button.click(
                reset_bone_rotvec,
                inputs=[bone_dropdown],
                outputs=[
                    model3d,
                    download_params_button,
                    measurements_summary,
                    x_slider,
                    y_slider,
                    z_slider,
                ],
            )
            # Launch the Gradio app
            demo.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
