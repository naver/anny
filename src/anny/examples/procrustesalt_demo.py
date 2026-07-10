# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Interactive demo for the experimental "procrustesalt" Anny model.

This is a proof-of-concept variant of ``interactive_demo.py`` that poses and
shapes a single fixed model whose bone orientations are solved with the
"procrustesalt" method. That method is not exposed through
``create_fullbody_model`` yet, so the model is assembled by monkey-patching,
exactly as in ``procrustesalt_example.py``.
"""
import os
from functools import partial

import anny
import roma
import torch
import gradio as gr
import tempfile
import json
import trimesh
from anny.paths import get_anny_root_dir
import anny.skinning.skinning as skinning


def build_procrustesalt_model(dtype=torch.float32):
    """
    Build the experimental procrustesalt model, reproducing the setup from
    ``procrustesalt_example.py``.

    The precomputed data in ``data/procrustes/default.pth`` was generated for
    ``rig="default"``, ``local_changes="all"``, ``facial_actions=False``, so the
    base model must use the same configuration. Blend shape and bone counts are
    independent of the topology, hence the SMPL topology can be used here.
    """
    model = anny.create_fullbody_model(
        rig="default",
        topology="smpl",
        local_changes="all",
        bone_orientation="blender-rootidentity",
        pose_parameterization="local-ref"
    )
    model = model.to(dtype=dtype)
    data = torch.load(os.path.join(get_anny_root_dir(), "data/procrustes/default.pth"))

    # Monkey patching to enable the procrustesalt bone orientation method.
    # The precomputed tensors are stored as float64; cast them to the model dtype.
    assert len(model.bone_labels) == len(data["bone_labels"]) and [model.bone_labels[i] == data["bone_labels"][i] for i in range(len(model.bone_labels))]
    model.bone_template_orientation_matrices = data["bone_template_orientation_matrices"].to(dtype=dtype)
    model.bone_orientation_blendshapes = data["bone_orientation_blendshapes"].to(dtype=dtype)
    model._bone_orientation_method = "procrustesalt"
    model.reference_bone_orientations = data["reference_bone_orientations"].to(dtype=dtype)

    def _get_procrustesalt_rest_model(self, blendshape_coeffs: torch.Tensor) -> dict[str, torch.Tensor]:
        rest_vertices = self.get_rest_vertices(blendshape_coeffs)
        rest_bone_heads = skinning.apply_linear_blendshape(self.template_bone_heads, self.bone_heads_blendshapes, blendshape_coeffs)
        # b: batch
        # k: bone
        # a: blend shape
        B = torch.einsum("ba,akij->bkij", blendshape_coeffs, self.bone_orientation_blendshapes)
        M = self.bone_template_orientation_matrices[None] + B
        rest_bone_orientation = roma.special_procrustes(M)
        rest_bone_poses = roma.Rigid(linear=rest_bone_orientation, translation=rest_bone_heads).to_homogeneous()
        return dict(rest_vertices=rest_vertices, rest_bone_heads=rest_bone_heads, rest_bone_poses=rest_bone_poses)

    model._get_procrustesalt_rest_model = partial(_get_procrustesalt_rest_model, model)

    def get_rest_model(self, blendshape_coeffs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._bone_orientation_method == "tail":
            return self._get_tail_rest_model(blendshape_coeffs)
        if self._bone_orientation_method == "procrustes":
            return self._get_procrustes_rest_model(blendshape_coeffs)
        elif self._bone_orientation_method == "procrustesalt":
            return self._get_procrustesalt_rest_model(blendshape_coeffs)
        raise ValueError(f"Unknown bone orientation method: {self._bone_orientation_method!r}")

    model.get_rest_model = partial(get_rest_model, model)

    return model


def main(server_name: str = None, server_port: int = None):
    dtype = torch.float32

    with (tempfile.NamedTemporaryFile(suffix=".glb") as temp_file,
          tempfile.NamedTemporaryFile(suffix=".json") as temp_params_file):

        mesh_filename = temp_file.name

        model = build_procrustesalt_model(dtype=dtype)
        bones_rotvec = torch.zeros((len(model.bone_labels), 3), dtype=dtype)
        phenotype_kwargs = {key: 0.5 for key in model.phenotype_labels}
        local_changes_kwargs = {key: 0. for key in model.local_change_labels}
        show_bones = False

        def export_mesh():
            nonlocal show_bones
            bones_rotmat = roma.rotvec_to_rotmat(torch.deg2rad(bones_rotvec))
            pose_parameters = roma.Rigid(bones_rotmat, torch.zeros((len(bones_rotmat), 3), dtype=dtype))[None].to_homogeneous()
            output = model(pose_parameters=pose_parameters,
                           phenotype_kwargs=phenotype_kwargs,
                           local_changes_kwargs=local_changes_kwargs)
            vertices = output["vertices"]
            faces = model.faces

            # Save parameters to file
            with open(temp_params_file.name, "w") as f:
                data = dict(phenotype_kwargs={key: value for key, value in phenotype_kwargs.items() if value != 0.5},
                            local_changes_kwargs={key: value for key, value in local_changes_kwargs.items() if value != 0.},
                            pose_parameterization=model.pose_parameterization,
                            pose_parameters={key: matrix for key, matrix in zip(model.bone_labels, pose_parameters.squeeze(dim=0).cpu().numpy().tolist())})
                json.dump(data, f)

            scene = trimesh.Scene()
            axis = trimesh.creation.axis(origin_size=0.01, axis_radius=0.005, axis_length=1.0)
            scene.add_geometry(axis)
            mesh = trimesh.Trimesh(vertices=vertices.squeeze(dim=0).cpu().numpy(), faces=faces.cpu().numpy())
            alpha = 0.5 if show_bones else 1.0
            material = trimesh.visual.material.PBRMaterial(baseColorFactor=[0.4, 0.8, 0.8, alpha],
                                                           metallicFactor=0.5,
                                                           doubleSided=False if show_bones else True,
                                                           alphaMode='BLEND' if show_bones else 'OPAQUE')
            mesh.visual = trimesh.visual.TextureVisuals(material=material)
            scene.add_geometry(mesh, node_name="body")

            if show_bones:
                # Add bones visualization
                bone_heads = output['bone_poses'][..., :3, 3].detach().cpu().squeeze(dim=0)

                bone_colors = [[0.8, 0.3, 0.3, 1.0]]
                bone_visuals = [trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=color,
                                                                        metallicFactor=0.,
                                                                        roughnessFactor=1.,
                                                                        doubleSided=True,
                                                                        alphaMode='BLEND')) for color in bone_colors]
                for i in range(len(bone_heads)):
                    bone_head = bone_heads[i]
                    # Connect to the parent bone if it exists
                    parent_id = model.bone_parents[i]
                    if parent_id >= 0:
                        bone_tail = bone_heads[parent_id]
                        cylinder = trimesh.creation.cylinder(radius=0.005, height=torch.norm(bone_tail - bone_head).item(), sections=16)
                        t = (bone_head + bone_tail) / 2
                        M = roma.special_gramschmidt(torch.stack([bone_tail - bone_head, torch.tensor([0., 0., 1.], dtype=dtype)], dim=-1))
                        R = torch.stack([M[:, 2], M[:, 1], M[:, 0]], dim=-1)
                        cylinder.visual = bone_visuals[i % len(bone_colors)]
                        scene.add_geometry(cylinder, transform=roma.Rigid(R, t).to_homogeneous().numpy(),
                                           node_name=f"bone_{model.bone_labels[i]}")

                # Show each bone pose as a coordinate frame (axes) to visualize orientation
                bone_poses = output["bone_poses"].squeeze(dim=0).cpu()
                for i in range(len(bone_poses)):
                    frame = trimesh.creation.axis(origin_size=0.004, axis_radius=0.002, axis_length=0.1)
                    scene.add_geometry(frame, transform=bone_poses[i].numpy(), node_name=f"pose_{model.bone_labels[i]}")

            # The gradio Model3D component does not use a Z-up camera orientation by default. We apply a scene rotation to compensate.
            view_transform = roma.Rigid(roma.euler_to_rotmat('x', [-90.], degrees=True), torch.zeros(3)).to_homogeneous().numpy()
            scene.apply_transform(view_transform)
            scene.export(mesh_filename)

            return mesh_filename, temp_params_file.name

        description = gr.Markdown(
            "\n".join([
                "### Procrustesalt model (experimental)",
                f"- Vertices: {len(model.template_vertices)}",
                f"- Faces: {len(model.faces)}",
                f"- Bones: {len(model.bone_labels)}",
                f"- Blendshapes: {model.blendshapes.shape[0]}",
                f"- Max influencing bones: {model.vertex_bone_weights.shape[1]}",
            ])
        )

        with gr.Blocks(title="Anny Procrustesalt Model", css="#control-column { max-width: 60pt; }") as demo:
            with gr.Row():
                with gr.Column("compact", elem_id="control-column"):
                    show_bones_checkbox = gr.Checkbox(label="Show bones", value=show_bones, visible=True, interactive=True)
                    description.render()
                    phenotype_dropdown = gr.Dropdown(label="Phenotype", choices=model.phenotype_labels, value=model.phenotype_labels[0])
                    macrodetail_slider = gr.Slider(label="Value", minimum=0., maximum=1., step=0.05, value=phenotype_kwargs[model.phenotype_labels[0]])
                    if len(model.local_change_labels) > 0:
                        local_change_dropdown = gr.Dropdown(label="Local change", choices=model.local_change_labels, value=model.local_change_labels[0], interactive=True, visible=True)
                        local_changes_slider = gr.Slider(label="Value", minimum=-1., maximum=1., step=0.05, value=0., interactive=True, visible=True)
                    else:
                        local_change_dropdown = gr.Dropdown(label="Local change", choices=["None"], value="None", interactive=False, visible=False)
                        local_changes_slider = gr.Slider(label="Value", minimum=-1., maximum=1., step=0.05, value=0., interactive=False, visible=False)
                    reset_shape_button = gr.Button("Reset shape")
                    bone_dropdown = gr.Dropdown(label="Bone orientation", choices=model.bone_labels, type="index", value=model.bone_labels[0])
                    x_slider = gr.Slider(label="X", minimum=-180, maximum=180, step=1, value=0)
                    y_slider = gr.Slider(label="Y", minimum=-180, maximum=180, step=1, value=0)
                    z_slider = gr.Slider(label="Z", minimum=-180, maximum=180, step=1, value=0)
                    reset_pose_button = gr.Button("Reset pose")
                    download_params_button = gr.DownloadButton(label="Download parameters", value=temp_params_file.name)
                filename, _ = export_mesh()
                model3d = gr.Model3D(value=filename, height="100vh")

            def update_show_bones(show_bones_value):
                nonlocal show_bones
                show_bones = show_bones_value
                return export_mesh()
            show_bones_checkbox.change(update_show_bones, inputs=[show_bones_checkbox], outputs=[model3d, download_params_button])

            def update_phenotype_label(macrodetail_label):
                return phenotype_kwargs[macrodetail_label]
            phenotype_dropdown.change(update_phenotype_label, inputs=phenotype_dropdown, outputs=macrodetail_slider)

            def update_phenotype_slider(macrodetail_label, value):
                phenotype_kwargs[macrodetail_label] = value
                return export_mesh()
            macrodetail_slider.change(update_phenotype_slider, inputs=[phenotype_dropdown, macrodetail_slider], outputs=[model3d, download_params_button])

            def update_local_changes_label(local_changes_label):
                if len(local_changes_kwargs) == 0:
                    return 0.
                return local_changes_kwargs[local_changes_label]
            local_change_dropdown.change(update_local_changes_label, inputs=local_change_dropdown, outputs=local_changes_slider)

            def update_local_changes_slider(local_changes_label, value):
                if local_changes_label in local_changes_kwargs:
                    local_changes_kwargs[local_changes_label] = value
                return export_mesh()
            local_changes_slider.change(update_local_changes_slider, inputs=[local_change_dropdown, local_changes_slider], outputs=[model3d, download_params_button])

            def reset_shape(macrodetail_label, local_change_label):
                for key in model.phenotype_labels:
                    phenotype_kwargs[key] = 0.5
                for key in list(local_changes_kwargs.keys()):
                    local_changes_kwargs[key] = 0.
                local_change_output = local_changes_kwargs[local_change_label] if local_change_label in local_changes_kwargs else 0.
                return *export_mesh(), phenotype_kwargs[macrodetail_label], local_change_output
            reset_shape_button.click(reset_shape, inputs=[phenotype_dropdown, local_change_dropdown], outputs=[model3d, download_params_button, macrodetail_slider, local_changes_slider])

            def update_bone_label(bone_index):
                index = bone_index
                return [bones_rotvec[index, 0].item(), bones_rotvec[index, 1].item(), bones_rotvec[index, 2].item()]
            bone_dropdown.change(update_bone_label, inputs=bone_dropdown, outputs=[x_slider, y_slider, z_slider])

            def update_bone_rotvec(bone_index, x, y, z):
                index = bone_index
                bones_rotvec[index, 0] = x
                bones_rotvec[index, 1] = y
                bones_rotvec[index, 2] = z
                return export_mesh()
            x_slider.change(update_bone_rotvec, inputs=[bone_dropdown, x_slider, y_slider, z_slider], outputs=[model3d, download_params_button])
            y_slider.change(update_bone_rotvec, inputs=[bone_dropdown, x_slider, y_slider, z_slider], outputs=[model3d, download_params_button])
            z_slider.change(update_bone_rotvec, inputs=[bone_dropdown, x_slider, y_slider, z_slider], outputs=[model3d, download_params_button])

            def reset_bone_rotvec(bone_index):
                bones_rotvec.zero_()
                return *export_mesh(), bones_rotvec[bone_index, 0].item(), bones_rotvec[bone_index, 1].item(), bones_rotvec[bone_index, 2].item()
            reset_pose_button.click(reset_bone_rotvec, inputs=[bone_dropdown], outputs=[model3d, download_params_button, x_slider, y_slider, z_slider])

            # Launch the Gradio app
            demo.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
