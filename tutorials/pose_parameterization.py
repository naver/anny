# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Parameterizing poses with Anny
#
# First we start by instantiating the model, and a few helper objects for visualization.

# %%
import torch
import roma # A PyTorch library useful to deal with space transformations.
import anny # The main library for the Anny model.
import trimesh # For 3D mesh visualization.

# Instantiate the model.
# Remark: the first instantiation may take a while. Latter calls will be faster thanks to caching.
anny_model = anny.Anny(rig="default-notongue", topology="default-notongue", remove_unattached_vertices=True).to(dtype=torch.float32, device='cpu')

# Some helper objects for visualization.
trimesh_scene_transform = roma.Rigid(linear=roma.euler_to_rotmat('x', [-90.], degrees=True), translation=None).to_homogeneous().cpu().numpy()

mesh_material = trimesh.visual.material.PBRMaterial(baseColorFactor=[0.6, 0.8, 0.7, 0.5],
                                                        metallicFactor=0.5,
                                                        doubleSided=False,
                                                        alphaMode='BLEND')

world_axis = trimesh.creation.axis(axis_length=1.)
axis = trimesh.creation.axis(axis_length=0.1)

# %% [markdown]
# ### Pose parameterization
#
# A pose is described by a set of 4x4 rigid transformation matrices, one per bone. Here we set all bone transformations to be the identity, which corresponds to the rest pose:
#
# *Remark: we use direct transforms and column vectors conventions, so that coordinates $X_i$ of a vector are transformed by a matrix $M_{i,j}$ into $Y_i = \sum_k M_{ik} X_k$.*

# %%
# The code supports batched inputs. Here we use a batch size of 1.
batch_size = 1
pose_parameters = {label: torch.eye(4)[None].repeat(batch_size, 1, 1) for label in anny_model.bone_labels}
output = anny_model(pose_parameters=pose_parameters)

mesh = trimesh.Trimesh(vertices=output['vertices'].squeeze(0), faces=anny_model.faces, process=False)
mesh.visual = trimesh.visual.TextureVisuals(material=mesh_material)
scene = trimesh.Scene()
scene.add_geometry(world_axis)
scene.add_geometry(mesh)
scene.apply_transform(trimesh_scene_transform)
scene.show()

# %% [markdown]
# Alternatively, pose parameters can be specified by a single tensor stacking all bone transformations. The ordering of the bones is consistent with the one of `anny_model.bone_labels`.

# %%
pose_parameters = torch.eye(4, dtype=torch.float32)[None, None].repeat(batch_size, anny_model.bone_count, 1, 1)

output = anny_model(pose_parameters=pose_parameters)

mesh = trimesh.Trimesh(vertices=output['vertices'].squeeze(0), faces=anny_model.faces, process=False)
mesh.visual = trimesh.visual.TextureVisuals(material=mesh_material)
scene = trimesh.Scene()
scene.add_geometry(world_axis)
scene.add_geometry(mesh)
scene.apply_transform(trimesh_scene_transform)
scene.show()

# %% [markdown]
# ### Setting bone pose
#
# The *root* bone is the first bone of the kinematic chain of the model.
#
# Anny supports different pose parameterizations that behave differently.
# The choice of parameterization can be changed at instantiation through the `pose_parameterization` argument, or at each call of Anny using the `pose_parameterization` argument.
#
# ### pose_parameterization="local-bone"
#
# In *local-bone* mode, the transformation associated with the root bone describes the pose of the root bone, with respect to the world coordinate system. When using the identity tranformation for the root bone, its coordinate system is therefore aligned with the world coordinate system.
#
# The transformation associated to a bone affects this bone and its children along the kinematic chain of the model.
# It describes the transformation applied to the bone with respect to its parent bone, and is expressed with respect to the rest coordinates system of this bone.
#
# Let us consider a particular bone, the left shoulder for example. We show below its default pose:

# %%
bone_id = anny_model.bone_labels.index('shoulder01.L')
print(f"Bone 'shoulder01.L' has id {bone_id}.")

pose_parameters = {label: torch.eye(4)[None] for label in anny_model.bone_labels}
output = anny_model(pose_parameters=pose_parameters)
mesh = trimesh.Trimesh(vertices=output['vertices'].squeeze(), faces=anny_model.faces, process=False)
mesh.visual = trimesh.visual.TextureVisuals(material=mesh_material)
scene = trimesh.Scene()
scene.add_geometry(mesh)
scene.add_geometry(world_axis)
scene.add_geometry(axis, transform=output["bone_poses"].squeeze(0)[bone_id].cpu().numpy())
scene.apply_transform(trimesh_scene_transform)
scene.show()

# %% [markdown]
# Let us apply a rotation around the Z axis (in blue) to rotate this bone with respect to its parent. Note how it affects the whole arm.

# %%

pose_parameters = {label: torch.eye(4)[None] for label in anny_model.bone_labels}
pose_parameters["shoulder01.L"] = roma.Rigid(roma.euler_to_rotmat("z", [30.], degrees=True), translation=None).to_homogeneous()[None]
output = anny_model(pose_parameters=pose_parameters)
mesh = trimesh.Trimesh(vertices=output['vertices'].squeeze(), faces=anny_model.faces, process=False)
mesh.visual = trimesh.visual.TextureVisuals(material=mesh_material)
scene = trimesh.Scene()
scene.add_geometry(mesh)
scene.add_geometry(world_axis)
scene.add_geometry(axis, transform=output["bone_poses"].squeeze(0)[bone_id].cpu().numpy())
scene.apply_transform(trimesh_scene_transform)
scene.show()

# %% [markdown]
# #### ```pose_parameterization="world"```
# In *world* mode, the pose parameterization consists of rigid transformations that describe the pose of each bone individually in the the world coordinate system. It allows to specify per-bone transformations easily, regardless of the kinematic skeleton.

# %%
# Retrieve an absolute parameterization of the previous output mesh.
absolute_parameters = output["bone_poses"].clone()

# Translate and scale the left upperarm bone (note how it does not affect the other bones of the kinematic chain).
bone_id = anny_model.bone_labels.index('upperarm01.L')
absolute_parameters[:,bone_id,2,3] += 0.2 # translate of 20cm in absolute Z direction
absolute_parameters[:,bone_id,:3,:3] *= 1.2 # scale the bone by 20%

new_output = anny_model(pose_parameters=absolute_parameters, pose_parameterization="world")
mesh = trimesh.Trimesh(vertices=new_output['vertices'].squeeze().cpu().numpy(), faces=anny_model.faces, process=False)
mesh.visual = trimesh.visual.TextureVisuals(material=mesh_material)
scene = trimesh.Scene()
scene.add_geometry(world_axis)
scene.add_geometry(mesh)
scene.add_geometry(axis, transform=new_output["bone_poses"].squeeze(0)[bone_id])
scene.apply_transform(trimesh_scene_transform)
scene.show()

# %% [markdown]
# #### ```pose_parameterization="world-orient"```
# In *world-orient* mode, the parameterization of the root bone does not change, but parameterization of the other bones consist in 3D rotations describing their orientation in the world coordinate system. The bone location is automatically resolved through forward kinematics.

# %%
output = anny_model(pose_parameterization="local-bone") # rest pose
pose_parameters = anny_model.get_pose_parameterization(output, pose_parameterization="world-orient")
# Manually set the bone orientation for the left upper arm
bone_id1 = anny_model.bone_labels.index("upperarm01.L")
bone_id2 = anny_model.bone_labels.index("upperarm02.L")
pose_parameters[:,bone_id1] = roma.Rigid(roma.euler_to_rotmat("z", [-90.], degrees=True), translation=None).to_homogeneous()[None]
pose_parameters[:,bone_id2] = roma.Rigid(roma.euler_to_rotmat("z", [-90.], degrees=True), translation=None).to_homogeneous()[None]
output = anny_model(pose_parameters=pose_parameters, pose_parameterization="world-orient")
mesh = trimesh.Trimesh(vertices=output['vertices'].squeeze(), faces=anny_model.faces, process=False)
mesh.visual = trimesh.visual.TextureVisuals(material=mesh_material)
scene = trimesh.Scene()
scene.add_geometry(mesh)
scene.add_geometry(world_axis)
scene.add_geometry(axis, transform=output["bone_poses"].squeeze(0)[bone_id1].cpu().numpy())
scene.add_geometry(axis, transform=output["bone_poses"].squeeze(0)[bone_id2].cpu().numpy())
scene.apply_transform(trimesh_scene_transform)
scene.show()
