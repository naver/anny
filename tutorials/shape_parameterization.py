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
# # Parameterizing shapes with Anny
#

# %% [markdown]
# ### Instantiate the model
# Anny is shipped as a Python package that can be easily installed (see the README for details).

# %%
from IPython.display import Markdown, display
import torch
import roma # A PyTorch library useful to deal with space transformations.
import anny # The main library for the Anny model.
import trimesh # For 3D mesh visualization.

# Instantiate the model, with all shape parameters available.
# Remark: the first instantiation may take a while. Latter calls will be faster thanks to caching.
anny_model = anny.Anny(all_phenotypes=True, local_changes="default", remove_unattached_vertices=True)
# Use 32bit floating point precision on the CPU for this demo.
dtype = torch.float32
device = torch.device('cpu')
anny_model = anny_model.to(device=device, dtype=dtype)

# A simple transform to get a better view angle in 3D mesh visualizations.
trimesh_scene_transform = roma.Rigid(linear=roma.euler_to_rotmat('x', [-90.], degrees=True), translation=None).to_homogeneous().cpu().numpy()

# %% [markdown]
# ### Template mesh
# This is the template mesh of Anny:

# %%
display(Markdown(f"{anny_model.template_vertices.shape[0]} vertices -- {anny_model.faces.shape[0]} faces composed of {anny_model.faces.shape[1]} vertices each."))
trimesh.Trimesh(vertices=anny_model.template_vertices.cpu().numpy(), faces=anny_model.faces.cpu().numpy()).apply_transform(trimesh_scene_transform).show()

# %% [markdown]
# ## Shape parameterization
#
# Anny can model a diversity of morphologies.
# We follow MakeHuman terminology, and parameterize diversity of body shapes using a set of *phenotype* parameters, typically between 0 and 1.
#
# *Note:* the values *african*, *caucasian* and *asian* parameters are normalized so that they sum to 1.
#
# **Word of caution regarding phenotypes:**
# *Phenotypes are based on preconceptions of artists regarding particular human traits. As a result, they encode by design stereotypes of MakeHuman artists, and one should not expect phenotype parameters to faithfully encode identity-related characteristics, such as gender, age or ethnicity.*

# %%
# List phenotype parameters
Markdown("**List of phenotype parameters**: " + ", ".join([f"{label}" for label in anny_model.phenotype_labels]))

# %% [markdown]
# ### Example
#
# Here we show an example of how the **age** parameter influences the resulting mesh.

# %%
batch_size = 5  # We can process multiple bodies at once in a batch. 

phenotype_kwargs = {key : torch.full((batch_size,), fill_value=0.5, dtype=dtype, device=device) for key in anny_model.phenotype_labels}
phenotype_kwargs['age'] = torch.linspace(0., 1., batch_size, dtype=dtype, device=device) # Example: vary the age parameter across the batch.
output = anny_model(phenotype_kwargs=phenotype_kwargs)

scene = trimesh.Scene()
for i in range(batch_size):
    # Create a mesh for each body in the batch.
    mesh = trimesh.Trimesh(vertices=output['vertices'][i].squeeze().cpu().numpy(), faces=anny_model.faces.cpu().numpy())
    transform = roma.Rigid(linear=None, translation=torch.tensor([i * 1., 0., 0.], dtype=dtype, device=device)).to_homogeneous().cpu().numpy()
    scene.add_geometry(mesh, transform=transform)

scene.apply_transform(trimesh_scene_transform)  # Rotate the scene to have a better view.
scene.show()  # This will open a window to visualize the scene with all the bodies in it.

# %% [markdown]
# ## Local changes
# Additionnally one specify some more local morphological changes.
# Local change parameters values are typically expected to be chosen between -1 and 1, but one can use values outside this range to extrapolate changes even further.
#
# *Note: it is easy to produce unrealistic meshes when using significant local changes.*

# %%
display(Markdown("**List of local changes parameters:** " + ", ".join(anny_model.local_change_labels)))

# %% [markdown]
# #### Local change example
#
# We show here the effect of the *stomach-pregnant-incr* local change parameter, as an example.

# %%
batch_size = 3  # We can process multiple faces at once in a batch.
pose_parameters = roma.Rigid.identity(dim=3, batch_shape=(batch_size, anny_model.bone_count)).to_homogeneous()
phenotype_kwargs = {key : torch.full((batch_size,), fill_value=0.5) for key in anny_model.phenotype_labels}
# In this example, we start from a stereotypical young adult woman mesh
phenotype_kwargs['age'].fill_(0.67) 
phenotype_kwargs['gender'].fill_(1.)

local_changes = {'stomach-pregnant-incr': torch.linspace(0, 1., batch_size)}  # Example: vary the upperarm fat increment across the batch.
output = anny_model(phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes)

scene = trimesh.Scene()
for i in range(batch_size):
    # Create a mesh for each face in the batch.
    mesh = trimesh.Trimesh(vertices=output['vertices'][i].squeeze().cpu().numpy(), faces=anny_model.faces.cpu().numpy())
    transform = roma.Rigid(linear=None, translation=torch.tensor([i * 1., 0., 0.])).to_homogeneous().cpu().numpy()
    scene.add_geometry(mesh, transform=transform)
scene.apply_transform(trimesh_scene_transform)  # Rotate the scene to have a better view.
scene.show()  # This will open a window to visualize the scene with all the faces in

# %% [markdown]
# ## Phenotype distribution

# %%
import anny.shape_distribution
import anny.anthropometry
import matplotlib.pyplot as plt

phenotype_distribution = anny.shape_distribution.SimpleShapeDistribution(anny_model,
            morphological_age_distribution=torch.distributions.Uniform(low=0.0, high=60.0))

real_age, phenotype_kwargs = phenotype_distribution.sample(batch_size=200)
output = anny_model(phenotype_kwargs=phenotype_kwargs)

scene = trimesh.Scene()
i = -1
for u in range(4):
    for v in range(5):
        i += 1
        assert i < output['vertices'].shape[0], "Batch size is too small for the grid."
        # Create a mesh for each face in the batch.
        mesh = trimesh.Trimesh(vertices=output['vertices'][i].squeeze().cpu().numpy(), faces=anny_model.faces.cpu().numpy())
        transform = roma.Rigid(linear=None, translation=torch.tensor([v * 1., u * 1., 0.], dtype=dtype, device=device)).to_homogeneous().cpu().numpy()
        scene.add_geometry(mesh, transform=transform)
scene.apply_transform(trimesh_scene_transform)  # Rotate the scene to have a better view.
scene.show()  # This will open a window to visualize the scene with all the faces in

# %% [markdown]
# ### Body measures
#
# We additionnally provide a class to estimate some anthropometric measurements, assuming a body buoyancy of .98 in water. 

# %%
real_age, phenotype_kwargs = phenotype_distribution.sample(batch_size=1000)
output = anny_model(phenotype_kwargs=phenotype_kwargs)

measurements = anny.anthropometry.Anthropometry(anny_model)
measures = measurements(output['rest_vertices'])

fig, axes = plt.subplots(1,3, squeeze=True, figsize=(10, 5))
axes[0].scatter(real_age.cpu().numpy(), measures['height'].cpu().numpy())
axes[0].set_xlabel('Morphological age (year)')
axes[0].set_ylabel('Height (m)')
axes[1].scatter(real_age.cpu().numpy(), measures['waist_circumference'].cpu().numpy())
axes[1].set_xlabel('Morphological age (year)')
axes[1].set_ylabel('Waist circumference (m)')
axes[2].scatter(real_age.cpu().numpy(), measures['bmi'].cpu().numpy())
axes[2].set_xlabel('Morphological age (year)')
axes[2].set_ylabel('Body Mass Index estimate')
fig.tight_layout()
plt.show()
