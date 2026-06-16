# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
from anny.models.rigged_model import RiggedModelWithBoneTails, RiggedModelWithBoneVertices
from typing import Union
import anny.utils.interpolation
import anny.utils.relu


class BufferDict(torch.nn.Module):
    def __init__(self, input_dict):
        super().__init__()
        for k,v in input_dict.items():
            self.register_buffer(k, v)

    def __getitem__(self, key):
        return getattr(self, key)


def to_batched_tensor(value, device, dtype):
    """
    Helper function to accept float inputs
    """
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.dim() == 0:
        return value.unsqueeze(dim=0)
    return value

PHENOTYPE_VARIATIONS = dict(
            race=["african", "asian", "caucasian"],
            gender=["male", "female"],
            age=["newborn", "baby", "child", "young", "old"],
            muscle=["minmuscle", "averagemuscle", "maxmuscle"],
            weight=["minweight", "averageweight", "maxweight"],
            height=["minheight", "maxheight"],
            proportions=["idealproportions", "uncommonproportions"],
            cupsize=["mincup", "averagecup", "maxcup"],
            firmness=["minfirmness", "averagefirmness", "maxfirmness"])

PHENOTYPE_LABELS = [key for key in PHENOTYPE_VARIATIONS.keys() if key != "race"] + PHENOTYPE_VARIATIONS["race"]
EXCLUDED_PHENOTYPES = ['cupsize', 'firmness'] + PHENOTYPE_VARIATIONS["race"]



class _AnnyMeta(type(torch.nn.Module)):
    """Metaclass making ``Anny(...)`` build a model via the factory.

    Subclass of ``type`` (compatible with the ``nn.Module`` metaclass) so it can be
    mixed into the rigged-model class hierarchy. It only customizes how the base class
    ``Anny`` is *called* — concrete subclasses are constructed normally.
    """

    def __call__(cls, *args, **kwargs):
        if cls is Anny:
            # ``anny.Anny(...)`` builds a full-body model through the factory.
            from anny.models import create_fullbody_model  # lazy import avoids a cycle
            return create_fullbody_model(*args, **kwargs)
        # Concrete subclasses (e.g. via ``from_model_data``) use normal new+init.
        return super().__call__(*args, **kwargs)


class Anny(metaclass=_AnnyMeta):
    """Base class of every Anny phenotype model, and the public constructor.

    Calling ``Anny(...)`` builds a full-body model (forwarding to
    :func:`anny.create_fullbody_model`), while ``isinstance(model, Anny)`` is ``True``
    for any Anny phenotype model (full-body, hand or head, tail- or procrustes-based).

    As a base it provides phenotype parameter handling, and must be combined with a
    :class:`RiggedModelWithLinearBlendShapes` subclass.
    """

    def _init_phenotype_parameters(self,
                                   stacked_phenotype_blend_shapes_mask,
                                   local_change_labels,
                                   base_mesh_vertex_indices,
                                   extrapolate_phenotypes,
                                   all_phenotypes):
        self.stacked_phenotype_blend_shapes_mask = torch.nn.Buffer(stacked_phenotype_blend_shapes_mask, persistent=False)
        self.local_change_labels = local_change_labels
        self.base_mesh_vertex_indices = base_mesh_vertex_indices
        self.extrapolate_phenotypes = extrapolate_phenotypes
        self.all_phenotypes = all_phenotypes

        self.phenotype_labels = PHENOTYPE_LABELS if self.all_phenotypes else [x for x in PHENOTYPE_LABELS if x not in EXCLUDED_PHENOTYPES]

        self.anchors = BufferDict(self._make_phenotype_anchors())

    @property
    def dtype(self):
        return self.stacked_phenotype_blend_shapes_mask.dtype

    @property
    def device(self):
        return self.stacked_phenotype_blend_shapes_mask.device


    def _make_phenotype_anchors(self) -> dict:
        anchors = {'age': torch.linspace(-1/3, 1., len(PHENOTYPE_VARIATIONS['age']), dtype=self.dtype, device=self.device)}
        for label in ['gender', 'muscle', 'weight', 'height', 'proportions', 'cupsize', 'firmness']:
            anchors[label] = torch.linspace(0., 1., len(PHENOTYPE_VARIATIONS[label]), dtype=self.dtype, device=self.device)
        return anchors


    def parse_phenotype_kwargs(self, phenotype_kwargs):
        if type(phenotype_kwargs) is torch.Tensor:
            assert phenotype_kwargs.shape[1] == len(self.phenotype_labels), f"phenotype_kwargs tensor must have shape [bs, {len(self.phenotype_labels)}], got {phenotype_kwargs.shape}"
            phenotype_kwargs = {key: phenotype_kwargs[:,i] for i, key in enumerate(self.phenotype_labels)}
        return phenotype_kwargs

    def get_phenotype_blendshape_coefficients(self,
        gender: Union[float, torch.Tensor] = 0.5,
        age: Union[float, torch.Tensor] = 0.5,
        muscle: Union[float, torch.Tensor] = 0.5,
        weight: Union[float, torch.Tensor] = 0.5,
        height: Union[float, torch.Tensor] = 0.5,
        proportions: Union[float, torch.Tensor] = 0.5,
        cupsize: Union[float, torch.Tensor] = 0.5,
        firmness: Union[float, torch.Tensor] = 0.5,
        african: Union[float, torch.Tensor] = 0.5,
        asian: Union[float, torch.Tensor] = 0.5,
        caucasian: Union[float, torch.Tensor] = 0.5,
        local_changes: dict = dict()):
        """Return blendshape coefficients corresponding to the input phenotype description."""
        dtype = self.dtype
        device = self.device
        anchors = self.anchors
        batch_size = 1
    
        weight_dicts = {}
        for feature, value in zip(
            ['age', 'gender', 'muscle', 'weight', 'height', 'proportions', 'cupsize', 'firmness'],
            [age, gender, muscle, weight, height, proportions, cupsize, firmness]):
            interpolation_coeffs = anny.utils.interpolation.linear_interpolation_coefficients(
                to_batched_tensor(value, device, dtype), anchors[feature], extrapolate=self.extrapolate_phenotypes)
            weight_dicts[feature] = {key: interpolation_coeffs[:, i] for i, key in enumerate(PHENOTYPE_VARIATIONS[feature])}
            batch_size = max(batch_size, interpolation_coeffs.shape[0])
    
        race_values = torch.stack([to_batched_tensor(v, device, dtype) for v in (african, asian, caucasian)], dim=1)
        race_weights = torch.nan_to_num(race_values / torch.sum(race_values, dim=1, keepdim=True), 1/3, 1/3, 1/3)
    
        dict_phens = {
            **weight_dicts['age'], **weight_dicts['gender'], **weight_dicts['muscle'],
            **weight_dicts['weight'], **weight_dicts['height'], **weight_dicts['proportions'],
            **weight_dicts['cupsize'], **weight_dicts['firmness'],
            'african': race_weights[:, 0], 'asian': race_weights[:, 1], 'caucasian': race_weights[:, 2],
        }
        phens = torch.stack(
            [dict_phens[key].expand(batch_size) for key_list in PHENOTYPE_VARIATIONS.values() for key in key_list],
            dim=1,
        )  # (batch_size, n_phen_components)
    
        masked_phens = phens.unsqueeze(1) * self.stacked_phenotype_blend_shapes_mask.unsqueeze(0)
        wi = torch.prod(masked_phens + (1 - self.stacked_phenotype_blend_shapes_mask.unsqueeze(0)), dim=-1)
        batch_size = len(wi)
    
        if len(self.local_change_labels) > 0:
            local_weights = torch.zeros((batch_size, 2 * len(self.local_change_labels)), device=device, dtype=dtype)
            for i, key in enumerate(self.local_change_labels):
                try:
                    value = to_batched_tensor(local_changes[key], device, dtype)
                    local_weights[:, 2*i] = anny.utils.relu.relu_with_gradient_at_zero(value)
                    local_weights[:, 2*i+1] = anny.utils.relu.relu_with_gradient_at_zero(-value)
                except KeyError:
                    pass
            wi = torch.cat([wi, local_weights], dim=1)
        return wi

    def forward(self,
                pose_parameters=None,
                phenotype_kwargs=dict(),
                local_changes_kwargs=dict(),
                pose_parameterization=None,
                return_bone_ends=False):
        phenotype_kwargs = self.parse_phenotype_kwargs(phenotype_kwargs)
        assert set(phenotype_kwargs) <= set(self.phenotype_labels), f"Invalid phenotype: {set(phenotype_kwargs) - set(self.phenotype_labels)}; available: {self.phenotype_labels}"
        blendshape_coeffs = self.get_phenotype_blendshape_coefficients(**phenotype_kwargs, local_changes=local_changes_kwargs)
            
        return super().forward(pose_parameters, blendshape_coeffs, pose_parameterization=pose_parameterization, return_bone_ends=return_bone_ends)


class RiggedModelWithPhenotypeParameters(Anny, RiggedModelWithBoneTails):
    """
    A class to deal with a rigged human model with phenotype parameters, using tail-based bone orientation.
    """
    def __init__(self,
                 template_vertices,
                 faces,
                 texture_coordinates,
                 face_texture_coordinate_indices,
                 blendshapes,
                 template_bone_heads,
                 bone_heads_blendshapes,
                 template_bone_tails,
                 bone_tails_blendshapes,
                 bone_rolls_rotmat,
                 bone_parents,
                 bone_labels,
                 vertex_bone_weights,
                 vertex_bone_indices,
                 skinning_method,
                 pose_parameterization,
                 stacked_phenotype_blend_shapes_mask,
                 local_change_labels,
                 base_mesh_vertex_indices,
                 extrapolate_phenotypes=False,
                 all_phenotypes=False,
                 bone_orientation="blender-rootidentity"):
        """
        Initialize the RiggedModelWithPhenotypeParameters class.
        Args:
            template_vertices (torch.Tensor): The vertices of the template mesh.
            faces (torch.Tensor): The faces of the mesh.
            blendshapes (torch.Tensor): The blendshapes of the mesh.
            base_mesh_vertex_indices (torch.Tensor): Indices of vertices in the base mesh.
        """
        super().__init__(
            template_vertices=template_vertices,
            faces=faces,
            texture_coordinates=texture_coordinates,
            face_texture_coordinate_indices=face_texture_coordinate_indices,
            blendshapes=blendshapes,
            template_bone_heads=template_bone_heads,
            bone_heads_blendshapes=bone_heads_blendshapes,
            template_bone_tails=template_bone_tails,
            bone_tails_blendshapes=bone_tails_blendshapes,
            bone_rolls_rotmat=bone_rolls_rotmat,
            bone_parents=bone_parents,
            bone_labels=bone_labels,
            vertex_bone_weights=vertex_bone_weights,
            vertex_bone_indices=vertex_bone_indices,
            skinning_method=skinning_method,
            pose_parameterization=pose_parameterization,
            bone_orientation=bone_orientation)
        self._init_phenotype_parameters(
            stacked_phenotype_blend_shapes_mask=stacked_phenotype_blend_shapes_mask,
            local_change_labels=local_change_labels,
            base_mesh_vertex_indices=base_mesh_vertex_indices,
            extrapolate_phenotypes=extrapolate_phenotypes,
            all_phenotypes=all_phenotypes)

    def to_model_data(self):
        from anny.models.model_data import ModelData, ModelMetadata
        bone_orientation = "blender-rootidentity" if self.root_identity_orientation else self.bone_orientation
        return ModelData(
            metadata=ModelMetadata(
                model_type="tail",
                bone_parents=self.bone_parents,
                bone_labels=self.bone_labels,
                local_change_labels=self.local_change_labels,
                pose_parameterization=self.pose_parameterization,
                skinning_method=self._skinning_method_name,
                all_phenotypes=self.all_phenotypes,
                extrapolate_phenotypes=self.extrapolate_phenotypes,
                bone_orientation=bone_orientation,
            ),
            template_vertices=self.template_vertices,
            faces=self.faces,
            texture_coordinates=self.texture_coordinates,
            face_texture_coordinate_indices=self.face_texture_coordinate_indices,
            blendshapes=self.blendshapes,
            stacked_phenotype_blend_shapes_mask=self.stacked_phenotype_blend_shapes_mask,
            template_bone_heads=self.template_bone_heads,
            bone_heads_blendshapes=self.bone_heads_blendshapes,
            vertex_bone_weights=self.vertex_bone_weights,
            vertex_bone_indices=self.vertex_bone_indices,
            base_mesh_vertex_indices=self.base_mesh_vertex_indices,
            template_bone_tails=self.template_bone_tails,
            bone_tails_blendshapes=self.bone_tails_blendshapes,
            bone_rolls_rotmat=self.bone_rolls_rotmat,
        )

    @classmethod
    def from_model_data(cls, data):
        assert data.metadata.model_type == "tail", (
            f"Expected model_type='tail', got {data.metadata.model_type!r}")
        assert data.template_bone_tails is not None
        assert data.bone_tails_blendshapes is not None
        assert data.bone_rolls_rotmat is not None
        return cls(
            template_vertices=data.template_vertices,
            faces=data.faces,
            texture_coordinates=data.texture_coordinates,
            face_texture_coordinate_indices=data.face_texture_coordinate_indices,
            blendshapes=data.blendshapes,
            template_bone_heads=data.template_bone_heads,
            template_bone_tails=data.template_bone_tails,
            bone_heads_blendshapes=data.bone_heads_blendshapes,
            bone_tails_blendshapes=data.bone_tails_blendshapes,
            bone_rolls_rotmat=data.bone_rolls_rotmat,
            bone_parents=data.metadata.bone_parents,
            bone_labels=data.metadata.bone_labels,
            vertex_bone_weights=data.vertex_bone_weights,
            vertex_bone_indices=data.vertex_bone_indices,
            skinning_method=data.metadata.skinning_method,
            pose_parameterization=data.metadata.pose_parameterization,
            stacked_phenotype_blend_shapes_mask=data.stacked_phenotype_blend_shapes_mask,
            local_change_labels=data.metadata.local_change_labels,
            base_mesh_vertex_indices=data.base_mesh_vertex_indices,
            extrapolate_phenotypes=data.metadata.extrapolate_phenotypes,
            all_phenotypes=data.metadata.all_phenotypes,
            bone_orientation=data.metadata.bone_orientation,
        )


class RiggedModelWithProcrustesAndPhenotypeParameters(Anny, RiggedModelWithBoneVertices):
    """
    A class to deal with a rigged human model with phenotype parameters, using procrustes-based bone orientation.
    """
    def __init__(self,
                 template_vertices,
                 faces,
                 texture_coordinates,
                 face_texture_coordinate_indices,
                 blendshapes,
                 template_bone_heads,
                 bone_heads_blendshapes,
                 bone_parents,
                 bone_labels,
                 vertex_bone_weights,
                 vertex_bone_indices,
                 bone_nonzeroweight_mask,
                 bone_vertex_indices,
                 bone_vertex_weights,
                 template_bone_vertices,
                 skinning_method,
                 pose_parameterization,
                 reference_bone_orientations,
                 stacked_phenotype_blend_shapes_mask,
                 local_change_labels,
                 base_mesh_vertex_indices,
                 extrapolate_phenotypes=False,
                 all_phenotypes=False,
                 ):
        super().__init__(
            template_vertices=template_vertices,
            faces=faces,
            texture_coordinates=texture_coordinates,
            face_texture_coordinate_indices=face_texture_coordinate_indices,
            blendshapes=blendshapes,
            template_bone_heads=template_bone_heads,
            bone_heads_blendshapes=bone_heads_blendshapes,
            bone_parents=bone_parents,
            bone_labels=bone_labels,
            vertex_bone_weights=vertex_bone_weights,
            vertex_bone_indices=vertex_bone_indices,
            bone_nonzeroweight_mask=bone_nonzeroweight_mask,
            bone_vertex_indices=bone_vertex_indices,
            bone_vertex_weights=bone_vertex_weights,
            template_bone_vertices=template_bone_vertices,
            skinning_method=skinning_method,
            pose_parameterization=pose_parameterization,
            reference_bone_orientations=reference_bone_orientations)
        self._init_phenotype_parameters(
            stacked_phenotype_blend_shapes_mask=stacked_phenotype_blend_shapes_mask,
            local_change_labels=local_change_labels,
            base_mesh_vertex_indices=base_mesh_vertex_indices,
            extrapolate_phenotypes=extrapolate_phenotypes,
            all_phenotypes=all_phenotypes)

    def to_model_data(self):
        from anny.models.model_data import ModelData, ModelMetadata
        return ModelData(
            metadata=ModelMetadata(
                model_type="procrustes",
                bone_parents=self.bone_parents,
                bone_labels=self.bone_labels,
                local_change_labels=self.local_change_labels,
                pose_parameterization=self.pose_parameterization,
                skinning_method=self._skinning_method_name,
                all_phenotypes=self.all_phenotypes,
                extrapolate_phenotypes=self.extrapolate_phenotypes,
                bone_orientation=None,
            ),
            template_vertices=self.template_vertices,
            faces=self.faces,
            texture_coordinates=self.texture_coordinates,
            face_texture_coordinate_indices=self.face_texture_coordinate_indices,
            blendshapes=self.blendshapes,
            stacked_phenotype_blend_shapes_mask=self.stacked_phenotype_blend_shapes_mask,
            template_bone_heads=self.template_bone_heads,
            bone_heads_blendshapes=self.bone_heads_blendshapes,
            vertex_bone_weights=self.vertex_bone_weights,
            vertex_bone_indices=self.vertex_bone_indices,
            base_mesh_vertex_indices=self.base_mesh_vertex_indices,
            bone_nonzeroweight_mask=self.bone_nonzeroweight_mask,
            bone_vertex_indices=self.bone_vertex_indices,
            bone_vertex_weights=self.bone_vertex_weights,
            template_bone_vertices=self.template_bone_vertices,
            reference_bone_orientations=self.reference_bone_orientations,
        )

    @classmethod
    def from_model_data(cls, data):
        assert data.metadata.model_type == "procrustes", (
            f"Expected model_type='procrustes', got {data.metadata.model_type!r}")
        assert data.bone_nonzeroweight_mask is not None
        assert data.bone_vertex_indices is not None
        assert data.bone_vertex_weights is not None
        assert data.template_bone_vertices is not None
        return cls(
            template_vertices=data.template_vertices,
            faces=data.faces,
            texture_coordinates=data.texture_coordinates,
            face_texture_coordinate_indices=data.face_texture_coordinate_indices,
            blendshapes=data.blendshapes,
            template_bone_heads=data.template_bone_heads,
            bone_heads_blendshapes=data.bone_heads_blendshapes,
            bone_parents=data.metadata.bone_parents,
            bone_labels=data.metadata.bone_labels,
            vertex_bone_weights=data.vertex_bone_weights,
            vertex_bone_indices=data.vertex_bone_indices,
            bone_nonzeroweight_mask=data.bone_nonzeroweight_mask,
            bone_vertex_indices=data.bone_vertex_indices,
            bone_vertex_weights=data.bone_vertex_weights,
            template_bone_vertices=data.template_bone_vertices,
            skinning_method=data.metadata.skinning_method,
            pose_parameterization=data.metadata.pose_parameterization,
            reference_bone_orientations=data.reference_bone_orientations,
            stacked_phenotype_blend_shapes_mask=data.stacked_phenotype_blend_shapes_mask,
            local_change_labels=data.metadata.local_change_labels,
            base_mesh_vertex_indices=data.base_mesh_vertex_indices,
            extrapolate_phenotypes=data.metadata.extrapolate_phenotypes,
            all_phenotypes=data.metadata.all_phenotypes,
        )
