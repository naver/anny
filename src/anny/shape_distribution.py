# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from anny.paths import get_anny_root_dir
import torch
from pathlib import Path
import anny.utils.interpolation


def _none_or_to_tensor(value, dtype=torch.float64):
    """
    Convert a value to a tensor if it is not None, otherwise return None.
    """
    return None if value is None else torch.as_tensor(value, dtype=dtype)


class MorphologicalAgeMapping(torch.nn.Module):
    def __init__(
        self, anny_age_anchors=None, morphological_age_anchors=None, dtype=torch.float64
    ):
        super().__init__()
        self.register_buffer(
            "anny_age_anchors", _none_or_to_tensor(anny_age_anchors, dtype=dtype)
        )
        self.register_buffer(
            "morphological_age_anchors",
            _none_or_to_tensor(morphological_age_anchors, dtype=dtype),
        )

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.anny_age_anchors = torch.empty_like(state_dict["anny_age_anchors"])
        self.morphological_age_anchors = torch.empty_like(
            state_dict["morphological_age_anchors"]
        )
        return super().load_state_dict(state_dict, *args, **kwargs)

    def morphological_to_anny_age(self, morphological_age):
        coeffs = anny.utils.interpolation.linear_interpolation_coefficients(
            morphological_age, self.morphological_age_anchors, extrapolate=True
        )
        anny_age = torch.einsum("bk, k -> b", coeffs, self.anny_age_anchors)
        return anny_age

    def anny_to_morphological_age(self, anny_age):
        coeffs = anny.utils.interpolation.linear_interpolation_coefficients(
            anny_age, self.anny_age_anchors, extrapolate=True
        )
        morphological_age = torch.einsum(
            "bk, k -> b", coeffs, self.morphological_age_anchors
        )
        return morphological_age


class ConditionalBetaDistribution(torch.nn.Module):
    """
    Generic conditional Beta distribution conditioned on age.
    """

    def __init__(
        self,
        age_anchors=None,
        alpha_anchors=None,
        beta_anchors=None,
        dtype=torch.float64,
    ):
        super().__init__()
        self.register_buffer(
            "age_anchors", _none_or_to_tensor(age_anchors, dtype=dtype)
        )
        self.register_buffer(
            "alpha_anchors", _none_or_to_tensor(alpha_anchors, dtype=dtype)
        )
        self.register_buffer(
            "beta_anchors", _none_or_to_tensor(beta_anchors, dtype=dtype)
        )

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.age_anchors = torch.empty_like(state_dict["age_anchors"])
        self.alpha_anchors = torch.empty_like(state_dict["alpha_anchors"])
        self.beta_anchors = torch.empty_like(state_dict["beta_anchors"])
        return super().load_state_dict(state_dict, *args, **kwargs)

    def get_distribution_params(self, age):
        coefs = anny.utils.interpolation.linear_interpolation_coefficients(
            age, anchors=self.age_anchors, extrapolate=False
        )
        alpha = torch.einsum("bk, k -> b", coefs, self.alpha_anchors)
        beta = anny.utils.interpolation.linear_interpolation_coefficients(
            age, anchors=self.age_anchors, extrapolate=False
        )
        beta = torch.einsum("bk, k -> b", coefs, self.beta_anchors)
        return alpha, beta

    def get_torch_distribution(self, age):
        """
        Get a Beta distribution parameterized by the given age.

        :param age: The age for which to get the distribution parameters.
        :return: A torch.distributions.Beta distribution.
        """
        alpha, beta = self.get_distribution_params(age)
        return torch.distributions.Beta(alpha, beta)


class ConditionalHeightDistribution(ConditionalBetaDistribution):
    """Conditional Beta distribution for Height (given age)."""

    pass  # inherits everything


class ConditionalWeightDistribution(ConditionalBetaDistribution):
    """Conditional Beta distribution for Weight (given age)."""

    pass  # inherits everything


class ConditionalMuscleDistribution(ConditionalBetaDistribution):
    """Conditional Beta distribution for Muscle (given age)."""

    pass  # inherits everything


class ConditionalProportionsDistribution(ConditionalBetaDistribution):
    """Conditional Beta distribution for Proportions (given age)."""

    pass  # inherits everything


class SimpleShapeDistribution(torch.nn.Module):
    """
    A simple phenotype distribution based on a handcrafted 'morphological age' mapping,
    calibrated to match the height vs age distribution of WHO data.
    """

    def __init__(
        self, model, morphological_age_distribution=None, gender_distribution=None
    ):
        super().__init__()
        if morphological_age_distribution is None:
            morphological_age_distribution = torch.distributions.Uniform(
                low=torch.tensor(0.0, dtype=model.dtype, device=model.device),
                high=torch.tensor(90.0, dtype=model.dtype, device=model.device),
            )
        self.morphological_age_distribution = morphological_age_distribution
        if gender_distribution is None:
            gender_distribution = torch.distributions.Uniform(
                low=torch.tensor(0.0, dtype=model.dtype, device=model.device),
                high=torch.tensor(1.0, dtype=model.dtype, device=model.device),
            )
        self.gender_distribution = gender_distribution

        # --- Boys distribution ---
        boys_state_dict = torch.load(
            Path(get_anny_root_dir() / "data" / "shape_calibration/boys.pth"),
            weights_only=True,
            map_location="cpu",
        )
        self.boys_conditional_height_distribution = ConditionalHeightDistribution()
        self.boys_conditional_height_distribution.load_state_dict(
            boys_state_dict["conditional_height_distribution"]
        )
        self.boys_conditional_height_distribution.to(
            device=model.device, dtype=model.dtype
        )

        self.boys_conditional_weight_distribution = ConditionalWeightDistribution()
        self.boys_conditional_weight_distribution.load_state_dict(
            boys_state_dict["conditional_weight_distribution"]
        )
        self.boys_conditional_weight_distribution.to(
            device=model.device, dtype=model.dtype
        )

        self.boys_conditional_muscle_distribution = ConditionalMuscleDistribution()
        self.boys_conditional_muscle_distribution.load_state_dict(
            boys_state_dict["conditional_muscle_distribution"]
        )
        self.boys_conditional_muscle_distribution.to(
            device=model.device, dtype=model.dtype
        )

        self.boys_conditional_proportions_distribution = (
            ConditionalProportionsDistribution()
        )
        self.boys_conditional_proportions_distribution.load_state_dict(
            boys_state_dict["conditional_proportions_distribution"]
        )
        self.boys_conditional_proportions_distribution.to(
            device=model.device, dtype=model.dtype
        )

        # --- Girls distributions ---
        girls_state_dict = torch.load(
            Path(get_anny_root_dir() / "data" / "shape_calibration/girls.pth"),
            weights_only=True,
            map_location="cpu",
        )

        self.girls_conditional_height_distribution = ConditionalHeightDistribution()
        self.girls_conditional_height_distribution.load_state_dict(
            girls_state_dict["conditional_height_distribution"]
        )
        self.girls_conditional_height_distribution.to(
            device=model.device, dtype=model.dtype
        )

        self.girls_conditional_weight_distribution = ConditionalWeightDistribution()
        self.girls_conditional_weight_distribution.load_state_dict(
            girls_state_dict["conditional_weight_distribution"]
        )
        self.girls_conditional_weight_distribution.to(
            device=model.device, dtype=model.dtype
        )

        self.girls_conditional_muscle_distribution = ConditionalMuscleDistribution()
        self.girls_conditional_muscle_distribution.load_state_dict(
            girls_state_dict["conditional_muscle_distribution"]
        )
        self.girls_conditional_muscle_distribution.to(
            device=model.device, dtype=model.dtype
        )

        self.girls_conditional_proportions_distribution = (
            ConditionalProportionsDistribution()
        )
        self.girls_conditional_proportions_distribution.load_state_dict(
            girls_state_dict["conditional_proportions_distribution"]
        )
        self.girls_conditional_proportions_distribution.to(
            device=model.device, dtype=model.dtype
        )

        for key in boys_state_dict["morphological_age_mapping"].keys():
            assert torch.all(
                boys_state_dict["morphological_age_mapping"][key]
                == girls_state_dict["morphological_age_mapping"][key]
            ), f"Mismatch in morphological age mapping: {key}"

        self.morphological_age_mapping = MorphologicalAgeMapping()
        self.morphological_age_mapping.load_state_dict(
            boys_state_dict["morphological_age_mapping"]
        )
        self.morphological_age_mapping.to(device=model.device, dtype=model.dtype)

        self.phenotype_labels = model.phenotype_labels

    def sample(self, batch_size):
        """
        Sample a batch of phenotype parameters for the input body model.

        :param batch_size: The number of samples to generate.
        :return: A dictionary containing sampled phenotype parameters.
        """
        morphological_age = self.morphological_age_distribution.sample((batch_size,))
        age = self.morphological_age_mapping.morphological_to_anny_age(
            morphological_age
        )

        gender = self.gender_distribution.sample((batch_size,))

        boys_height = self.boys_conditional_height_distribution.get_torch_distribution(
            age
        ).sample()
        girls_height = (
            self.girls_conditional_height_distribution.get_torch_distribution(
                age
            ).sample()
        )
        height = torch.where(gender <= 0.5, boys_height, girls_height)

        boys_weight = self.boys_conditional_weight_distribution.get_torch_distribution(
            age
        ).sample()
        girls_weight = (
            self.girls_conditional_weight_distribution.get_torch_distribution(
                age
            ).sample()
        )
        weight = torch.where(gender <= 0.5, boys_weight, girls_weight)

        boys_muscle = self.boys_conditional_muscle_distribution.get_torch_distribution(
            age
        ).sample()
        girls_muscle = (
            self.girls_conditional_muscle_distribution.get_torch_distribution(
                age
            ).sample()
        )
        muscle = torch.where(gender <= 0.5, boys_muscle, girls_muscle)

        boys_proportions = (
            self.boys_conditional_proportions_distribution.get_torch_distribution(
                age
            ).sample()
        )
        girls_proportions = (
            self.girls_conditional_proportions_distribution.get_torch_distribution(
                age
            ).sample()
        )
        proportions = torch.where(gender <= 0.5, boys_proportions, girls_proportions)

        phenotype_kwargs = {
            key: torch.rand(batch_size, dtype=height.dtype, device=height.device)
            for key in self.phenotype_labels
            if key not in ["height", "age", "gender"]
        }
        phenotype_kwargs["height"] = height
        phenotype_kwargs["weight"] = weight
        phenotype_kwargs["muscle"] = muscle
        phenotype_kwargs["proportions"] = proportions
        phenotype_kwargs["age"] = age
        phenotype_kwargs["gender"] = gender
        return morphological_age, phenotype_kwargs
