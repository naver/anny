# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""Bake the default skinning-weight cleanup into the default weights file.

The default skinning weights used to be cleaned up at model-build time by two passes
(`symmetrize_skinning_weights` and `remove_skinning_islands`) controlled by the
`enforce_skinning_weights_symmetry` / `remove_skinning_islands` arguments. Those arguments
have been removed; instead this one-shot script applies the same passes offline and writes
the cleaned weights back to `weights.default.json`.

It reads the raw weights from `weights.legacy.json` (the original MakeHuman export) and
writes the improved weights to `weights.default.json`. `edit_mesh` only rewrites faces, not
vertex indices, so the cleaned per-vertex weights stay keyed to the same raw vertex IDs and
can be written straight back into the sparse JSON format.

Re-run this script whenever the legacy weights or the cleanup transforms change:

    uv run python scripts/compute_skinning_weights.py
"""
import json
import os

from anny.models import model_transforms
from anny.models.full_model import ANNY_ROOT_DIR, load_data

_STANDARD_DIR = os.path.join(ANNY_ROOT_DIR, "data/mpfb2/rigs/standard")
_RIG_FILENAME = os.path.join(_STANDARD_DIR, "rig.default.json")
_SRC_WEIGHTS = os.path.join(_STANDARD_DIR, "weights.legacy.json")
_OUT_WEIGHTS = os.path.join(_STANDARD_DIR, "weights.default.json")


def compute_cleaned_weights():
    """Return the legacy default weights after symmetry + island cleanup, keyed by raw vertex id."""
    # eyes=True, tongue=True mirrors the default full-body build so the mesh connectivity
    # used for island detection matches. remove_unattached_vertices is intentionally NOT
    # applied: it reindexes vertices, and we need the raw ids to write the JSON back.
    data = load_data(
        rig_filename=_RIG_FILENAME,
        weights_filename=_SRC_WEIGHTS,
        eyes=True,
        tongue=True,
    )
    data = model_transforms.edit_mesh(data)
    data = model_transforms.symmetrize_skinning_weights(data)
    data = model_transforms.remove_skinning_islands(data)
    return data


def export_weights(data, src_weights_filename, out_filename):
    """Write `data`'s sparse skinning weights to a MakeHuman-style weights JSON file."""
    with open(src_weights_filename) as f:
        src = json.load(f)

    bone_labels = data.metadata.bone_labels
    vertex_bone_indices = data.vertex_bone_indices
    vertex_bone_weights = data.vertex_bone_weights

    # Pre-seed every bone present in the source file so none silently disappears and
    # load_data emits no "joints without associated weights" warning.
    weights = {bone_name: [] for bone_name in src["weights"]}

    n_vertices, n_slots = vertex_bone_weights.shape
    for vertex_id in range(n_vertices):
        for slot in range(n_slots):
            weight = vertex_bone_weights[vertex_id, slot].item()
            if weight <= 0:
                continue
            bone_name = bone_labels[vertex_bone_indices[vertex_id, slot].item()]
            weights.setdefault(bone_name, []).append([vertex_id, weight])

    out = dict(src)
    # These weights are derived from the original MakeHuman export but cleaned up
    # (symmetrized + island-free), so reflect that in the header. license/version are
    # inherited from the source.
    out["name"] = "Anny default skinning weights"
    out["description"] = (
        "Default Anny skinning weights, symmetrized and island-free, baked from "
        "weights.legacy.json by scripts/compute_skinning_weights.py."
    )
    out["copyright"] = "Copyright (C) 2025 NAVER Corp."
    out["weights"] = weights
    with open(out_filename, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    data = compute_cleaned_weights()
    export_weights(data, _SRC_WEIGHTS, _OUT_WEIGHTS)
    print(f"Wrote cleaned default skinning weights to {_OUT_WEIGHTS}")
