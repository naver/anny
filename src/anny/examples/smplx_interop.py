# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import jsonargparse
import torch
import anny
import trimesh

def retarget_mesh(output_dirname: str):
        dtype = torch.float32
        model = anny.create_fullbody_model(remove_unattached_vertices=False).to(dtype=dtype)

        output = model()
        
        anny_vertices = output["vertices"]
        mesh = trimesh.Trimesh(vertices=output["vertices"].squeeze(dim=0).cpu().numpy(), faces=model.faces.cpu().numpy())
        anny_output_filename = output_dirname + "/anny_mesh.ply"
        print(f"Exporting Anny mesh to {anny_output_filename}")
        mesh.export(anny_output_filename)

        # From Anny to SMPLX
        anny2smplx = anny.VertexRegressor(type="anny_to_smplx")
        smplx_vertices = anny2smplx(anny_vertices.repeat(8,1,1)).to(dtype=dtype) # checking batch_size
        smplx_vertices = anny2smplx(anny_vertices).to(dtype=dtype)

        # From SMPLX to Anny
        smplx2anny = anny.VertexRegressor(type="smplx_to_anny")
        smplx2anny_vertices = smplx2anny(smplx_vertices).to(dtype=dtype)
        mesh = trimesh.Trimesh(vertices=smplx2anny_vertices.squeeze(dim=0).cpu().numpy(), faces=model.faces.cpu().numpy())
        smplx2anny_output_filename = output_dirname + "/anny_mesh_back.ply"
        print(f"Exporting Anny (again after moved to SMPL-X) mesh to {smplx2anny_output_filename}")
        mesh.export(smplx2anny_output_filename)


if __name__ == "__main__":
      parser = jsonargparse.ArgumentParser(description="Export a mesh from the Anny model.")
      parser.add_argument("--output_dirname", type=str, help="Output filename for the exported mesh", default = ".")
      args = parser.parse_args()
      
      retarget_mesh(args.output_dirname)