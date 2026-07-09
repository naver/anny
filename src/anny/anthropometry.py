# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch

BASE_MESH_WAIST_VERTICES = [4121, 10763, 10760, 10757, 10777, 10776, 10779, 10780, 10778, 10781, 10771, 10773, 10772, 10775, 10774, 10814, 10834, 10816, 10817, 10818, 10819, 10820, 10821, 4181, 4180, 4179, 4178, 4177, 4176, 4175, 4196, 4173, 4131, 4132, 4129, 4130, 4128, 4138, 4135, 4137, 4136, 4133, 4134, 4108, 4113, 4118]

class Anthropometry:
    def __init__(self, model):
        base_mesh_vertex_indices = model.base_mesh_vertex_indices.detach().cpu().numpy().tolist()
        self.model = model
        self.faces = model.faces
        if self.faces.shape[1] != 3:
            raise ValueError("Faces must be triangles (3 vertices per face). Use a triangulated Anny model.")
        try:
            self.waist_vertex_indices = [base_mesh_vertex_indices.index(i) for i in BASE_MESH_WAIST_VERTICES]
        except ValueError:
            raise ValueError("Base mesh vertex indices do not contain all waist vertices.")
        
    def height(self, rest_vertices):
        return torch.max(rest_vertices[...,2], dim=1)[0] - torch.min(rest_vertices[...,2], dim=1)[0]
    
    def waist_circumference(self, rest_vertices):
        waist_vertices  = rest_vertices[:,self.waist_vertex_indices]
        waist_vertices_rolled = torch.roll(waist_vertices, shifts=1, dims=1)
        waist_circumference = torch.sum(torch.linalg.norm(waist_vertices_rolled - waist_vertices, dim=-1), dim=-1)
        return waist_circumference

    def volume(self, rest_vertices):
        faces = self.faces

        v0 = rest_vertices[:,faces[:, 0]]  # (F,3)
        v1 = rest_vertices[:,faces[:, 1]]  # (F,3)
        v2 = rest_vertices[:,faces[:, 2]]  # (F,3)

        cross = torch.cross(v0, v1, dim=-1)          # (F,3)
        signed = (cross * v2).sum(dim=-1) / 6.0      # (F,)
        volume = signed.sum(dim=-1).abs()            # scalar
        return volume

    def mass(self, rest_vertices):
        volume = self.volume(rest_vertices)
        density = 980 # Assuming density of 980 kg/m^3 for simplicity
        mass = volume * density
        return mass

    def bmi(self, rest_vertices):
        """ Return Body Mass Index (BMI) """
        height = self.height(rest_vertices)
        mass = self.mass(rest_vertices)
        bmi = mass / (height ** 2)
        return bmi
    
    def __call__(self, rest_vertices):
        return dict(height=self.height(rest_vertices),
                    waist_circumference=self.waist_circumference(rest_vertices),
                    volume=self.volume(rest_vertices),
                    mass=self.mass(rest_vertices),
                    bmi=self.bmi(rest_vertices)
                    )