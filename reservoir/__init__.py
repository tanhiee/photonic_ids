"""
reservoir/__init__.py — Reservoir layer public API.
"""
from reservoir.projection import build_input_mask
from reservoir.recurrent_matrix import build_recurrent_matrix
from reservoir.readout import RidgeReadout
from reservoir.mrr_mesh import PhotonicMesh
__all__ = ['build_input_mask', 'build_recurrent_matrix', 'RidgeReadout', 'PhotonicMesh']
