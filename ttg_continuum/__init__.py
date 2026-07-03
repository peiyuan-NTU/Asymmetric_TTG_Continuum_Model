"""Exactly-commensurate continuum model for generator-built asymmetric TTG.

Public API:
    RigidTTG        the model (geometry + certificate + BM + V_layer/beta/lambda)
    theta_com, f_com, best_mn   commensurate-angle helpers
    A0, HBAR_VF     generator lattice constant (2.47 A), velocity scale (eV*A)
"""
from .rigid_ttg import (RigidTTG, theta_com, f_com, best_mn, A0, HBAR_VF,
                        hex_lattice, recip, rotmat, K_of, int_matrix)

__all__ = ["RigidTTG", "theta_com", "f_com", "best_mn", "A0", "HBAR_VF",
           "hex_lattice", "recip", "rotmat", "K_of", "int_matrix"]
