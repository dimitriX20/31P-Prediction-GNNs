from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import mendeleev
from sklearn.preprocessing import OneHotEncoder
from rdkit import Chem
from rdkit.Chem import Descriptors


# -----------------------------
# Encoders (must match training)
# -----------------------------
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.OTHER,
]
_hyb_values = np.array([int(hyb) for hyb in HYBRIDIZATIONS]).reshape(-1, 1)

try:
    _hyb_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:  # sklearn<1.2
    _hyb_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
_hyb_encoder.fit(_hyb_values)

ALL_FORMAL_CHARGES = [-2, -1, 0, 1, 3]
_fc_values = np.array(ALL_FORMAL_CHARGES).reshape(-1, 1)
try:
    _fc_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:  # sklearn<1.2
    _fc_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
_fc_encoder.fit(_fc_values)


def one_hot_encode_hybridization(hybridization) -> np.ndarray:
    v = np.array([[int(hybridization)]])
    return _hyb_encoder.transform(v).flatten()


def one_hot_encode_formal_charge(formal_charge: int) -> np.ndarray:
    v = np.array([[int(formal_charge)]])
    return _fc_encoder.transform(v).flatten()


def get_ring_size(obj, max_size: int = 12) -> int:
    if not obj.IsInRing():
        return 0
    for size in range(3, max_size + 1):
        if obj.IsInRingSize(size):
            return size
    return max_size


# Mendeleev element cache (speed + reproducibility)
_el_map: Dict[int, "mendeleev.models.Element"] = {}


def getMendeleevElement(nr: int):
    if nr not in _el_map:
        _el_map[nr] = mendeleev.element(nr)
    return _el_map[nr]


def _safe_float(x) -> float:
    try:
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _safe_call0(obj, method_name: str) -> float:
    """Call a no-arg method; return 0.0 on failure."""
    try:
        m = getattr(obj, method_name, None)
        if m is None:
            return 0.0
        return _safe_float(m())
    except Exception:
        return 0.0


def get_atom_features(mol: Chem.Mol, atom: Chem.Atom) -> List[float]:
    me = getMendeleevElement(atom.GetAtomicNum())
    feats: List[float] = []
    feats.append(atom.GetAtomicNum())
    feats.append(atom.GetDegree())
    feats.append(_safe_float(getattr(me, "atomic_radius", None)))
    feats.append(_safe_float(getattr(me, "atomic_volume", None)))
    feats.extend(one_hot_encode_formal_charge(atom.GetFormalCharge()).tolist())
    feats.append(_safe_float(getattr(me, "covalent_radius", None)))
    feats.append(_safe_float(getattr(me, "vdw_radius", None)))
    feats.append(_safe_float(getattr(me, "dipole_polarizability", None)))
    feats.append(_safe_float(getattr(me, "electron_affinity", None)))
    feats.append(_safe_call0(me, "electrophilicity"))
    feats.append(_safe_float(getattr(me, "en_pauling", None)))
    feats.append(_safe_float(getattr(me, "electrons", None)))
    feats.append(_safe_float(getattr(me, "neutrons", None)))
    feats.append(int(atom.GetChiralTag()))
    feats.append(int(atom.IsInRing()))
    feats.append(int(atom.GetIsAromatic()))
    feats.extend(one_hot_encode_hybridization(atom.GetHybridization()).tolist())
    feats.append(_safe_float(atom.GetMass()))
    feats.append(_safe_float(atom.GetExplicitValence()))
    feats.append(_safe_float(atom.GetTotalValence()))
    feats.append(float(get_ring_size(atom)))

    try:
        gasteiger_charge = float(atom.GetProp("_GasteigerCharge"))
    except Exception:
        gasteiger_charge = 0.0
    feats.append(gasteiger_charge)

    num_atom_rings = sum(1 for ring in mol.GetRingInfo().AtomRings() if atom.GetIdx() in ring)
    feats.append(float(num_atom_rings))

    feats.append(1.0 if atom.GetAtomicNum() == 15 else 0.0)  # is_phosphorus

    aromatic_bond_count = sum(1 for b in atom.GetBonds() if b.GetIsAromatic())
    feats.append(float(aromatic_bond_count))

    bond_order_sum = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
    feats.append(float(bond_order_sum))

    heavy_neighbor_count = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() != 1)
    feats.append(float(heavy_neighbor_count))

    non_carbon_neighbor_count = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() != 6)
    feats.append(float(non_carbon_neighbor_count))

    charge_diff = gasteiger_charge - float(atom.GetFormalCharge())
    feats.append(float(charge_diff))

    return feats


def get_bond_features(bond: Chem.Bond) -> List[float]:
    feats: List[float] = []
    feats.append(float(bond.GetBondTypeAsDouble()))
    feats.append(float(bond.IsInRing()))
    feats.append(float(bond.GetIsConjugated()))
    feats.append(float(bond.GetIsAromatic()))
    feats.append(float(get_ring_size(bond)))
    feats.append(float(bond.GetStereo()))
    feats.append(1.0 if (bond.GetBondType() == Chem.rdchem.BondType.SINGLE and not bond.IsInRing()) else 0.0)
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    feats.append(float(abs(a1.GetDegree() - a2.GetDegree())))
    feats.append(float(a1.GetDegree() + a2.GetDegree()))
    return feats


def sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
    t = t.clone()
    t[torch.isnan(t)] = 0.0
    t[torch.isposinf(t)] = 0.0
    t[torch.isneginf(t)] = 0.0
    return t


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    return mol


@dataclass(frozen=True)
class GlobalFeatures:
    num_n: float
    ratio_p: float
    formal_charge: float
    tpsa: float
    mol_mr: float
    num_aromatic_rings: float
    ratio_aromatic: float
    avg_mass: float
    aromatic_count: float


def compute_global_features(mol: Chem.Mol) -> GlobalFeatures:
    num_atoms = mol.GetNumAtoms()
    num_n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    p_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 15)
    ratio_p = (p_count / num_atoms) if num_atoms else 0.0
    formal_charge = float(Chem.GetFormalCharge(mol))

    tpsa = float(Descriptors.TPSA(mol))
    mol_mr = float(Descriptors.MolMR(mol))
    num_ar_rings = float(Descriptors.NumAromaticRings(mol))

    aromatic_count = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    ratio_aromatic = (aromatic_count / num_atoms) if num_atoms else 0.0
    avg_mass = (sum(a.GetMass() for a in mol.GetAtoms()) / num_atoms) if num_atoms else 0.0

    return GlobalFeatures(
        num_n=float(num_n),
        ratio_p=float(ratio_p),
        formal_charge=float(formal_charge),
        tpsa=float(tpsa),
        mol_mr=float(mol_mr),
        num_aromatic_rings=float(num_ar_rings),
        ratio_aromatic=float(ratio_aromatic),
        avg_mass=float(avg_mass),
        aromatic_count=float(aromatic_count),
    )
