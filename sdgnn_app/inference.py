from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit import RDLogger

from .features import (
    compute_global_features,
    get_atom_features,
    get_bond_features,
    sanitize_tensor,
    smiles_to_mol,
)
from .model import SDGNN


@dataclass
class PredictionResult:
    smiles: str
    pred_ppm: float
    node_dim: int
    edge_dim: int
    global_dim: int
    hidden_dim: int


def _torch_load_compat(path: str, map_location="cpu") -> Dict[str, Any]:
    """Torch 2.0+ supports weights_only; older versions don't."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_graph_from_smiles(smiles: str, edge_dim_fallback: int = 9) -> Data:
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    mol = smiles_to_mol(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")

    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    atom_features_list = [get_atom_features(mol, atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features_list, dtype=torch.float)

    edge_index = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        edge_index.append((i, j)); edge_attr_list.append(bf)
        edge_index.append((j, i)); edge_attr_list.append(bf)

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, int(edge_dim_fallback)), dtype=torch.float)

    gf = compute_global_features(mol)
    u = torch.tensor([[
        gf.num_n,
        gf.ratio_p,
        gf.formal_charge,
        gf.tpsa,
        gf.mol_mr,
        gf.num_aromatic_rings,
        gf.ratio_aromatic,
        gf.avg_mass,
        gf.aromatic_count,
    ]], dtype=torch.float)

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)
    g.smiles = smiles

    # sanitize like training
    g.x = sanitize_tensor(g.x)
    if g.edge_attr is not None and g.edge_attr.numel() > 0:
        g.edge_attr = sanitize_tensor(g.edge_attr)
    g.u = sanitize_tensor(g.u)
    return g


class SDGNNPredictor:
    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        self.ckpt_path = str(ckpt_path)
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = _torch_load_compat(self.ckpt_path, map_location="cpu")
        self.scalers = ckpt["scalers"]
        self.hidden_dim = int(ckpt["params"]["hd"])

        # Infer expected feature dims directly from scalers when possible
        self.node_dim = int(getattr(self.scalers["node"], "n_features_in_", 0) or 0)
        self.edge_dim = int(getattr(self.scalers.get("edge", None), "n_features_in_", 0) or 0) if self.scalers.get("edge", None) is not None else 0
        self.global_dim = int(getattr(self.scalers["global"], "n_features_in_", 0) or 0)

        # Fallback: build a probe graph and read dims if scalers lack n_features_in_
        if self.node_dim == 0 or self.global_dim == 0 or (self.scalers.get("edge", None) is not None and self.edge_dim == 0):
            probe = build_graph_from_smiles("CP")  # simple phosphine probe
            self.node_dim = probe.x.shape[1]
            self.global_dim = probe.u.shape[1]
            self.edge_dim = probe.edge_attr.shape[1] if probe.edge_attr is not None else 9

        self.model = SDGNN(self.node_dim, self.edge_dim, self.global_dim, self.hidden_dim).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _apply_scalers(self, g: Data) -> Data:
        gg = g.clone()
        gg.x = torch.from_numpy(self.scalers["node"].transform(gg.x.cpu().numpy())).float()
        gg.u = torch.from_numpy(self.scalers["global"].transform(gg.u.cpu().numpy())).float()

        if self.scalers.get("edge", None) is not None:
            if gg.edge_attr is not None and gg.edge_attr.numel() > 0:
                gg.edge_attr = torch.from_numpy(self.scalers["edge"].transform(gg.edge_attr.cpu().numpy())).float()
            else:
                # Keep correct feature dimension for empty edge_attr
                gg.edge_attr = torch.empty((0, self.edge_dim), dtype=torch.float)
        return gg

    @torch.no_grad()
    def predict(self, smiles: str) -> PredictionResult:
        g = build_graph_from_smiles(smiles, edge_dim_fallback=self.edge_dim or 9)
        gs = self._apply_scalers(g)
        gs = gs.to(self.device)
        # Single-graph forward: ensure batch attribute exists
        if not hasattr(gs, "batch") or gs.batch is None:
            gs.batch = torch.zeros(gs.x.size(0), dtype=torch.long, device=self.device)

        pred = self.model(gs).view(-1).detach().cpu().numpy()
        pred_ppm = float(pred[0])

        return PredictionResult(
            smiles=smiles,
            pred_ppm=pred_ppm,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            global_dim=self.global_dim,
            hidden_dim=self.hidden_dim,
        )
