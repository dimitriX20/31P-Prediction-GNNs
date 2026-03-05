# SDGNN SMILES App (Streamlit)

Minimal-App: SMILES eingeben → SDGNN (Checkpoint `SDGNN_grouped_BEST.pt`) → 31P-Vorhersage (ppm).

## 1) Checkpoint ablegen

Lege deinen Checkpoint hier ab:

```
models/SDGNN_grouped_BEST.pt
```

Oder setze eine Umgebungsvariable:

```
export SDGNN_CKPT=/path/to/SDGNN_grouped_BEST.pt
```

> Empfehlung: Wenn du `.pt` in Git committen willst, nutze Git LFS.

## 2) Lokal starten (Conda/Mamba empfohlen)

```
mamba env create -f environment.yml
mamba activate sdgnn-smiles-app
streamlit run app.py
```

## 3) Lokal starten (pip-only, best effort)

PyTorch Geometric ist pip-seitig manchmal heikel (CUDA/CPU Wheels). Wenn du pip nutzen willst:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Wenn `torch-scatter` / `torch-geometric` Probleme machen: installiere die passenden Wheels gemäß der offiziellen PyG-Install-Anleitung.

## 4) Docker

```
docker build -t sdgnn-smiles-app .
docker run --rm -p 8501:8501 -v $PWD/models:/app/models sdgnn-smiles-app
```

Dann im Browser: http://localhost:8501

## Feature-Definition

Die Feature-Definitionen (`sdgnn_app/features.py`) sind aus dem bereitgestellten Notebook übernommen:
- RDKit + AddHs
- Gasteiger Charges
- Node features mit mendeleev-Properties + OneHot(Hybridization/FormalCharge) + Ring/Neighbors
- Edge features (9 dims)
- Global features (9 dims): `[N, ratio_P, formal_charge, TPSA, MolMR, NumAromaticRings, ratio_aromatic, avg_mass, aromatic_count]`

**Wichtig:** Für sinnvolle Predictions muss diese Definition exakt mit dem Training/Checkpoint übereinstimmen.
