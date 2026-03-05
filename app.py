import os
from pathlib import Path

import streamlit as st

from sdgnn_app import SDGNNPredictor


st.set_page_config(page_title="SDGNN SMILES Predictor", page_icon="🧪", layout="centered")

st.title("🧪 SDGNN: 31P NMR Shift Predictor")
st.caption("SMILES → RDKit/mendeleev Featurization → SDGNN_grouped_BEST.pt → Vorhersage (ppm).")

with st.sidebar:
    st.header("Model")
    default_ckpt = os.environ.get("SDGNN_CKPT", "models/SDGNN_grouped_BEST.pt")
    ckpt_path = st.text_input("Checkpoint path", value=default_ckpt)
    st.write("Tipp: große `.pt`-Dateien am besten via Git LFS oder als Docker-Volume einbinden.")

@st.cache_resource(show_spinner=False)
def load_predictor(path: str) -> SDGNNPredictor:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {p.resolve()}")
    return SDGNNPredictor(str(p))

smiles = st.text_input("SMILES", value="CP", help="Beispiele: CP, O=P(C)(C)C, [PH+](C)(C)C")

if st.button("Vorhersage", type="primary"):
    try:
        predictor = load_predictor(ckpt_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Featurizing + Predicting..."):
        try:
            res = predictor.predict(smiles.strip())
        except Exception as e:
            st.error(f"Fehler bei SMILES/Inference: {e}")
            st.stop()

    st.success(f"Vorhersage: **{res.pred_ppm:.3f} ppm**")
    with st.expander("Model-/Feature-Details"):
        st.json({
            "node_dim": res.node_dim,
            "edge_dim": res.edge_dim,
            "global_dim": res.global_dim,
            "hidden_dim": res.hidden_dim,
            "device": str(predictor.device),
        })

st.markdown("---")
st.markdown(
    "📌 Lege den Checkpoint unter `models/SDGNN_grouped_BEST.pt` ab (oder setze `SDGNN_CKPT`). "
    "Wenn du ihn ins Repo committen willst: nutze **Git LFS**."
)
