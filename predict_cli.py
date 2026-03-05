import argparse
from sdgnn_app import SDGNNPredictor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/SDGNN_grouped_BEST.pt")
    ap.add_argument("smiles")
    args = ap.parse_args()

    pred = SDGNNPredictor(args.ckpt).predict(args.smiles)
    print(f"{pred.smiles}\t{pred.pred_ppm:.6f} ppm")

if __name__ == "__main__":
    main()
