"""K-fold linear readout (person / no person) on CUDA, trained on real and random targets.

Examples:
  python script/linear_readouts.py --model linear --data cls_embeddings.pt
  python script/linear_readouts.py --model conv1d --data image_embeddings.pt --epochs 100
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


class LinearReadout(torch.nn.Module):
    """Input: batch x features."""

    def __init__(self, features):
        super().__init__()
        self.linear = torch.nn.Linear(features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class LinearReadoutTokens(torch.nn.Module):
    """Input: batch x channels x features; conv1d (k=1) mixes channels, then linear."""

    def __init__(self, channels, features):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(channels, 1, kernel_size=1)
        self.linear = torch.nn.Linear(features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(self.conv1d(x).squeeze(1)))


def build_model(kind, x):
    if kind == "linear":
        return LinearReadout(x.shape[1])
    return LinearReadoutTokens(x.shape[1], x.shape[2])


def kfold(x, y, kind, n_splits, lr, epochs, seed, device, tag):
    accs, aucs = [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(np.arange(len(x)))):
        torch.manual_seed(seed + fold)
        model = build_model(kind, x).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = torch.nn.BCELoss()
        x_tr, y_tr = x[tr].to(device), y[tr].to(device)
        x_va, y_va = x[va].to(device), y[va]

        for epoch in range(epochs):  # full batch
            opt.zero_grad()
            loss = crit(model(x_tr).squeeze(-1), y_tr)
            loss.backward()
            opt.step()
            if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                print(f"[{tag}] fold {fold + 1} epoch {epoch + 1}/{epochs} loss {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            probs = model(x_va).squeeze(-1).cpu()
        acc = ((probs > 0.5).float() == y_va).float().mean().item()
        auc = roc_auc_score(y_va.numpy(), probs.numpy())
        accs.append(acc)
        aucs.append(auc)
        print(f"[{tag}] fold {fold + 1} acc {acc:.4f} auc {auc:.4f}")

    print(f"[{tag}] mean acc {np.mean(accs):.4f} +- {np.std(accs):.4f}, "
          f"mean auc {np.mean(aucs):.4f} +- {np.std(aucs):.4f}")
    return accs, aucs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["linear", "conv1d"], required=True,
                   help="linear: batch x features; conv1d: batch x channels x features")
    p.add_argument("--class0_dir", default="/u/fdammeier/artifacts/310other")
    p.add_argument("--class1_dir", default="/u/fdammeier/artifacts/310person")
    p.add_argument("--data", required=True, help="embedding filename inside each class dir")
    p.add_argument("--n_splits", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="optional JSON file for results")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device("cuda")

    x0 = torch.load(os.path.join(args.class0_dir, args.data))
    x1 = torch.load(os.path.join(args.class1_dir, args.data))
    x = torch.cat([x0, x1]).float()
    # If linear, flatten all dimensions except the batch dimension
    if args.model == "linear":
        x = x.view(x.size(0), -1)

    y = torch.cat([torch.zeros(len(x0)), torch.ones(len(x1))])

    expected_dims = 2 if args.model == "linear" else 3
    if x.dim() != expected_dims:
        raise ValueError(f"--model {args.model} expects {expected_dims}D input, got {tuple(x.shape)}")

    g = torch.Generator().manual_seed(args.seed)
    y_random = torch.randint(0, 2, y.shape, generator=g).to(y.dtype)

    results = {}
    for tag, target in (("target", y), ("random", y_random)):
        accs, aucs = kfold(x, target, args.model, args.n_splits, args.lr,
                           args.epochs, args.seed, device, tag)
        results[tag] = {"accuracy": accs, "auc": aucs}

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()