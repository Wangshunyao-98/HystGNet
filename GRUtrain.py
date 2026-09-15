from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
from torch.optim import Adam

from model import HystGNet

try:
    import scipy.io
except ImportError:
    scipy = None


PARAMETER_NAMES = [
    "lambda_b",
    "lambda_c",
    "R_t",
    "f_c_pre",
    "f_c_cast",
    "rho_y",
    "f_y_c",
    "rho_x",
    "f_y_b",
    "rho_v",
    "f_v",
    "n",
]

DATA_ALIASES = {
    "train_seq": ["Input_train", "X_train", "x_seq_train", "Displacement_train", "Input_seq_train"],
    "val_seq": ["Input_val", "Input_valid", "X_val", "X_valid", "x_seq_val", "x_seq_valid"],
    "test_seq": ["Input_test", "X_test", "x_seq_test", "Displacement_test", "Input_seq_test"],
    "train_params": [
        "Param_train",
        "Params_train",
        "Parameter_train",
        "Parameters_train",
        "Input_train_P",
        "Input_train_param",
        "Input_train_params",
        "P_train",
    ],
    "val_params": [
        "Param_val",
        "Params_val",
        "Parameter_val",
        "Parameters_val",
        "Input_val_P",
        "Input_valid_P",
        "P_val",
        "P_valid",
    ],
    "test_params": [
        "Param_test",
        "Params_test",
        "Parameter_test",
        "Parameters_test",
        "Input_test_P",
        "Input_test_param",
        "Input_test_params",
        "P_test",
    ],
    "train_target": ["Target_train", "Y_train", "y_train", "Force_train"],
    "val_target": ["Target_val", "Target_valid", "Y_val", "Y_valid", "y_val", "y_valid"],
    "test_target": ["Target_test", "Y_test", "y_test", "Force_test"],
}


class MinMaxScalerLite:

    def __init__(self, feature_range: tuple[float, float] = (-1.0, 1.0)) -> None:
        self.feature_range = feature_range
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "MinMaxScalerLite":
        self.data_min_ = np.min(data, axis=0)
        self.data_max_ = np.max(data, axis=0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        low, high = self.feature_range
        denom = self.data_max_ - self.data_min_
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        scaled = (data - self.data_min_) / denom
        return scaled * (high - low) + low


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "hystgnet")
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seq-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def find_array(mat: dict, aliases: Iterable[str], required: bool = True) -> np.ndarray | None:
    lookup = {key.lower(): key for key in mat if not key.startswith("__")}
    for alias in aliases:
        key = lookup.get(alias.lower())
        if key is not None:
            return np.asarray(mat[key])
    if required:
        raise KeyError(
            "Missing fields: "
            + ", ".join(aliases)
            + f". Available fields: {', '.join(sorted(lookup.values()))}"
        )
    return None


def ensure_sequence(array: np.ndarray, seq_len: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array[None, :, None]
    elif array.ndim == 2:
        array = array[:, :, None]
    elif array.ndim != 3:
        raise ValueError(f"Invalid sequence shape: {array.shape}")
    if array.shape[1] == seq_len:
        return array

    old_grid = np.linspace(0.0, 1.0, array.shape[1])
    new_grid = np.linspace(0.0, 1.0, seq_len)
    resampled = np.empty((array.shape[0], seq_len, array.shape[2]), dtype=np.float32)
    for item_idx in range(array.shape[0]):
        for feature_idx in range(array.shape[2]):
            resampled[item_idx, :, feature_idx] = np.interp(
                new_grid,
                old_grid,
                array[item_idx, :, feature_idx],
            )
    return resampled


def ensure_params(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Invalid parameter shape: {array.shape}")
    return array


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    residual = y_true - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return r2, mae, rmse


def normalize_sequences(train: np.ndarray, *others: np.ndarray) -> tuple[MinMaxScalerLite, list[np.ndarray]]:
    scaler = MinMaxScalerLite(feature_range=(-1, 1))
    scaler.fit(train.reshape(-1, train.shape[-1]))
    normalized = [
        scaler.transform(array.reshape(-1, array.shape[-1])).reshape(array.shape)
        for array in (train, *others)
    ]
    return scaler, normalized


def normalize_params(train: np.ndarray, *others: np.ndarray) -> tuple[MinMaxScalerLite, list[np.ndarray]]:
    scaler = MinMaxScalerLite(feature_range=(-1, 1))
    scaler.fit(train)
    return scaler, [scaler.transform(array) for array in (train, *others)]


def build_loader(
    x_seq: np.ndarray,
    params: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> Data.DataLoader:
    dataset = Data.TensorDataset(
        torch.as_tensor(x_seq, dtype=torch.float32),
        torch.as_tensor(params, dtype=torch.float32),
        torch.as_tensor(target, dtype=torch.float32),
    )
    return Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, loader: Data.DataLoader, criterion: nn.Module) -> tuple[float, float, float, float]:
    model.eval()
    device = next(model.parameters()).device
    losses = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_seq, params, target in loader:
            x_seq = x_seq.to(device)
            params = params.to(device)
            target = target.to(device)
            prediction = model(x_seq, params)
            losses.append(criterion(prediction, target).item())
            y_true.append(target.detach().cpu().numpy())
            y_pred.append(prediction.detach().cpu().numpy())

    y_true_array = np.concatenate(y_true, axis=0).reshape(-1, 1)
    y_pred_array = np.concatenate(y_pred, axis=0).reshape(-1, 1)
    eval_r2, eval_mae, eval_rmse = regression_metrics(y_true_array, y_pred_array)
    return float(np.mean(losses)), eval_r2, eval_mae, eval_rmse


def load_dataset(data_path: Path, seq_len: int) -> dict[str, np.ndarray]:
    if data_path.suffix.lower() == ".npz":
        loaded = np.load(data_path)
        mat = {key: loaded[key] for key in loaded.files}
    else:
        if scipy is None:
            raise ImportError("scipy is required to read .mat files.")
        mat = scipy.io.loadmat(data_path)

    train_seq = ensure_sequence(find_array(mat, DATA_ALIASES["train_seq"]), seq_len)
    test_seq = ensure_sequence(find_array(mat, DATA_ALIASES["test_seq"]), seq_len)
    val_seq_raw = find_array(mat, DATA_ALIASES["val_seq"], required=False)
    val_seq = ensure_sequence(val_seq_raw, seq_len) if val_seq_raw is not None else None

    train_params = find_array(mat, DATA_ALIASES["train_params"], required=False)
    test_params = find_array(mat, DATA_ALIASES["test_params"], required=False)
    val_params = find_array(mat, DATA_ALIASES["val_params"], required=False)

    if train_params is None or test_params is None:
        raise KeyError("Missing train/test parameter arrays.")

    train_target = ensure_sequence(find_array(mat, DATA_ALIASES["train_target"]), seq_len)
    test_target = ensure_sequence(find_array(mat, DATA_ALIASES["test_target"]), seq_len)
    val_target_raw = find_array(mat, DATA_ALIASES["val_target"], required=False)
    val_target = ensure_sequence(val_target_raw, seq_len) if val_target_raw is not None else None

    data = {
        "train_seq": train_seq,
        "train_params": ensure_params(train_params),
        "train_target": train_target,
        "test_seq": test_seq,
        "test_params": ensure_params(test_params),
        "test_target": test_target,
    }

    if val_seq is not None and val_params is not None and val_target is not None:
        data.update(
            {
                "val_seq": val_seq,
                "val_params": ensure_params(val_params),
                "val_target": val_target,
            }
        )

    return data


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data, args.seq_len)
    has_val = "val_seq" in data

    seq_scaler, seq_arrays = normalize_sequences(
        data["train_seq"],
        data["val_seq"] if has_val else data["test_seq"],
        data["test_seq"],
    )
    target_scaler, target_arrays = normalize_sequences(
        data["train_target"],
        data["val_target"] if has_val else data["test_target"],
        data["test_target"],
    )
    param_scaler, param_arrays = normalize_params(
        data["train_params"],
        data["val_params"] if has_val else data["test_params"],
        data["test_params"],
    )

    train_seq = seq_arrays[0]
    val_seq = seq_arrays[1] if has_val else None
    test_seq = seq_arrays[2]
    train_target = target_arrays[0]
    val_target = target_arrays[1] if has_val else None
    test_target = target_arrays[2]
    train_params = param_arrays[0]
    val_params = param_arrays[1] if has_val else None
    test_params = param_arrays[2]

    param_size = train_params.shape[1]

    train_loader = build_loader(train_seq, train_params, train_target, args.batch_size, shuffle=True)
    validation_loader = (
        build_loader(val_seq, val_params, val_target, args.batch_size, False)
        if has_val
        else None
    )
    test_loader = build_loader(test_seq, test_params, test_target, args.batch_size, shuffle=False)

    model = HystGNet(
        input_size=train_seq.shape[2],
        hidden_size=args.hidden_size,
        param_size=len(PARAMETER_NAMES),
        output_size=train_target.shape[2],
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    if param_size != len(PARAMETER_NAMES):
        raise ValueError(f"Expected {len(PARAMETER_NAMES)} structural parameters, got {param_size}.")

    history = []
    print(model)
    print(
        "Shapes:",
        f"train_seq={train_seq.shape}",
        f"train_params={train_params.shape}",
        f"train_target={train_target.shape}",
        f"test_seq={test_seq.shape}",
        f"test_params={test_params.shape}",
        f"test_target={test_target.shape}",
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x_seq, params, target in train_loader:
            x_seq = x_seq.to(device)
            params = params.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            prediction = model(x_seq, params)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        eval_loader = validation_loader if validation_loader is not None else test_loader
        eval_loss, eval_r2, eval_mae, eval_rmse = evaluate(model, eval_loader, criterion)
        train_loss = float(np.mean(train_losses))
        history.append([epoch, train_loss, eval_loss, eval_r2, eval_mae, eval_rmse])

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:05d}",
                f"train_loss={train_loss:.6f}",
                f"eval_loss={eval_loss:.6f}",
                f"eval_r2={eval_r2:.4f}",
                f"eval_mae={eval_mae:.6f}",
                f"eval_rmse={eval_rmse:.6f}",
            )

    torch.save(model.state_dict(), args.output_dir / "hystgnet.pt")
    np.savetxt(
        args.output_dir / "history.csv",
        np.asarray(history),
        delimiter=",",
        header="epoch,train_loss,eval_loss,eval_r2,eval_mae,eval_rmse",
        comments="",
    )
    with open(args.output_dir / "scalers.pkl", "wb") as file:
        pickle.dump(
            {
                "seq_scaler": seq_scaler,
                "param_scaler": param_scaler,
                "target_scaler": target_scaler,
                "parameter_names": PARAMETER_NAMES,
            },
            file,
        )

    test_loss, test_r2, test_mae, test_rmse = evaluate(model, test_loader, criterion)
    print(
        "Best model test metrics:",
        f"loss={test_loss:.6f}",
        f"R2={test_r2:.4f}",
        f"MAE={test_mae:.6f}",
        f"RMSE={test_rmse:.6f}",
    )


if __name__ == "__main__":
    main()
