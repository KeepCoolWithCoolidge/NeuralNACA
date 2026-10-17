"""Generate a small XFOIL data set, train a model on it and evaluate it (needs XFOIL; a few minutes)."""

from neuralnaca.data import generate_dataset, pack_dataset
from neuralnaca.evaluation import error_metrics
from neuralnaca.model import predict_tensors, save_model
from neuralnaca.support import SupportModel
from neuralnaca.training import TrainConfig, train_ensemble
from neuralnaca.xfoil import detect_xfoil

if __name__ == "__main__":                        # required: XFOIL runs in spawned worker processes
    cfg = detect_xfoil()
    train = generate_dataset(1024, "sobol", seed=42, cfg=cfg, path="example_train.npz")
    test = generate_dataset(300, "random", seed=1042, cfg=cfg, path="example_test.npz")
    net = train_ensemble(pack_dataset(train), TrainConfig(ensemble=3, steps=4000, seed=42))
    save_model(net, "example_model.npz", {"note": "example"})
    SupportModel.from_dataset(train).save("example_model.support.npz")
    pred = predict_tensors(net, pack_dataset(test, with_targets=False)).mean(0)
    m = error_metrics(pred, test.targets())
    print(f"MAE  CL {m['CL_MAE']:.4f}   ln CD {m['lnCD_MAE']:.4f}   CM {m['CM_MAE']:.5f}")
