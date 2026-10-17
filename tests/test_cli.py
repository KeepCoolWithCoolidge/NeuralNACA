import json

import pytest

from neuralnaca.cli import _seeds, build_parser, main


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and "neuralnaca" in capsys.readouterr().out


def test_predict_table(capsys):
    assert main(["predict", "2412", "--re", "1e6", "--alpha", "0", "4", "--step", "2"]) == 0
    out = capsys.readouterr().out
    assert "NACA 2412" in out
    assert len([l for l in out.splitlines() if l.strip().startswith(("0.00", "2.00", "4.00"))]) == 3


def test_predict_json_and_angle_list(capsys):
    assert main(["predict", "0012", "--re", "5e5", "--alpha", "-2", "0", "2", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["alpha"] == [-2.0, 0.0, 2.0] and len(doc["CL"]) == 3
    assert doc["CL"][0] == pytest.approx(-doc["CL"][2], abs=1e-6)


def test_predict_marks_out_of_domain(capsys):
    assert main(["predict", "2412", "--re", "1e6", "--alpha", "20"]) == 0
    assert "outside validated domain" in capsys.readouterr().out


def test_bad_designation_is_an_error():
    with pytest.raises(ValueError):
        main(["predict", "12345", "--re", "1e6"])


def test_info(capsys):
    assert main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["training"]["n_train_runs"] == 4096 and "torch" in info["environment"]


def test_seed_ranges():
    assert _seeds("0-2,5") == [0, 1, 2, 5]
    assert _seeds("3") == [3]


def test_every_subcommand_has_help():
    ap = build_parser()
    for cmd in ("predict", "info", "xfoil-check", "generate", "train", "evaluate", "bench", "reproduce", "report",
                "figures"):
        with pytest.raises(SystemExit) as e:
            ap.parse_args([cmd, "--help"])
        assert e.value.code == 0


def test_train_and_evaluate_commands(tmp_path, capsys):
    import numpy as np

    from neuralnaca.data import XfoilDataset
    from neuralnaca.domain import sample_points

    from ._solvers import analytic_teacher

    for name, kind, n in (("tr", "sobol", 128), ("te", "random", 40)):
        pts = sample_points(n, kind, 0)
        XfoilDataset(pts, np.array([analytic_teacher(p, None) for p in pts])).save(tmp_path / f"{name}.npz")
    assert main(["train", "--data", str(tmp_path / "tr.npz"), "--out", str(tmp_path / "m.npz"), "--ensemble", "2",
                 "--steps", "40"]) == 0
    assert (tmp_path / "m.support.npz").exists()
    assert main(["evaluate", "--weights", str(tmp_path / "m.npz"), "--data", str(tmp_path / "te.npz"),
                 "--neuralfoil", "", "--out", str(tmp_path / "e.json")]) == 0
    assert "40 XFOIL-converged test points" in capsys.readouterr().out
    assert json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))["n_test"] == 40


def test_report_and_reproduce_commands(tmp_path, capsys, repo_root):
    if not (repo_root / "benchmarks" / "results" / "main_s0.json").exists():
        pytest.skip("needs the shipped results")
    assert main(["report", "--root", str(repo_root), "--out", str(tmp_path / "t.md")]) == 0
    assert (tmp_path / "headline.md").exists()
    pytest.importorskip("neuralfoil")
    assert main(["reproduce", "--level", "0", "--seeds", "0", "--root", str(repo_root),
                 "--json", str(tmp_path / "r.json")]) == 0
    assert "PASSED" in capsys.readouterr().out
