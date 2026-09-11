import json
import runpy
import sys
from pathlib import Path


def test_collect_results_table(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "metrics.json").write_text(
        json.dumps(
            dict(
                summary=dict(
                    method="baseline",
                    variant="occ0",
                    vis_gate=None,
                    keep_first=None,
                    J=0.5,
                    F=0.4,
                    JF=0.45,
                    leak=float("nan"),
                    vis_auc=0.7,
                    recovery_median=2.0,
                    never_recovered=1,
                    n_episodes=3,
                )
            )
        )
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "collect_results.py"
    sys.argv = [str(script), str(tmp_path)]
    runpy.run_path(str(script), run_name="__main__")
    out = capsys.readouterr().out
    assert (
        "| a | baseline | occ0 | – | – | 0.500 | 0.400 | 0.450 | – | 0.700 | 2.000 | 1 / 3 |" in out
    )
