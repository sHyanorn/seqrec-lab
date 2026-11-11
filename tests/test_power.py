import math

import numpy as np
import pytest

from seqrec.analysis import power


@pytest.mark.parametrize(("sigma", "n"), [(0.006, 6), (0.0075, 7), (0.01, 10)])
def test_seed_count_matches_the_methods_examples(sigma, n):
    assert power.seed_count(sigma) == n


def test_directional_power_at_sigma_001_and_ten_seeds_is_078():
    assert power.directional_power(0.01, 10) == pytest.approx(0.78, abs=0.005)
    assert power.equivalence_power(0.01, 10) >= 0.8


def test_power_is_finite_and_small_sigma_needs_five_seeds():
    for sigma in np.linspace(0.001, 0.012, 45):
        for n in power.N_RANGE:
            assert np.isfinite(power.directional_power(sigma, n))
    for sigma in (0.002, 0.00225, 0.0035, 0.0039, 0.0047):
        assert power.seed_count(sigma) == 5
    assert power.directional_power(0.0035, 5) == pytest.approx(0.9985, abs=5e-4)


def test_directional_power_matches_simulation():
    rng = np.random.default_rng(0)
    sigma, n = 0.008, 6
    se, df = sigma * math.sqrt(2 / n), 2 * n - 2
    from scipy import stats

    crit = stats.t.ppf(1 - 0.05 / 24 / 2, df)
    a = rng.normal(power.DELTA, sigma, (40000, n))
    b = rng.normal(0.0, sigma, (40000, n))
    sp = np.sqrt((a.var(1, ddof=1) + b.var(1, ddof=1)) / 2)
    t = (a.mean(1) - b.mean(1)) / (sp * math.sqrt(2 / n))
    assert np.mean(np.abs(t) > crit) == pytest.approx(power.directional_power(sigma, n), abs=0.01)
    assert se > 0


def test_pooled_sd():
    groups = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 12.0])]
    expected = math.sqrt((2 * 1.0 + 1 * 2.0) / 3)
    assert power.pooled_sd(groups) == pytest.approx(expected)


def test_sigma_from_an_export_file(tmp_path):
    path = tmp_path / "pilot.csv"
    path.write_text(
        "arm,tag,dataset,split,shuffle,status,validation_ndcg@10\n"
        "sasrec-ce,sigma,ml-1m,loo,none,trained,0.18\nsasrec-ce,sigma,ml-1m,loo,none,trained,0.182\n"
        "bert4rec,sigma,ml-1m,loo,none,trained,0.15\nbert4rec,sigma,ml-1m,loo,none,trained,0.151\n"
        "sasrec-ce,time-small,ml-1m,loo,none,trained,0.1\nmostpop,sigma,ml-1m,loo,none,trained,0.05\n"
        "sasrec-ce,sigma,steam,loo,none,trained,0.07\nsasrec-ce,sigma,ml-1m,loo,train_eval,trained,0.1\n"
        "sasrec-ce,sigma,ml-1m,gts,none,trained,0.1\nsasrec-ce,sigma,ml-1m,loo,none,failed,0.01\n",
        encoding="utf-8",
    )
    sigma, groups = power.sigma_from_export(path, "sigma", "ml-1m")
    assert sorted(groups) == ["bert4rec", "sasrec-ce"] and all(len(v) == 2 for v in groups.values())
    logs = [np.log([0.18, 0.182]), np.log([0.15, 0.151])]
    assert sigma == pytest.approx(power.pooled_sd(logs))
    assert "pooled sigma" in power.report(path, "sigma", "ml-1m")


def _final_export(path, drop=None, status=None):
    """Final runs of five arms with seeds 1 to 7; seeds 6 and 7 are far from the first five."""
    lines = ["arm,seed,tag,dataset,split,shuffle,status,validation_ndcg@10"]
    for a, arm in enumerate(power.NEURAL_ARMS):
        for seed in range(1, 8):
            value = 0.15 + 0.01 * a + (0.001 * seed if seed <= 5 else 0.05)
            state = (status or {}).get((arm, seed), "trained")
            if (arm, seed) != drop:
                lines.append(f"{arm},{seed},,ml-1m,loo,none,{state},{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_sigma_is_estimated_again_from_the_first_five_final_seeds(tmp_path):
    """METHODS 7.5: after the first 5 final seeds per arm; later seeds do not enter."""
    path = _final_export(tmp_path / "final.csv")
    sigma, groups = power.sigma_from_export(path, "", "ml-1m", seeds=range(1, 6))
    assert sorted(groups) == sorted(power.NEURAL_ARMS)
    assert all(len(v) == 5 for v in groups.values())
    expected = [np.log(0.15 + 0.01 * a + 0.001 * np.arange(1, 6)) for a in range(5)]
    assert sigma == pytest.approx(power.pooled_sd(expected))
    assert sigma < power.sigma_from_export(path, "", "ml-1m")[0]  # all seven seeds
    report = power.report(path, "", "ml-1m", seeds=range(1, 6))
    assert f"pooled sigma {sigma:.5f}" in report and "5 runs" in report


def test_a_seed_missing_from_an_arm_stops_the_estimate(tmp_path):
    """An arm with four of the five seeds would shrink its group without notice."""
    path = _final_export(tmp_path / "missing.csv", drop=("gsasrec", 3))
    with pytest.raises(ValueError, match="gsasrec.*seeds 1, 2, 4, 5"):
        power.sigma_from_export(path, "", "ml-1m", seeds=range(1, 6))
    path = _final_export(tmp_path / "failed.csv", status={("bert4rec", 5): "failed"})
    with pytest.raises(ValueError, match="bert4rec.*seeds 1, 2, 3, 4"):
        power.sigma_from_export(path, "", "ml-1m", seeds=range(1, 6))
