"""Tests for run_post_processing_vectorized.py -- the plotting counterpart
to run_core_training_vectorized.py. Verifies (1) the module imports and its
figure registry is well-formed, (2) fig_NN_..._vectorized() functions raise
VectorizedDataMissingError -- never fabricate data -- when their required
"_vectorized" cache is absent, and (3) a representative spread of figures
across every Core A-K render real end-to-end PDFs from the real
"_vectorized" cache entries already present in results/core_cache/ from
this session's own Core A-K test runs (not synthetic fixtures -- genuine,
if tiny-scale, cached output)."""

from unittest.mock import patch

import pytest

import run_post_processing_vectorized as rppv


def test_figure_registry_well_formed():
    assert len(rppv.FIGURE_FUNCTIONS) == 41
    for fig_num, fn in rppv.FIGURE_FUNCTIONS.items():
        assert callable(fn)
        assert fn.__name__.endswith("_vectorized")
    # Explicitly out-of-scope figures must never appear.
    for excluded in (2, 3, 4, 19, 34, 35, 48, 49, 50, 52):
        assert excluded not in rppv.FIGURE_FUNCTIONS


def test_raises_on_missing_cache_no_fallback():
    """Proves the no-fallback guarantee: with load_cache mocked to always
    return None, every figure must raise VectorizedDataMissingError, never
    silently substitute placeholder data."""
    with patch.object(rppv, "load_cache", return_value=None):
        for fig_num in (5, 8, 11, 13, 15, 18, 23, 39, 40, 43, 58):
            with pytest.raises(rppv.VectorizedDataMissingError):
                rppv.FIGURE_FUNCTIONS[fig_num]()


def test_raises_on_missing_second_cache_for_multi_cache_figures():
    """fig_12/fig_33/fig_55/fig_56 need two cache entries -- missing either
    one must raise, not silently render from whichever one is present."""
    real_load_cache = rppv.load_cache

    def _only_core_a(name):
        if name == rppv._CORE_A_CACHE:
            return real_load_cache(name)
        return None

    with patch.object(rppv, "load_cache", side_effect=_only_core_a):
        with pytest.raises(rppv.VectorizedDataMissingError):
            rppv.fig_12_convergence_bands_vectorized()


@pytest.mark.parametrize("fig_num", [5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 20,
                                      21, 22, 23, 24, 25, 29, 31, 33, 36, 37, 38,
                                      39, 40, 41, 42, 43, 44, 45, 46, 47, 53, 54,
                                      55, 56, 58])
def test_renders_real_pdf_from_existing_vectorized_cache(fig_num):
    """End-to-end render against the REAL "_vectorized" cache entries
    already in results/core_cache/ from this session's actual Core A-K
    test/smoke runs -- genuine (if tiny-scale) cached output, not a
    fabricated fixture. Skips cleanly if that real cache entry isn't
    present in this checkout (e.g. a fresh clone that hasn't run the
    corresponding `--core X` yet) rather than failing the suite."""
    try:
        rppv.FIGURE_FUNCTIONS[fig_num]()
    except rppv.VectorizedDataMissingError as e:
        pytest.skip(f"real vectorized cache not present for fig {fig_num}: {e}")

    matches = list(rppv.FIGURES_DIR.glob(f"fig_{fig_num:02d}_*_vectorized.pdf"))
    assert matches, f"expected a saved PDF for fig {fig_num} under {rppv.FIGURES_DIR}"


def test_fig_32_and_57_correctly_decline_on_undersized_real_cache():
    """These two figures legitimately cannot compute a meaningful statistic
    (Cohen's d needs >1 run per method; bootstrap CI needs >=10 episodes)
    from the tiny n_runs=1/n_episodes=2 smoke-scale cache already present
    in this checkout -- confirms the no-fallback guarantee holds even when
    a cache IS present but too small, not just when it's wholly absent."""
    for fig_num in (32, 57):
        try:
            rppv.FIGURE_FUNCTIONS[fig_num]()
        except rppv.VectorizedDataMissingError:
            continue
        except Exception:
            raise
        else:
            # If the real cache has since grown past the tiny smoke scale,
            # a clean render is equally acceptable -- just confirm the file.
            matches = list(rppv.FIGURES_DIR.glob(f"fig_{fig_num:02d}_*_vectorized.pdf"))
            assert matches
