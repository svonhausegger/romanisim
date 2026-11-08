"""The threaded / blocked L1-L2 paths against the original single-call code."""
import copy

import galsim
import numpy as np
import pytest
from astropy.time import Time

from romanisim import l1, ramp, persistence as rpers
from romanisim.models import nonlinearity, parameters

RP = [[1], [2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12], [13]]


def _setup(ny=96, nx=130, seed=3):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rng.uniform(0, 5000, (ny, nx))).astype('f8')
    counts[5, :20] = 3e5
    co = np.zeros((4, ny, nx), 'f4')
    co[1] = 1 + rng.normal(0, 0.01, (ny, nx))
    co[2] = 3e-6
    co[3] = -1e-11
    nl = nonlinearity.Nonlinearity(
        coeffs=co, getdq=True,
        gain=(1.8 + 0.1 * rng.standard_normal((ny, nx))).astype('f4'),
        saturation=np.full((ny, nx), 60000., 'f4'))
    nl.inl_lookup = np.linspace(-1000, 70000, 32)
    nl.inl_corrs = {c: rng.normal(0, 3, 32) for c in range(1, 3)}
    pers = rpers.Persistence()
    img = np.zeros((ny, nx))
    img[40:44, 10:60] = 1e5
    pers.update(img, Time('2025-12-31T23:50:00').mjd)
    return counts, nl, pers


def _run(mode, counts, nl, pers, ref, **kw):
    p = copy.deepcopy(pers)
    res, dq = l1.apportion_counts_to_resultants(
        counts, l1.read_pattern_to_tij(RP), pedestal=parameters.pedestal,
        pedestal_extra_noise=parameters.pedestal_extra_noise,
        inv_linearity=nl, crparam=dict(flux=200), persistence=p,
        tstart=Time('2026-01-01T00:00:00'), rng=galsim.UniformDeviate(7),
        reference_read=ref, mode=mode, **kw)
    return res, dq, p


@pytest.mark.parametrize('ref', [False, True])
def test_exact_mode_is_bit_identical_to_legacy(ref):
    counts, nl, pers = _setup()
    r0, d0, p0 = _run('legacy', counts, nl, pers, ref)
    assert np.any(d0 & parameters.dqbits['jump_det'])  # CRs were simulated
    for nthreads, rpb in ((1, 64), (4, 7), (3, 1000)):
        r1, d1, p1 = _run('exact', counts, nl, pers, ref,
                          nthreads=nthreads, rows_per_block=rpb)
        assert np.array_equal(r0, r1)
        assert np.array_equal(d0, d1)
        assert np.array_equal(p0.x, p1.x)
        assert np.array_equal(p0.index, p1.index)


def test_parallel_mode_seed_semantics():
    counts, nl, pers = _setup()
    r0, d0, _ = _run('legacy', counts, nl, pers, False)
    ra, da, _ = _run('parallel', counts, nl, pers, False, nthreads=1,
                     rows_per_block=16)
    rb, db, _ = _run('parallel', counts, nl, pers, False, nthreads=5,
                     rows_per_block=16)
    # independent of the thread count, a different realization than legacy
    assert np.array_equal(ra, rb) and np.array_equal(da, db)
    assert not np.array_equal(ra, r0)
    # cosmic rays and their flags come from the shared CR stream
    assert np.array_equal(da, d0)


def test_parallel_mode_statistics():
    tij = l1.read_pattern_to_tij(RP)
    N = 300
    counts = np.full((400, 400), N, 'f8')
    res, _ = l1.apportion_counts_to_resultants(
        counts, tij, rng=galsim.UniformDeviate(11), mode='parallel',
        nthreads=4)
    T = tij[-1][-1]
    for i, t in enumerate(tij):
        q = np.asarray(t) / T
        mean = N * q.mean()
        assert abs(res[i].mean() - mean) < 5 * np.sqrt(N / res[i].size) + 1e-6
    assert np.all(res[-1] == N)


@pytest.mark.parametrize('ref', [False, True])
def test_dark_decay_matches_original_formula(ref, monkeypatch):
    dd = dict(amplitude=3.7, time_constant=23.0, sca=5)
    tij = l1.read_pattern_to_tij(RP, reference_read=ref)
    rng = np.random.default_rng(1)
    r0 = rng.normal(1000, 50, (len(tij),) + (4088, 4088)).astype('f4')
    expect = r0.copy()
    for i in range(expect.shape[0]):
        expect[i] += -1 * np.mean([l1.dark_decay_for_read(dd, t)
                                   for t in tij[i]], axis=0)
    for n in ('1', '3'):
        monkeypatch.setenv('ROMANISIM_NTHREADS', n)
        out = r0.copy()
        l1.apply_dark_decay(out, dd, RP, sign=-1, reference_read=ref)
        assert np.array_equal(out, expect)


def test_threaded_nonlinearity_and_ramp_fit(monkeypatch):
    counts, nl, _ = _setup(ny=128, nx=96)
    monkeypatch.setattr(nonlinearity, '_PARALLEL_MIN_SIZE', 1)
    img3 = np.random.default_rng(4).uniform(0, 60000, (3, 128, 96))
    monkeypatch.setenv('ROMANISIM_NTHREADS', '1')
    a2 = nl.apply(counts.copy(), electrons=True)
    a3 = nl.apply(img3.copy())
    monkeypatch.setenv('ROMANISIM_NTHREADS', '4')
    assert np.array_equal(a2, nl.apply(counts.copy(), electrons=True))
    assert np.array_equal(a3, nl.apply(img3.copy()))

    tij = l1.read_pattern_to_tij(RP)
    res, _ = l1.apportion_counts_to_resultants(
        counts, tij, rng=galsim.UniformDeviate(2), mode='parallel')
    res = res.reshape(len(RP), -1).astype('f4')
    dq = (np.random.default_rng(5).random(res.shape) < 0.05).astype('i4')
    rn = np.full(res.shape[1], 10, 'f4')
    one = ramp.romanisim.ramp_fit_casertano.fit_ramps(res, dq, rn, RP)
    many = ramp._fit_ramps_threaded(res, dq, rn, RP, nthreads=4,
                                    min_pixels=1)
    for k in one:
        assert np.array_equal(one[k], many[k]), k


def test_non_image_counts_use_legacy():
    tij = l1.read_pattern_to_tij(RP)
    counts = np.random.default_rng(9).poisson(500, 1000)
    a, _ = l1.apportion_counts_to_resultants(
        counts, tij, rng=galsim.UniformDeviate(3), mode='parallel')
    b, _ = l1.apportion_counts_to_resultants(
        counts, tij, rng=galsim.UniformDeviate(3), mode='legacy')
    assert a.shape == (len(RP), 1000)
    assert np.array_equal(a, b)
