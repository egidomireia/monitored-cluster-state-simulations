"""Fixed physical Pauli measurements with sign-free local Clifford frames.

Graph representatives alone do not specify subsequent physical bases. When
locally complementing v, U_v maps G to tau_v(G): sqrt(X) on v and sqrt(Z)
on its neighbours. Updating C -> C U_v^dagger changes C^dagger P C by
Y<->Z at v and X<->Y at its neighbours. Signs affect measurement outcomes,
not graph connectivity or entanglement, and are intentionally not sampled.

References: Hein, Eisert & Briegel, PRA 69, 062311 (2004);
Van den Nest et al., PRA 69, 022316 (2004).
"""
from pathlib import Path
import os
import pickle
import numpy as np

ALGORITHM = 'fixed-physical-pauli-lc-frame-gf2-v1'


def bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def gf2_rank(rows):
    """Exact binary rank of rows encoded as Python integer bitsets."""
    pivots = {}
    for row in rows:
        row = int(row)
        while row:
            pivot = row.bit_length() - 1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                break
    return len(pivots)


def square_adjacency(L, periodic=False):
    if L < 2:
        raise ValueError('L must be >=2.')
    rows = [0] * (L*L)
    for x in range(L):
        for y in range(L):
            a = x*L+y
            for dx, dy in ((1,0), (0,1)):
                u,v = x+dx,y+dy
                if periodic:
                    u,v = u % L,v % L
                if u < L and v < L:
                    b = u*L+v
                    if a != b:
                        rows[a] |= 1 << b
                        rows[b] |= 1 << a
    return rows


def measure_graph(adjacency, measured, basis='Y', order=None):
    """Return a graph LC-equivalent to the conditional surviving state.

    `basis` is the fixed physical basis, or one X/Y/Z character per vertex.
    `order` may permute the measured vertices (useful for validation).
    No physical correction gates are applied to the surviving system.
    """
    rows = adjacency.copy()
    n = len(rows)
    bases = [basis]*n if isinstance(basis, str) and len(basis)==1 else list(basis)
    if len(bases)!=n or any(b not in ('X','Y','Z') for b in bases):
        raise ValueError('Supply X, Y, Z, or one such label per vertex.')
    measured = [int(v) for v in measured]
    if len(set(measured))!=len(measured) or any(v<0 or v>=n for v in measured):
        raise ValueError('Invalid measured vertices.')
    order = measured if order is None else [int(v) for v in order]
    if sorted(order)!=sorted(measured):
        raise ValueError('order must be a permutation of measured vertices.')
    active = (1 << n)-1

    def complement(v):
        neighbours = rows[v]
        # Each neighbour row is toggled once; exclude its diagonal.
        for u in bits(neighbours):
            rows[u] ^= neighbours ^ (1 << u)
            bases[u] = {'X':'Y', 'Y':'X', 'Z':'Z'}[bases[u]]
        bases[v] = {'X':'X', 'Y':'Z', 'Z':'Y'}[bases[v]]

    for a in order:
        if bases[a]=='X' and rows[a]:
            b = (rows[a] & -rows[a]).bit_length()-1
            complement(b)  # Maps the effective X at a to Y.
        if bases[a]=='Y':
            complement(a)  # Maps the effective Y at a to Z.
        # Effective Z is vertex deletion; any isolated measurement is too.
        assert bases[a]=='Z' or rows[a]==0
        for b in bits(rows[a]):
            rows[b] &= ~(1 << a)
        rows[a] = 0
        active &= ~(1 << a)
    return rows, active


def components(rows, active):
    remaining = active
    while remaining:
        frontier = remaining & -remaining
        component = 0
        remaining ^= frontier
        while frontier:
            component |= frontier
            neighbours = 0
            for v in bits(frontier):
                neighbours |= rows[v]
            frontier = neighbours & remaining
            remaining &= ~frontier
        yield component


def entropy_cut(rows, component, A_mask):
    A = component & A_mask
    B = component & ~A_mask
    return gf2_rank(rows[a] & B for a in bits(A))


def simulate_realisation(L, p, basis, rng, periodic=False, adjacency=None):
    if not 0 <= p <= 1:
        raise ValueError('p must be between 0 and 1.')
    adjacency = square_adjacency(L, periodic) if adjacency is None else adjacency
    measured = np.flatnonzero(rng.random(L*L) < p)
    rows, active = measure_graph(adjacency, measured, basis)
    candidates, largest = [], 0
    for comp in components(rows, active):
        size = comp.bit_count()
        if size > largest:
            candidates, largest = [comp], size
        elif size == largest:
            candidates.append(comp)
    selected = candidates[int(rng.integers(len(candidates)))] if candidates else 0
    # node id = x*L+y; cut at x=floor(L/2), as in the supplied code.
    A_mask = (1 << ((L//2)*L))-1
    return dict(entropy=entropy_cut(rows, selected, A_mask),
                lcc_size=largest, measured_count=len(measured))


def _atomic_save(path, data):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_simulations(output_file, *, sizes, probabilities, realisations=200,
                    basis='Y', periodic=False, seed=20261104, checkpoint_every=10,
                    progress=True):
    """Serial, resumable sampling; one writer per output file.

    Trial RNG is SeedSequence([seed, basis code, L, p-index, trial-index]).
    Increasing `realisations` appends samples; existing samples are retained.
    Restart must use identical axes, boundary conditions, basis and seed.
    """
    sizes = [int(L) for L in sizes]
    probabilities = [float(p) for p in probabilities]
    basis = basis.upper()
    if (not sizes or len(set(sizes))!=len(sizes) or min(sizes)<2
        or not probabilities or len(set(probabilities))!=len(probabilities)
        or any(not 0 <= p <= 1 for p in probabilities)
        or basis not in ('X','Y','Z') or realisations<2
        or checkpoint_every<1 or seed<0):
        raise ValueError('Invalid simulation configuration.')
    config = dict(algorithm=ALGORITHM, basis=basis, periodic=bool(periodic),
                  L_values=sizes, p_values=probabilities, seed=int(seed),
                  cut='x < L//2', lcc_tie_break='uniform',
                  entropy='rank_GF2_cross_cut', outcomes='signs omitted; entropy invariant')
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open('rb') as handle:
            data = pickle.load(handle)
        if data.get('config') != config:
            raise ValueError('Existing file has a different protocol/configuration. Use a new filename.')
    else:
        data = dict(schema_version=1, config=config, L_values=np.array(sizes),
                    p_values=np.array(probabilities), results={})
    code = {'X':1,'Y':2,'Z':3}[basis]
    for L in sizes:
        adjacency = square_adjacency(L, periodic)
        for pi, p in enumerate(probabilities):
            result = data['results'].setdefault(p, {}).setdefault(L, dict(
                dv_entropies=[], LCC_sizes=[], measured_counts=[], trial_ids=[]))
            counts = [len(result[k]) for k in ('dv_entropies','LCC_sizes','measured_counts','trial_ids')]
            if len(set(counts))!=1 or result['trial_ids']!=list(range(counts[0])):
                raise ValueError('Inconsistent checkpoint sample records.')
            start = counts[0]
            for trial in range(start, realisations):
                rng = np.random.default_rng(np.random.SeedSequence([seed,code,L,pi,trial]))
                sample = simulate_realisation(L,p,basis,rng,periodic,adjacency)
                result['dv_entropies'].append(sample['entropy'])
                result['LCC_sizes'].append(sample['lcc_size'])
                result['measured_counts'].append(sample['measured_count'])
                result['trial_ids'].append(trial)
                if (trial+1)%checkpoint_every==0:
                    _atomic_save(path,data)
            if len(result['dv_entropies']) != start or not path.exists():
                _atomic_save(path,data)
            if progress:
                print(f'{basis}: L={L}, p={p:.6f}, N={len(result["dv_entropies"])}',flush=True)
    return data


def load_statistics(path):
    with Path(path).open('rb') as handle:
        raw = pickle.load(handle)
    if raw.get('config',{}).get('algorithm')!=ALGORITHM:
        raise ValueError('Not a corrected Pauli raw-data file.')
    sizes, ps = np.sort(raw['L_values']),np.sort(raw['p_values'])
    mean = np.full((len(sizes),len(ps)),np.nan)
    sem = np.full_like(mean,np.nan)
    counts = np.zeros_like(mean,dtype=int)
    for i,L in enumerate(sizes):
        for j,p in enumerate(ps):
            samples = np.asarray(raw['results'].get(p,{}).get(L,{}).get('dv_entropies',[]))
            counts[i,j] = len(samples)
            if len(samples): mean[i,j] = samples.mean()
            if len(samples)>1: sem[i,j] = samples.std(ddof=1)/np.sqrt(len(samples))
    return dict(L=sizes,p=ps,mean=mean,sem=sem,counts=counts,config=raw['config'])
