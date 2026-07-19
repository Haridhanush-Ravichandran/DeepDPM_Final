"""
pairwise_constraint_overrides.py

Adds pairwise (must-link / cannot-link) constraint overrides on top of the
Bayesian, Hastings-ratio-driven split/merge decisions used by ClusterNet /
DeepDPM (see split_merge_operations.py).

--------------------------------------------------------------------------
Rules implemented (as specified)
--------------------------------------------------------------------------
4.1  Two elements that must be in different clusters end up in the SAME
     cluster                      -> FORCE SPLIT of that cluster.
4.2  A merge would put two elements that must be different into the SAME
     cluster                      -> PREVENT that merge.
4.3  No constraint spans the two candidate clusters (contrastive loss of
     the merged result is 0 / unchanged) -> FORCE MERGE.
4.4  A split is statistically accepted for a cluster that has 0
     contrastive loss (no internal cannot-link violation) -> PREVENT the
     split.

Because 4.1/4.4 and 4.2/4.3 are complementary, together they cover every
case: whenever a constraint signal exists for a cluster (or cluster pair),
it *replaces* the statistical decision rather than merely nudging it.
Set `hard_override=False` if you'd rather fall back to the statistical
(Hastings-ratio) decision in the "no violation" branch instead of forcing
one way or the other.

--------------------------------------------------------------------------
Your data: only cannot-link (negative) pairs
--------------------------------------------------------------------------
`pair_labels` follows the same convention already used in
`ClusterNetModel.cluster_net_pretraining` (`z`): 1 = must-link (same
cluster), 0 = cannot-link (different cluster). Since you only have
negative pairs, `pair_labels` will be all zeros -- the must-link code
path below is simply inert for you, but it's there if that ever changes.
"""

import torch


# ---------------------------------------------------------------------
# 1. Collect per-pair cluster assignments (call once per epoch)
# ---------------------------------------------------------------------

def get_pair_assignments(pair_resp_a, pair_resp_b, pair_labels, must_link_label=1):
    """
    Args:
        pair_resp_a, pair_resp_b: [N_pairs, K] cluster responsibilities
            (softmax logits from `cluster_net`) for the two elements of
            every pair seen this epoch, accumulated in matching order.
        pair_labels: [N_pairs] tensor, 1 = must-link, 0 = cannot-link.
        must_link_label: which label value marks a must-link pair.

    Returns a dict with hard (argmax) cluster assignments split by
    constraint type:
        {
          "cl_a", "cl_b": cannot-link pair assignments (what you have),
          "ml_a", "ml_b": must-link pair assignments (empty for you now),
        }
    """
    assign_a = pair_resp_a.argmax(-1)
    assign_b = pair_resp_b.argmax(-1)

    cannot_link_mask = pair_labels != must_link_label
    must_link_mask = pair_labels == must_link_label

    return {
        "cl_a": assign_a[cannot_link_mask],
        "cl_b": assign_b[cannot_link_mask],
        "ml_a": assign_a[must_link_mask],
        "ml_b": assign_b[must_link_mask],
    }


# ---------------------------------------------------------------------
# 2. Violation checks
# ---------------------------------------------------------------------

def cluster_cannot_link_violations(k, cl_a, cl_b):
    """4.1 / 4.4 - does cluster k contain a cannot-link pair (both
    endpoints currently assigned to k)?

    Returns (has_violation: bool, n_violations: int).
    """
    if cl_a.numel() == 0:
        return False, 0
    both_in_k = (cl_a == k) & (cl_b == k)
    n = int(both_in_k.sum().item())
    return n > 0, n


def merge_cannot_link_violation(k1, k2, cl_a, cl_b):
    """4.2 / 4.3 - would merging k1 and k2 place a cannot-link pair into
    the same cluster (one endpoint in k1, the other in k2)?
    """
    if cl_a.numel() == 0:
        return False
    cross = ((cl_a == k1) & (cl_b == k2)) | ((cl_a == k2) & (cl_b == k1))
    return bool(cross.any().item())


# ---------------------------------------------------------------------
# 3. Decision resolvers - call these inside split_rule / merge_rule
# ---------------------------------------------------------------------

def resolve_split_decision(k, stat_decision, cl_a, cl_b, hard_override=False):
    """
    stat_decision: the Hastings-ratio decision from log_Hastings_ratio_split.

    4.1: violation inside k                -> force split (True). This is
         hard evidence a split is needed, so it always overrides the stats.
    4.4: stats say split, but no cannot-link pair happens to fall inside k.
         With sparse/negative-only pair coverage, "no violation found" is
         weak evidence (could just mean the pair sample didn't touch this
         region) -- NOT proof the split is unwarranted. Default
         (hard_override=False) trusts the statistics here; set
         hard_override=True to force-reject instead (rule 4.4 as originally
         specified), but note this caused severe under-splitting in
         practice (see epoch-125 divergence in testing, where this rule
         blocked a split the Hastings ratio strongly supported, H=3662,
         and the run never recovered above K=3 for the rest of training).
    """
    has_violation, n_violations = cluster_cannot_link_violations(k, cl_a, cl_b)

    if has_violation:
        return True, f"forced_split(rule 4.1, {n_violations} violating pair(s))"

    if stat_decision:
        if hard_override:
            return False, "prevented_split(rule 4.4, zero contrastive loss)"
        return True, "stat_split(no constraint signal, hard_override=False)"

    return False, "stat_no_split"


# ---------------------------------------------------------------------
# 4. Seeded-bipartition fallback for degenerate subcluster proposals
# ---------------------------------------------------------------------
#
# The subclustering net sometimes proposes a degenerate split for cluster
# k (e.g. all N points on one side, 0 on the other), which split_rule
# rejects outright ("subclusters too small") before rule 4.1 ever gets a
# chance to fire -- even when k has real cannot-link violations. This
# section lets split_rule fall back to a constraint-seeded 2-means split
# instead of giving up, whenever that happens.

def get_cannot_link_codes(pair_codes_a, pair_codes_b, pair_labels, must_link_label=1):
    """
    Companion to get_pair_assignments: extracts the raw embeddings for
    cannot-link pairs (same mask, same order, so index i here lines up
    with index i in get_pair_assignments()['cl_a'/'cl_b']).

    Args:
        pair_codes_a, pair_codes_b: [N_pairs, D] embeddings (codes_a /
            codes_b) for every pair seen this epoch, accumulated in the
            same order as pair_resp_a/pair_resp_b.
        pair_labels: [N_pairs], 1 = must-link, 0 = cannot-link.

    Returns:
        cl_codes_a, cl_codes_b: [N_cannot_link, D] embeddings.
    """
    cannot_link_mask = pair_labels != must_link_label
    return pair_codes_a[cannot_link_mask], pair_codes_b[cannot_link_mask]


def get_violation_seeds(k, cl_a, cl_b, cl_codes_a, cl_codes_b):
    """Embeddings of the two endpoints of every cannot-link pair that
    currently falls entirely inside cluster k. These anchor the two
    sides of the forced split: every seed_a point must end up opposite
    its paired seed_b point.
    """
    if cl_a.numel() == 0:
        empty = cl_codes_a[:0]
        return empty, empty
    mask = (cl_a == k) & (cl_b == k)
    return cl_codes_a[mask], cl_codes_b[mask]


def seeded_bipartition(codes_k, seeds_a, seeds_b, n_iters=10):
    """Constrained 2-means: initialise the two centroids from the
    cannot-link seed pairs (side 1 = mean of seeds_a, side 2 = mean of
    seeds_b) instead of randomly/from the (degenerate) net proposal,
    then run a few rounds of nearest-centroid assignment + recentre.

    Args:
        codes_k: [N, D] embeddings of all points currently in cluster k.
        seeds_a, seeds_b: [M, D] embeddings from get_violation_seeds.
            Requires M > 0 (only call this when has_violation is True).

    Returns:
        assignment: [N] bool tensor, True -> side 2, False -> side 1.
        mu1, mu2: [D] resulting centroids.
        degenerate: bool, True if the constrained split still collapsed
            (all points landed on one side) -- caller should treat this
            like the original "too small" rejection.
    """
    mu1 = seeds_a.mean(dim=0)
    mu2 = seeds_b.mean(dim=0)
    assignment = torch.zeros(len(codes_k), dtype=torch.bool)

    for _ in range(n_iters):
        d1 = ((codes_k - mu1) ** 2).sum(dim=1)
        d2 = ((codes_k - mu2) ** 2).sum(dim=1)
        assignment = d2 < d1
        if assignment.all() or (~assignment).all():
            return assignment, mu1, mu2, True
        new_mu1 = codes_k[~assignment].mean(dim=0)
        new_mu2 = codes_k[assignment].mean(dim=0)
        if torch.allclose(new_mu1, mu1, atol=1e-6) and torch.allclose(new_mu2, mu2, atol=1e-6):
            mu1, mu2 = new_mu1, new_mu2
            break
        mu1, mu2 = new_mu1, new_mu2

    return assignment, mu1, mu2, False


def resolve_merge_decision(k1, k2, stat_decision, cl_a, cl_b, hard_override=False):
    """
    4.2: merge would violate a cannot-link constraint -> force NOT merge.
         Hard evidence, always overrides the stats.
    4.3: no cannot-link pair spans k1/k2. As with 4.4, this is weak/absent
         evidence rather than proof the merge is safe -- default trusts
         the statistics instead of force-merging. Set hard_override=True
         to force-merge instead (rule 4.3 as originally specified).
    """
    violates = merge_cannot_link_violation(k1, k2, cl_a, cl_b)

    if violates:
        return False, "prevented_merge(rule 4.2, cannot-link violation)"

    if hard_override:
        return True, "forced_merge(rule 4.3, no constraint between clusters)"

    return stat_decision, "stat_merge(no constraint signal, hard_override=False)"
