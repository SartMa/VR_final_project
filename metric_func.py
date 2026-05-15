def hit_at_k(ret, rel_id, k, nr=None):
    return 1.0 if rel_id in ret[:k] else 0.0

def recall_at_k(ret, rel_id, k, nr):
    return sum(1 for r in ret[:k] if r == rel_id) / nr if nr else 0.0

def ndcg_at_k(ret, rel_id, k, nr):
    dcg   = sum(1/np.log2(i+2) for i,r in enumerate(ret[:k]) if r == rel_id)
    ideal = sum(1/np.log2(i+2) for i in range(min(nr, k)))
    return dcg/ideal if ideal else 0.0

def map_at_k(ret, rel_id, k, nr):
    h, s = 0, 0.0
    for i, r in enumerate(ret[:k]):
        if r == rel_id: h += 1; s += h/(i+1)
    return s/nr if nr else 0.0
