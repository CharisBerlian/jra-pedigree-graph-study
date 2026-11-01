"""Frozen-space ancestor-only encoding, independent of model fitting."""
import math


def encode(horse, parents, vectors, centers, dimensions=64):
    assert horse and len(centers)==64 and all(len(c)==dimensions for c in centers)
    frontier={horse}
    for depth in range(1,4):
        frontier={parents.get((child,relation)) for child in frontier for relation in ('sire','dam')}
        frontier.discard(None);frontier.discard('');frontier.discard(horse)
        available=sorted(node for node in frontier if node in vectors)
        if available:
            assert all(len(vectors[node])==dimensions for node in available)
            vec=tuple(sum(float(vectors[node][i]) for node in available)/len(available) for i in range(dimensions))
            assert all(math.isfinite(v) for v in vec)
            distances=[sum((vec[j]-float(center[j]))**2 for j in range(dimensions)) for center in centers]
            assert all(math.isfinite(x) for x in distances)
            return {'embedding_status':'fallback','fallback_depth':depth,'ancestor_count':len(available),
                    'embedding_missing':0,'graph_cluster':str(min(range(len(centers)),key=lambda i:distances[i])),
                    'vector':vec}
    return {'embedding_status':'missing','fallback_depth':None,'ancestor_count':0,
            'embedding_missing':1,'graph_cluster':'MISSING','vector':(0.0,)*dimensions}
