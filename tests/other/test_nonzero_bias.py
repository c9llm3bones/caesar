import pickle, numpy as np
p = pickle.load(open("salad_weights/ae_params/small_inner-200k.jax","rb"))

def maxabs(x): return float(np.max(np.abs(np.array(x))))

for k,v in sorted(p.items()):
    if k.startswith("structure_autoencoder/encoder/~prepare_features/") and isinstance(v, dict) and "b" in v:
        print(k, "b maxabs =", maxabs(v["b"]), "shape", np.array(v["b"]).shape)
