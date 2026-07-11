def blockscore(q, k, out):
    out.copy_(q @ k.transpose(-1, -2))
