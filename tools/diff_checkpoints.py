"""Compare two checkpoints structurally, then numerically.

Two TeSCo checkpoints can differ by architecture rather than by training, and a
state dict loaded with ``strict=False`` will not tell you -- it silently skips
whatever does not match. This prints the key sets first, so an architecture
mismatch is visible before any tensor is compared.

    python tools/diff_checkpoints.py a.pt b.pt [--sample 400]
"""
import argparse

import torch


def state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "module"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    # Training checkpoints carry a DDP "module." prefix; strip it so the two
    # formats are comparable.
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in obj.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--sample", type=int, default=400,
                        help="how many shared tensors to compare exactly")
    args = parser.parse_args()

    a, b = state_dict(args.a), state_dict(args.b)
    ka, kb = set(a), set(b)
    shared = sorted(ka & kb)

    print(f"A {args.a}\n  params: {len(ka)}")
    print(f"B {args.b}\n  params: {len(kb)}")
    print(f"shared keys: {len(shared)}   only in A: {len(ka - kb)}   only in B: {len(kb - ka)}")

    if ka - kb or kb - ka:
        print("\n*** ARCHITECTURE MISMATCH -- these are not the same model ***")
        for label, keys, src in (("only in A", ka - kb, a), ("only in B", kb - ka, b)):
            if keys:
                print(f"  {label}:")
                for k in sorted(keys):
                    shape = tuple(src[k].shape) if hasattr(src[k], "shape") else ""
                    print(f"    {k} {shape}")

    step = max(1, len(shared) // args.sample) if args.sample else 1
    checked = sorted(shared[::step][:args.sample])
    equal = sum(1 for k in checked
                if a[k].shape == b[k].shape and torch.equal(a[k].float(), b[k].float()))
    print(f"\nsampled {len(checked)} shared tensors: identical {equal}, differing {len(checked) - equal}")
    if not (ka - kb) and not (kb - ka) and equal == len(checked):
        print("=> same architecture, and every sampled tensor matches: almost certainly the same weights")


if __name__ == "__main__":
    main()
