"""Index the rendered clips so the project page can list them without a server.

The page is static, so it cannot read a directory; it fetches this manifest
instead. Labels come from the filename the renderer produces,
<take-prefix>_<object>.mp4, with the object made readable again.
"""
import argparse, json, os, re


def label_of(fname):
    stem = os.path.splitext(fname)[0]
    take, _, obj = stem.partition("_")
    obj = re.sub(r"-(\d+)$", r" \1", obj.replace("-", " ")).strip()
    return f"{obj}  ·  take {take}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="docs/page/clips")
    args = ap.parse_args()
    out = {}
    for d in sorted(os.listdir(args.root)):
        sub = os.path.join(args.root, d)
        if not os.path.isdir(sub):
            continue
        clips = sorted(f for f in os.listdir(sub) if f.endswith(".mp4"))
        out[d] = [{"file": f, "label": label_of(f)} for f in clips]
        print(f"  {d}: {len(clips)} clips")
    path = os.path.join(args.root, "manifest.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
