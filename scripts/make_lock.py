"""Write requirements.lock = pinned transitive closure of requirements.in (installed versions)."""
import re, sys
from importlib import metadata
from packaging.requirements import Requirement

def norm(n): return re.sub(r"[-_.]+", "-", n).lower()
top = [norm(l.strip()) for l in open("requirements.in") if l.strip() and not l.startswith("#")]
seen, stack = {}, list(top)
while stack:
    n = stack.pop()
    if n in seen: continue
    try: d = metadata.distribution(n)
    except metadata.PackageNotFoundError: continue
    seen[n] = d.version
    for r in d.requires or []:
        req = Requirement(r)
        if req.marker and not req.marker.evaluate({"extra": ""}): continue
        stack.append(norm(req.name))
with open("requirements.lock", "w") as f:
    f.write(f"# python {sys.version.split()[0]}\n")
    for k in sorted(seen): f.write(f"{k}=={seen[k]}\n")
print(len(seen), "packages pinned")
