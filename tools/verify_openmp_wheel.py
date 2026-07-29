"""Reject a wheel which bundles or directly imports an OpenMP runtime."""

import re
import sys
import zipfile
from pathlib import Path

OPENMP_NAMES = (b"libomp", b"libiomp", b"vcomp")
DLL_PATTERN = re.compile(rb"(?i)(?:libi?omp[^/\x00]*|vcomp\d*)\.dll")


def main(wheel_name: str) -> int:
    wheel = Path(wheel_name)
    problems = []
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            lowered = member.filename.lower().encode()
            if lowered.endswith(b".dll") and any(x in lowered for x in OPENMP_NAMES):
                problems.append(f"bundled runtime: {member.filename}")
            if lowered.endswith((b".dll", b".pyd")):
                for match in sorted(set(DLL_PATTERN.findall(archive.read(member)))):
                    problems.append(
                        f"{member.filename} imports "
                        f"{match.decode('ascii', errors='replace')}"
                    )

    if problems:
        print("Unsafe OpenMP dependency found:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"Verified: {wheel} has no native OpenMP runtime dependency.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} WHEEL")
    raise SystemExit(main(sys.argv[1]))
