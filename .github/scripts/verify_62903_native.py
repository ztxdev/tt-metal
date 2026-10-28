"""Exercise source-built TTNN; never load the saved LD_PRELOAD component."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as data:
        for chunk in iter(lambda: data.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["wormhole_b0", "blackhole"], required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    build = repo / "build_62903_ci"
    artifacts = repo / "artifacts"
    artifacts.mkdir(exist_ok=True)
    config = (build / "CMakeCache.txt").read_text()
    for setting in ["ENABLE_DISTRIBUTED:BOOL=ON", "TT_UMD_BUILD_SIMULATION:BOOL=ON", "CMAKE_BUILD_TYPE:STRING=Release"]:
        if setting not in config:
            raise SystemExit("Required build setting not found: " + setting)
    for name in ["_ttnn.so", "_ttnncpp.so"]:
        src = build / "ttnn" / name
        if not src.is_file():
            raise SystemExit("Missing newly built extension: " + str(src))
        dst = repo / "ttnn/ttnn" / name
        if dst.is_symlink():
            if dst.resolve() != src.resolve():
                raise SystemExit("Extension points outside this build: " + str(dst))
        elif dst.exists():
            raise SystemExit("Refusing to replace existing extension: " + str(dst))
        else:
            dst.symlink_to(src)
    version = (repo / "tt_metal/ttsim-version").read_text().strip()
    sim = artifacts / ("sim-" + args.arch)
    sim.mkdir(exist_ok=True)
    asset = "libttsim_wh.so" if args.arch == "wormhole_b0" else "libttsim_bh.so"
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            version,
            "--repo",
            "tenstorrent/ttsim",
            "--pattern",
            asset,
            "--dir",
            str(sim),
            "--clobber",
        ],
        check=True,
    )
    (sim / asset).replace(sim / "libttsim.so")
    descriptor = "wormhole_b0_80_arch.yaml" if args.arch == "wormhole_b0" else "blackhole_140_arch.yaml"
    (sim / "soc_descriptor.yaml").write_bytes((repo / "tt_metal/soc_descriptors" / descriptor).read_bytes())
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    env.pop("TT_METAL_CACHE", None)
    env.update(
        PYTHONPATH=str(repo / "ttnn") + ":" + str(repo),
        LD_LIBRARY_PATH=":".join(
            sorted({str(p.parent) for p in build.rglob("*.so")})
            + [part for part in env.get("LD_LIBRARY_PATH", "").split(":") if part]
        ),
        TT_METAL_HOME=str(repo),
        TT_METAL_RUNTIME_ROOT=str(repo),
        TT_METAL_SIMULATOR_HOME=str(sim),
        TT_METAL_SIMULATOR=str(sim / "libttsim.so"),
        TT_METAL_SLOW_DISPATCH_MODE="1",
        TT_METAL_DISABLE_SFPLOADMACRO="1",
        ARCH_NAME=args.arch,
        HWLOC_COMPONENTS="-x86",
    )
    proof = """import json, pathlib, ttnn
build = pathlib.Path('build_62903_ci').resolve()
assert pathlib.Path(ttnn._ttnn.__file__).resolve().is_relative_to(build)
maps = pathlib.Path('/proc/self/maps').read_text().splitlines()
backend = [line.split()[-1] for line in maps if '_ttnncpp.so' in line]
assert backend and all(pathlib.Path(p).resolve().is_relative_to(build) for p in backend), backend
print(json.dumps({'package': ttnn.__file__, 'extension': str(pathlib.Path(ttnn._ttnn.__file__).resolve()), 'backend': sorted(set(backend))}))
"""
    loaded = subprocess.check_output([sys.executable, "-c", proof], cwd=repo, env=env, text=True)
    (artifacts / ("loaded-" + args.arch + ".txt")).write_text(loaded)
    print(loaded, flush=True)
    info = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "arch": args.arch,
        "sim_version": version,
        "sim_sha256": digest(sim / "libttsim.so"),
        "extensions": {p.name: digest(p) for p in (build / "ttnn").glob("_ttnn*.so")},
    }
    # Compatibility workaround required by the pinned Wormhole simulator, not a submitted kernel change.
    needle = b"TT_UNPACR(SrcA, 0, 0, context_id, 0, 1 /* Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 1, 0, 0, 0, 1);"
    backups = {}
    try:
        if args.arch == "wormhole_b0":
            for header in (repo / "tt_metal/tt-llk").rglob("cunpack_common.h"):
                if "wormhole_b0" in str(header):
                    original = header.read_bytes()
                    changed = b"".join(line for line in original.splitlines(True) if needle not in line)
                    if changed != original:
                        backups[header] = original
                        header.write_bytes(changed)
        info["runtime_header_workarounds"] = [str(p.relative_to(repo)) for p in backups]
        (artifacts / ("manifest-" + args.arch + ".json")).write_text(json.dumps(info, indent=2) + "\n")
        selection = json.loads((repo / ".github/scripts/62903-selection.json").read_text())[args.arch]
        # Root conftest.py appends a timestamp to JUnit filenames in GitHub Actions.
        # Only accept a report created by this invocation, never a stale artifact.
        reports_before = {p: digest(p) for p in artifacts.glob(args.arch + "*.xml")}
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-xq", *selection, "--junitxml=" + str(artifacts / (args.arch + ".xml"))],
            cwd=repo,
            env=env,
        )
    finally:
        for header, original in backups.items():
            header.write_bytes(original)
    if result.returncode == 0:
        reports = [
            p for p in artifacts.glob(args.arch + "*.xml") if reports_before.get(p) != digest(p)
        ]
        if len(reports) != 1:
            raise SystemExit("Expected exactly one fresh JUnit report: " + str(reports))
        print("Validating JUnit report: " + str(reports[0]), flush=True)
        suites = ET.parse(reports[0]).getroot().iter("testsuite")
        totals = {key: 0 for key in ["tests", "failures", "errors", "skipped"]}
        for suite in suites:
            for key in totals:
                totals[key] += int(suite.get(key, 0))
        expected = 94 if args.arch == "wormhole_b0" else 62
        if totals != dict(tests=expected, failures=0, errors=0, skipped=0):
            raise SystemExit("Unexpected regression coverage: " + str(totals))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
