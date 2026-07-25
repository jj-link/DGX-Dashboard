#!/usr/bin/env python3
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("container")
parser.add_argument("output")
parser.add_argument("--since", default="1s")
args = parser.parse_args()

with open(args.output, "xb", buffering=0) as output:
    completed = subprocess.run(
        ["docker", "logs", "--since", args.since, "-f", args.container],
        stdout=output,
        stderr=subprocess.STDOUT,
        check=False,
    )
raise SystemExit(completed.returncode)
