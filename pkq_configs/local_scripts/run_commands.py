"""Command(s) each run executed: uncommented PKQ_RUN_COMMAND lines of the pkqbatch1.sh copied into its backup.

    python3 -I pkq_configs/local_scripts/run_commands.py <log.tex> <backup_root>
"""

import os
import re
import sys

tex, root = sys.argv[1], sys.argv[2]
runs = sorted(set(re.findall(r"(2026\d{4})\\?_(\d{6})", open(tex).read())))
for d, t in runs:
    ts = f"{d}_{t}"
    path = os.path.join(root, ts, "pkq_configs", "remote", "pkqbatch1.sh")
    if not os.path.exists(path):
        print(f"{ts}: no pkqbatch1.sh")
        continue
    lines = open(path).read().split("\n")
    cmds = []
    for i, line in enumerate(lines):
        if line.startswith("PKQ_RUN_COMMAND="):
            j, cmd = i, line
            while cmd.rstrip().endswith("\\"):
                j += 1
                cmd = cmd.rstrip()[:-1] + " " + lines[j].strip()
            gpu = lines[i - 1] if i > 0 and lines[i - 1].startswith("PKQ_GPU_NUM=") else ""
            cmds.append((gpu + " " + cmd).strip())
    print(f"{ts}:")
    for c in cmds or ["(none uncommented)"]:
        print(f"    {c}")
