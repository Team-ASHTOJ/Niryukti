# Benchmark data

`afiro.mps`, `adlittle.mps`, `israel.mps`, and `e226.mps` are public Netlib LP instances, cached from a pinned public HiGHS test-data mirror for an offline demonstration. `manifest.json` records source URLs, revision, sizes and SHA-256 checksums. `MIRROR-LICENSE.txt` retains the mirror's license. No upstream solver code is included with these model files.

Refresh the small suite with `python3 benchmark/download_netlib.py`. The downloader retains download failures in its manifest. It computes checksums of retrieved content; the pinned revision and stored manifest support subsequent integrity checking, rather than representing an independently signed upstream checksum.

Original collections for broader later evaluation:

- Netlib LP: https://www.netlib.org/lp/data/
- MIPLIB: https://miplib.zib.de/
- QPLIB: https://qplib.zib.de/
- Mittelmann benchmarks: https://plato.asu.edu/bench.html

This prototype has not completed those entire collections. Check each dataset's terms before redistributing additional material, and filter unsupported quadratic/discrete classes explicitly.

`refinery_large.json` and `.mps`, when generated, are synthetic and ignored by git. They can be recreated using the commands in the main README. They are not MRPL or Netlib data.
