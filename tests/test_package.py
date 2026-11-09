import os
import subprocess
import sys


def test_import_pins_triton_fp_fusion():
    """Importing the package fixes Triton's FMA fusion on, whatever the environment says (D037)."""
    env = os.environ | {"TRITON_DEFAULT_FP_FUSION": "0"}
    out = subprocess.run(
        [sys.executable, "-c", "import os, seqrec; print(os.environ['TRITON_DEFAULT_FP_FUSION'])"],
        env=env, capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert out.stdout.strip() == "1"
