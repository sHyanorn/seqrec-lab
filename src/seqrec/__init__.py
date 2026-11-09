import os

# Triton fuses multiplies and adds into FMA by default in the locked triton-windows 3.9.
# Fixed here, before torch or Triton load, so that an upgrade or an inherited environment
# cannot change compiled numerics unnoticed (D037).
os.environ["TRITON_DEFAULT_FP_FUSION"] = "1"
