"""
Run this on any candidate GPU machine BEFORE committing a real measurement
session to it (Kaggle notebook, college cluster node, etc.). Confirms: GPU
identity, driver version, and whether the hardware energy counter this
project's methodology depends on is actually available.

Background: Kaggle's T4 was originally the planned platform (confirmed
Turing-generation, NVML counter supported) but Session 1 errored out
mid-run 2026-08-13; the project switched to the college GPU cluster for
all measurement sessions. The cluster's GPU model/NVML support has NOT
been verified yet — run this the moment you have access, before spending
any real GPU time. Volta-generation or newer (V100, T4, A100, RTX 20-series+,
L4, H100, ...) is required; Pascal or older (P100, P40, GTX 10-series, K80)
lacks nvmlDeviceGetTotalEnergyConsumption entirely.

Install: pip install nvidia-ml-py
Run:     python check_nvml_energy_support.py
"""
import pynvml

pynvml.nvmlInit()
count = pynvml.nvmlDeviceGetCount()
driver = pynvml.nvmlSystemGetDriverVersion()
if isinstance(driver, bytes):
    driver = driver.decode()

print(f"Driver version: {driver}")
print(f"GPU count: {count}")

for i in range(count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode()
    print(f"\nGPU {i}: {name}")

    try:
        energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        print(f"  [OK]   nvmlDeviceGetTotalEnergyConsumption supported "
              f"(current reading: {energy_mj} mJ) -> use this as the "
              f"primary meter, matches the project's Session 1 protocol as-is.")
    except pynvml.NVMLError as e:
        print(f"  [FAIL] nvmlDeviceGetTotalEnergyConsumption NOT supported: {e}")
        print(f"         -> this GPU cannot use the primary hardware-counter "
              f"method. Fall back to integrating nvmlDeviceGetPowerUsage at "
              f"20Hz (see RESEARCH_PLAN.md SS3), and disclose it as a WARN "
              f"in the paper's threats-to-validity section.")

    try:
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        print(f"  [OK]   nvmlDeviceGetPowerUsage (fallback path) supported "
              f"(current reading: {power_mw} mW)")
    except pynvml.NVMLError as e:
        print(f"  [FAIL] nvmlDeviceGetPowerUsage NOT supported either: {e}")
        print(f"         -> no viable energy-measurement path on this GPU. "
              f"Do not run Session 1 here.")

pynvml.nvmlShutdown()
