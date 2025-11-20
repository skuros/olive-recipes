from pathlib import Path
import subprocess
import sys

from sam2_1_hiera_small import export_sam2_to_onnx


def main() -> None:
    # Ensure encoder ONNX exists before running Olive
    export_sam2_to_onnx()

    config_path = Path(__file__).with_name("sam2.1_hiera_small_ov_config.json")
    cmd = [sys.executable, "-m", "olive.cli.launcher", "run", "--config", str(config_path)]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
from pathlib import Path
import subprocess
import sys

from sam2_1_hiera_small import export_sam2_to_onnx


def main() -> None:
    # Ensure encoder ONNX exists
    export_sam2_to_onnx()

    # Run Olive on the config in this directory
    config_path = Path(__file__).with_name("sam2.1_hiera_small_ov_config.json")
    cmd = [sys.executable, "-m", "olive.cli.launcher", "run", "--config", str(config_path)]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
