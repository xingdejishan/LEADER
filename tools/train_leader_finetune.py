import sys

from tools.train_magic_nclt import main


if __name__ == "__main__":
    if any(arg == "--fusion-mode" or arg.startswith("--fusion-mode=") for arg in sys.argv[1:]):
        raise SystemExit("This entry point fixes fusion-mode to lidar_finetune")
    sys.argv.extend(["--fusion-mode", "lidar_finetune"])
    main()
