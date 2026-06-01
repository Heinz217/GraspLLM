# Stage-3 entry point. Forwards CLI args to train._train.
import sys

sys.path.append(".")
sys.path.append("./utils")

from train import _train

if __name__ == "__main__":
    _train()
