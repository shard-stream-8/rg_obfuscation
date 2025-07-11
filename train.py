import sys
import yaml

from reinforce.trainer import train as single_model_train
from reinforce.shoggoth_face_trainer import train as shoggoth_face_train
from reinforce.trainer import Config


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r") as f:
        cfg = Config(yaml.safe_load(f))

    if cfg.shoggoth_name is None:
        single_model_train(config_path)
    else:
        shoggoth_face_train(config_path)


if __name__ == "__main__":
    main() 