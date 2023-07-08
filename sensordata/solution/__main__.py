import argparse
import os
import pathlib
import sys

# ----------------------------Setting root path------------------------------------
### setting path
## set the base path
os.chdir(pathlib.Path(os.path.realpath(__file__)).parents[1])
## set path to sys for Lambda function
sys.path.insert(0, os.getcwd())


# ----------------------------Loading environment should be first to be loaded------------------
def __parseArguments():
    parser = argparse.ArgumentParser(description="load the component")

    parser.add_argument(
        "--module",
        dest="MODULE",
        help="Put the name of module, module can be sensordata or something else",
        default="sensordata",
    )

    parser.add_argument(
        "--actions",
        dest="ACTIONS",
        choices=[
            "validate",
            "train",
            "predict",
        ],
        help="Which action or action pipeline to perform",
        nargs="+",
        default="[predict]",
    )

    args = parser.parse_args()

    return args


args = __parseArguments()

# ----------------------- Service execution START--------------------------------
from solution.workflows.SensorData import predict, train, validate


class Creator:
    _services = {
        "sensordata": {
            "validate": validate,
            "train": train,
            "predict": predict,
        }
    }

    def __init__(self, serviceNames: str):
        self.service = self._services[serviceNames]

    def run(self, action):
        self.service[action]()


if __name__ == "__main__":
    service = Creator(args.MODULE)
    for action in args.ACTIONS:
        service.run(action)
