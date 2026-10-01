from ._predict import Predictor
import numpy as np


class axionemulator:
    def __init__(self, version=2):
        #self.predictor = Predictor.from_path(f"/mn/stornext/d8/data/hansw/Dennis/axionemu_test/emulators/lightning_logs/version_{version}")
        self.predictor = Predictor.from_path(f"./emulators/lightning_logs/version_{version}")

    def __call__(
        self,
        params,
        return_tensors: bool = False,
    ):
        inputs = np.array(params)
        return self.predictor(inputs).reshape(-1)



if __name__ == "__main__":
    import numpy as np

    emulator = axionemulator(version=2)
    params = np.random.rand(10, 6)  # Example input parameters
    predictions = emulator(params)
    print(predictions)

