from ._predict import Predictor
import numpy as np


class axionemulator:
    def __init__(self, version=2):
        #self.predictor = Predictor.from_path(f"/mn/stornext/d8/data/hansw/Dennis/axionemu_test/emulators/lightning_logs/version_{version}")
        self.predictor = Predictor.from_path(f"./emulators/lightning_logs/version_{version}")

    def __call__(
        self,
        Omega_cdm=0.2637,
        A_s=2.1e-9,
        f=0.6,
        m=1e-25,
        z=0.0,
        k=np.logspace(np.log10(0.01), np.log10(10), num=100),
        return_tensors: bool = False,
    ):
        # Convert all parameters to arrays
        Omega_cdm, A_s, f, m, z, k = np.broadcast_arrays(
            Omega_cdm, A_s, f, m, z, k
        )

        # Construct input array
        inputs = np.stack(
            [
                Omega_cdm,
                np.log10(A_s),
                np.log10(f),
                np.log10(m),
                z,
                np.log10(k),
            ],
            axis=-1,
        )

        # Flatten for prediction
        original_shape = inputs.shape[:-1]
        inputs = inputs.reshape(-1, 6)
        
        # Predict and restore original shape
        predictions = self.predictor(
            inputs, return_tensors=return_tensors
        )

        return predictions.reshape(original_shape)


if __name__ == "__main__":
    import numpy as np

    emulator = axionemulator(version=2)
    params = np.random.rand(10, 6)  # Example input parameters
    predictions = emulator(params)
    print(predictions)

