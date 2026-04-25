# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import numpy as np
from pathlib import Path

device = spy.create_device(spy.DeviceType.automatic, enable_debug_layers=True, include_paths=[Path(__file__).parent])
module = spy.Module.load_from_file(device, "compress.slang")

# Load some materials.
data_path = Path(__file__).parent
image = spy.Tensor.load_from_image(device, data_path.joinpath("slangstars.png"), linearize=True)


class NetworkParameters(spy.InstanceList):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        self.inputs = inputs
        self.outputs = outputs

        # Biases and weights for the layer.
        self.biases = spy.Tensor.from_numpy(device, np.zeros(outputs).astype("float32"))
        self.weights = spy.Tensor.from_numpy(
            device, np.random.uniform(-0.5, 0.5, (outputs, inputs)).astype("float32")
        )

        # Gradients for the biases and weights.
        self.biases_grad = spy.Tensor.zeros_like(self.biases)
        self.weights_grad = spy.Tensor.zeros_like(self.weights)

        # Temp data for Adam optimizer.
        self.m_biases = spy.Tensor.zeros_like(self.biases)
        self.m_weights = spy.Tensor.zeros_like(self.weights)
        self.v_biases = spy.Tensor.zeros_like(self.biases)
        self.v_weights = spy.Tensor.zeros_like(self.weights)

    # Calls the Slang 'optimize' function for biases and weights
    def optimize(self, learning_rate: float, optimize_counter: int):
        module.optimizer_step(
            self.biases,
            self.biases_grad,
            self.m_biases,
            self.v_biases,
            learning_rate,
            optimize_counter,
        )
        module.optimizer_step(
            self.weights,
            self.weights_grad,
            self.m_weights,
            self.v_weights,
            learning_rate,
            optimize_counter,
        )


class LatentTexture(spy.InstanceList):
    def __init__(self, width: int, height: int, num_latents: int):
        super().__init__(module[f"LatentTexture<{num_latents}>"])
        self.width = width
        self.height = height
        self.num_latents = num_latents

        # Initialize to random latent texture
        initial_latents = np.random.uniform(0.0, 1.0, (height, width, num_latents)).astype("float32")
        self.texture = spy.Tensor.from_numpy(device, initial_latents)

        # Gradients for the latent texture
        self.texture_grads = spy.Tensor.zeros_like(self.texture)

        # Temp data for Adam optimizer.
        self.m_texture = spy.Tensor.zeros_like(self.texture)
        self.v_texture = spy.Tensor.zeros_like(self.texture)

    # Calls the Slang 'optimize' function for biases and weights
    def optimize(self, learning_rate: float, optimize_counter: int):
        module.optimizer_step(
            self.texture,
            self.texture_grads,
            self.m_texture,
            self.v_texture,
            learning_rate,
            optimize_counter,
        )


class Network(spy.InstanceList):
    def __init__(self):
        super().__init__(module["Network"])
        self.latent_texture = LatentTexture(32, 32, 4)
        self.layer0 = NetworkParameters(4, 32)
        self.layer1 = NetworkParameters(32, 32)
        self.layer2 = NetworkParameters(32, 3)

    # Calls the Slang 'optimize' function for the layer.
    def optimize(self, learning_rate: float, optimize_counter: int):
        self.latent_texture.optimize(learning_rate, optimize_counter)
        self.layer0.optimize(learning_rate, optimize_counter)
        self.layer1.optimize(learning_rate, optimize_counter)
        self.layer2.optimize(learning_rate, optimize_counter)


network = Network()

optimize_counter = 0

# Slang will compile the shaders the first time we call into them (i.e. in the first iteration)
print("Compiling shaders... this may take a while")

res = spy.int2(256, 256)
# Train a batch of samples at a time. Smaller batches train faster, but are more "jittery"
# A better strategy is to use small batches at the start, and slowly increase them over time
batch_size = (64, 64)

learning_rate = 0.001

for optimize_counter in range(100_000):
    module.calculate_grads(
        seed=spy.wang_hash(seed=optimize_counter, warmup=2),
        batch_index=spy.grid(batch_size),
        batch_size=spy.int2(batch_size),
        reference=image,
        network=network,
    )
    network.optimize(learning_rate, optimize_counter+1)

    if optimize_counter % 100 == 0:
        # Show loss between neural texture and reference texture.
        loss_output = spy.Tensor.empty_like(image)
        module.loss(
            pixel=spy.call_id(), resolution=res, network=network, reference=image, _result=loss_output
        )

        mse = np.mean(loss_output.to_numpy())
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        print(f"{optimize_counter} Loss: {mse:.6f} PSNR: {psnr:.4f} dB")

output = spy.Tensor.empty_like(image)
module.render(pixel=spy.call_id(), resolution=res, network=network, _result=output)
spy.Bitmap(output.to_numpy()).convert(component_type=spy.Bitmap.ComponentType.uint8, srgb_gamma=True).write("out.png")
