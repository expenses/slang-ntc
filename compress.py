# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import numpy as np
from pathlib import Path
import time
import sys

device = spy.create_device(spy.DeviceType.automatic, enable_debug_layers=True, include_paths=[Path(__file__).parent])
module = spy.Module.load_from_file(device, "compress.slang")

# Load some materials.
data_path = Path(__file__).parent
image = spy.Tensor.load_from_image(device, sys.argv[1], linearize=True)
print(image.shape)

class NetworkParameters(spy.InstanceList):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        self.inputs = inputs
        self.outputs = outputs

        # Biases and weights for the layer.
        self.biases = spy.Tensor.from_numpy(device, np.zeros(outputs).astype("float16"))
        self.weights = spy.Tensor.from_numpy(
            device, np.random.uniform(-0.5, 0.5, (outputs, inputs)).astype("float16")
        )

        # Gradients for the biases and weights.
        self.biases_grad = spy.Tensor.from_numpy(device, np.zeros(outputs).astype("float32"))
        self.weights_grad = spy.Tensor.from_numpy(device, np.zeros((outputs, inputs)).astype("float32"))

        # Temp data for Adam optimizer.
        self.m_biases = spy.Tensor.zeros_like(self.biases_grad)
        self.m_weights = spy.Tensor.zeros_like(self.weights_grad)
        self.v_biases = spy.Tensor.zeros_like(self.biases_grad)
        self.v_weights = spy.Tensor.zeros_like(self.weights_grad)

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
    def __init__(self, width: int, height: int):
        super().__init__(module["LatentTexture"])
        self.width = width
        self.height = height

        # Initialize to random latent texture
        initial_latents = np.random.uniform(0.0, 1.0, (height, width, 3)).astype("float16")
        self.texture = spy.Tensor.from_numpy(device, initial_latents)

        # Gradients for the latent texture
        self.texture_grads = spy.Tensor.from_numpy(device,np.zeros((height, width, 3)).astype("float32"))

        # Temp data for Adam optimizer.
        self.m_texture = spy.Tensor.zeros_like(self.texture_grads)
        self.v_texture = spy.Tensor.zeros_like(self.texture_grads)

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
    def __init__(self, shape):
        super().__init__(module["Network"])
        self.latent_texture_1 = LatentTexture(shape[0]//4, shape[1]//4)
        self.latent_texture_2 = LatentTexture(shape[0]//4, shape[1]//4)
        self.latent_texture_3 = LatentTexture(shape[0]//8, shape[1]//8)
        self.latent_texture_4 = LatentTexture(shape[0]//8, shape[1]//8)
        self.layer0 = NetworkParameters(12, 32)
        self.layer1 = NetworkParameters(32, 32)
        self.layer2 = NetworkParameters(32, 3)

    # Calls the Slang 'optimize' function for the layer.
    def optimize(self, learning_rate: float, optimize_counter: int):
        self.latent_texture_1.optimize(learning_rate, optimize_counter)
        self.latent_texture_2.optimize(learning_rate, optimize_counter)
        self.latent_texture_3.optimize(learning_rate, optimize_counter)
        self.latent_texture_4.optimize(learning_rate, optimize_counter)
        self.layer0.optimize(learning_rate, optimize_counter)
        self.layer1.optimize(learning_rate, optimize_counter)
        self.layer2.optimize(learning_rate, optimize_counter)

network = Network(image.shape)

res = spy.int2(*image.shape)
# Train a batch of samples at a time. Smaller batches train faster, but are more "jittery"
# A better strategy is to use small batches at the start, and slowly increase them over time
batch_size = (64, 64)
learning_rate = 0.001
loss_output = spy.Tensor.empty_like(image)

for optimize_counter in range(100_000):
    module.calculate_grads(
        seed=spy.wang_hash(seed=optimize_counter, warmup=2),
        batch_index=spy.grid(batch_size),
        batch_size=spy.int2(batch_size),
        reference=image,
        network=network,
    )
    network.optimize(learning_rate, optimize_counter+1)

    if optimize_counter == 0:
        start = time.time()

    if optimize_counter % 100 == 0:
        print(f"{optimize_counter}")
    if optimize_counter % 1000 == 0:
        module.loss(
            pixel=spy.call_id(), resolution=res, network=network, reference=image, _result=loss_output
        )
        mse = np.mean(loss_output.to_numpy())
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        print(f"Loss: {mse:.8f} PSNR: {psnr:.4f} dB")
end = time.time()
print(end - start)

output = spy.Tensor.empty_like(image)
module.render(pixel=spy.call_id(), resolution=res, network=network, _result=output)
spy.Bitmap(output.to_numpy()).convert(component_type=spy.Bitmap.ComponentType.uint8, srgb_gamma=True).write("out.png")

#np.save("out.net", network.latent_texture.texture.to_numpy())
