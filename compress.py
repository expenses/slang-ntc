# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import numpy as np
from pathlib import Path
import time
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--srgb", dest="srgb", default=[], nargs="+", action="append")
parser.add_argument("--nonsrgb", dest="nonsrgb", default=[], nargs="+", action="append")
parser.add_argument("--size", dest="size", type=int, default=1024)
parser.add_argument("--steps", dest="steps", type=int, default=10_000)
args = parser.parse_args()

filenames = [(filename, True) for filenames in args.srgb for filename in filenames] + [
    (filename, False) for filenames in args.nonsrgb for filename in filenames
]

device = spy.create_device(
    spy.DeviceType.automatic,
    enable_debug_layers=True,
    include_paths=[Path(__file__).parent],
)
print(device.features)
module = spy.Module.load_from_file(device, "compress.slang")

# Load some materials.
data_path = Path(__file__).parent
tex = []
loader = spy.TextureLoader(device)
for filepath, is_srgb in filenames:
    opts = spy.TextureLoader.Options()
    opts.load_as_srgb = is_srgb
    opts.generate_mips = True
    tex.append(loader.load_texture(filepath, options=opts))
print(tex)
tex_size = tex[0].width
num_channels = len(tex) * 3


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
        self.biases_grad = spy.Tensor.from_numpy(
            device, np.zeros(outputs).astype("float32")
        )
        self.weights_grad = spy.Tensor.from_numpy(
            device, np.zeros((outputs, inputs)).astype("float32")
        )

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
    def __init__(self, size: int):
        super().__init__(module["LatentTexture"])
        self.size = size

        num_latents = size * size * 3

        self.num_mip_levels = 1
        while size > 4:
            self.num_mip_levels += 1
            size >>= 1
            num_latents += size * size * 3
        print(num_latents)

        # Initialize to random latent texture
        initial_latents = np.random.uniform(0.0, 1.0, num_latents).astype("float16")
        self.texture = spy.Tensor.from_numpy(device, initial_latents)

        # Gradients for the latent texture
        self.texture_grads = spy.Tensor.from_numpy(
            device, np.zeros(num_latents).astype("float32")
        )

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
    def __init__(self, size, num_channels):
        hidden_layer_size = 53
        super().__init__(module[f"Network<{hidden_layer_size}, {num_channels}>"])
        self.latent_texture_1 = LatentTexture(size)
        self.latent_texture_2 = LatentTexture(size)
        self.latent_texture_3 = LatentTexture(size // 2)
        self.latent_texture_4 = LatentTexture(size // 2)
        self.layer0 = NetworkParameters(12, hidden_layer_size)
        self.layer1 = NetworkParameters(hidden_layer_size, hidden_layer_size)
        self.layer2 = NetworkParameters(hidden_layer_size, num_channels)

    # Calls the Slang 'optimize' function for the layer.
    def optimize(self, learning_rate: float, optimize_counter: int):
        self.latent_texture_1.optimize(learning_rate, optimize_counter)
        self.latent_texture_2.optimize(learning_rate, optimize_counter)
        self.latent_texture_3.optimize(learning_rate, optimize_counter)
        self.latent_texture_4.optimize(learning_rate, optimize_counter)
        self.layer0.optimize(learning_rate, optimize_counter)
        self.layer1.optimize(learning_rate, optimize_counter)
        self.layer2.optimize(learning_rate, optimize_counter)


network = Network(args.size, num_channels)

# Train a batch of samples at a time. Smaller batches train faster, but are more "jittery"
# A better strategy is to use small batches at the start, and slowly increase them over time
batch_size = (64, 64)
learning_rate = 0.001

samp = device.create_sampler(spy.SamplerDesc())
print(samp)

for optimize_counter in range(args.steps):
    module.calculate_grads(
        seed=spy.wang_hash(seed=optimize_counter, warmup=2),
        batch_index=spy.grid(batch_size),
        batch_size=spy.int2(batch_size),
        reference=tex,
        network=network,
        samp=samp,
    )
    network.optimize(learning_rate, optimize_counter + 1)

    if optimize_counter == 0:
        start = time.time()

    if optimize_counter % 100 == 0:
        print(f"{optimize_counter}")
    if optimize_counter % 1000 == 0 or optimize_counter == args.steps - 1:
        loss_output = spy.Tensor.from_numpy(device, np.zeros((1,)).astype("float32"))
        module.sum_loss(
            pixel=spy.grid((tex_size, tex_size)),
            resolution=tex_size,
            network=network,
            reference=tex,
            total=loss_output,
            samp=samp,
        )
        mae = loss_output.to_numpy()[0] / tex_size / tex_size / num_channels
        psnr = 20 * np.log10(1.0 / mae) if mae > 0 else float("inf")
        print(f"Loss: {mae:.8f} PSNR: {psnr:.4f} dB")
end = time.time()
print(end - start)

for mip in range(tex[0].mip_count):
    output = spy.Tensor.from_numpy(
        device,
        np.zeros((tex_size >> mip, tex_size >> mip, num_channels)).astype("float16"),
    )
    module.render(
        pixel=spy.call_id(),
        resolution=tex_size >> mip,
        network=network,
        mip=mip,
        _result=output,
    )
    output = output.to_numpy()
    outputs = [output[:, :, i * 3 : (i + 1) * 3] for i in range(len(tex))]
    for i, (_, is_srgb) in enumerate(filenames):
        spy.Bitmap(outputs[i]).convert(
            component_type=spy.Bitmap.ComponentType.uint8, srgb_gamma=is_srgb
        ).write(f"{i}_m{mip}.png")
