# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import numpy as np
from pathlib import Path
import time
import sys
import argparse


class LatentTexture(spy.InstanceList):
    def __init__(self, size: int):
        super().__init__(module["LatentTexture"])
        self.size = size

        size_in_blocks = size // 4
        num_blocks = size_in_blocks * size_in_blocks

        self.num_mip_levels = 1
        while size_in_blocks > 1:
            self.num_mip_levels += 1
            size_in_blocks >>= 1
            num_blocks += size_in_blocks * size_in_blocks

        self.endpoint_a = spy.Tensor.from_numpy(
            device, np.random.uniform(0.0, 1.0, num_blocks * 3).astype("float32")
        ).with_grads()
        self.m_endpoint_a = spy.Tensor.zeros_like(self.endpoint_a)
        self.v_endpoint_a = spy.Tensor.zeros_like(self.endpoint_a)

        self.endpoint_b = spy.Tensor.from_numpy(
            device, np.random.uniform(0.0, 1.0, num_blocks * 3).astype("float32")
        ).with_grads()
        self.m_endpoint_b = spy.Tensor.zeros_like(self.endpoint_b)
        self.v_endpoint_b = spy.Tensor.zeros_like(self.endpoint_b)

        self.alpha = spy.Tensor.from_numpy(
            device, np.random.uniform(0.0, 1.0, num_blocks * 16).astype("float32")
        ).with_grads()
        self.m_alpha = spy.Tensor.zeros_like(self.alpha)
        self.v_alpha = spy.Tensor.zeros_like(self.alpha)

    # Calls the Slang 'optimize' function for biases and weights
    def optimize(self, learning_rate: float, optimize_counter: int):
        module.optimizer_step(
            self.alpha,
            self.alpha.grad,
            self.m_alpha,
            self.v_alpha,
            learning_rate,
            optimize_counter,
        )
        module.optimizer_step(
            self.endpoint_a,
            self.endpoint_a.grad,
            self.m_endpoint_a,
            self.v_endpoint_a,
            learning_rate,
            optimize_counter,
        )
        module.optimizer_step(
            self.endpoint_b,
            self.endpoint_b.grad,
            self.m_endpoint_b,
            self.v_endpoint_b,
            learning_rate,
            optimize_counter,
        )


class Network(spy.InstanceList):
    def __init__(self, size, num_channels):
        super().__init__(module[f"Network<{num_channels}>"])
        self.latent_texture_1 = LatentTexture(size)
        self.latent_texture_2 = LatentTexture(size)
        self.latent_texture_3 = LatentTexture(size // 2)
        self.latent_texture_4 = LatentTexture(size // 2)

        initial = np.concatenate(
            [
                np.random.uniform(-0.5, 0.5, 16 * 64).astype("float32"),
                np.zeros(64).astype("float32"),
                np.random.uniform(-0.5, 0.5, 64 * 64).astype("float32"),
                np.zeros(64).astype("float32"),
                np.random.uniform(-0.5, 0.5, 64 * 16).astype("float32"),
                np.zeros(16).astype("float32"),
            ]
        )

        self.weights_and_biases = spy.Tensor.from_numpy(device, initial)
        self.weights_and_biases_grad = spy.Tensor.zeros_like(self.weights_and_biases)
        # Temp data for Adam optimizer.
        self.m = spy.Tensor.zeros_like(self.weights_and_biases)
        self.v = spy.Tensor.zeros_like(self.weights_and_biases)

    # Calls the Slang 'optimize' function for the layer.
    def optimize(self, learning_rate: float, optimize_counter: int):
        self.latent_texture_1.optimize(learning_rate, optimize_counter)
        self.latent_texture_2.optimize(learning_rate, optimize_counter)
        self.latent_texture_3.optimize(learning_rate, optimize_counter)
        self.latent_texture_4.optimize(learning_rate, optimize_counter)
        module.optimizer_step(
            self.weights_and_biases,
            self.weights_and_biases_grad,
            self.m,
            self.v,
            learning_rate,
            optimize_counter,
        )


def render_to_tensor(device, module, network, num_channels, tex_size, mip):
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
    return output


def compress_blocks(device, module, texture):
    blocks = spy.Tensor.from_numpy(
        device,
        np.zeros((texture.endpoint_a.shape[0] // 3, 4)).astype("uint16"),
    )
    module.compress_latent_texture(texture=texture, block=spy.call_id(), _result=blocks)
    return blocks


def train(args, device, module):
    filenames = [
        (filename, True) for filenames in args.srgb for filename in filenames
    ] + [(filename, False) for filenames in args.nonsrgb for filename in filenames]

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

    args.size = args.size or tex_size

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
            loss_output = spy.Tensor.from_numpy(
                device, np.zeros((1,)).astype("float32")
            )
            module.sum_loss(
                pixel=spy.grid((tex_size, tex_size)),
                resolution=tex_size,
                network=network,
                reference=tex,
                total=loss_output,
                samp=samp,
            )
            mae = loss_output.to_numpy()[0] / tex_size / tex_size / num_channels
            psnr = 10 * np.log10(1.0 / mae) if mae > 0 else float("inf")
            print(f"Loss: {mae:.8f} PSNR: {psnr:.4f} dB")
    end = time.time()
    print(end - start)

    for mip in range(tex[0].mip_count):
        output = render_to_tensor(
            device, module, network, num_channels, tex_size, mip
        ).to_numpy()
        outputs = [output[:, :, i * 3 : (i + 1) * 3] for i in range(len(tex))]
        for i, (_, is_srgb) in enumerate(filenames):
            spy.Bitmap(outputs[i]).convert(
                component_type=spy.Bitmap.ComponentType.uint8, srgb_gamma=is_srgb
            ).write(f"{i}_m{mip}.png")

    # blocks = compress_blocks(device, module, network.latent_texture_1)
    # desc = spy.TextureDesc()
    # desc.width = network.latent_texture_1.size
    # desc.height = network.latent_texture_1.size
    # desc.format = spy.Format.bc1_unorm
    # desc.usage = spy.TextureUsage.shader_resource
    # desc.mip_count = network.latent_texture_1.num_mip_levels
    # tex_out = device.create_texture(desc)
    # offset = 0
    # for mip in range(desc.mip_count):
    #     size_in_blocks = desc.width >> 2 >> mip
    #     tex_out.copy_from_numpy(
    #         blocks.to_numpy()[offset : offset + size_in_blocks * size_in_blocks],
    #         mip=mip,
    #     )
    #     offset += size_in_blocks * size_in_blocks

    # ress = spy.Tensor.from_numpy(
    #     device, np.zeros(((tex_out.width), (tex_out.width), 4)).astype("float32")
    # )

    # module.render_texture(
    #     texture=tex_out,
    #     pixel=spy.call_id(),
    #     _result=ress,
    #     samp=samp,
    #     resolution=spy.int2(tex_out.width, tex_out.width),
    # )
    # spy.Bitmap(ress.to_numpy()).write(f"bc1.exr")
    # module.render_tensors(
    #     texture=network.latent_texture_1,
    #     pixel=spy.call_id(),
    #     _result=ress,
    #     resolution=spy.int2(tex_out.width, tex_out.width),
    # )
    # spy.Bitmap(ress.to_numpy()).write(f"tens.exr")

    if args.output:
        np.savez(
            args.output,
            size=network.latent_texture_1.size,
            num_channels=num_channels,
            lt1_endpoint_a=network.latent_texture_1.endpoint_a.to_numpy(),
            lt1_endpoint_b=network.latent_texture_1.endpoint_b.to_numpy(),
            lt1_alpha=network.latent_texture_1.alpha.to_numpy(),
            lt2_endpoint_a=network.latent_texture_2.endpoint_a.to_numpy(),
            lt2_endpoint_b=network.latent_texture_2.endpoint_b.to_numpy(),
            lt2_alpha=network.latent_texture_2.alpha.to_numpy(),
            lt3_endpoint_a=network.latent_texture_3.endpoint_a.to_numpy(),
            lt3_endpoint_b=network.latent_texture_3.endpoint_b.to_numpy(),
            lt3_alpha=network.latent_texture_3.alpha.to_numpy(),
            lt4_endpoint_a=network.latent_texture_4.endpoint_a.to_numpy(),
            lt4_endpoint_b=network.latent_texture_4.endpoint_b.to_numpy(),
            lt4_alpha=network.latent_texture_4.alpha.to_numpy(),
            weights_and_biases=network.weights_and_biases.to_numpy(),
        )


def eval(args, device, module):
    data = np.load(args.input)

    size = int(data["size"])
    num_channels = int(data["num_channels"])

    network = Network(size, num_channels)

    def load_tensor(tensor, arr):
        tensor.copy_from_numpy(arr.astype("float32"))

    load_tensor(network.latent_texture_1.endpoint_a, data["lt1_endpoint_a"])
    load_tensor(network.latent_texture_1.endpoint_b, data["lt1_endpoint_b"])
    load_tensor(network.latent_texture_1.alpha, data["lt1_alpha"])
    load_tensor(network.latent_texture_2.endpoint_a, data["lt2_endpoint_a"])
    load_tensor(network.latent_texture_2.endpoint_b, data["lt2_endpoint_b"])
    load_tensor(network.latent_texture_2.alpha, data["lt2_alpha"])
    load_tensor(network.latent_texture_3.endpoint_a, data["lt3_endpoint_a"])
    load_tensor(network.latent_texture_3.endpoint_b, data["lt3_endpoint_b"])
    load_tensor(network.latent_texture_3.alpha, data["lt3_alpha"])
    load_tensor(network.latent_texture_4.endpoint_a, data["lt4_endpoint_a"])
    load_tensor(network.latent_texture_4.endpoint_b, data["lt4_endpoint_b"])
    load_tensor(network.latent_texture_4.alpha, data["lt4_alpha"])
    load_tensor(network.layer0.biases, data["layer0_biases"])
    load_tensor(network.layer0.weights, data["layer0_weights"])
    load_tensor(network.layer1.biases, data["layer1_biases"])
    load_tensor(network.layer1.weights, data["layer1_weights"])
    load_tensor(network.layer2.biases, data["layer2_biases"])
    load_tensor(network.layer2.weights, data["layer2_weights"])

    output = render_to_tensor(device, module, network, num_channels, size, 0).to_numpy()

    spy.Bitmap(output).convert(
        component_type=spy.Bitmap.ComponentType.uint8, srgb_gamma=True
    ).write(args.output)


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="mode", required=True)

train_parser = subparsers.add_parser("train")
train_parser.add_argument("--srgb", dest="srgb", default=[], nargs="+", action="append")
train_parser.add_argument(
    "--nonsrgb", dest="nonsrgb", default=[], nargs="+", action="append"
)
train_parser.add_argument("--size", dest="size", type=int)
train_parser.add_argument("--steps", dest="steps", type=int, default=10_000)
train_parser.add_argument("--output", dest="output")

eval_parser = subparsers.add_parser("eval")
eval_parser.add_argument("--input", dest="input", required=True)
eval_parser.add_argument("--output", dest="output", required=True)

args = parser.parse_args()

device = spy.create_device(
    spy.DeviceType.automatic,
    enable_debug_layers=True,
    include_paths=[Path(__file__).parent],
)
print(device.features)
module = spy.Module.load_from_file(device, "compress.slang")

if args.mode == "train":
    train(args, device, module)
elif args.mode == "eval":
    eval(args, device, module)
