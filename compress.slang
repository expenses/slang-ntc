// SPDX-License-Identifier: Apache-2.0

import slangpy;

[Differentiable]
half fma(half a, half b, half c) {
    return a * b + c;
}

uint16_t quantize(float value, uint16_t bits) {
    let size = (1 << bits) - 1;
    return uint16_t(round(value * size));
}

uint16_t3 quantize(float3 value, uint16_t3 bits) {
    let size = (uint3(1) << bits) - 1;
    return uint16_t3(round(value * size));
}

[BackwardDerivative(as_quantized_f_bwd)]
float as_quantized(float value, uint16_t bits) {
    let size = (1 << bits) - 1;
    return float(quantize(value, bits)) / size;
}

void as_quantized_f_bwd(inout DifferentialPair<float> value, uint bits, float grad) {
    // pass gradient straight through
    value = DifferentialPair<float>(value.p, grad);
}

[BackwardDerivative(as_quantized_v_bwd)]
float3 as_quantized(float3 value, uint16_t3 bits) {
    let size = (uint3(1) << bits) - 1;
    return float3(quantize(value, bits)) / size;
}

void as_quantized_v_bwd(inout DifferentialPair<float3> value, uint3 bits, float3 grad) {
    // pass gradient straight through
    value = DifferentialPair<float3>(value.p, grad);
}

[Differentiable]
float sigmoid(float x) {
    let expX = exp(x);
    return expX / (expX + 1.0f);
}

[Differentiable]
float3 sigmoid(float3 x) {
    let expX = exp(x);
    return expX / (expX + 1.0f);
}

struct LatentTexture {
    DiffTensor<float3, 1> endpoint_a;
    DiffTensor<float3, 1> endpoint_b;
    DiffTensor<float, 1> alpha;

    [Differentiable]
    float3 get_value(int mip, int2 coord) {
        let block_offset = get_mip_offset(mip) / 16;
        let block_size = get_mip_size(mip) / 4;
        let block_index = block_offset + (coord.x / 4) + (coord.y / 4) * block_size;
        let a = endpoint_a.load(block_index);
        let b = endpoint_b.load(block_index);
        let alpha = alpha.load(block_index * 16 + (coord.x % 4) + (coord.y % 4) * 4);
        return lerp(
            as_quantized(sigmoid(a), uint16_t3(5,6,5)),
            as_quantized(sigmoid(b), uint16_t3(5,6,5)),
            as_quantized(sigmoid(alpha), 2)
        );
    }

    int size;
    int num_mip_levels;

    int get_mip_size(int mip_level) {
        return size >> mip_level;
    }

    int get_mip_offset(int mip_level) {
        int offset = 0;
        for (int level = 0; level < mip_level; ++level) {
            offset += (size * size) >> level * 2;
        }
        return offset;
    }

    [Differentiable]
    half3 sample(no_diff float2 uv, no_diff float mip_level) {
        mip_level = clamp(mip_level, 0.0, float(num_mip_levels - 1));
        int mip0 = int(mip_level);
        int mip1 = min(mip0 + 1, num_mip_levels - 1);
        float mip_fractional = mip_level - mip0;

        int mip0_size = get_mip_size(mip0);
        int mip1_size = get_mip_size(mip1);

        let mip0_uv = uv * float2(mip0_size) - 0.5;
        let mip0_c0 = clamp(int2(mip0_uv), int2(0), mip0_size-1);
        let mip0_c1 = min(mip0_c0 + 1, mip0_size - 1);
        let mip0_fractional_uv = mip0_uv - mip0_c0;

        let mip1_uv = uv * float2(mip1_size) - 0.5;
        let mip1_c0 = clamp(int2(mip1_uv), int2(0), mip1_size-1);
        let mip1_c1 = min(mip1_c0 + 1, mip1_size - 1);
        let mip1_fractional_uv = mip1_uv - mip1_c0;

        let v_00_m0 = get_value(mip0, int2(mip0_c0.x, mip0_c0.y));
        let v_10_m0 = get_value(mip0, int2(mip0_c1.x, mip0_c0.y));
        let v_01_m0 = get_value(mip0, int2(mip0_c0.x, mip0_c1.y));
        let v_11_m0 = get_value(mip0, int2(mip0_c1.x, mip0_c1.y));
        let v_00_m1 = get_value(mip1, int2(mip1_c0.x, mip1_c0.y));
        let v_10_m1 = get_value(mip1, int2(mip1_c1.x, mip1_c0.y));
        let v_01_m1 = get_value(mip1, int2(mip1_c0.x, mip1_c1.y));
        let v_11_m1 = get_value(mip1, int2(mip1_c1.x, mip1_c1.y));
        return half3(
            lerp(
                lerp(
                    lerp(v_00_m0, v_10_m0, mip0_fractional_uv.x),
                    lerp(v_01_m0, v_11_m0, mip0_fractional_uv.x),
                    mip0_fractional_uv.y
                ),
                lerp(
                    lerp(v_00_m1, v_10_m1, mip1_fractional_uv.x),
                    lerp(v_01_m1, v_11_m1, mip1_fractional_uv.x),
                    mip1_fractional_uv.y
                ),
                mip_fractional
            )
        );
    }
}

struct Network<int NumChannels> {
    LatentTexture latent_texture_1;
    LatentTexture latent_texture_2;
    LatentTexture latent_texture_3;
    LatentTexture latent_texture_4;
    Tensor<float, 1> weights_and_biases;
    AtomicTensor<float, 1> weights_and_biases_grad;

    float load(int index) {
        return weights_and_biases.load(index);
    }

    [BackwardDerivativeOf(load)]
    void load_bwd(int index, float grad)
    {
        weights_and_biases_grad.add(index, grad);
    }

    [Differentiable]
    half[Outputs] forward<int Inputs, int Outputs>(half[Inputs] x, int offset) {
        half[Outputs] y;
        // For optimal runtime performance, using [ForceUnroll] here can be better. However, the downstream
        // compiler on some platforms (in particular, the DXC compiler on D3D12) may not cope well with
        // unrolled loops and compilation times grow exponentially. Using MaxIters on the outer loop
        // here helps avoid this.
        [MaxIters(Outputs)]
        for (int row = 0; row < Outputs; ++row) {
            // Sum up bias plus weighted inputs for 1 neuron
            half sum = half(0.0h);
            [ForceUnroll]
            for (int col = 0; col < Inputs; ++col) {
                sum = fma(half(load(col * Outputs + row + offset)), x[col], sum);
            }
            y[row] = sum + half(load(Outputs * Inputs + row + offset));
        }

        return y;
    }

    [Differentiable]
    half[16] eval_from_inputs(half[16] inputs) {
        half output0[64] = forward<16, 64>(inputs, 0);
        [ForceUnroll]
        for (int i = 0; i < 64; ++i)
            output0[i] = relu(output0[i]);
        half output1[64] = forward<64, 64>(output0, 16*64 + 64);
        [ForceUnroll]
        for (int i = 0; i < 64; ++i)
            output1[i] = relu(output1[i]);
        return forward<64, 16>(output1, 16*64 + 64 + 64*64 + 64);
    }

    [Differentiable]
    half[16] eval(no_diff float2 uv, no_diff float mip_level) {
        let s_1 = latent_texture_1.sample(uv, mip_level);
        let s_2 = latent_texture_2.sample(uv+0.5, mip_level);
        let s_3 = latent_texture_3.sample(uv, mip_level-1);
        let s_4 = latent_texture_4.sample(uv+0.5, mip_level-1);
        half inputs[16] = {
            s_1[0],
            s_1[1],
            s_1[2],
            s_2[0],
            s_2[1],
            s_2[2],
            s_3[0],
            s_3[1],
            s_3[2],
            s_4[0],
            s_4[1],
            s_4[2],
            0.0h,
            0.0h,
            0.0h,
            0.0h
        };
        return eval_from_inputs(inputs);
    }
}

[Differentiable]
half relu(half x) {
    return max(x, half(0.0));
}

half[NumChannels] render<int NumChannels>(
    int2 pixel,
    int resolution,
    int mip,
    Network<NumChannels> network
) {
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    let eval = network.eval(uv, mip);
    half[NumChannels] output;
    for (int i = 0; i < NumChannels; ++i) {
        output[i] = eval[i];
    }
    return output;
}

float[NumChannels] calculate_loss<int NumChannels>(
    int2 pixel,
    int resolution,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network<NumChannels> network
) {
    return loss(
        (float2(pixel) + 0.5f) / float2(resolution),
        0.0f,
        reference,
        samp,
        network
    );
}

[Differentiable]
float3 L2(float3 value, float3 reference) {
    let diff = value - reference;
    return diff * diff;
}

[Differentiable]
float3 L1(float3 value, float3 reference) {
    let diff = value - reference;
    return abs(diff);
}

[Differentiable]
float[NumChannels] loss<int NumChannels>(
    no_diff float2 uv,
    no_diff float mip_level,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network<NumChannels> network
) {
    let color = network.eval(uv, mip_level);
    float[NumChannels] error;
    [ForceUnroll]
    for (int i = 0; i < NumChannels/3; ++i) {
        let value = float3(color[i*3], color[i*3+1], color[i*3+2]);
        let ref_sample = (no_diff reference[i].SampleLevel(samp, uv, mip_level)).rgb;
        let loss = L2(value, ref_sample);

        error[i*3] = loss[0];
        error[i*3+1] = loss[1];
        error[i*3+2] = loss[2];
    }
    return error;
}

struct LCG {
    uint state;

    __init(uint seed) { state = seed; }

    [mutating]
    uint next_uint() {
        const uint A = 1664525u;
        const uint C = 1013904223u;
        state = (A * state + C);
        return state;
    }

    [mutating]
    float next_float() {
        // Convert to float in range [0, 1)
        return (next_uint() >> 8) * 0x1p-24;
    }
};

void optimizer_step(
    inout float primal,
    inout float grad,
    inout float mean,
    inout float variance,
    float learning_rate,
    int iteration
) {
    // Default Adam values.
    const float ADAM_BETA_1 = 0.9;
    const float ADAM_BETA_2 = 0.999;
    const float ADAM_EPSILON = 1e-8;

    // Running average of gradient and square gradient (== variance)
    mean = lerp(grad, mean, ADAM_BETA_1);
    variance = lerp(grad * grad, variance, ADAM_BETA_2);

    // The running average is initialized with zeros, and underestimates
    // the true values at the beginning. Adam divides by a compensation factor
    float mHat = mean / (1.0f - pow(ADAM_BETA_1, iteration));
    float vHat = variance / (1.0f - pow(ADAM_BETA_2, iteration));
    // Compute the adam step by scaling the mean gradient by the learning rate and the variance estimate
    float step = learning_rate * mHat / (sqrt(vHat) + ADAM_EPSILON);

    // Take an adam step
    primal -= step;

    // Because we now accumulate gradients, we need to clear the gradient to zero before the next iteration
    grad = 0.0f;
}


void sum_loss<int NumChannels>(
    int2 pixel,
    int resolution,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network<NumChannels> network,
    AtomicTensor<float,1> total
) {
    let loss = loss(
        (float2(pixel) + 0.5f) / float2(resolution),
        0.0f,
        reference,
        samp,
        network
    );
    float local_sum = 0.0;
    for (int i = 0; i < NumChannels; ++i) {
        local_sum += loss[i];
    }
    let wave_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) {
        total.add(0, wave_sum);
    }
}

void calculate_grads<int NumChannels>(
    uint seed,
    int2 batch_index,
    int2 batch_size,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network<NumChannels> network
) {
    // Pick a random pixel for computing the loss
    // First, sample a random UV within our cell of the training batch grid
    LCG lcg = LCG(seed);
    float2 uv = (batch_index + float2(lcg.next_float(), lcg.next_float())) / float2(batch_size);
    let num_mip_levels = network.latent_texture_1.num_mip_levels;
    float mip;
    if (lcg.next_float() < 0.1f) {
        mip = lcg.next_float() * num_mip_levels; // uniform
    } else {
        float u = max(lcg.next_float(), 1e-6f);
        mip = clamp(-log(u) / log(4.0f), 0.0f, float(num_mip_levels - 1));
    }

    // Backpropagate through the loss
    float[NumChannels] gradient_seed;
    for (int i = 0; i < NumChannels; ++i)
        gradient_seed[i] = 1.0f;
    bwd_diff(loss)(uv, mip, reference, samp, network, gradient_seed);
}

uint16_t4 compress_latent_texture(LatentTexture texture, int block) {
    let endpoint_a = quantize(sigmoid(texture.endpoint_a.load(block)), uint16_t3(5,6,5));
    let endpoint_b = quantize(sigmoid(texture.endpoint_b.load(block)), uint16_t3(5,6,5));

    let e1 = endpoint_a.b | (endpoint_a.g << 5) | (endpoint_a.r << 11);
    let e2 = endpoint_b.b | (endpoint_b.g << 5) | (endpoint_b.r << 11);
    // bc1 has a seperate mode if e2 >= e1 that we want to avoid
    let max_e = max(e1, e2);
    let min_e = min(e1, e2);

    uint alphas = 0;
    for (int i = 0; i < 16; ++i) {
        var alpha = uint(quantize(sigmoid(texture.alpha.load(block * 16 + i)), 2));

        // If the endpoints were switched, make sure we flip the bits
        if (max_e != e1) {
            alpha = 3 - alpha;
        // If the end points are identical then we're in mode 2 and should just use the first endpoint
        } else if (max_e == min_e) {
            alpha = 0;
        }

        // bc1 has the second bit as the bit that uses endpoint 2.
        if (alpha == 3) {
            alpha = 1;
        } else if (alpha == 1 || alpha == 2) {
            alpha ++;
        }
        alphas |= alpha << i * 2;
    }

    return uint16_t4(
        max_e,
        min_e,
        uint16_t(alphas),
        uint16_t(alphas >> 16),
    );
}

float4 render_tensors(
    LatentTexture texture,
    int2 pixel,
    int2 resolution,
) {
    let uv = (float2(pixel) + 0.5f) / float2(resolution);
    return float4(texture.sample(uv, 0), 1);
}

float4 render_texture(
    Texture2D<float4> texture,
    SamplerState samp,
    int2 pixel,
    int2 resolution,
) {
    let uv = (float2(pixel) + 0.5f) / float2(resolution);
    return texture.SampleLevel(samp, uv, 0);
}
