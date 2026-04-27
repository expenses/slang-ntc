// SPDX-License-Identifier: Apache-2.0

import slangpy;

[Differentiable]
float16_t fma(float16_t a, float16_t b, float16_t c) {
    return a * b + c;
}

struct NetworkParameters<int Inputs, int Outputs> {
    DiffTensor<float, 1> biases;
    DiffTensor<float, 2> weights;

    // Evaluate all neurons in the layer
    [Differentiable]
    float16_t[Outputs] forward(float16_t[Inputs] x) {
        float16_t[Outputs] y;
        // For optimal runtime performance, using [ForceUnroll] here can be better. However, the downstream
        // compiler on some platforms (in particular, the DXC compiler on D3D12) may not cope well with
        // unrolled loops and compilation times grow exponentially. Using MaxIters on the outer loop
        // here helps avoid this.
        [MaxIters(Outputs)]
        for (int row = 0; row < Outputs; ++row) {
            // Sum up bias plus weighted inputs for 1 neuron
            float16_t sum = float16_t(0.0h);
            [ForceUnroll]
            for (int col = 0; col < Inputs; ++col) {
                sum = fma(float16_t(weights.load(row, col)), x[col], sum);
            }
            y[row] = sum + float16_t(biases.load(row));
        }

        return y;
    }
}

uint8_t quantize(float value, uint8_t bits) {
    let size = (1 << bits) - 1;
    return uint8_t(round(value * size));
}

uint8_t3 quantize(float3 value, uint8_t3 bits) {
    let size = (uint3(1) << bits) - 1;
    return uint8_t3(round(value * size));
}

[BackwardDerivative(as_quantized_f_bwd)]
float as_quantized(float value, uint8_t bits) {
    let size = (1 << bits) - 1;
    return float(quantize(value, bits)) / size;
}

void as_quantized_f_bwd(inout DifferentialPair<float> value, uint bits, float grad) {
    // pass gradient straight through
    value = DifferentialPair<float>(value.p, grad);
}

[BackwardDerivative(as_quantized_v_bwd)]
float3 as_quantized(float3 value, uint8_t3 bits) {
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
            as_quantized(sigmoid(a), uint8_t3(5,6,5)),
            as_quantized(sigmoid(b), uint8_t3(5,6,5)),
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
    float16_t3 sample(no_diff float2 uv, no_diff float mip_level) {
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
        return float16_t3(
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

struct Network<int HiddenLayerSize, int NumChannels> {
    LatentTexture latent_texture_1;
    LatentTexture latent_texture_2;
    LatentTexture latent_texture_3;
    LatentTexture latent_texture_4;
    NetworkParameters<12, HiddenLayerSize>  layer0;
    NetworkParameters<HiddenLayerSize, HiddenLayerSize> layer1;
    NetworkParameters<HiddenLayerSize, NumChannels>  layer2;

    [Differentiable]
    float16_t[NumChannels] eval(no_diff float2 uv, no_diff float mip_level) {
        let s_1 = latent_texture_1.sample(uv, mip_level);
        let s_2 = latent_texture_2.sample(uv+0.5, mip_level);
        let s_3 = latent_texture_3.sample(uv, mip_level-1);
        let s_4 = latent_texture_4.sample(uv+0.5, mip_level-1);
        float16_t half_inputs[12] = {
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
        };
        float16_t output0[HiddenLayerSize] = layer0.forward(half_inputs);
        [ForceUnroll]
        for (int i = 0; i < HiddenLayerSize; ++i)
            output0[i] = relu(output0[i]);
        float16_t output1[HiddenLayerSize] = layer1.forward(output0);
        [ForceUnroll]
        for (int i = 0; i < HiddenLayerSize; ++i)
            output1[i] = relu(output1[i]);
        return layer2.forward(output1);
    }
}

[Differentiable]
float16_t relu(float16_t x) {
    return max(x, float16_t(0.0));
}

// "Render" the neural texture
float16_t[NumChannels] render<int HiddenLayerSize, int NumChannels>(
    int2 pixel,
    int resolution,
    int mip,
    Network<HiddenLayerSize, NumChannels> network
) {
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return network.eval(uv, mip);
}

float[NumChannels] calculate_loss<int HiddenLayerSize, int NumChannels>(
    int2 pixel,
    int resolution,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network<HiddenLayerSize, NumChannels> network
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
float[NumChannels] loss<int HiddenLayerSize, int NumChannels>(
    no_diff float2 uv,
    no_diff float mip_level,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network<HiddenLayerSize, NumChannels> network
) {
    let color = network.eval(uv, mip_level);
    float[NumChannels] error;
    [ForceUnroll]
    for (int i = 0; i < NumChannels/3; ++i) {
        float3 sample = (no_diff reference[i].SampleLevel(samp, uv, mip_level)).rgb;
        error[i*3] = abs(sample[0] - float(color[i*3]));
        error[i*3+1] = abs(sample[1] - float(color[i*3+1]));
        error[i*3+2] = abs(sample[2] - float(color[i*3+2]));
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


void sum_loss<int HiddenLayerSize, int NumChannels>(
    int2 pixel,
    int resolution,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network<HiddenLayerSize, NumChannels> network,
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

void calculate_grads<int HiddenLayerSize, int NumChannels>(
    uint seed,
    int2 batch_index,
    int2 batch_size,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network<HiddenLayerSize, NumChannels> network
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
    let endpoint_a = uint16_t3(quantize(sigmoid(texture.endpoint_a.load(block)), uint8_t3(5,6,5)));
    let endpoint_b = uint16_t3(quantize(sigmoid(texture.endpoint_b.load(block)), uint8_t3(5,6,5)));

    uint16_t top = 0;
    for (int i = 0; i < 8; ++i) {
        let alpha = quantize(sigmoid(texture.alpha.load(block*16+i)), 2);
        top |= alpha << i * 2;
    }

    uint16_t bottom = 0;
    for (int i = 0; i < 8; ++i) {
        let alpha = quantize(sigmoid(texture.alpha.load(block*16+i+8)), 2);
        bottom |= alpha << i * 2;
    }

    return uint16_t4(
        endpoint_a.b | (endpoint_a.g << 5) | (endpoint_a.r << 11),
        endpoint_b.b | (endpoint_b.g << 5) | (endpoint_b.r << 11),
        0,//top,//f,//endpoint_a.b | (endpoint_a.g << 5) | (endpoint_a.r << 11),
        0,//bottom,//f,//endpoint_b.b | (endpoint_b.g << 5) | (endpoint_b.r << 11),
    );
}

float4 render_texture(
    Texture2D<float4> texture,
    int2 pixel,
) {
    return texture[pixel];
}
