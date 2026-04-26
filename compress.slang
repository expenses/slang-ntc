// SPDX-License-Identifier: Apache-2.0

import slangpy;

[Differentiable]
float16_t fma(float16_t a, float16_t b, float16_t c) {
    return a * b + c;
}

struct NetworkParameters<int Inputs, int Outputs> {
    Tensor<float16_t, 1> biases;
    Tensor<float16_t, 2> weights;

    // Unlike the mipmapping example, where the parameters for each texel are independent and
    // each texel only affect the material for one sample, the network weights are shared for
    // the entire neural texture. In that case, we need to sum the weight gradients from all
    // the training samples in the batch. We can do this using the atomic tensor
    AtomicTensor<float, 1> biases_grad;
    AtomicTensor<float, 2> weights_grad;

    [Differentiable]
    float16_t get_bias(int neuron) {
        return biases.load(neuron);
    }

    [Differentiable]
    float16_t get_weight(int neuron, int input) {
        return weights.load(neuron, input);
    }

    [BackwardDerivativeOf(get_bias)]
    void get_bias_bwd(int neuron, float16_t grad) {
        // .set on an atomic tensor atomically adds the result
        biases_grad.add(neuron, grad);
    }

    [BackwardDerivativeOf(get_weight)]
    void get_weight_bwd(int neuron, int input, float16_t grad) {
        // .set on an atomic tensor atomically adds the result
        weights_grad.add(neuron, input, grad);
    }

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
                sum = fma(get_weight(row, col), x[col], sum);
            }
            y[row] = sum + get_bias(row);
        }

        return y;
    }
}

struct LatentTexture {
    Tensor<float16_t3, 1> texture;
    AtomicTensor<float3, 1> texture_grads;

    [BackwardDerivative(get_value_bwd)]
    float3 get_value(int index) {
        return float3(texture[index]);
    }

    void get_value_bwd(int index, float3 latent_grad) {
        texture_grads.add(index, latent_grad);
    }

    int size;
    int num_mip_levels;

    int get_mip_size(int mip_level) {
        return size >> mip_level;
    }

    int get_mip_offset(int mip_level) {
        int offset = 0;
        for (int l = 0; l < mip_level; ++l) {
            int s = size >> l;
            offset += s * s;
        }
        return offset;
    }

    [Differentiable]
    float16_t3 sample(no_diff float2 uv, no_diff float mip_level) {
        int mip0 = int(clamp(mip_level, 0.0, float(num_mip_levels - 1)));
        int mip1 = min(mip0 + 1, num_mip_levels - 1);
        float mip_fractional = mip_level - mip0;

        int mip0_size = get_mip_size(mip0);
        int mip0_offset = get_mip_offset(mip0);
        int mip1_size = get_mip_size(mip1);
        int mip1_offset = get_mip_offset(mip1);

        let mip0_uv = uv * float2(mip0_size) - 0.5;
        let mip0_c0 = clamp(int2(mip0_uv), int2(0), mip0_size-1);
        let mip0_c1 = min(mip0_c0 + 1, mip0_size - 1);
        let mip0_fractional_uv = mip0_uv - mip0_c0;

        let mip1_uv = uv * float2(mip1_size) - 0.5;
        let mip1_c0 = clamp(int2(mip1_uv), int2(0), mip1_size-1);
        let mip1_c1 = min(mip1_c0 + 1, mip1_size - 1);
        let mip1_fractional_uv = mip1_uv - mip1_c0;

        let v_00_m0 = get_value(mip0_offset + mip0_c0.x + mip0_c0.y * mip0_size);
        let v_10_m0 = get_value(mip0_offset + mip0_c1.x + mip0_c0.y * mip0_size);
        let v_01_m0 = get_value(mip0_offset + mip0_c0.x + mip0_c1.y * mip0_size);
        let v_11_m0 = get_value(mip0_offset + mip0_c1.x + mip0_c1.y * mip0_size);
        let v_00_m1 = get_value(mip1_offset + mip1_c0.x + mip1_c0.y * mip1_size);
        let v_10_m1 = get_value(mip1_offset + mip1_c1.x + mip1_c0.y * mip1_size);
        let v_01_m1 = get_value(mip1_offset + mip1_c0.x + mip1_c1.y * mip1_size);
        let v_11_m1 = get_value(mip1_offset + mip1_c1.x + mip1_c1.y * mip1_size);
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
        let s_3 = latent_texture_3.sample(uv, saturate(mip_level-1));
        let s_4 = latent_texture_4.sample(uv+0.5, saturate(mip_level-1));
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
    int2 resolution,
    int mip,
    Network<HiddenLayerSize, NumChannels> network
) {
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return network.eval(uv, mip);
}

float[NumChannels] loss<int HiddenLayerSize, int NumChannels>(
    int2 pixel,
    int2 resolution,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network<HiddenLayerSize, NumChannels> network
) {
    return loss_uv(
        (float2(pixel) + 0.5f) / float2(resolution),
        0.0f,
        reference,
        samp,
        network
    );
}

[Differentiable]
float[NumChannels] loss_uv<int HiddenLayerSize, int NumChannels>(
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
    inout float16_t primal,
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
    primal -= float16_t(step);

    // Because we now accumulate gradients, we need to clear the gradient to zero before the next iteration
    grad = 0.0f;
}

void calculate_grads<int HiddenLayerSize, int NumChannels>(
    uint seed,
    int2 batch_index,
    int2 batch_size,
    int2 resolution,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network<HiddenLayerSize, NumChannels> network
) {
    // Pick a random pixel for computing the loss
    // First, sample a random UV within our cell of the training batch grid
    LCG lcg = LCG(seed);
    float2 uv = (batch_index + float2(lcg.next_float(), lcg.next_float())) / float2(batch_size);
    float[NumChannels] gradient_seed;
    for (int i = 0; i < NumChannels; ++i)
        gradient_seed[i] = 1.0f;
    let num_mip_levels = network.latent_texture_1.num_mip_levels;
    float mip;
    if (lcg.next_float() < 0.1f) {
        mip = lcg.next_float() * num_mip_levels; // uniform
    } else {
        float u = max(lcg.next_float(), 1e-6f);
        mip = clamp(-log(u) / log(4.0f), 0.0f, float(num_mip_levels - 1));
    }
    
    // Backpropagate through the loss
    bwd_diff(loss_uv)(uv, mip, reference, samp, network, gradient_seed);
}
