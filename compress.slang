// SPDX-License-Identifier: Apache-2.0

import slangpy;

[Differentiable]
float16_t fma(float16_t a, float16_t b, float16_t c)
{
    return a * b + c;
}

struct NetworkParameters<int Inputs, int Outputs>
{
    Tensor<float16_t, 1> biases;
    Tensor<float16_t, 2> weights;

    // Unlike the mipmapping example, where the parameters for each texel are independent and
    // each texel only affect the material for one sample, the network weights are shared for
    // the entire neural texture. In that case, we need to sum the weight gradients from all
    // the training samples in the batch. We can do this using the atomic tensor
    AtomicTensor<float, 1> biases_grad;
    AtomicTensor<float, 2> weights_grad;

    [Differentiable]
    float16_t get_bias(int neuron)
    {
        return biases.load(neuron);
    }
    [Differentiable]
    float16_t get_weight(int neuron, int input)
    {
        return weights.load(neuron, input);
    }

    [BackwardDerivativeOf(get_bias)]
    void get_bias_bwd(int neuron, float16_t grad)
    {
        // .set on an atomic tensor atomically adds the result
        biases_grad.add(neuron, grad);
    }

    [BackwardDerivativeOf(get_weight)]
    void get_weight_bwd(int neuron, int input, float16_t grad)
    {
        // .set on an atomic tensor atomically adds the result
        weights_grad.add(neuron, input, grad);
    }

    // Evaluate all neurons in the layer
    [Differentiable]
    float16_t[Outputs] forward(float16_t[Inputs] x)
    {
        float16_t[Outputs] y;
        // For optimal runtime performance, using [ForceUnroll] here can be better. However, the downstream
        // compiler on some platforms (in particular, the DXC compiler on D3D12) may not cope well with
        // unrolled loops and compilation times grow exponentially. Using MaxIters on the outer loop
        // here helps avoid this.
        [MaxIters(Outputs)]
        for (int row = 0; row < Outputs; ++row)
        {
            // Sum up bias plus weighted inputs for 1 neuron
            float16_t sum = float16_t(0.0h);
            [ForceUnroll]
            for (int col = 0; col < Inputs; ++col)
                sum = fma(get_weight(row, col), x[col], sum);
            y[row] = sum + get_bias(row);
        }

        return y;
    }
}

struct LatentTexture {
    Tensor<float16_t, 1> texture;
    uint width;
    uint height;

    AtomicTensor<float, 1> texture_grads;

    [BackwardDerivative(get_latent_bwd)]
    float get_latent(int x, int y, int latent_index)
    {
        let index = latent_index * (width*height) + y * width + x;
        return float(texture[index]);
    }

    void get_latent_bwd(int x, int y, int latent_index, float latent_grad)
    {
        let index = latent_index * (width*height) + y * width + x;
        texture_grads.add(index, latent_grad);
    }

    [Differentiable]
    float16_t3 sample(no_diff float2 uv)
    {
        int2 size = {width, height};
        // Convert to (fractional) texel position
        uv *= float2(size - 1);
        // Get (clamped) integer positions of texel corners
        int2 c0 = clamp(int2(uv), int2(0), size - 1);
        int2 c1 = min(c0 + 1, size - 1);

        // Get fractional part of texel position for interpolation
        uv = saturate(uv - float2(c0));

        float16_t3 result;
        [ForceUnroll]
        for (int i = 0; i < 3; ++i) {
            // Simple linear interpolation of latent texture
            float latent00 = get_latent(c0.x, c0.y, i);
            float latent10 = get_latent(c1.x, c0.y, i);
            float latent01 = get_latent(c0.x, c1.y, i);
            float latent11 = get_latent(c1.x, c1.y, i);
            result[i] = float16_t(lerp(lerp(latent00, latent10, uv.x), lerp(latent01, latent11, uv.x), uv.y));
        }

        return result;
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
    float[NumChannels] eval(no_diff float2 uv)
    {
        let s_1 = latent_texture_1.sample(uv);
        let s_2 = latent_texture_2.sample(uv+0.5);
        let s_3 = latent_texture_3.sample(uv);
        let s_4 = latent_texture_4.sample(uv+0.5);
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
        float16_t output2[NumChannels] = layer2.forward(output1);
        float full_outputs[NumChannels];
        [ForceUnroll]
        for (int i = 0; i < NumChannels; ++i)
            full_outputs[i] = float(output2[i]);
        return full_outputs;
    }
}

[Differentiable]
float16_t relu(float16_t x)
{
    return max(x, float16_t(0.0));
}

// "Render" the neural texture
[Differentiable]
float[NumChannels] render<int HiddenLayerSize, int NumChannels>(int2 pixel, int2 resolution, Network<HiddenLayerSize, NumChannels> network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return network.eval(uv);
}

[Differentiable]
float[NumChannels] loss<int HiddenLayerSize, int NumChannels>(
    int2 pixel,
    int2 resolution,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network<HiddenLayerSize, NumChannels> network
) {
    return loss_uv(
        (float2(pixel) + 0.5f) / float2(resolution),
        reference,
        samp,
        network
    );
}

[Differentiable]
float[NumChannels] loss_uv<int HiddenLayerSize, int NumChannels>(
    no_diff float2 uv,
    Texture2D<float4>[] reference,
    SamplerState samp,
    Network<HiddenLayerSize, NumChannels> network
) {
    let color = network.eval(uv);
    float[NumChannels] error;
    [ForceUnroll]
    for (int i = 0; i < NumChannels/3; ++i) {
        float3 sample = (no_diff reference[i].SampleLevel(samp, uv, 0)).rgb;
        error[i*3] = abs(sample[0] - color[i*3]);
        error[i*3+1] = abs(sample[1] - color[i*3+1]);
        error[i*3+2] = abs(sample[2] - color[i*3+2]);
    }
    return error;
}

struct LCG
{
    uint state;

    __init(uint seed) { state = seed; }

    [mutating]
    uint next_uint()
    {
        const uint A = 1664525u;
        const uint C = 1013904223u;
        state = (A * state + C);
        return state;
    }

    [mutating]
    float next_float()
    {
        // Convert to float in range [0, 1)
        return (next_uint() >> 8) * 0x1p-24;
    }
};


void optimizer_step(inout float16_t primal, inout float grad, inout float mean, inout float variance, float learning_rate, int iteration)
{
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

void calculate_grads<int HiddenLayerSize, int NumChannels>(uint seed, int2 batch_index, int2 batch_size, int2 resolution, Texture2D<float4>[] reference, SamplerState samp, Network<HiddenLayerSize, NumChannels> network)
{
    // Pick a random pixel for computing the loss
    // First, sample a random UV within our cell of the training batch grid
    LCG lcg = LCG(seed);
    float2 uv = (batch_index + float2(lcg.next_float(), lcg.next_float())) / float2(batch_size);
    float[NumChannels] gradient_seed;
    for (int i = 0; i < NumChannels; ++i)
        gradient_seed[i] = 1.0f;
    // Backpropagate through the loss
    bwd_diff(loss_uv)(uv, reference, samp, network, gradient_seed);
}
