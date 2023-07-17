### Model structure

1. clip text encoder -> text embedding
2. stable diffusion + control net main -> main diffusion steps (20 times)
3. VAE auto encoder/decoder (decoder from latent space)

### Progress

- [x] 1. Pipeline optimization (completed by trt, convert SD+controlnet in one onnx model)
    - Maybe no plugin needed here?
2. python workflow optimization
3. clip model and VAE convert 


