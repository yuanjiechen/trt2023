### Model structure

1. clip text encoder -> text embedding
2. stable diffusion + control net main -> main diffusion steps (20 times)
3. VAE auto encoder/decoder (decoder from latent space)

### Progress

- [x] 1. Pipeline optimization (completed by trt, convert SD+controlnet in one onnx model)
    - Maybe no plugin needed here?
2. python workflow optimization
3. clip model and VAE convert 


### Tools

1. nsignt system
2. trt engine viewer
3. polygraphy

### Optimize direction

1. Reduce dtoh, htod mem copy
2. Reduce tensor create (create in __init__)
3. Remove unused data calculation
4. Optimizae the time comsuming layers
5. Focus on main loop
6. use GPU rather than numpy
7. cv2.canny
8. ddim_hacked.py -> make_schedule 49~58, self.model.parameterization (Done)
9. controlnet timestep calculation -> to initial
10. VAE decoder ->nonlinearity -> silu (Done)
11. VAE decoder -> attention -> sqrt(d) -> constant
12. attention groupnorm -> layernorm ?