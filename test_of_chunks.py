import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

model_path = "/local/home/hhamidi/t_dif/diffusers/examples/text_to_image/mimic-2"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

# image = pipe(prompt="").images[0]
# image.save("test.png")