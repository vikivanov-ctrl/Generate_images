#импорт библиотек
import torch

from diffusers import StableDiffusionPipeline

#загрузка модели
model_id = "dreamlike-art/dreamlike-diffusion-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")

#генерация изображения
images = pipe(
    #запрос
    prompt = "YOUR PROMPT",
    #размеры
    height = 512,
    width = 512,
    num_inference_steps = 100,
    guidance_scale = 0.5,
    num_images_per_prompt = 1
).images

#вывод изображения
images[0].save("output.png")
