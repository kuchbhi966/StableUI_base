import os
import random
import uuid
import torch
import numpy as np
import gradio as gr
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

# Constants
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344
SAVE_DIR = "/content/images"
MODEL_PATH = '/content/StableUI_base/model_link.safetensors'

# Setup
os.makedirs(SAVE_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
os.system(f'wget -O {MODEL_PATH} "https://civitai.com/api/download/models/926965?type=Model&format=SafeTensor&size=pruned&fp=fp16"')

pipe = StableDiffusionXLPipeline.from_single_file(MODEL_PATH, use_safetensors=True, torch_dtype=torch.float16).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
print("\033[1;32mDone!\033[0m")


def infer(prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps):
    if seed == -1:  # -1 indicates random seed
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps, 
        width=width, 
        height=height,
        generator=generator,
    ).images[0]
    
    image_filename = f"{uuid.uuid4()}.png"
    image_path = os.path.join(SAVE_DIR, image_filename)
    image.save(image_path)
    
    return image

# UI setup
css = """
#col-container {
    margin: 0 auto;
    max-width: 580px;
}
footer {
    display: none !important;
}
"""

examples = [
    "a cat",
    "a cat in the hat",
    "a cat in the cowboy hat",
]

with gr.Blocks(css=css, theme='ParityError/Interstellar') as app:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
    # Stable Diffusion <a href="https://www.patreon.com/marat_ai">by marat_ai</a> 
    <a href="https://www.youtube.com/@marat_ai">
        <img src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube" style="display: inline;"/>
    </a>
    <a href="https://www.patreon.com/marat_ai">
        <img src="https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white" alt="Patreon" style="display: inline;"/>
    </a>

    Google Colab's free tier offers about 4 hours of GPU usage per day. No authorization, no data storing or tracking. Your session data will be deleted when this session closes.
""")

        with gr.Group():
            with gr.Row():
                prompt = gr.Text(label="Prompt", show_label=False, lines=1, max_lines=7,
                                 placeholder="Enter your prompt", container=False, scale=4)
                run_button = gr.Button("🚀 Run", scale=1, variant='primary')
        
        result = gr.Image(label="Result", show_label=False)
        
        with gr.Group():
            with gr.Accordion("⚙️ Settings", open=False):
                negative_prompt = gr.Text(label="Negative prompt", placeholder="Enter a negative prompt",
                                          lines=3, value='lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature')
                
                seed = gr.Slider(label="Seed (-1 for random)", minimum=-1, maximum=MAX_SEED, step=1, value=-1)
                
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                    height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                
                with gr.Row():
                    guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=10.0, step=0.1, value=5.0)
                    num_inference_steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=20)

        gr.Examples(examples=examples, inputs=[prompt])
    
    run_button.click(
        fn=infer,
        inputs=[prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps],
        outputs=result,
    )

if __name__ == "__main__":
    app.launch(share=True, inline=False, inbrowser=False, debug=True)
