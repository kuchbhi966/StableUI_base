import os
import random
import uuid
import torch
import numpy as np
import gradio as gr
from diffusers import (
    StableDiffusionXLPipeline, EulerDiscreteScheduler, DDIMScheduler, 
    PNDMScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler
)

# Constants
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344
SAVE_DIR = "./images"
os.makedirs(SAVE_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables to hold models and scheduler state
models = {}
current_model_key = None
current_scheduler = "Euler"

# Function to download and load models
def load_model(model_url):
    global models, current_model_key
    
    model_filename = model_url.split("/")[-1]
    model_path = os.path.join(SAVE_DIR, model_filename)
    
    if not os.path.exists(model_path):
        os.system(f'wget -O {model_path} "{model_url}"')
    
    # Load the model if not already loaded
    if model_filename not in models:
        pipe = StableDiffusionXLPipeline.from_single_file(model_path, use_safetensors=True, torch_dtype=torch.float16).to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)  # Default to Euler scheduler
        models[model_filename] = pipe
    
    current_model_key = model_filename
    return f"Model loaded: {model_filename}"

# Function to switch schedulers dynamically
def switch_scheduler(scheduler_name):
    global current_scheduler
    if current_model_key is None:
        return "No model loaded."
    
    # Get the current model's pipeline
    pipe = models[current_model_key]
    
    # Map scheduler names to their corresponding classes
    schedulers = {
        "Euler": EulerDiscreteScheduler,
        "DDIM": DDIMScheduler,
        "PNDM": PNDMScheduler,
        "LMS": LMSDiscreteScheduler,
        "DPM-Solver": DPMSolverMultistepScheduler,
        "DPM-Solver++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True),
        "DPM2": DPMSolverSinglestepScheduler
    }
    
    # Load the corresponding scheduler
    if scheduler_name in schedulers:
        scheduler_class = schedulers[scheduler_name]
        if callable(scheduler_class):  # For Karras or any special instantiation
            pipe.scheduler = scheduler_class(pipe.scheduler.config)
        else:
            pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
        current_scheduler = scheduler_name
        return f"Switched to scheduler: {scheduler_name}"
    else:
        return f"Scheduler {scheduler_name} not found."

# Inference function
def infer(prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps):
    if current_model_key is None:
        return "No model loaded."
    
    if seed == -1:  # -1 indicates random seed
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    pipe = models[current_model_key]
    
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
    # Stable Diffusion Model Switcher with Schedulers
    """)

        # Model URL input and model loading button
        model_url_input = gr.Textbox(label="Model URL", placeholder="Enter model download link")
        load_model_button = gr.Button("Load Model")
        load_model_output = gr.Markdown()
        
        # Scheduler switcher
        scheduler_switch = gr.Dropdown(
            label="Switch Scheduler",
            choices=["Euler", "DDIM", "PNDM", "LMS", "DPM-Solver", "DPM-Solver++ Karras", "DPM2"],
            value="Euler",
        )
        switch_scheduler_button = gr.Button("Switch Scheduler")
        switch_scheduler_output = gr.Markdown()
        
        with gr.Group():
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt", max_lines=3)
                run_button = gr.Button("🚀 Run")
        
        result = gr.Image(label="Result")
        
        # Settings accordion
        with gr.Group():
            with gr.Accordion("⚙️ Settings", open=False):
                negative_prompt = gr.Textbox(label="Negative prompt", placeholder="Enter a negative prompt",
                                          lines=3, value='lowres, text, error, cropped, worst quality, low quality')
                seed = gr.Slider(label="Seed (-1 for random)", minimum=-1, maximum=MAX_SEED, step=1, value=-1)
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                    height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                with gr.Row():
                    guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=10.0, step=0.1, value=5.0)
                    num_inference_steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=20)

        # Model load action
        load_model_button.click(
            fn=load_model,
            inputs=model_url_input,
            outputs=load_model_output
        )

        # Scheduler switch action
        switch_scheduler_button.click(
            fn=switch_scheduler,
            inputs=scheduler_switch,
            outputs=switch_scheduler_output
        )

        # Inference action
        run_button.click(
            fn=infer,
            inputs=[prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps],
            outputs=result,
        )

if __name__ == "__main__":
    app.launch(share=True, inline=False, inbrowser=False, debug=True)
