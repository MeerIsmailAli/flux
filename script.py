# import torch
# from diffusers import FluxPipeline


# #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


# # Set the default tensor type to CPU
# torch.set_default_tensor_type(torch.FloatTensor)

# # Load the pipeline
# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float32)

# # Move the pipeline to CPU explicitly
# pipe.to('cpu')

# prompt = "A homeless man holding a board that says 'samiullah robbed me!' "

# try:
#     image = pipe(
#         prompt,
#         height=512,
#         width=512,
#         guidance_scale=3.0,
#         num_inference_steps=30,
#         max_sequence_length=256,
#         generator=torch.Generator("cpu").manual_seed(0)
#     ).images[0]

#     image.save("trial.png")
#     print("Image generated and saved successfully.")
# except Exception as e:
#     print(f"An error occurred: {e}")

import torch
from diffusers import FluxPipeline
from transformers import AutoTokenizer

# Ensure we're using CPU
device = "cpu"

# Load the tokenizer separately
tokenizer = AutoTokenizer.from_pretrained("black-forest-labs/FLUX.1-dev")

# Load the pipeline with 8-bit quantization
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float32,
    load_in_8bit=True,
    device_map="auto",
    tokenizer=tokenizer
)

prompt = "A cat holding a sign that says hello world"

try:
    image = pipe(
        prompt,
        height=256,
        width=256,
        guidance_scale=3.0,
        num_inference_steps=20,
        max_sequence_length=64,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]

    image.save("flux-dev-8bit.png")
    print("Image generated and saved successfully.")
except Exception as e:
    print(f"An error occurred: {e}")