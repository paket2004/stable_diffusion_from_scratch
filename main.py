from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import io
from PIL import Image
from logic_functions.preload_model import preload_models_from_standard_weights
from logic_functions.generate import generate
import sys
import os
import numpy as np
app = FastAPI()
@app.get("/")
def get_root():
    return {"Hello" : "world"}
# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# model_file = f"model_setup\v1-5-pruned-emaonly.ckpt"
# models = preload_models_from_standard_weights(model_file, "cpu")


def generate_image_from_text(text: str):
    sampler="ddpm"
    image = generate(
        prompt=text,
        uncond_prompt=None,
        strength=0.8,
        do_cfg=True,
        input_image=None,
        sampler_name=sampler,
        cfg_scale=7.5,
        n_inference_steps=50,
        models={},
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
    )
    return image

@app.get("/generate_image")
async def create_image(text: str):
    try:
        img = generate_image_from_text(text)
        img = Image.fromarray(np.uint8(img))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Return the image as a response
        return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
