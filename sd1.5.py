import torch
import datetime
from PIL import Image
from diffusers import StableDiffusionPipeline


# Path to your .ckpt file
ckpt_path = "C:/Users/leong/Downloads/AUmodel1.ckpt" #change ckpt file path

output_dir = "C:/Users/leong/Desktop/generated_images" #change output file

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

try:
    # Load the pipeline directly from the checkpoint
    pipe = StableDiffusionPipeline.from_single_file(
        ckpt_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Move the pipeline to GPU if available
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Enable memory efficient attention if using CUDA
    if torch.cuda.is_available():
        pipe.enable_attention_slicing()

    print("Model loaded successfully. Ready to generate images!")
    print(f"Images will be saved in: {output_dir}")

    # Initialize a counter for the session
    session_counter = 1

    while True:
        # Get user input for positive prompt
        prompt = input("Enter your positive prompt (or 'quit' to exit): ")
        
        if prompt.lower() == 'quit':
            break

        # Get user input for negative prompt
        negative_prompt = input("Enter your negative prompt (optional, press Enter to skip): ")

        # Get number of images to generate
        while True:
            try:
                num_images = int(input("How many images do you want to generate? "))
                if num_images > 0:
                    break
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")

        # Generate images
        for i in range(num_images):
            # Generate an image
            image = pipe(
                prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=50
            ).images[0]

            # Create a filename with a simple numbering system
            filename = f"generated_image_{session_counter:04d}.png"
            full_path = os.path.join(output_dir, filename)

            # Save the generated image
            image.save(full_path)
            print(f"Image saved as {full_path}")

            # Increment the counter
            session_counter += 1

        print(f"{num_images} image(s) generated successfully!")

except Exception as e:
    print(f"An error occurred: {str(e)}")