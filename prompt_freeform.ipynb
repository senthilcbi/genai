"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""

from pathlib import Path
import google.generativeai as genai

genai.configure(api_key="AIzaSyBNxQYAe67nbq9fxxBhdpiBDDQIOA_CieM")

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
  "stop_sequences": [
    "10",
  ],
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro-vision-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

print("The current path in co-lab is as below");

from google.colab import drive
drive.mount('/content/gdrive')





# Validate that an image is present
if not (img := Path("/content/gdrive/MyDrive/AI_prompt_inputfiles/indianparliment.png")).exists():
  raise FileNotFoundError(f"Could not find image: {img}")

image_parts = [
  {
    "mime_type": "image/png",
    "data": Path("/content/gdrive/MyDrive/AI_prompt_inputfiles/indianparliment.png").read_bytes()
  },
]

prompt_parts = [
  "look at the following picture and tell me what is the number of floors?",
  image_parts[0],
]

response = model.generate_content(prompt_parts)
print(response.text)
