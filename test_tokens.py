import sys
sys.path.append(r'c:\Users\Funky\Desktop\cs4248\api-inference')
from transformers import AutoProcessor
from PIL import Image
processor = AutoProcessor.from_pretrained(r'c:\Users\Funky\Desktop\cs4248\models\Qwen3-VL-2B-Instruct')
image = Image.new("RGB", (768, 768), (0, 0, 0))
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe"}
        ]
    }
]
prompt = processor.apply_chat_template(messages, tokenize=False)
inputs = processor(text=[prompt], images=[image], return_tensors="pt")
print('Input tokens:', inputs.input_ids.shape)