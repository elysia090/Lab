import pandas as pd
import random
import torch
import os
import contextlib
from gemma.config import get_config_for_2b, get_config_for_7b
from gemma.model import GemmaForCausalLM

# Function to load Gemma model and tokenizer
def load_gemma_model_and_tokenizer(variant, machine_type, weights_dir):
    # Model Config
    model_config = get_config_for_2b() if "2b" in variant else get_config_for_7b()
    model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")
    model_config.quant = "quant" in variant

    # Model
    device = torch.device(machine_type)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = GemmaForCausalLM(model_config)
        ckpt_path = os.path.join(weights_dir, f'gemma-{variant}.ckpt')
        model.load_weights(ckpt_path)
        model = model.to(device).eval()
    
    return model, model.tokenizer, device

# Function to generate rewritten texts
def generate_rewritten_texts(original_texts, rewrite_prompts, model, tokenizer, device):
    rewrite_data = []

    for original_text in original_texts:
        rewrite_prompt = random.choice(rewrite_prompts)
        prompt = f'{rewrite_prompt}\n{original_text}'
        
        # Generate rewritten text
        with torch.no_grad():
            rewritten_tokens = model.generate(
                f'<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n',
                device=device,
                output_len=100,
            )
        
        # Decode token IDs to text
        rewritten_text = tokenizer.decode(rewritten_tokens.tolist()[0])

        rewrite_data.append({
            'original_text': original_text,
            'rewrite_prompt': rewrite_prompt,
            'rewritten_text': rewritten_text,
        })

    return pd.DataFrame(rewrite_data)

def main():
    # Load forum messages data
    forum_messages_df = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')
    
    # Select first 5 messages for testing
    original_texts = forum_messages_df['Message'][:5].tolist()
    
    # Rewrite prompts
    rewrite_prompts = [
        'Explain this to me like I\'m five.',
        'Convert this into a sea shanty.',
        'Make this rhyme.',
    ]
    
    # Set up Gemma model
    variant = "7b-it-quant" 
    machine_type = "cuda" 
    weights_dir = '/kaggle/input/gemma/pytorch/7b-it-quant/2' 

    # Load Gemma model and tokenizer
    model, tokenizer, device = load_gemma_model_and_tokenizer(variant, machine_type, weights_dir)
    
    # Generate rewritten texts
    rewrite_data_df = generate_rewritten_texts(original_texts, rewrite_prompts, model, tokenizer, device)
    
    # Print the first row to check if it makes sense
    print(rewrite_data_df.head(1).values)

if __name__ == "__main__":
    main()

