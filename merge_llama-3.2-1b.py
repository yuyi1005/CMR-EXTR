import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


if __name__ == "__main__":
    # Load tokenizer
    model_path = "/path/to/models"
    model_name = "meta-llama-3.2-1b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, model_name))
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="fp8", 
        bnb_8bit_use_double_quant=True,
        bnb_8bit_compute_dtype="float16"
    )

    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_path, model_name),
        device_map="auto",
        quantization_config=bnb_config,
    )

    # Load LoRA adapter
    lora_path = "./llama-3.2-1b-finetuned"  # path where LoRA adapter was saved
    model = PeftModel.from_pretrained(model, lora_path)

    merged_model_path = "./cmrextr-1b"
    model = model.merge_and_unload()
    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"Merged model saved at {merged_model_path}")
    
