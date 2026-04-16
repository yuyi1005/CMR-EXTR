import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import json
import re
import hashlib
import torch


values_to_extract = {
    'HEIGHT': 'Height of the Patient in cm',
    'WEIGHT': 'Weight of the Patient in kg',
    'BSA': 'Body Surface Area (BSA) of the Patient in m^2',
    'SBP': 'Systolic Blood Pressure (BP) (or the first value of BP) of the Patient in mmHg',
    'DBP': 'Diastolic Blood Pressure (BP) (or the second value of BP) of the Patient in mmHg',
    'BHR': 'Baseline Heart Rate (HR) of the Patient in BPM',

    'LVEDV': 'End-Diastolic Volume (EDV) of the Left Ventricle (LV) in mL',
    'RVEDV': 'End-Diastolic Volume (EDV) of the Right Ventricle (RV) in mL',
    'LVESV': 'End-Systolic Volume (ESV) of the Left Ventricle (LV) in mL',
    'RVESV': 'End-Systolic Volume (ESV) of the Right Ventricle (RV) in mL',
    'LVCO': 'Cardiac Output (CO) of the Left Ventricle (LV) in L/min',
    'RVCO': 'Cardiac Output (CO) of the Right Ventricle (RV) in L/min',
    'LVMASS': 'Total Mass (MASS) of the Left Ventricle (LV) in g',
    'RVMASS': 'Total Mass (MASS) of the Right Ventricle (RV) in g',
    'LVSV': 'Stroke Volume (SV) of the Left Ventricle (LV) in mL',
    'RVSV': 'Stroke Volume (SV) of the Right Ventricle (RV) in mL',
    'LVEF': 'Ejection Fraction (EF) of the Left Ventricle (LV) in %',
    'RVEF': 'Ejection Fraction (EF) of the Right Ventricle (RV) in %',

    'LVEDVI': 'End-Diastolic Volume (EDV) of the Left Ventricle (LV) Indexed to BSA in mL/m^2',
    'RVEDVI': 'End-Diastolic Volume (EDV) of the Right Ventricle (RV) Indexed to BSA in mL/m^2',
    'LVESVI': 'End-Systolic Volume (ESV) of the Left Ventricle (LV) Indexed to BSA in mL/m^2',
    'RVESVI': 'End-Systolic Volume (ESV) of the Right Ventricle (RV) Indexed to BSA in mL/m^2',
    'LVCOI': 'Cardiac Output (CO) of the Left Ventricle (LV) Indexed to BSA in L/min/m^2',
    'RVCOI': 'Cardiac Output (CO) of the Right Ventricle (RV) Indexed to BSA in L/min/m^2',
    'LVMASSI': 'Total Mass (MASS) of the Left Ventricle (LV) Indexed to BSA in g/m^2',
    'RVMASSI': 'Total Mass (MASS) of the Right Ventricle (RV) Indexed to BSA in g/m^2',
    'LVSVI': 'Stroke Volume (SV) of the Left Ventricle (LV) Indexed to BSA in mL/m^2',
    'RVSVI': 'Stroke Volume (SV) of the Right Ventricle (RV) Indexed to BSA in mL/m^2',

    'LVEDD': 'End-Diastolic Diameter (EDD) of the Left Ventricle (LV) in cm',
    'RVEDD': 'End-Diastolic Diameter (EDD) of the Right Ventricle (RV) in cm',
    'LVESD': 'End-Systolic Diameter (ESD) of the Left Ventricle (LV) in cm',
    'RVESD': 'End-Systolic Diameter (ESD) of the Right Ventricle (RV) in cm',
    'LVAWT': 'Anteroseptal Wall Thickness of the Left Ventricle (LV) in cm',
    'LVIWT': 'Inferolateral Wall Thickness of the Left Ventricle (LV) in cm',
    'LAV': 'Volume of the Left Atrium (LA) in mL, not area',
    'LAVI': 'Volume of the Left Atrium (LA) Indexed to BSA in mL/m^2',
    'RAV': 'Volume of the Right Atrium (RA) in mL, not area',
    'RAVI': 'Volume of the Right Atrium (RA) Indexed to BSA in mL/m^2',
    'LAA2CH': 'Area of the Left Atrium (LA) in 2 Chamber View in cm^2, not volume',
    'RAA2CH': 'Area of the Right Atrium (RA) in 2 Chamber View in cm^2, not volume',
    'LAA4CH': 'Area of the Left Atrium (LA) in 4 Chamber View in cm^2, not volume',
    'RAA4CH': 'Area of the Right Atrium (RA) in 4 Chamber View in cm^2, not volume',
    'LAL2CH': 'Length of the Left Atrium (LA) in 2 Chamber View in cm, not diameter',
    'RAL2CH': 'Length of the Right Atrium (RA) in 2 Chamber View in cm, not diameter',
    'LAL4CH': 'Length of the Left Atrium (LA) in 4 Chamber View in cm, not diameter',
    'RAL4CH': 'Length of the Right Atrium (RA) in 4 Chamber View in cm, not diameter',

    'PRET1M': 'Pre-contrast T1 of myocardium in msec',
    'PRET1B': 'Pre-contrast T1 of cavity (or blood) in msec',
    'POSTT1M': 'Post-contrast T1 of myocardium in msec',
    'POSTT1B': 'Post-contrast T1 of cavity (or blood) in msec',
    'HCT': 'Hematocrit (HCT) in %',
    'ECV': 'Extracellular Volume Fraction (ECV) in %',
}


def load_dataset(path, fold=0):
    data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if "_FullReport.txt" in file:
                # Use the checksum of file name to split train/val
                if int(hashlib.md5(file.encode()).hexdigest(), 16) % 5 == fold:
                    continue

                report_file = os.path.join(path, file)
                gt_file = report_file.replace('_FullReport.txt', '_GT.json')

                with open(report_file, 'r') as f:
                    report_text = f.read()
                    report_text = report_text.replace('²', '^2')
                    report_text = re.sub(r'\n+', '\n', report_text)
                    report_text = re.sub(r"-+", "-", report_text)
                    report_text = re.sub(r"=+", "=", report_text)
                    remove_pattern = re.compile(r"MRN|Name|DOB|Date|Account|Physician|Nurse|Technologist", re.IGNORECASE)
                    report_text = "\n".join(line for line in report_text.splitlines() if not remove_pattern.search(line))
                    report_text = report_text.encode('ascii', errors='ignore').decode('ascii')

                with open(gt_file, 'r') as f:
                    item = json.load(f)
                    gt_text = json.dumps(item['structured'] | {'CATEGORY': item['diseasetag']})

                prompt = (
                    "### Instruction:\n"
                    "From the following report, extract these metrics and output as JSON.\n"
                    f"{values_to_extract}\n\n"
                    "### Input:\n"
                    f"{report_text}\n\n"
                    "### Response:\n"
                )
                data.append({"input": prompt, "output": gt_text})

    print(f'{len(data)} data loaded. The first data is being printed.\n{data[0]["input"]}{data[0]["output"]}')
    return data


if __name__ == "__main__":
    # 1. Load dataset
    train_data = load_dataset('/path/to/datasets/CMR-Extracted-GPT-OSS')

    # 2. Save dataset in HF format
    dataset = Dataset.from_list(train_data)

    # 3. Load model + tokenizer
    model_path = "/path/to/models"
    model_name = "meta-llama-3.2-1b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, model_name))
    tokenizer.pad_token = tokenizer.eos_token

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="fp8", 
        bnb_8bit_use_double_quant=True,
        bnb_8bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_path, model_name),
        device_map={"": torch.device(f"cuda:{local_rank}")},
        quantization_config=bnb_config,
    )

    # 4. Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)

    # 5. Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(examples["input"] + examples["output"], truncation=True, padding="max_length", max_length=4096)
        input_len = len(tokenizer(examples["input"], truncation=True, max_length=4096)["input_ids"])
        tokenized["labels"] = tokenized["input_ids"].copy()
        tokenized["labels"][:input_len] = [-100] * input_len
        return tokenized

    tokenized = dataset.map(tokenize_function, remove_columns=dataset.column_names)

    # 6. Training arguments
    args = TrainingArguments(
        output_dir="./llama-3.2-1b-finetuned",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        ddp_find_unused_parameters=False
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer
    )

    trainer.train()

    # 8. Save adapter
    model.save_pretrained("./llama-3.2-1b-finetuned")
