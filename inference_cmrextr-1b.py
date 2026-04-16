from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import json
import re
import hashlib
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


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


json_schema = {
        "type": "object",
        "properties": {
            k: {"type": "string", "enum": ["CAD", "HCM", "DCM", "Ebstein", "PAH"], "description": v} if k == "CATEGORY" \
                else {"type": ["number", "null"], "description": v} for k, v in values_to_extract.items()
        },
        "required": list(values_to_extract.keys())
    }
print(json_schema)


def load_dataset(path, fold=0):
    data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if "_FullReport.txt" in file:
                # If json file exist, skip
                if any(f.endswith("_Extr.json") for f in os.listdir(root)):
                    continue
                
                report_file = os.path.join(root, file)
                with open(report_file, 'r', encoding="utf-8", errors="ignore") as f:
                    report_text = f.read()
                    report_text = report_text.replace('²', '^2')
                    report_text = re.sub(r'\n+', '\n', report_text)
                    report_text = re.sub(r"-+", "-", report_text)
                    report_text = re.sub(r"=+", "=", report_text)
                    remove_pattern = re.compile(r"MRN|Name|DOB|Date|Account|Physician|Nurse|Technologist", re.IGNORECASE)
                    report_text = "\n".join(line for line in report_text.splitlines() if not remove_pattern.search(line))
                    report_text = report_text.encode('ascii', errors='ignore').decode('ascii')

                prompt = (
                    "### Instruction:\n"
                    "From the following report, extract these metrics and output as JSON.\n"
                    f"{values_to_extract}\n\n"
                    "### Input:\n"
                    f"{report_text}\n\n"
                    "### Response:\n"
                )
                data.append({"input": prompt,
                             'patient': file.replace('_FullReport.txt', ''),
                             'root': root})

    print(f'{len(data)} data loaded. The first data is being printed.\n{data[0]["input"]}')
    return data


if __name__ == "__main__":
    # 1. Load dataset
    train_data = load_dataset('./data')

    # Split data
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    train_data = train_data[rank::world_size]
    print(f'rank {rank} is dealing with {len(train_data)} data')

    # Create output folder
    dst_dir = './data'
    os.makedirs(dst_dir, exist_ok=True)

    llm = LLM(model="./cmrextr-1b",
             gpu_memory_utilization=0.9)

    structured_outputs_params = StructuredOutputsParams(json=json_schema)
    params = SamplingParams(max_tokens=4096, temperature=0.0, structured_outputs=structured_outputs_params)

    sum_correct = 0
    sum_wrong = 0
    for data in train_data:
        prompt = data["input"]

        try:
            outputs = llm.generate([prompt], sampling_params=params)[0]
            generated_text = outputs.outputs[0].text
            extracted = json.loads(generated_text)

            output_file = os.path.join(data["root"], data["patient"] + '_Extr.json')
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(extracted, f, ensure_ascii=True, indent=4)
            
        except Exception as e:
            print(f"{data['patient']} Failed.\n{generated_text}\n{e}")
            print(f"Finish reason: {outputs.outputs[0].finish_reason}")
            print(f"Prompt tokens: {len(outputs.prompt_token_ids)}")
            print(f"Output tokens: {len(outputs.outputs[0].token_ids)}")
            print(f"Total tokens: {len(outputs.prompt_token_ids) + len(outputs.outputs[0].token_ids)}")
