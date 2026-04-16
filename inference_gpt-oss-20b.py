import os
import numpy as np
import re
import ast
import shutil
import time
import json
from vllm import LLM, SamplingParams


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


def confidence_function(a, b, k=2.0, eps=1e-8):
    if a == None and b == None:
        return 1.0
    if a == None or b == None:
        return 0.5
    diff = abs(a - b) / (abs(a + b) / 2 + eps)
    c = np.exp(-k * diff)
    return c


def find_last_number_end(text):
    # Match a number at the very end of the string
    match = re.search(r'(\d+(?:\.\d+)?)\s*$', text)
    if match:
        return float(match.group(1))
    return None


def majority_vote(v):
    """
    Given a list or tuple v of length 3, return the most-agreed value.
    If all three values differ, return v[0].
    None is treated as a valid comparable value.
    """
    if v[0] == v[1] or v[0] == v[2]:
        return v[0]
    elif v[1] == v[2]:
        return v[1]
    else:
        return v[0]
    

def extract_one_report(path, llm):
    with open(path, 'r') as f:
        report_text = f.read()
        report_text = report_text.replace('²', '^2')
        report_text = re.sub(r'\n+', '\n', report_text)
        report_text = re.sub(r"-+", "-", report_text)
        report_text = re.sub(r"=+", "=", report_text)
        remove_pattern = re.compile(r"MRN|Name|DOB|Date|Account|Physician|Nurse|Technologist", re.IGNORECASE)
        report_text = "\n".join(line for line in report_text.splitlines() if not remove_pattern.search(line))
        report_text = report_text.encode('ascii', errors='ignore').decode('ascii')
    
    structured = {}
    confidence = {}
    reasonings = {}
    for k, v in values_to_extract.items():
        try:
            prompts = (
                f"Instruction:\n"
                f"- From the following CMR report, extract {v}.\n"
                f"- Only use information explicitly stated in the report.\n"
                f"- Do not compute from other values (such as CO from HBR or Indexed values from BSA).\n"
                f"- Do not guess, infer, or reinterpret mistakes.\n"
                f"- If the value is not present, return None.\n"
                f"- Output exactly one number or None, with no extra text, unit, or symbol.\n\n"
                f"Report:\n{report_text}"
            )
            params = SamplingParams(temperature=0.3, max_tokens=2048, top_p=0.8, top_k=0, n=1)
            texts = []
            for _ in range(10):
                outputs = llm.generate([prompts], sampling_params=params)
                # print(f"{outputs[0].outputs[0].text}\n----------\n")
                if outputs[0].outputs[0].finish_reason == "stop":
                    texts.append(outputs[0].outputs[0].text)
                if len(texts) >= 3:
                    break

            reasonings[k] = [texts[0], texts[1], texts[2]]
            v = list(map(find_last_number_end, texts))
            structured[k] = majority_vote(v)
            confidence[k] = [(confidence_function(v[0], v[1]) +
                              confidence_function(v[0], v[2]) +
                              confidence_function(v[1], v[2])) / 3]

        except Exception as e:
            print(f"{e}")
            structured[k] = None
            confidence[k] = [0.0]
            reasonings[k] = None

    return structured, confidence, reasonings


def verify_formula(S, formula_str):
    try:
        lhs_var, rhs_expr = formula_str.split("=")
        lhs_var = lhs_var.strip()
        rhs_expr = rhs_expr.strip()

        # Evaluate RHS using values from S
        rhs_value = eval(rhs_expr, {}, S)
        lhs_value = S.get(lhs_var)

        if lhs_value is None:
            raise ValueError(f"{lhs_var} is None.")

        # Compute confidence
        return confidence_function(lhs_value, rhs_value)

    except Exception as e:
        print(f"Error verifying formula '{formula_str}': {e}")
        return None


def confidence_consisteny(structured, confidence):
    formulas = {
        'BSA = (HEIGHT * WEIGHT / 3600)**0.5': ['BSA', 'HEIGHT', 'WEIGHT'],

        'LVEF = (LVEDV - LVESV) / LVEDV * 100': ['LVEF', 'LVEDV', 'LVESV'],
        'RVEF = (RVEDV - RVESV) / RVEDV * 100': ['RVEF', 'RVEDV', 'RVESV'],
        'LVSV = LVEDV - LVESV': ['LVSV', 'LVEDV', 'LVESV'],
        'RVSV = RVEDV - RVESV': ['RVSV', 'RVEDV', 'RVESV'],
        'LVCO = LVSV * BHR / 1000': ['LVCO', 'LVSV', 'BHR'],
        'RVCO = RVSV * BHR / 1000': ['RVCO', 'RVSV', 'BHR'],
        'LVEF = LVSV / LVEDV * 100': ['LVEF', 'LVSV', 'LVEDV'],
        'RVEF = RVSV / RVEDV * 100': ['RVEF', 'RVSV', 'RVEDV'],

        'BSA = LVEDV / LVEDVI': ['BSA', 'LVEDV', 'LVEDVI'],
        'BSA = RVEDV / RVEDVI': ['BSA', 'RVEDV', 'RVEDVI'],
        'BSA = LVESV / LVESVI': ['BSA', 'LVESV', 'LVESVI'],
        'BSA = RVESV / RVESVI': ['BSA', 'RVESV', 'RVESVI'],
        'BSA = LVCO / LVCOI': ['BSA', 'LVCO', 'LVCOI'],
        'BSA = RVCO / RVCOI': ['BSA', 'RVCO', 'RVCOI'],
        'BSA = LVMASS / LVMASSI': ['BSA', 'LVMASS', 'LVMASSI'],
        'BSA = RVMASS / RVMASSI': ['BSA', 'RVMASS', 'RVMASSI'],
        'BSA = LVSV / LVSVI': ['BSA', 'LVSV', 'LVSVI'],
        'BSA = RVSV / RVSVI': ['BSA', 'RVSV', 'RVSVI'],
        'BSA = LAV / LAVI': ['BSA', 'LAV', 'LAVI'],
        'BSA = RAV / RAVI': ['BSA', 'RAV', 'RAVI'],

        'ECV = (100 - HCT) * (1 / POSTT1M - 1 / PRET1M) / (1 / POSTT1B - 1 / PRET1B)': ['ECV', 'HCT', 'POSTT1M', 'PRET1M', 'POSTT1B', 'PRET1B'],
    }

    verified = {}
    for f, keys in formulas.items():
        c = verify_formula(structured, f)
        if c is not None:
            for k in keys:
                if k not in verified.keys():
                    verified[k] = []
                verified[k].append(c)
    
    for k in confidence.keys():
        if k in verified.keys():
            confidence[k].append(sum(verified[k]) / len(verified[k]))
        else:
            confidence[k].append(0.7)

    return confidence


def min_normalized_prob(value, dist):
    if dist is None:
        return None
    distances = [abs(value - mean) / (std * 6) for mean, std in dist if std != 0]
    return np.exp(-0.5 * min(distances)**2) if distances else None


def confidence_distribution(structured, confidence):
    distributions = {
        "LVEDV": [[143.0, 31.0], [113.0, 23.0]],
        "LVESV": [[52.0, 18.0], [39.0, 13.0]],
        "LVCO": [[6.1, 1.3], [4.9, 1.1]],
        "LVMASS": [[105.0, 23.0], [74.0, 14.0]],
        "LVSV": [[91.0, 19.0], [74.0, 15.0]],
        "LVEF": [[64.0, 7.0], [66.0, 6.0]],
        "LVEDVI": [[75.0, 14.0], [68.0, 11.0]],
        "LVESVI": [[28.0, 8.0], [23.0, 6.0]],
        "LVCOI": [[3.2, 0.6], [3.0, 0.6]],
        "LVMASSI": [[55.0, 8.0], [45.0, 6.0]],
        "LVSVI": [[47.0, 9.0], [44.0, 8.0]],
        "RVEDV": [[152.0, 40.0], [115.0, 29.0]],
        "RVESV": [[65.0, 20.0], [46.0, 15.0]],
        "RVCO": [[5.5, 1.6], [4.6, 1.2]],
        "RVMASS": [[36.0, 9.0], [29.0, 7.0]],
        "RVSV": [[89.0, 29.0], [71.0, 20.0]],
        "RVEF": [[58.0, 7.0], [61.0, 7.0]],
        "RVEDVI": [[82.0, 18.0], [71.0, 14.0]],
        "RVESVI": [[34.0, 9.0], [28.0, 8.0]],
        "RVCOI": [[2.9, 0.9], [2.7, 0.7]],
        "RVMASSI": [[18.0, 4.0], [16.0, 4.0]],
        "RVSVI": [[48.0, 14.0], [45.0, 10.0]],
        "LVEDD": [[5.2, 0.5], [4.8, 0.4]],
        "LVESD": [[3.4, 0.3], [3.1, 0.4]],
        "LVAWT": [[0.9, 0.16], [0.74, 0.14]],
        "LVIWT": [[0.78, 0.14], [0.63, 0.12]],
        "LAA2CH": [[21.0, 4.0], [18.0, 4.0]],
        "LAA4CH": [[21.0, 4.0], [19.0, 4.0]],
        "LAL2CH": [[4.9, 0.7], [4.6, 0.7]],
        "LAL4CH": [[5.7, 0.7], [5.4, 0.7]],
        "RAA2CH": [[23.0, 4.0], [21.0, 4.0]],
        "RAA4CH": [[21.0, 5.0], [18.0, 4.0]],
        "RAL2CH": [[5.5, 0.7], [5.1, 0.6]],
        "RAL4CH": [[5.2, 0.7], [4.9, 0.7]]
    }

    for k, v in structured.items():
        score = 0.7
        try:
            if k in distributions.keys() and v is not None:
                c = min_normalized_prob(v, distributions[k])
                if c is not None:
                    score = c
        except Exception as e:
            print(f"{e}")
        confidence[k].append(score)
    return confidence


if __name__ == "__main__":
    src_dir = '/fs/ess/PDE0069/Data/CMR'
    dst_dir = '/fs/ess/PDE0069/Data/CMR-Extracted-GPT-OSS'
    os.makedirs(dst_dir, exist_ok=True)
    confidence_file = os.path.join(dst_dir, 'confidence_scores.txt')
    open(confidence_file, "w").close()

    # 1. Load dataset
    train_data = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if "_FullReport.txt" in file:
                train_data.append([root, file])

    # Split data
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    train_data = train_data[rank::world_size]
    print(f'rank {rank} is dealing with {len(train_data)} data')
    
    llm = LLM(model="/fs/scratch/PCON0521/yuyi1005/openai-gpt-oss-20b",
             gpu_memory_utilization=0.9,
             disable_custom_all_reduce=True,
             async_scheduling=True)

    for root, file in train_data:
        if "_FullReport.txt" in file:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")

            structured, confidence, reasonings = extract_one_report(dst_file, llm)
            confidence = confidence_consisteny(structured, confidence)
            confidence = confidence_distribution(structured, confidence)
            # print(confidence)
            for k in confidence.keys():
                confidence[k] = round(sum(confidence[k]) / len(confidence[k]), 4)
            min_confidence = min(confidence.values())

            label_file = src_file.replace('_FullReport.txt', '_Label.txt')
            with open(label_file, 'r') as f:
                diseasetag = f.read().strip()
    
            gt_file = dst_file.replace('_FullReport.txt', '_GT.json')
            with open(gt_file, "w", encoding="utf-8") as f:
                json.dump({
                    'diseasetag': diseasetag,
                    'structured': structured,
                    'confidence': confidence,
                    'reasonings': reasonings
                    }, f, ensure_ascii=True, indent=4)

            with open(confidence_file, "a", encoding="utf-8") as f:
                f.write(f"{file} -> Confidence: {min_confidence}\n")
