# Uncertainty-Aware Structured Data Extraction from Full CMR Reports via Distilled LLMs

CMR-EXTR is a lightweight framework for extracting structured data from free-text cardiac magnetic resonance (CMR) reports. The model is trained via a teacher–student distillation pipeline and supports efficient offline inference with uncertainty-aware outputs.

---

## Model

Download the pretrained model from Hugging Face:

https://huggingface.co/yuyi1005/cmrextr-1b

---

## Inference

To run inference with the pretrained CMR-EXTR model:

```bash
python inference_cmrextr-1b.py
```

- Example inputs are provided in `./data/`
- The script runs a demo using these example reports

---

## Training

The training pipeline follows a teacher–student distillation framework.

### Step 1: Generate pseudo-labels with teacher model

Run GPT-OSS-20B inference:

```bash
python inference_gpt-oss-20b.py
```

---

### Step 2: Review the labels based on the scores

In the json files output by inference_gpt-oss-20b.py, there are the extracted values and the corresponding scores. Review and correct them if necessary.

---

### Step 3: Train the student model

Fine-tune LLaMA-3.2-1B:

```bash
python finetune_llama-3.2-1b.py
```

---

### Step 4: Merge LoRA adapter

Merge the trained adapter into the base model to obtain the final model:

```bash
python merge_llama-3.2-1b.py
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yu2026uncertainty,
  title={Uncertainty-Aware Structured Data Extraction from Full CMR Reports via Distilled LLMs},
  author={Yu, Yi and Martin, Parker and Bu, Zhenyu and Liu, Yixuan and Zheng, Yi-Yu and Simonetti, Orlando and Han, Yuchi and Xue, Yuan},
  booktitle={IEEE 23rd International Symposium on Biomedical Imaging (ISBI)},
  year={2026}
}
```

---

## Contact

For questions or collaborations, please open an issue.
