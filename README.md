<p align="center">
  <h1 align="center">Uncertainty-Aware Structured Data Extraction from Full CMR Reports via Distilled LLMs</h1>
  <p align="center">
    Yi Yu&ensp;
    Parker Martin&ensp;
    Zhenyu Bu&ensp;
    Yixuan Liu&ensp;
    Yi-Yu Zheng&ensp;
    Orlando Simonetti&ensp;
    Yuchi Han&ensp;
    Yuan Xue
  </p>
  <div align="center">
      <a href='https://arxiv.org/abs/2605.08045'><img src='https://img.shields.io/badge/arXiv-2605.08045-brown.svg?logo=arxiv&logoColor=white'></a>
  </div>
  <p align="center">
    If you find our work helpful, please consider giving us a ⭐!
  </p>
</p>

## Introduction

CMR-EXTR is a lightweight framework for extracting structured data from free-text cardiac magnetic resonance (CMR) reports. The model is trained via a teacher–student distillation pipeline and supports efficient offline inference with uncertainty-aware outputs.

## Model

Download the pretrained model from Hugging Face:

https://huggingface.co/yuyi1005/cmrextr-1b

## Inference

To run inference with the pretrained CMR-EXTR model:

```bash
python inference_cmrextr-1b.py
```

- Example inputs are provided in `./data/`
- The script runs a demo using these example reports

## Training

The training pipeline follows a teacher–student distillation framework.

### Step 1: Generate pseudo-labels with teacher model

Run GPT-OSS-20B inference:

```bash
python inference_gpt-oss-20b.py
```

### Step 2: Review the labels based on the scores

In the json files output by inference_gpt-oss-20b.py, there are the extracted values and the corresponding scores. Review and correct them if necessary.

### Step 3: Train the student model

Fine-tune LLaMA-3.2-1B:

```bash
python finetune_llama-3.2-1b.py
```

### Step 4: Merge LoRA adapter

Merge the trained adapter into the base model to obtain the final model:

```bash
python merge_llama-3.2-1b.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yu2026uncertainty,
  title={Uncertainty-Aware Structured Data Extraction from Full CMR Reports via Distilled LLMs},
  author={Yu, Yi and Martin, Parker and Bu, Zhenyu and Liu, Yixuan and Zheng, Yi-Yu and Simonetti, Orlando and Han, Yuchi and Xue, Yuan},
  booktitle={IEEE International Symposium on Biomedical Imaging},
  year={2026}
}
```

## Contact

For questions or collaborations, please open an issue.
