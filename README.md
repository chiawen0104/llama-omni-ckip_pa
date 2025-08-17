# llama-omni-ckip_pa
A CKIP Lab project on applying [Llama-Omni](https://github.com/ictnlp/LLaMA-Omni) for english pronunciation assessment. This repository adapts the training code of Llama-Omni reproduced by [wntg](https://github.com/wntg/LLaMA-Omni).

## Create Conda Environment
1. Clone the repository.
   ```
   git clone https://github.com/ictnlp/LLaMA-Omni
   cd LLaMA-Omni
   ```
2. Install packages.
   ```
   conda create -n llama-omni python=3.10
   conda activate llama-omni
   pip install pip==24.0
   pip install -e .
   ```
3. Install `fairseq`.
   ```
   git clone https://github.com/pytorch/fairseq
   cd fairseq
   pip install -e . --no-build-isolation
   ```
4. Install `flash-attention` (v2).
   ```
   pip install flash-attn --no-build-isolation
   ```
   If the installation fails, please visit [here](https://github.com/Dao-AILab/flash-attention/releases) to download the wheel files, and then rerun the above command.

## Installation
1. Clone this repository.
   ```
   git https://github.com/chiawen0104/llama-omni-ckip_pa
   cd llama-omni-ckip_pa
   ```
2. Download the `Llama-3.1-8B-Omni` model from [Huggingface](https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni).
   ```
   pip install huggingface_hub
   huggingface-cli login
   ```
   ```
   huggingface-cli download ICTNLP/Llama-3.1-8B-Omni --local-dir ./Llama-3.1-8B-Omni
   ```
3. Download the `Whisper-large-v3` model.
   ```
   import whisper
   model = whisper.load_model("large-v3", download_root="models/speech_encoder/")
   ```
4. Download the unit-based HiFi-GAN vocoder.
   ```
   wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 -P vocoder/
   wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json -P vocoder/
   ```
5. Download the `HuBERT` model and quantizer from [fairseq](https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_to_speech/docs/textless_s2st_real_data.md#hubert).
   ```
   wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt -P models/wav2unit
   wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin -P models/wav2unit
   ```
6. Download wav files (`speechocean/WAVE`) from [speechocean762](https://github.com/jimbozhang/speechocean762).

## Usage
### Train
#### Stage1    
```
bash omni_speech/train/run_stage1.sh
```
#### Stage2
```
bash omni_speech/train/run_stage2.sh
```
### Inference
#### Stage1
```
bash omni_speech/infer/run_infer1.sh speechocean/
```
#### Stage2
```
bash omni_speech/infer/run_infer2.sh speechocean/
```
### wav2unit (prepare training data for stage2)
Step1: ```bash run_dump_hubert_feature.sh```   
Step2: ```bash run_dump_km_label.sh```



