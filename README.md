#Explanation Bottleneck Models (MINT Workshop @ NeurIPS2024 / AAAI2025)
## Requirements
### Software Requirements
* CUDA >= 12.3
### Python Requirements
* Please see `requirements.txt` or `apptainer.def`

## Preparations
Here, we describe the preparation for the experiments on StanfordCars.
You can use other datasets by modifying the preparation scripts.
### Download pre-trained BLIP backbone (from [the official repository](https://github.com/salesforce/BLIP))
```sh
mkdir pretrained && cd pretrained
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth
cd ../
```
### Target Datset: Car
  1. Download the dataset from [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) including `{train,test}_annos.npz`
  2. Install the dataset into `./data/StanfordCars`
  3. Run the preparation script as follows:
```sh
  cd ./data/StanfordCars/
  python3 ../script/split_train_test.py
  cd ../../
```

## Example
### Run Training

```sh
python3 train_exbm.py --config=configs/car_exbm.yaml
```
