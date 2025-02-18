# [Explanation Bottleneck Models](https://arxiv.org/abs/2409.17663) (AAAI2025 Oral)
XBM is an interpretable model that generates text explanations for the input embedding with respect to target tasks and then predicts final task labels from the explanations.

<img width="1229" alt="image" src="https://github.com/user-attachments/assets/036f0198-bd9f-4f6a-b435-d882b61826c7" />

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
## Citation
```
@inproceedings{yamaguchi_AAAI25_XBM,
  title={Explanation Bottleneck Models,
  author={Yamaguchi, Shin'ya and Nishida, Kosuke},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
