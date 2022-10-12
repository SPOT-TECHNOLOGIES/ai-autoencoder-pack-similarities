# ai-autoencoder-pack-similarities
Autoencoder-based similiraty computation for packages
![image](https://user-images.githubusercontent.com/2066122/195456099-b57321a6-85ac-4a6a-b6eb-72bcdb6230d7.png)

## Dependencies
- python > 3.7
- pytorch 1.11.0+cu102
- tqdm


## Inputs
json file with two images (for now):
```
{
"pallet_a":b64
"pallet_b":b64
}

```

## Outputs (for now)
json file:
```
{
"similarity": float
}

```
## Training
You can train this autoencoder using the following file format inside autoencoder/data/datasets/
```
-datasets
--- val/
--- train/
--- test/
```
and running:
```
python autoencoder/tools/do_train.py
```

## Request url

Soon
