# ai-autoencoder-pack-similarities
Autoencoder-based similiraty computation for packages
![image](https://user-images.githubusercontent.com/2066122/195456099-b57321a6-85ac-4a6a-b6eb-72bcdb6230d7.png)

## Dependencies
- python > 3.7
- pytorch 1.11.0+cu102
- tqdm


## Inputs
json file with a query image and a collection for consulting:
```
{ "data":{
      "query":[
          {
              "pallet_id": int,
              "base64": b64
          }
      ]
      "collection":[
          {
              "pallet_id":int,
              "base64": b64
          },
          
          ...
      
      ]
  }
}

```

## Outputs (for now)
json file:
```
{
	"data": {
		"similars": [
			{
				"pallet_id": int,   ## first similar
				"similarity": float
			},
			{
				"pallet_id": int, ## second similar
				"similarity": float
			}
		]
	}
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
