### Setting Up the Data

After setting up the environment, run `data.py` to process the data from the source:

```bash
python src/dog_breed_classifier/data.py
```
That will download the original dataset, preprocess it and save the train, test and validation datasets in the /data folder.

---

### Running Training Sessions

#### Building from scratch
In Linux, Git Bash, or WSL, you can build both the processed dataset and the model by running
```bash
chmod +x scripts/run_all.sh
./scripts/run_all.sh
```

#### Train A Model Individually
To train a model on a specific configuration, run
```bash
python src/dog_breed_classifier/train.py local
```
You'll find the hyperparameters for a single training session in /config/config.yaml

To do the training with wandb, run it with the flag wandb-run

```bash
python src/dog_breed_classifier/train.py wandb-run
```

#### Parameter sweep
The config file with the setup can be found at /configs/sweep.yaml.
To run this session:

```bash
python src/dog_breed_classifier/train.py sweep
```


### Updating `requirements.txt`

To update the `requirements.txt` file based on your current environment:

1. Install `pipreqs`:
   ```bash
   pip install pipreqs
   ```

2. Generate an updated `requirements.txt`:
   ```bash
   pipreqs . --force
   ```

3. When installing from source, bear in mind that the pytorch version will differ depending on the available hardware, preferrably delete it from requirements.txt and install it as explained before.
