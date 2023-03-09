# Invariant Neural ODEs

Pytorch implementation of [Invariant Neural ODEs](https://arxiv.org/abs/2302.13262) paper.

## Replicating the experiments
The code was developed and tested with `python3.8` and `pytorch 1.13`. You can install the relevant packages via 
```
pip install -r requirements.txt
```

## Data
Data is generated on the fly in the folder `data`. In this folder you can also find the data configuration file: `config.yml`. In total the code enables to generate 5 different datasets:
- Sinusoidal data
- Lotka Volterra data
- Rotating MNIST
- Moving MNIST
- Bouncing Balls

Upon generation a folder for the corresponding dataset is created with a train, valid, and test dataset, respectively. 

## Train

To the train the models run the `main.py` file with passing the relevant arguments. For example commands for each dataset please see the `commands.txt` file. 

An example commands for sin data with SINODE model:
```
python main.py --task sin --solver dopri5 --batch_size 10 --T_in 3 --T_inv 10 --ode_latent_dim 4 --inv_latent_dim 4 --Nepoch 1500
```

## Test

In the `test` folder you can find .ipynb for each dataset where the trained model can be evaluated on the test data. 


