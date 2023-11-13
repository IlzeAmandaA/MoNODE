# Modulated Neural ODEs (MoNODE)

Pytorch implementation of Modulated Neural ODEs

## Replicating the experiments
The code was developed and tested with `python3.8` and `pytorch 1.13`. You can install the relevant packages via 
```
pip install -r requirements.txt
```

## Data
Data is generated on the fly in the folder `data`. In this folder you can also find the data configuration file: `config.yml`. In total the code enables to generate 5 different datasets:
- Sinusoidal data
- Predator Prey data
- Bouncing Balls
- Rotating MNIST
- Mocap

Upon generation a folder for the corresponding dataset is created with a train, valid, and test dataset, respectively. 

For Mocap data, please download the files reported in the appendix from http://mocap.cs.cmu.edu/subjects.php. 

## Train

To the train the models run the `main.py` file with passing the relevant arguments. For example commands for each dataset please see the `commands.txt` file. 

An example commands for sine data with MoNODE model:
```
python main.py --task sin --model node --solver rk4 --batch_size 10 --T_in 3 --T_inv 10 --ode_latent_dim 4 --modulator_dim 4 --Nepoch 600
```

## Results

All log files, figures and trained model will be stored automatically in a results folder.

