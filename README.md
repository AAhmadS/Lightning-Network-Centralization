# LNChannelOpening
Lightning Network Channel and Capacity Selection


<div>

[![Made with Python](https://img.shields.io/badge/Python->=3.8-red?logo=python&logoColor=white)](https://python.org "Go to Python homepage")
[![pandas - up-to-date](https://img.shields.io/static/v1?label=pandas&message=up-to-date&color=blueviolet)](https://pandas.pydata.org/)
[![Gym - >=0.2](https://img.shields.io/static/v1?label=Gym&message=>%3D0.2&color=black)](https://github.com/openai/gym)
[![Stable-baselines3 - >= 1.6](https://img.shields.io/static/v1?label=Stable-baselines3&message=>%3D+1.6&color=2ea44f)](https://stable-baselines3.readthedocs.io/en/master/)
[![Sb3-contrib - >= 1.6](https://img.shields.io/static/v1?label=Sb3-contrib&message=>%3D+1.6&color=green)](https://sb3-contrib.readthedocs.io/en/master/)

[![NetworkX - ~2.8.5](https://img.shields.io/static/v1?label=NetworkX&message=~2.8.5&color=brightgreen)](https://networkx.org/)
[![tensorboard - ~=2.9.1](https://img.shields.io/static/v1?label=tensorboard&message=~%3D2.9.1&color=orange)](https://www.tensorflow.org/tensorboard/get_started#:~:text=TensorBoard%20is%20a%20tool%20for,dimensional%20space%2C%20and%20much%20more.)


</div>



## How to Use

### Prerequisite


Please make sure you have installed `python3.8` or higher versions of python.


### Install


```
git clone https://github.com/AAhmadS/Lightning-Network-Centralization
cd Lightning-Network-Centralization
```
All dependencies will be handled using the command :

```pip install -r requirements.txt```


### Run

```
python3 -m scripts.ln_channel_opening --algo PPO --model Transformer --tb_name PPO_tensorboard
python3 -m scripts.baselines --strategy random
python3 -m scripts.evaluation --model --model Transformer
```

### Parameters
ln_channel_opening:

| Parameter              | Default | choices                                      |
|------------------------|--------|----------------------------------------------|
| _--algo_               | PPO    | PPO, A2C     |
| _--total_timesteps_    | 200000 | Arbitrary Integer                            |
| _--max_episode_length_ | 10    | Arbitrary Integer less than total_timesteps  |
| _--counts_             | [10, 10, 10] | List of Integers                             |
| _--amounts_            | [10000, 50000, 100000] | List of Integers |
| _--epsilons_           | [.6, .6, .6] | List of floats between 0 and 1               |
| _--capacity_upper_scale_bound_           | 25 | Arbitrary Integer               |
| _--max_capacity_           | 1e7 | Arbitrary Integer               |



- You can modify the transaction sampling parameters by changing counts, amounts and epsilons
  - `counts` contains count of each transaction type. 
  - `amounts` contains amount of each transaction type in satoshi.
  - `epsilons` is the ratio of merchants in final sampling.
- Please note that length of counts, amounts and epsilons lists should be the same.
baselines:

| Parameter              | Default | choices                                      |
|------------------------|--------|----------------------------------------------|
| _--strategy_            | top_k_betweenness | random, top_k_betweenness, top_k_degree, bottom_k_betweenness, bottom_k_degree   |





You can check the results on tensorboard.

```
tensorboard --logdir plotting/tb_results/
```

## Trouble-shootings

If you are facing problems with tensorboard, run the command below in terminal :

```
python3 -m tensorboard.main --logdir plotting/tb_results/
```

## License


This work is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](http://creativecommons.org/licenses/by-nc-nd/4.0/) license.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a>


## Citation





