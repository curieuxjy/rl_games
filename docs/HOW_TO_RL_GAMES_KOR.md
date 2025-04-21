# [rl_games](https://github.com/Denys88/rl_games/) 소개 - 새로운 환경과 알고리즘 확장
**작성자** - [Anish Diwan](https://www.anishdiwan.com/)

이 글에서는 [rl_games](https://github.com/Denys88/rl_games/)라는 강화학습 라이브러리의 전반적인 동작 방식과 함께, 이를 확장하여 새로운 환경과 알고리즘을 추가하는 방법을 설명합니다. 특히 [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)와 유사한 구조로 작업하는 방법을 안내합니다. 본 문서에서 다루는 주제는 다음과 같습니다:

1. rl_games의 구성 요소들 (runner, 알고리즘, 환경 등)
2. rl_games를 직접 활용하는 방법
    - 새로운 gym 기반 환경 추가하기
    - gym 기반이 아닌 환경과 시뮬레이터를 rl_games 알고리즘과 함께 사용하기 (예시는 [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) 참고)
    - 새로운 알고리즘 추가하기

---

## rl_games 기본 구조

rl_games는 `runner.py`라는 메인 파이썬 스크립트를 사용하며, `--train` 또는 `--play` 플래그를 통해 training 또는 정책 playing(inferencing)할 수 있고, 설정 파일은 `--file` 인자로 넘깁니다. 예를 들어, Pong 환경에서 PPO 훈련 및 실행은 아래처럼 가능합니다. 이에 대한 PPO 설정은 `rl_games/configs/atari/ppo_pong.yaml`에 정의되어 있습니다.:

```
python runner.py --train --file rl_games/configs/atari/ppo_pong.yaml
python runner.py --play --file rl_games/configs/atari/ppo_pong.yaml --checkpoint nn/PongNoFrameskip.pth
```

rl_games는 알고리즘 정의, 환경 인스턴스 생성, 로그 기록을 위해 아래의 클래스들을 사용합니다:

---

### 1. 메인 스크립트 - `rl_games.torch_runner.Runner`
- 설정에 따라 알고리즘 인스턴스를 생성하고 training 또는 playing을 수행하는 주요 클래스입니다.
- 모든 알고리즘과 플레이어 인스턴스는 `rl_games.common.ObjectFactory()`의 `register_builder()` 메서드를 통해 자동 등록됩니다.
- 전달된 args에 따라 `self.run_train()` 또는 `self.run_play()`가 실행됩니다.
해당 문장을 한국어로 해석하면 다음과 같습니다:
- Runner는 훈련 중 메트릭을 기록하는 알고리즘 observer도 설정합니다. observer가 별도로 지정되지 않은 경우, 기본적으로 `DefaultAlgoObserver()`를 사용하며, 이는 텐서보드의 summarywriter를 이용해 알고리즘이 접근할 수 있는 메트릭들을 기록합니다.
  - 로그와 체크포인트는 기본적으로 `nn`이라는 디렉토리에 자동으로 생성됩니다.
  - 필요에 따라 사용자 정의 알고리즘이나 observer를 추가로 지정할 수도 있습니다 (아래에서 더 자세히 설명됩니다).


### 2. 알고리즘 생성 - `rl_games.common.Objectfactory()`

- 알고리즘 또는 플레이어를 생성합니다. `register_builder(self, name, builder)` 메서드를 가지고 있으며, 이 메서드는 어떤 객체를 생성할지를 반환하는 함수를 `(name: str)` 형식으로 등록합니다. 예를 들어, 아래 코드는 `'a2c_continuous'`라는 이름을 `A2CAgent`를 반환하는 람다 함수와 함께 등록합니다:

  ```python
  register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
  ```

- 또한 `create(self, name, **kwargs)` 메서드를 통해 등록된 builder 중 하나를 이름으로 호출하여 해당 객체를 생성할 수 있습니다.


### 3. 강화학습 알고리즘


- rl_games에는 여러 가지 강화학습 알고리즘이 포함되어 있습니다. 대부분은 `rl_games.algos_torch.A2CBase`와 같은 기본 알고리즘 클래스를 상속합니다.
- rl_games에서는 환경이 알고리즘에 의해 생성됩니다. 설정 파일(config)의 구성에 따라 여러 개의 환경을 병렬로 실행하는 것도 가능합니다.


### 4. 환경 구성 - `rl_games.common.vecenv` & `rl_games.common.env_configurations`


- `vecenv` 스크립트는 환경의 타입에 따라 다양한 환경을 생성하는 클래스들을 포함하고 있습니다. rl_games는 상당히 범용적인 라이브러리이기 때문에, openAI gym 환경, brax 환경, cule 환경 등 여러 종류의 환경을 지원합니다. 이러한 환경 타입과 그에 대응하는 베이스 클래스들은 `rl_games.common.vecenv.vecenv_config` 딕셔너리에 저장되어 있습니다. 이 환경 클래스들은 다중 병렬 환경 실행이나 멀티 에이전트 환경 실행 등의 기능을 가능하게 합니다. 기본적으로 사용 가능한 모든 환경들이 이미 등록되어 있습니다. 새로운 환경을 추가하는 방법은 아래에서 설명됩니다.

- `rl_games.common.env_configurations.configurations`는 또 다른 딕셔너리로, `env_name: {'vecenv_type', 'env_creator'}` 정보를 저장합니다. 예를 들어, 아래는 `"CartPole-v1"`이라는 환경 이름에 대해 타입과 해당 gym 환경을 생성하는 람다 함수를 저장하는 예시입니다:

  ```python
  'CartPole-v1' : {
      'vecenv_type' : 'RAY',
      'env_creator' : lambda **kwargs : gym.make('CartPole-v1'),
  }
  ```

- 이 구조의 일반적인 작동 방식은 다음과 같습니다:
  - 알고리즘의 베이스 클래스(예: `A2CAgent`)는 config 파일에 명시된 `env_name`(예: `'CartPole-v1'`)을 참고하여 환경을 생성합니다.
  - 내부적으로 `'CartPole-v1'`이라는 이름을 `rl_games.common.env_configurations.configurations`에서 찾아 `vecenv_type`을 얻고, 이 타입을 통해 `vecenv.vecenv_config`에서 실제 환경 클래스를 불러옵니다(예: `RayVecEnv`).
  - 이후 이 환경 클래스(`RayVecEnv`)는 내부적으로 `'env_creator'` 키에 등록된 함수(예: `lambda **kwargs : gym.make('CartPole-v1')`)를 호출하여 환경을 생성합니다.

- 구조가 다소 복잡해 보일 수 있지만, 이러한 방식 덕분에 config 파일에 단순히 환경 이름만 입력함으로써 실험을 바로 실행할 수 있습니다.

## rl_games 확장하기

기본 제공되는 환경과 알고리즘 외에도, rl_games는 새로운 연구나 개발의 출발점으로 훌륭합니다. 이 섹션에서는 **새로운 환경과 알고리즘을 추가하는 법**을 설명합니다. 구조는 [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)를 참고하며, 설정 관리는 [hydra](https://hydra.cc/docs/intro/)를 활용합니다.

`runner.py` 대신, **동적으로 알고리즘이나 환경을 삽입할 수 있는** `train.py`를 사용합니다.


디렉토리 구조 예시:

```
project dir
│   train.py (replacement to the runner.py script)
│
└───tasks dir (sometimes also called envs dir)
│   │   customenv.py
│   │   customenv_utils.py
|
└───cfg dir (main hydra configs)
│   │   config.yaml (main config for the setting up simulators etc. if needed)
│   │
│   └─── task dir (configs for the env)
│       │   customenv.yaml
│       │   otherenv.yaml
│       │   ...
|
│   └─── train dir (configs for training the algorithm)
│       │   customenvPPO.yaml
│       │   otherenvAlgo.yaml
│       │   ...
|
└───algos dir (custom wrappers for training algorithms in rl_games)
|   │   custom_network_builder.py
|   │   custom_algo.py
|   | ...
|
└───runs dir (generated automatically on executing train.py)
│   └─── env_name_alg_name_datetime dir (train logs)
│       └─── nn
|           |   checkpoints.pth
│       └─── summaries
            |   events.out...
```

### 새로운 gym 기반 환경 추가



새로운 환경을 rl_games 설정에서 사용하려면, 먼저 해당 환경의 **TYPE**을 정의해야 합니다. 새로운 환경 TYPE은 `vecenv.register(config_name, func)` 함수를 호출하여 `config_name: func` 쌍을 딕셔너리에 추가함으로써 등록할 수 있습니다. 예를 들어 아래 코드는 `'RAY'` 타입의 환경을 추가하며, 해당 타입은 `RayVecEnv` 클래스를 생성하는 람다 함수를 사용합니다. 이 `RayVecEnv` 클래스는 내부적으로 환경을 저장하는 `RayWorkers`를 포함하고 있으며, 이를 통해 다중 환경 병렬 학습이 자동으로 가능해집니다.

```python
register('RAY', lambda config_name, num_actors, **kwargs: RayVecEnv(config_name, num_actors, **kwargs))
```


gym 기반 환경(gym의 기본 클래스를 상속받은 환경)의 경우, TYPE은 단순히 rl_games의 `RayVecEnv`를 사용하면 됩니다. gym 기반 환경을 추가하는 과정은 본질적으로 `gym.Env`를 상속한 클래스를 정의한 뒤, 이 환경을 `'RAY'` 타입으로 `rl_games.common.env_configurations`에 등록하는 것입니다.

이 작업은 일반적으로 `env_name: {'vecenv_type', 'env_creator'}` 형태의 key-value 쌍을 `env_configurations.configurations`에 추가하여 수행됩니다.
그러나 이 방식은 rl_games 라이브러리를 직접 수정해야 하기 때문에,
만약 라이브러리를 수정하고 싶지 않다면 다음과 같은 대안이 있습니다:

- `register` 메서드를 통해 새로운 환경을 딕셔너리에 동적으로 추가하고,
- `RayVecEnv`와 `RayWorker` 클래스를 복사하여,
- `__init__` 메서드가 수정된 환경 설정 딕셔너리를 입력받도록 변경하는 방식입니다.

예시 코드:

**Within train.py**
```python
@hydra.main(version_base="1.1", config_name="custom_config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    from custom_envs.custom_env import SomeEnv
    from custom_envs.customenv_utils import CustomRayVecEnv
    from rl_games.common import env_configurations, vecenv

    def create_pusht_env(**kwargs):
        # Instantiate new env
        env =  SomeEnv()

        #Alternate example, env = gym.make('LunarLanderContinuous-v2')
        return env

    # Register the TYPE
    env_configurations.register('pushT', {
        'vecenv_type': 'CUSTOMRAY',
        'env_creator': lambda **kwargs: create_pusht_env(**kwargs),
    })

    # Provide the TYPE:func pair
    vecenv.register('CUSTOMRAY', lambda config_name, num_actors, **kwargs: CustomRayVecEnv(env_configurations.configurations, config_name, num_actors, **kwargs))
```

--------------------------------

**Custom Env TYPEs (enables adding new envs dynamically)**
```python
# Make a copy of RayVecEnv

class CustomRayVecEnv(IVecEnv):
    import ray

    def __init__(self, config_dict, config_name, num_actors, **kwargs):
        ### ADDED CHANGE ###
        # Explicityly passing in the dictionary containing env_name: {vecenv_type, env_creator}
        self.config_dict = config_dict

        self.config_name = config_name
        self.num_actors = num_actors
        self.use_torch = False
        self.seed = kwargs.pop('seed', None)


        self.remote_worker = self.ray.remote(CustomRayWorker)
        self.workers = [self.remote_worker.remote(self.config_dict, self.config_name, kwargs) for i in range(self.num_actors)]

        ...
        ...

# Make a copy of RayWorker

class CustomRayWorker:
    ## ADDED CHANGE ###
    # Add config_dict to init
    def __init__(self, config_dict, config_name, config):
        self.env = config_dict[config_name]['env_creator'](**config)

        ...
        ...
```


### gym이 아닌 환경 및 시뮬레이터 추가하기

gym이 아닌 환경들도 동일한 방식으로 추가할 수 있습니다.
하지만 이 경우에는 **새로운 TYPE 클래스**를 직접 정의해야 합니다.
[IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)에서는 이를 위해 IsaacGym 시뮬레이션 환경을 사용하는 `RLGPU`라는 새로운 TYPE을 정의합니다.

해당 예시는 IsaacGymEnvs 라이브러리에서 확인할 수 있으며,
구체적으로는 [`RLGPUEnv`](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/utils/rlgames_utils.py) 클래스에서 확인할 수 있습니다.

### rl_games에 새로운 알고리즘 및 옵저버 추가하기

사용자 정의 알고리즘(custom algorithm)을 추가하는 것은 본질적으로 `rl_games.torch_runner.Runner` 내부에 사용자만의 **builder**와 **player**를 등록하는 작업을 의미합니다.

IsaacGymEnvs에서는 이를 위해, `hydra` 데코레이터가 적용된 메인 함수 내부에서 아래와 같은 방식으로 알고리즘(이 경우 `AMP`)을 등록합니다.

**Within train.py**
```python
# register new AMP network builder and agent
def build_runner(algo_observer):
    runner = Runner(algo_observer)
    runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
    runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
    model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
    model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

    return runner
```

앞에서 본 것처럼, 필요한 데이터를 기록하기 위해 **custom observer**를 추가할 수도 있습니다.
이를 위해 `rl_games.common.algo_observer.AlgoObserver` 클래스를 상속받아 직접 구현하면 됩니다.

예를 들어, **scores**를 기록하고 싶다면,
사용자 정의 환경의 `info` 딕셔너리(환경을 `step` 했을 때 반환되는 값) 안에 `"scores"`라는 키가 반드시 포함되어 있어야 합니다.

### 전체 예시

다음은 `pushT`라는 새로운 gym 기반 환경을 만들고, 사용자 정의 옵저버를 사용해 메트릭을 기록하는 **사용자 정의 `train.py` 스크립트의 전체 예제**입니다.

```python
import hydra

from omegaconf import DictConfig, OmegaConf
from omegaconf import DictConfig, OmegaConf


# Hydra decorator to pass in the config. Looks for a config file in the specified path. This file in turn has links to other configs
@hydra.main(version_base="1.1", config_name="custom_config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    import logging
    import os

    from hydra.utils import to_absolute_path
    import gym
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner


    # Naming the run
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.run_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)


    # Creating a new function to return a pushT environment. This will then be added to rl_games env_configurations so that an env can be created from its name in the config
    from custom_envs.pusht_single_env import PushTEnv
    from custom_envs.customenv_utils import CustomRayVecEnv, PushTAlgoObserver

    def create_pusht_env(**kwargs):
        env =  PushTEnv()
        return env

    # env_configurations.register adds the env to the list of rl_games envs.
    env_configurations.register('pushT', {
        'vecenv_type': 'CUSTOMRAY',
        'env_creator': lambda **kwargs: create_pusht_env(**kwargs),
    })

    # vecenv register calls the following lambda function which then returns an instance of CUSTOMRAY.
    vecenv.register('CUSTOMRAY', lambda config_name, num_actors, **kwargs: CustomRayVecEnv(env_configurations.configurations, config_name, num_actors, **kwargs))

    # Convert to a big dictionary
    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # Build an rl_games runner. You can add other algos and builders here
    def build_runner():
        runner = Runner(algo_observer=PushTAlgoObserver())
        return runner

    # create runner and set the settings
    runner = build_runner()
    runner.load(rlg_config_dict)
    runner.reset()

    # Run either training or playing via the rl_games runner
    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        # 'checkpoint': cfg.checkpoint,
        # 'sigma': cfg.sigma if cfg.sigma != '' else None
    })


if __name__ == "__main__":
    launch_rlg_hydra()
```
