## Isaac Gym 결과

[https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)

아래의 내용은 동일 디렉토리의 `HOW_TO_RL_GAMES.md` 파일을 보완하기 위한 것이며,
클래스 내부 구현 설명과 훈련(또는 테스트) 루프, 모델 및 네트워크를 **사용자 정의(Customizing)**하는 방법에 중점을 둡니다. 예시로는 `IsaacGymEnvs`에 구현된 **AMP 알고리즘**이 사용되므로, 이 파일 내에서 해당 내용을 읽고 계신 것입니다.


## 프로그램 Entry Point

`IsaacGymEnvs`에서 훈련과 테스트의 주요 Entry Point은 `train.py` 스크립트입니다.
이 파일은 `rl_games.torch_runner.Runner` 클래스의 인스턴스를 초기화하며, 선택한 모드에 따라 `run_train` 또는 `run_play` 함수가 실행됩니다.
또한, `train.py`는 훈련 및 테스트 루프의 사용자 정의 구현과, 커스텀 네트워크 및 모델을 라이브러리에 통합할 수 있도록 `build_runner` 함수를 제공합니다.
이 과정을 **"registering"** 이라고 하며, 설정 파일 내에 적절한 이름을 지정함으로써 사용자 정의 코드를 실행하도록 구성할 수 있습니다.

RL Games에서는 훈련 알고리즘을 **"agent"**, 테스트용 알고리즘을 **"player"**라고 부릅니다.
`run_train` 함수에서는 agent가 생성되고 `agent.train`을 통해 훈련이 시작됩니다.
반면, `run_play` 함수에서는 player가 생성되며 `player.run`을 호출하여 테스트가 수행됩니다.
즉, RL Games의 핵심 진입점은 agent의 `train`, player의 `run` 함수입니다.

```python
def run_train(self, args):
    """알고리즘에 전달된 훈련 절차 실행"""
    print('Started to train')
    agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
    _restore(agent, args)
    _override_sigma(agent, args)
    agent.train()

def run_play(self, args):
    """알고리즘에 전달된 추론 절차 실행"""
    print('Started to play')
    player = self.create_player()
    _restore(player, args)
    _override_sigma(player, args)
    player.run()
```


## Training 알고리즘


에이전트의 생성은 아래 코드에서 볼 수 있듯이 `algo_factory`에 의해 처리됩니다.
기본적으로 `algo_factory`에는 continuous-action A2C, discrete-action A2C, 그리고 SAC가 등록되어 있습니다.
이 기본 등록은 `Runner` 클래스의 생성자(constructor) 안에 정의되어 있으며, 그 구현은 아래에 나와 있습니다.

이 중에서도 특히, `IsaacGymEnvs`에서 대부분의 연속 제어(continuous-control) 과제에 사용되는 주요 알고리즘인 `A2CAgent`에 초점을 맞출 것입니다.

```python
self.algo_factory.register_builder(
    'a2c_continuous',
    lambda **kwargs: a2c_continuous.A2CAgent(**kwargs)
)
self.algo_factory.register_builder(
    'a2c_discrete',
    lambda **kwargs: a2c_discrete.DiscreteA2CAgent(**kwargs)
)
self.algo_factory.register_builder(
    'sac',
    lambda **kwargs: sac_agent.SACAgent(**kwargs)
)
```

모든 RL Games 알고리즘의 최상위에는 추상 클래스 `BaseAlgorithm`이 있으며, 이 클래스는 `train`, `train_epoch` 등 핵심 훈련 함수들을 정의합니다.
`A2CBase` 클래스는 `BaseAlgorithm`을 상속하며, continuous 및 discrete A2C agent 모두에서 공유되는 기능들을 제공합니다.
예를 들어, rollout 데이터를 수집하는 `play_steps`, `play_steps_rnn`, 환경과 상호작용하는 `env_step`, `env_reset` 등이 이에 해당합니다.

하지만 실제 훈련과 관련된 함수들(`train`, `train_epoch`, `update_epoch`, `prepare_dataset`, `train_actor_critic`, `calc_gradients`)은 이 단계에서는 구현되어 있지 않고,
이들은 `ContinuousA2CBase` 클래스에서 구현되며, 실제로는 `A2CAgent` 클래스가 이를 상속하여 사용합니다.

`ContinuousA2CBase`는 에이전트 훈련의 핵심 로직을 담당하며, 특히 `train`, `train_epoch`, `prepare_dataset` 함수에서 주요 처리가 이루어집니다.
`train` 함수는 환경을 한 번 초기화한 후, 주요 훈련 루프를 실행하며 이 루프는 다음 세 단계를 포함합니다:

1. `update_epoch` 호출
2. `train_epoch` 실행
3. 에피소드 길이, 보상, 손실 등 주요 정보 로깅

---

`update_epoch`는 epoch 카운트를 증가시키며, `train_epoch`는 다음과 같은 순서로 작동합니다:

1. `play_steps` 또는 `play_steps_rnn` 함수가 호출되어 rollout 데이터를 생성하며, 이 데이터는 텐서들로 이루어진 딕셔너리인 `batch_dict` 형태로 저장됩니다. 수집되는 환경 스텝 수는 설정된 `horizon_length`와 동일합니다.
2. `prepare_dataset` 함수는 `batch_dict`에 포함된 텐서들을 수정하며, 설정에 따라 값이나 advantage를 정규화하는 작업이 포함될 수 있습니다.
3. 여러 개의 mini-epoch이 실행됩니다. 각 mini-epoch에서는 데이터셋이 여러 mini-batch로 나뉘고, 이 mini-batch들이 순차적으로 `train_actor_critic` 함수에 입력됩니다.
   `train_actor_critic` 함수는 `A2CAgent`에 구현되어 있으며, 내부적으로 `A2CAgent`의 `calc_grad` 함수를 호출합니다.


`ContinuousA2CBase`를 상속한 `A2CAgent` 클래스는 **gradient 계산**과 **모델 파라미터 최적화**라는 핵심 작업을 `calc_grad` 함수에서 수행합니다.

구체적으로 `calc_grad` 함수는 다음과 같은 절차로 작동합니다:

1. PyTorch의 gradient 및 연산 그래프 기능을 활성화한 상태로 policy 모델의 **forward pass**를 수행합니다.
2. 각 loss 항목과 총 scalar loss를 계산합니다.
3. **backward pass**를 실행하여 gradient를 계산합니다.
4. 필요하다면 gradient를 잘라냅니다(truncate).
5. optimizer를 통해 모델 파라미터를 업데이트합니다.
6. loss 항목 및 학습률과 같은 관련 훈련 메트릭을 로깅합니다.

이러한 기본 함수 구조를 이해하면, `A2CAgent`를 상속하여 특정 메서드만 오버라이딩함으로써 **Custom agent**를 쉽게 만들 수 있습니다. 대표적인 예는 `IsaacGymEnvs`에서 구현된 **AMP 알고리즘**으로, 여기서는 `AMPAgent`라는 클래스를 생성하고 `train.py`에서 다음과 같이 등록합니다:

```python
_runner.algo_factory.register_builder(
    'amp_continuous',
    lambda **kwargs: amp_continuous.AMPAgent(**kwargs)
)
```


## Players

training 알고리즘과 마찬가지로, 테스트용 알고리즘인 player들도 `Runner` 클래스의 `player_factory`에 기본 등록되어 있습니다.
등록된 예시로는 `PPOPlayerContinuous`, `PPOPlayerDiscrete`, `SACPlayer` 등이 있으며, 이들은 모두 `BasePlayer` 클래스를 상속합니다.
`BasePlayer`는 공통적으로 `run` 함수를 제공하며, 하위 클래스에서는 모델 복원(`restore`), RNN 초기화(`reset`), observation을 받아 action을 생성(`get_action`, `get_masked_action`)하는 등의 기능이 구현됩니다.

testing 루프는 training 루프보다 단순합니다:

1. 환경을 초기화하고 첫 observation을 얻음
2. `max_steps` 만큼 루프를 반복하며, observation → 행동 생성 → 환경에 적용 → 다음 observation, 보상 획득
3. 총 `n_games` 에피소드가 끝나면 평균 보상(average reward) 및 에피소드 길이(average episode lengths) 출력

이 testing 루프 역시 player 클래스를 상속받아 원하는 방식으로 오버라이딩함으로써 customizing이 가능합니다. 커스텀 player도 `train.py`의 `player_factory`에 등록해야 합니다:

```python
self.player_factory.register_builder(
    'a2c_continuous',
    lambda **kwargs: players.PpoPlayerContinuous(**kwargs)
)
self.player_factory.register_builder(
    'a2c_discrete',
    lambda **kwargs: players.PpoPlayerDiscrete(**kwargs)
)
self.player_factory.register_builder(
    'sac',
    lambda **kwargs: players.SACPlayer(**kwargs)
)

_runner.player_factory.register_builder(
    'amp_continuous',
    lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs)
)
```

---

## Models와 Networks

RL Games 버전 `1.6.1`에서는 **model**과 **network**의 개념이 혼동될 수 있습니다.
아래는 그 구조와 관계에 대한 개요입니다:

- **Network Builder:** `A2CBuilder`, `SACBuilder` 등의 Network builder 클래스는 `NetworkBuilder`를 상속하며, 내부에 `torch.nn.Module` 기반의 `Network` 클래스가 포함되어 있습니다. 이 클래스는 observation 등 텐서 딕셔너리를 입력받아 행동을 생성하는 텐서 튜플을 반환합니다. 변환은 `forward` 함수에서 이루어집니다.

- **Model:** `ModelA2C`, `ModelSACContinuous` 등은 `BaseModel`을 상속하며, 내부에 `Network` 클래스를 포함하고, `build` 함수로 이를 구성합니다.

- **알고리즘 내 Model & Network:** 기본 agent 또는 player 알고리즘에서 `self.model`은 **모델 네트워크 클래스의 인스턴스**를 가리키고, `self.network`는 **모델 클래스의 인스턴스**를 가리킵니다.

- **ModelBuilder:** `algos_torch.model_builder`의 `ModelBuilder` 클래스는 모델 로딩 및 관리를 담당하며, `load` 함수를 통해 모델 이름 기반으로 인스턴스를 생성합니다.

커스텀 모델을 만들려면 custom network builder와 model class를 작성하고
`train.py` 내에서 등록해야 합니다. AMP 구현 예시는 다음과 같습니다:

```python
# algos_torch.model_builder.NetworkBuilder.__init__
self.network_factory.register_builder(
    'actor_critic',
    lambda **kwargs: network_builder.A2CBuilder()
)
self.network_factory.register_builder(
    'resnet_actor_critic',
    lambda **kwargs: network_builder.A2CResnetBuilder()
)
self.network_factory.register_builder(
    'rnd_curiosity',
    lambda **kwargs: network_builder.RNDCuriosityBuilder()
)
self.network_factory.register_builder(
    'soft_actor_critic',
    lambda **kwargs: network_builder.SACBuilder()
)

# algos_torch.model_builder.ModelBuilder.__init__
self.model_factory.register_builder(
    'discrete_a2c',
    lambda network, **kwargs: models.ModelA2C(network)
)
self.model_factory.register_builder(
    'multi_discrete_a2c',
    lambda network, **kwargs: models.ModelA2CMultiDiscrete(network)
)
self.model_factory.register_builder(
    'continuous_a2c',
    lambda network, **kwargs: models.ModelA2CContinuous(network)
)
self.model_factory.register_builder(
    'continuous_a2c_logstd',
    lambda network, **kwargs: models.ModelA2CContinuousLogStd(network)
)
self.model_factory.register_builder(
    'soft_actor_critic',
    lambda network, **kwargs: models.ModelSACContinuous(network)
)
self.model_factory.register_builder(
    'central_value',
    lambda network, **kwargs: models.ModelCentralValue(network)
)

# isaacgymenvs.train.launch_rlg_hydra.build_runner
model_builder.register_model(
    'continuous_amp',
    lambda network, **kwargs: amp_models.ModelAMPContinuous(network),
)
model_builder.register_network(
    'amp',
    lambda **kwargs: amp_network_builder.AMPBuilder()
)
```
