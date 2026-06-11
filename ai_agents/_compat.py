"""
ai_agents/_compat.py — 게임별 하위 호환 shim 공통 팩토리.

ai_agents/<game>/__init__.py 들은 모두 "공통 d3qn_helper 로 위임"하는
같은 코드를 게임 이름만 바꿔 반복하고 있었다. 이 모듈의 install_shim() 을
호출하는 한 줄로 동일한 공개 API 를 구성한다.

사용 예 (ai_agents/breakout/__init__.py):
    from ai_agents._compat import install_shim
    install_shim(__name__, 'breakout', loader_alias='load_breakout_d3qn')

이후 기존 import 경로가 그대로 동작한다:
    from ai_agents.breakout import load_breakout_d3qn, get_q_values, ACTION_NAMES
"""
import sys


def install_shim(module_name: str, game_id: str, loader_alias: str = 'load_d3qn'):
    """module_name 모듈에 d3qn_helper 위임 API 를 주입한다.

    loader_alias — 게임별 기존 로더 함수 이름
                   (예: 'load_breakout_d3qn', 'load_enduro_d3qn', 기본 'load_d3qn')
    """
    from ai_agents import d3qn_helper as h

    def _loader(model_path: str, device: str = 'cpu'):
        return h.load_d3qn(game_id, model_path, device)

    _loader.__name__ = loader_alias
    _loader.__doc__ = f'{game_id} D3QN 모델을 로드합니다.'

    namespace = {
        'ACTION_NAMES':    h.GAME_CONFIGS[game_id]['action_names'],
        'DuelingDQN':      h.DuelingDQN,
        'AtariWrapper':    h.AtariWrapper,
        'GradRescale':     h.GradRescale,
        'get_q_values':    h.get_q_values,
        'analyze_episode': h.analyze_episode,
        loader_alias:      _loader,
    }

    module = sys.modules[module_name]
    for name, obj in namespace.items():
        setattr(module, name, obj)
    module.__all__ = sorted(namespace)
