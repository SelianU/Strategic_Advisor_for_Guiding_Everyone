"""
ai_agents/amidar/__init__.py — 하위 호환 shim.
공통 d3qn_helper 위임 코드는 ai_agents/_compat.py 참조.

기존 import 경로는 변경 없이 계속 동작합니다:
    from ai_agents.amidar import load_d3qn, get_q_values
"""
from ai_agents._compat import install_shim

install_shim(__name__, 'amidar', loader_alias='load_d3qn')
