"""
gomoku — Gomoku 게임 백엔드 패키지.

서버 진행 흐름에 필요한 모듈:
  state     — 게임 상태 싱글턴 + PPO 모델 + 보드 상수
  engine    — 순수 로직 (Q-value / 승률 / 승리선 / 결과)
  render    — cv2 보드 렌더링 + 프레임 인코딩
  sessions  — 디스크 영속화 + 직렬화 헬퍼
  handlers  — SocketIO 이벤트 핸들러 (게임 / 세션 / 분석 / counterfactual)

사용 예 (app.py):
    import gomoku.handlers   # noqa  (side effect: @socketio.on 등록)

다른 모듈에서 직접 호출 (예: routes.py):
    from gomoku.sessions import list_saved_sessions
"""