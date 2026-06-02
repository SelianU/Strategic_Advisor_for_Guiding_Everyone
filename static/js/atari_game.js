// atari_game.js — core: 캔버스, 키 입력, 게임 루프, 세션, UI 탭

// ── 캔버스 ────────────────────────────────────────────────────────────────────
const canvas  = document.getElementById('gameCanvas');
const ctx     = canvas.getContext('2d');
let gameActive = false, totalSteps = 0;
let sessionId  = 0;
let currentAction = 0;

// ── 키 상태: KEYBOARD_KEYS의 id를 키로 하는 동적 맵 ────────────────────────
// 예) { left: false, right: false, fire: false, accel: false, brake: false }
const keys = Object.fromEntries(KEYBOARD_KEYS.map(k => [k.id, false]));

// KEYBOARD_KEYS에서 각 키가 반응할 물리 키 코드 매핑 (설정 기반 자동 생성)
// key_combos의 키 이름을 분석해 물리 키와 연결
//   left  → ArrowLeft
//   right → ArrowRight
//   fire  → Space
//   accel → Space  (fire와 동일 or 'z')
//   brake → ArrowDown
// games/*.py에서 key_combos의 키 이름을 기반으로 매핑 추론
const KEY_MAP = (() => {
  // KEYBOARD_KEYS의 id마다 물리 키를 자동 추론
  const map = {};
  for (const k of KEYBOARD_KEYS) {
    const id = k.id.toLowerCase();
    if (id === 'left')        map[k.id] = ['ArrowLeft', 'a'];
    else if (id === 'right')  map[k.id] = ['ArrowRight', 'd'];
    else if (id === 'fire')   map[k.id] = [' ', 'c', 'z'];
    else if (id === 'accel')  map[k.id] = [' ', 'z', 'x'];
    else if (id === 'brake')  map[k.id] = ['ArrowDown', 's'];
    else if (id === 'up')     map[k.id] = ['ArrowUp', 'w'];
    else if (id === 'down')   map[k.id] = ['ArrowDown', 's'];
    else                      map[k.id] = [];  // 알 수 없는 키는 빈 배열
  }
  return map;
})();

// ── CONTROLS 패널 자동 생성 ───────────────────────────────────────────────────
(function buildControlsList() {
  const el = document.getElementById('controlsList');
  const labels = {
    left:  ['←', '왼쪽'],
    right: ['→', '오른쪽'],
    fire:  ['SPACE', '발사'],
    accel: ['SPACE', '가속'],
    brake: ['↓', '브레이크'],
    up:    ['↑', '위쪽'],
    down:  ['↓', '아래쪽'],
  };
  el.innerHTML = KEYBOARD_KEYS.map(k => {
    const fallback = labels[k.id] || [k.label, k.id];
    const keyLabel = k.label || fallback[0];
    const desc = (typeof CONTROL_DESCRIPTIONS !== 'undefined' && CONTROL_DESCRIPTIONS[k.id])
      ? CONTROL_DESCRIPTIONS[k.id]
      : fallback[1];
    return `<span class="key">${keyLabel}</span> ${desc}<br>`;
  }).join('');
})();

// ── 가상 키보드 빌드 ─────────────────────────────────────────────────────────
(function buildVkbd() {
  const vkbd = document.getElementById('vkbd');
  ['human', 'agent'].forEach(side => {
    const div = document.createElement('div');
    div.className = 'vkbd-side';
    div.innerHTML = `<div class="vkbd-label ${side === 'agent' ? 'agent' : ''}">${side.toUpperCase()}</div>`;
    const keysDiv = document.createElement('div');
    keysDiv.className = 'vkbd-keys';
    KEYBOARD_KEYS.forEach(k => {
      const el = document.createElement('div');
      el.className = 'vkey ' + (k.label.length <= 2 ? 'narrow' : 'wide');
      el.id = `vkey_${side}_${k.id}`;
      el.textContent = k.label;
      keysDiv.appendChild(el);
    });
    div.appendChild(keysDiv);
    vkbd.appendChild(div);
  });
})();

// ── 키 → 액션 변환 ───────────────────────────────────────────────────────────
function getAction() {
  // 현재 눌린 키 id들을 알파벳 순으로 정렬해 조합 문자열 생성
  const pressed = KEYBOARD_KEYS.map(k => k.id).filter(id => keys[id]);
  pressed.sort();
  const combo = pressed.join('+');
  // KEY_COMBOS에서 정확히 매칭되는 조합 찾기, 없으면 NOOP(0)
  return KEY_COMBOS[combo] ?? KEY_COMBOS[''] ?? 0;
}

// ── 가상 키보드 업데이트 ─────────────────────────────────────────────────────
function updateVkbd(humanAction, agentAction) {
  KEYBOARD_KEYS.forEach(k => {
    const hEl = document.getElementById(`vkey_human_${k.id}`);
    const aEl = document.getElementById(`vkey_agent_${k.id}`);
    const hPressed = k.actions.includes(humanAction ?? 0);
    const aPressed = k.actions.includes(agentAction ?? 0);
    if (hEl) hEl.classList.toggle('pressed', hPressed);
    if (aEl) aEl.classList.toggle('pressed', aPressed);
  });
}

// 게임에서 사용하는 모든 물리 키 집합 (preventDefault 대상)
const GAME_KEYS = new Set(
  Object.values(KEY_MAP).flat()
);

// ── 키보드 이벤트 ─────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  // 게임 키는 브라우저 기본 동작(스크롤 등) 전부 차단
  if (GAME_KEYS.has(e.key)) e.preventDefault();
  let changed = false;
  for (const k of KEYBOARD_KEYS) {
    if ((KEY_MAP[k.id] || []).includes(e.key)) {
      if (!keys[k.id]) { keys[k.id] = true; changed = true; }
    }
  }
  if (changed) currentAction = getAction();
});

document.addEventListener('keyup', e => {
  if (GAME_KEYS.has(e.key)) e.preventDefault();
  let changed = false;
  for (const k of KEYBOARD_KEYS) {
    if ((KEY_MAP[k.id] || []).includes(e.key)) {
      if (keys[k.id]) { keys[k.id] = false; changed = true; }
    }
  }
  if (changed) currentAction = getAction();
});

// ── 탭 전환 ──────────────────────────────────────────────────────────────────
function setRightTab(tab) {
  rightTab = tab;
  document.querySelectorAll('#rightTabs .right-tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.getElementById('resultPanel').classList.toggle('tab-open', tab === 'qvalue');
  document.getElementById('worstPanel').classList.toggle('tab-open', tab === 'worst');
}

function setFbTab(tab) {
  fbTab = tab;
  document.querySelectorAll('[data-fb-tab]').forEach(b => b.classList.toggle('active', b.dataset.fbTab === tab));
  const showFeedback = tab === 'feedback';
  document.getElementById('fbStructured').style.display = showFeedback ? '' : 'none';
  document.getElementById('fbSummary').classList.toggle('visible', !showFeedback);
}

function actionIds() {
  return Object.keys(ACTION_NAMES).map(Number).sort((a, b) => a - b);
}

function makeActionGrid(counts, el, asPct = false) {
  el.innerHTML = '';
  const total = Object.values(counts || {}).reduce((sum, value) => sum + Number(value || 0), 0);
  actionIds().forEach(id => {
    const name = ACTION_NAMES[id];
    const cnt = Number((counts && counts[name]) || 0);
    const val = asPct && total ? `${(cnt / total * 100).toFixed(1)}%` : cnt;
    el.innerHTML += `<div class="action-item"><div class="action-name">${name}</div><div class="action-count">${val}</div></div>`;
  });
}

function setAnalysisMode(mode, step = null) {
  const isStep = mode === 'step';
  document.querySelector('#rightTabs .right-tab[data-tab="qvalue"]').textContent = isStep ? 'STEP REVIEW' : 'Q-VALUE';
  document.querySelector('#rightTabs .right-tab[data-tab="worst"]').textContent = isStep ? '비교 요약' : '최악 행동';
  document.querySelector('#resultPanel .d3qn-title').textContent =
    isStep && step ? `STEP ${step}까지 결과` : 'D3QN Q-VALUE 분석';
  document.querySelector('#worstPanel .d3qn-title').textContent = isStep ? '비교 요약' : '최악 행동 분석';
  document.getElementById('analysisZone').classList.toggle('step-review-wide', isStep);
  document.getElementById('resultPanel').classList.toggle('step-review', isStep);
  document.getElementById('worstPanel').classList.toggle('step-review', isStep);
  document.getElementById('momentLossCard').style.display = isStep ? '' : 'none';

  const firstWorstTitle = document.querySelector('#worstPanel .qvalue-card:first-child > div:first-child');
  const worstTableCard = document.querySelector('#worstPanel .qvalue-card:nth-child(2)');
  if (firstWorstTitle) firstWorstTitle.textContent = isStep ? '' : '최악 행동';
  if (worstTableCard) worstTableCard.style.display = isStep ? 'none' : '';
}

function renderQStats(stats, opts = {}) {
  const agree = Number(stats?.agree_rate ?? 0);
  const avgLoss = Number(stats?.avg_loss ?? 0);
  document.getElementById('agreeRate').textContent = `${agree.toFixed(1)}%`;
  document.getElementById('avgLoss').textContent = avgLoss.toFixed(3);
  document.getElementById('momentLoss').textContent =
    opts.loss != null ? Number(opts.loss).toFixed(3) : '—';
  makeActionGrid(stats?.player_actions, document.getElementById('playerGrid'), Boolean(opts.asPct));
  makeActionGrid(stats?.ai_actions, document.getElementById('aiGrid'), Boolean(opts.asPct));
}

function formatValue(value) {
  return value === null || value === undefined ? '—' : value;
}

function renderComparisonSummary(data) {
  const s = data?.summary || {};
  const c = candidates.find(x => parseInt(x.entry_index) === pendingEI);
  const step = c?.step ?? s.step ?? '—';
  const loss = c?.loss != null ? Number(c.loss).toFixed(3) : formatValue(s.loss);
  document.getElementById('worstBox').innerHTML = `
    <div class="cf-meta-row">
      <div class="cf-meta-chip">
        <div class="cf-meta-label">STEP</div>
        <div class="cf-meta-val">${step}</div>
      </div>
      <div class="cf-meta-chip loss">
        <div class="cf-meta-label">LOSS</div>
        <div class="cf-meta-val">${loss}</div>
      </div>
    </div>
    <div class="cf-vs-header">
      <div class="cf-vs-h human">HUMAN</div>
      <div class="cf-vs-badge">VS</div>
      <div class="cf-vs-h agent">AGENT</div>
    </div>
    <div class="cf-vs-rows">
      <div class="cf-vs-row">
        <div class="cf-vs-cell human action">
          <div class="cf-vs-mini">액션</div>
          <div class="cf-vs-val">${s.human_action_name || '—'}</div>
        </div>
        <div class="cf-vs-sep"></div>
        <div class="cf-vs-cell agent action">
          <div class="cf-vs-mini">액션</div>
          <div class="cf-vs-val">${s.agent_action_name || '—'}</div>
        </div>
      </div>
      <div class="cf-vs-row">
        <div class="cf-vs-cell human">
          <div class="cf-vs-mini">점수 변화</div>
          <div class="cf-vs-val">${formatValue(s.human_score_delta)}</div>
        </div>
        <div class="cf-vs-sep"></div>
        <div class="cf-vs-cell agent">
          <div class="cf-vs-mini">점수 변화</div>
          <div class="cf-vs-val">${formatValue(s.agent_score_delta)}</div>
        </div>
      </div>
      <div class="cf-vs-row">
        <div class="cf-vs-cell human">
          <div class="cf-vs-mini">즉시 점수</div>
          <div class="cf-vs-val">${formatValue(s.human_first_reward_step)}</div>
        </div>
        <div class="cf-vs-sep"></div>
        <div class="cf-vs-cell agent">
          <div class="cf-vs-mini">즉시 점수</div>
          <div class="cf-vs-val">${formatValue(s.agent_first_reward_step)}</div>
        </div>
      </div>
    </div>
  `;
}

function renderStepReview(data) {
  const s = data.summary || {};
  const c = candidates.find(x => parseInt(x.entry_index) === pendingEI);
  const step = c?.step ?? s.step ?? null;
  setAnalysisMode('step', step);
  renderQStats(data.cumulative || {}, { loss: c?.loss ?? s.loss, asPct: true });
  renderComparisonSummary(data);
  setRightTab('qvalue');
}

// ── 분석 결과 적용 ────────────────────────────────────────────────────────────
function applyAnalysis(data) {
  latestAnalysis = data;
  setAnalysisMode('episode');
  document.getElementById('resultPanel').classList.add('visible');
  document.getElementById('worstPanel').classList.add('visible');
  document.getElementById('postgamePanel').classList.add('visible');
  document.getElementById('analysisZone').classList.add('visible');
  setRightTab(rightTab);
  renderQStats(data);

  const w = data.worst;
  if (w) {
    document.getElementById('worstBox').innerHTML =
      `<strong>스텝 ${w.step}</strong> — ${w.action_name}<br>` +
      `내 Q: <strong>${w.player_q}</strong> | 최선 Q: <strong style="color:var(--theme)">${w.best_q}</strong><br>` +
      `최선: <strong style="color:var(--cyan)">${w.best_action_name}</strong> | 손실: <strong style="color:var(--red)">${w.loss.toFixed(3)}</strong>`;
  }

  const tbody = document.getElementById('worstBody');
  tbody.innerHTML = '';
  candidates = data.worst_10 || [];
  candidateStatus.clear(); prefetchStarted = false;
  pendingEI = null;
  candidates.forEach(c => candidateStatus.set(parseInt(c.entry_index), 'waiting'));
  candidates.forEach(a => {
    const cls = a.loss > 0.3 ? 'loss-high' : (a.loss > 0.1 ? 'loss-mid' : 'loss-low');
    tbody.innerHTML += `<tr><td>${a.step}</td><td>${a.action_name}</td><td style="color:var(--cyan)">${a.best_action_name}</td><td class="${cls}">${a.loss.toFixed(3)}</td></tr>`;
  });
  setRightTab(rightTab);
  renderCandidateBar();
  maybePrefetch();
}

// ── 세션 목록 렌더 ────────────────────────────────────────────────────────────
function renderSessions(sessions) {
  const sel  = document.getElementById('sessionSelect');
  const prev = sel.value;
  sel.innerHTML = '<option value="">저장된 기록 선택</option>';
  (sessions || []).forEach(item => {
    const o = document.createElement('option');
    o.value = item.id;
    o.textContent = `${item.title} · ${item.saved_at || ''}`;
    sel.appendChild(o);
  });
  if (prev && (sessions || []).some(s => s.id === prev)) sel.value = prev;
}

// ── UI 초기화 ────────────────────────────────────────────────────────────────
function resetUI() {
  practiceActive = false;
  stopPracticeLoops();
  stopCompareCountdown();
  document.body.classList.remove('coach-visible');
  unlockedAchievementIds.clear();
  canvas.classList.remove('hidden');
  ['progressPanel','resultPanel','worstPanel','analysisZone','postgamePanel','compareStage','centerFeedback']
    .forEach(id => { const el = document.getElementById(id); el.classList.remove('visible','tab-open'); });
  document.getElementById('candidateBar').innerHTML = '';
  document.getElementById('replayTimer').style.display = 'none';
  document.getElementById('practiceBtn').style.display = 'none';
  document.getElementById('practiceExitBtn').style.display = 'none';
  document.getElementById('practiceView').style.display = 'none';
  document.getElementById('practiceResultCard').classList.remove('visible');
  document.getElementById('practiceOverOverlay').style.display = 'none';
  cfCache.clear(); candidateStatus.clear(); pendingEI = null; prefetchStarted = false;
  rightTab = 'qvalue'; setRightTab('qvalue'); setFbTab('feedback');
  document.getElementById('fbSource').textContent = '';
  // 구조화 섹션 숨기고 placeholder 텍스트 표시
  ['fbSituation','fbComparison','fbAdvice'].forEach(id => { document.getElementById(id).style.display = 'none'; });
  const ftEl = document.getElementById('fbText');
  ftEl.style.display = 'block';
  ftEl.textContent   = '후보를 선택하면 코칭 피드백이 여기에 표시됩니다.';
  document.getElementById('fbLoading').classList.remove('active');
  document.getElementById('score').textContent = '0';
  document.getElementById('steps').textContent = '0';
  document.getElementById('sessionLabel').textContent = 'LIVE';
  totalSteps = 0;
  // Grad-CAM 상태 초기화
  hasGradCam = false;
  gcamMode = 'normal';
  cfGcamHuman = []; cfGcamAgent = [];
  document.getElementById('gcamNoSupport').classList.remove('visible');
  document.getElementById('gcamLegend').classList.remove('visible');
  ['gcamNormalBtn','gcamHumanBtn','gcamAgentBtn','gcamSplitBtn'].forEach(id => {
    const btn = document.getElementById(id);
    btn.classList.toggle('active', id === 'gcamNormalBtn');
    btn.style.opacity = '';
  });
  ['viewNormal','viewHumanCam','viewAgentCam','viewSplit'].forEach(id => {
    document.getElementById(id).style.display = id === 'viewNormal' ? '' : 'none';
  });
}

// ── 버튼 이벤트 ──────────────────────────────────────────────────────────────
document.getElementById('startBtn').addEventListener('click', () => {
  gameActive = true;
  resetUI();
  document.body.classList.add('game-playing');
  // Rule Book 숨기기
  const introPanel = document.getElementById('gameIntroPanel');
  if (introPanel) introPanel.classList.add('hidden');
  document.getElementById('status').textContent = 'PLAYING...';
  document.getElementById('startBtn').disabled  = true;
  // 시작 시 FIRE/ACCEL 키가 있으면 눌린 상태로 시작
  const hasFireKey = KEYBOARD_KEYS.some(k => k.id === 'fire' || k.id === 'accel');
  currentAction = hasFireKey ? (KEY_COMBOS['fire'] ?? KEY_COMBOS['accel'] ?? 1) : 0;
  socket.emit(P + 'start');
});

document.getElementById('saveBtn').addEventListener('click', () => {
  const title = document.getElementById('sessionTitle').value.trim();
  if (!title) { alert('저장할 기록 이름을 입력해주세요.'); return; }
  socket.emit(P + 'save_session', { title });
});
document.getElementById('loadBtn').addEventListener('click', () => {
  const sid = document.getElementById('sessionSelect').value;
  if (sid) socket.emit(P + 'load_session', { session_id: sid });
});
document.getElementById('deleteBtn').addEventListener('click', () => {
  const sid = document.getElementById('sessionSelect').value;
  if (sid) socket.emit(P + 'delete_session', { session_id: sid });
});

document.querySelectorAll('.speed-btn[data-speed]').forEach(b =>
  b.addEventListener('click', () => setSpeed(parseFloat(b.dataset.speed))));

document.querySelectorAll('#rightTabs .right-tab').forEach(b =>
  b.addEventListener('click', () => setRightTab(b.dataset.tab)));
document.querySelectorAll('[data-fb-tab]').forEach(b =>
  b.addEventListener('click', () => setFbTab(b.dataset.fbTab)));

// ── 소켓 이벤트 ──────────────────────────────────────────────────────────────
socket.emit(P + 'list_sessions');

socket.on(P + 'frame', data => {
  if (data.session_id !== undefined) sessionId = data.session_id;
  if (!gameActive) return;
  const img = new Image();
  img.src = 'data:image/jpeg;base64,' + data.image;
  img.onload = () => {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    totalSteps++;
    document.getElementById('steps').textContent = totalSteps;
    if (data.score != null) document.getElementById('score').textContent = Math.round(data.score);
    setTimeout(() => {
      if (gameActive) socket.emit(P + 'action', { action: currentAction });
    }, 1000 / 60);
  };
});

socket.on(P + 'over', report => {
  gameActive = false;
  document.body.classList.remove('game-playing');
  document.body.classList.add('coach-visible');
  canvas.classList.add('hidden');
  document.getElementById('status').textContent = 'GAME OVER';
  document.getElementById('startBtn').disabled = false;
  if (report.has_model) document.getElementById('progressPanel').classList.add('visible');

  if (report.achievements) {
    buildAchievementState(report.achievements).forEach(a => {
      if (a.unlocked && a.id) unlockedAchievementIds.add(a.id);
    });
  }
});

socket.on(P + 'analysis_start', data => {
  if (data.session_id !== undefined && data.session_id !== sessionId) return;
  document.getElementById('progressText').textContent = `0 / ${data.total} 스텝 분석 중...`;
  document.getElementById('progressFill').style.width = '0%';
});
socket.on(P + 'analysis_progress', data => {
  if (data.session_id !== undefined && data.session_id !== sessionId) return;
  document.getElementById('progressFill').style.width = data.pct + '%';
  document.getElementById('progressText').textContent = `${data.current} / ${data.total} 분석 중...`;
});
socket.on(P + 'analysis_done', data => {
  if (data.session_id !== undefined && data.session_id !== sessionId) return;
  document.getElementById('progressPanel').classList.remove('visible');
  applyAnalysis(data);
});

socket.on(P + 'sessions_list',  data => renderSessions(data.sessions));
socket.on(P + 'session_saved',  data => { document.getElementById('sessionStatus').textContent = data.message || ''; if (data.ok) renderSessions(data.sessions); });
socket.on(P + 'session_deleted',data => { document.getElementById('sessionStatus').textContent = data.message || ''; renderSessions(data.sessions); });
socket.on(P + 'session_loaded', data => {
  if (!data.ok) { document.getElementById('sessionStatus').textContent = data.message || '불러오기 실패'; return; }
  document.body.classList.remove('game-playing');
  document.body.classList.add('coach-visible');
  sessionId = data.session_id;
  cfCache.clear(); candidateStatus.clear();
  (data.cached_counterfactuals || []).forEach(item => cfCache.set(parseInt(item.entry_index), item));
  canvas.classList.add('hidden');
  document.getElementById('status').textContent   = '';
  document.getElementById('startBtn').disabled    = false;
  document.getElementById('sessionLabel').textContent = data.meta?.title || 'LOADED';
  document.getElementById('score').textContent    = data.basic?.total_reward ?? '0';
  document.getElementById('steps').textContent    = data.basic?.total_steps  ?? '0';
  applyAnalysis(data.analysis || {});
  candidates.forEach(c => {
    const ei = parseInt(c.entry_index);
    candidateStatus.set(ei, cfCache.has(ei) ? 'ready' : 'waiting');
  });
  pendingEI = null; prefetchStarted = true;
  renderCandidateBar();
  document.getElementById('sessionTitle').value   = data.meta?.title || '';
  document.getElementById('sessionStatus').textContent = `불러온 기록: ${data.meta?.title || '이름 없음'}`;
  document.getElementById('compareStage').classList.remove('visible');
  document.getElementById('centerFeedback').classList.remove('visible');
  document.getElementById('fbSource').textContent = '';
  document.getElementById('fbText').textContent   = '후보를 선택하면 코칭 피드백이 여기에 표시됩니다.';
  setFbTab('feedback');
});
