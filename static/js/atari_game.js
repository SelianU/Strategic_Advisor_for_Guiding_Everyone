const REPLAY_FPS    = 60;
const REPLAY_HORIZON = REPLAY_FPS * 30;
const STATUS_LABELS = { waiting:'대기 중', generating:'생성 중', ready:'준비됨', error:'실패' };

// ── 소켓 / 캔버스 ────────────────────────────────────────────────────────────
const socket  = io({ transports: ['websocket'] });
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

// ── 카운터팩추얼 리플레이 ─────────────────────────────────────────────────────
let cfFrames = [], cfHActs = [], cfAActs = [];
// Grad-CAM 프레임 (인간/에이전트 각각, 서버에서 오버레이 완성본)
let cfGcamHuman = [], cfGcamAgent = [];
let hasGradCam = false;
let gcamMode = 'normal';  // 'normal' | 'human' | 'agent' | 'split'
let cfIdx = 0, cfSpeed = 1, cfTimer = null;
let candidates = [], pendingEI = null, prefetchStarted = false;
const cfCache = new Map(), candidateStatus = new Map();
let rightTab = 'qvalue', fbTab = 'feedback';
let latestAnalysis = null;

// ── Grad-CAM 뷰 전환 ─────────────────────────────────────────────────────────
function setGcamMode(mode) {
  if (!hasGradCam && mode !== 'normal') {
    document.getElementById('gcamNoSupport').classList.add('visible');
    return;
  }
  document.getElementById('gcamNoSupport').classList.remove('visible');
  gcamMode = mode;

  document.getElementById('gcamNormalBtn').classList.toggle('active', mode === 'normal');
  document.getElementById('gcamHumanBtn').classList.toggle('active', mode === 'human');
  document.getElementById('gcamAgentBtn').classList.toggle('active', mode === 'agent');
  document.getElementById('gcamSplitBtn').classList.toggle('active', mode === 'split');

  document.getElementById('viewNormal').style.display   = mode === 'normal' ? '' : 'none';
  document.getElementById('viewHumanCam').style.display = mode === 'human'  ? '' : 'none';
  document.getElementById('viewAgentCam').style.display = mode === 'agent'  ? '' : 'none';
  document.getElementById('viewSplit').style.display    = mode === 'split'  ? '' : 'none';

  document.getElementById('gcamLegend').classList.toggle('visible', mode !== 'normal');

  if (cfFrames.length) renderCfFrame(cfIdx);
}

document.getElementById('gcamNormalBtn').addEventListener('click', () => setGcamMode('normal'));
document.getElementById('gcamHumanBtn').addEventListener('click', () => setGcamMode('human'));
document.getElementById('gcamAgentBtn').addEventListener('click', () => setGcamMode('agent'));
document.getElementById('gcamSplitBtn').addEventListener('click', () => setGcamMode('split'));

// ── 프레임 렌더 ───────────────────────────────────────────────────────────────
function b64src(b64) { return b64 ? 'data:image/jpeg;base64,' + b64 : ''; }

function renderCfFrame(idx) {
  if (!cfFrames.length) return;
  cfIdx = Math.max(0, Math.min(idx, cfFrames.length - 1));

  // NORMAL: 기존 side-by-side
  document.getElementById('compareImg').src = b64src(cfFrames[cfIdx]);

  // Grad-CAM 뷰들 (데이터 있을 때만)
  if (hasGradCam) {
    const hCam = cfGcamHuman[cfIdx];
    const aCam = cfGcamAgent[cfIdx];
    // HUMAN CAM
    document.getElementById('imgHumanCamFull').src = b64src(hCam);
    // AGENT CAM
    document.getElementById('imgAgentCamFull').src = b64src(aCam);
    // SPLIT: 각 셀에 일반 + Grad-CAM
    // 일반 프레임은 side-by-side에서 각각 분리할 수 없으므로
    // Grad-CAM 오버레이(50% 투명도)와 원본 side-by-side 교대 표시
    // → 별도 저장된 hCam/aCam으로 채움, 일반 쪽은 side-by-side 원본 대신
    //   서버에서 보낸 frames[idx] 좌/우 절반을 canvas로 분리하는 대신
    //   frames와 gcam을 같이 표시 (좌=frames, 우=gcam)
    document.getElementById('imgHumanNormal').src = b64src(cfFrames[cfIdx]);  // side-by-side 전체
    document.getElementById('imgHumanCam').src    = b64src(hCam);
    document.getElementById('imgAgentNormal').src = b64src(cfFrames[cfIdx]);
    document.getElementById('imgAgentCam').src    = b64src(aCam);
  }

  updateVkbd(cfHActs[cfIdx], cfAActs[cfIdx]);
}

function setSpeed(speed) {
  cfSpeed = speed;
  document.querySelectorAll('.speed-btn[data-speed]').forEach(b =>
    b.classList.toggle('active', parseFloat(b.dataset.speed) === speed));
  if (cfTimer) { clearInterval(cfTimer); cfTimer = startCfTimer(); }
  if (practiceActive) startPracticeAgentReplay();
}

function startCfTimer() {
  return setInterval(() => {
    const nextIdx = (cfIdx + 1) % cfFrames.length;
    const looped = nextIdx === 0 && cfIdx !== 0;
    cfIdx = nextIdx;
    renderCfFrame(cfIdx);
    if (looped && !practiceActive) startCompareCountdown();
  }, 1000 / (REPLAY_FPS * cfSpeed));
}

function playCfFrames(data) {
  clearInterval(cfTimer);
  cfFrames    = data.frames        || [];
  cfGcamHuman = data.gradcam_human || [];
  cfGcamAgent = data.gradcam_agent || [];
  cfHActs     = data.human_actions || [];
  cfAActs     = data.agent_actions || [];
  hasGradCam  = !!data.has_gradcam;
  cfIdx = 0;

  // Grad-CAM 미지원 시 버튼 반투명 처리
  ['gcamHumanBtn','gcamAgentBtn','gcamSplitBtn'].forEach(id => {
    document.getElementById(id).style.opacity = hasGradCam ? '' : '0.35';
  });

  if (!cfFrames.length) return;
  document.getElementById('compareStage').style.display = '';
  document.getElementById('practiceView').style.display = 'none';
  document.getElementById('compareStage').classList.add('visible');
  // 뷰 전환 (모드 재적용)
  setGcamMode(hasGradCam ? gcamMode : 'normal');
  renderCfFrame(0);
  cfTimer = startCfTimer();
  startCompareCountdown();
  document.getElementById('practiceBtn').style.display = 'inline-block';
}

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

// ── 후보 목록 ────────────────────────────────────────────────────────────────
function setCandidateStatus(ei, status) {
  candidateStatus.set(ei, status);
  renderCandidateBar();
  updateAutoGenerationLoading();
}

function updateAutoGenerationLoading() {
  if (pendingEI !== null && pendingEI !== undefined) return;
  const generating = candidates.some(c => candidateStatus.get(parseInt(c.entry_index)) === 'generating');
  if (!generating) {
    if (!pendingEI) {
      document.getElementById('fbLoading').classList.remove('active');
    }
    return;
  }
  document.getElementById('centerFeedback').classList.add('visible');
  document.getElementById('fbSource').textContent = HAS_LLM ? '비교 리플레이 생성 및 코칭 요청 중...' : '비교 리플레이 생성 중...';
  document.getElementById('fbText').textContent = '';
  document.getElementById('fbLoading').classList.add('active');
  document.getElementById('llmStatus').textContent = HAS_LLM ? '현재: 외부 LLM 확인 중' : '현재: 비교 리플레이 생성 중';
}

function renderCandidateBar() {
  const bar = document.getElementById('candidateBar');
  bar.innerHTML = '';
  candidates.forEach((c, idx) => {
    const ei     = parseInt(c.entry_index);
    const status = candidateStatus.get(ei) || 'waiting';
    const btn    = document.createElement('button');
    btn.className = `candidate-pill state-${status}${pendingEI === ei ? ' active' : ''}`;
    btn.type = 'button'; btn.dataset.ei = ei;
    btn.innerHTML =
      `<span class="candidate-rank">TOP ${idx + 1}</span>` +
      `<span class="candidate-main"><span>STEP ${c.step}</span><span class="candidate-loss"> (LOSS ${c.loss.toFixed(3)})</span></span>` +
      `<span class="candidate-state">${STATUS_LABELS[status]}</span>`;
    btn.addEventListener('click', () => {
      document.querySelectorAll('.candidate-pill').forEach(el => el.classList.remove('active'));
      btn.classList.add('active');
      requestCf(parseInt(btn.dataset.ei));
    });
    bar.appendChild(btn);
  });
}

function maybePrefetch() {
  if (prefetchStarted || !candidates.length) return;
  prefetchStarted = true;
  candidates.forEach((c, i) => {
    const ei = parseInt(c.entry_index);
    if (cfCache.has(ei)) { setCandidateStatus(ei, 'ready'); return; }
    setTimeout(() => prefetchCf(ei), 150 * (i + 1));
  });
}

function prefetchCf(ei) {
  if (Number.isNaN(ei) || cfCache.has(ei)) return;
  const s = candidateStatus.get(ei);
  if (s === 'generating' || s === 'ready') return;
  setCandidateStatus(ei, 'generating');
  socket.emit(P + 'request_counterfactual', { entry_index: ei, horizon: REPLAY_HORIZON, session_id: sessionId });
}

function requestCf(ei) {
  if (Number.isNaN(ei)) return;
  practiceActive = false;
  stopPracticeLoops();
  stopCompareCountdown();
  document.getElementById('practiceView').style.display = 'none';
  document.getElementById('practiceResultCard').classList.remove('visible');
  document.getElementById('practiceExitBtn').style.display = 'none';
  pendingEI = ei;
	  if (cfCache.has(ei)) {
	    const cached = cfCache.get(ei);
	    setCandidateStatus(ei, 'ready');
	    playCfFrames(cached);
	    showFeedback(cached);
	    renderStepReview(cached);
	    return;
	  }
  const status = candidateStatus.get(ei);
  if (status !== 'generating') setCandidateStatus(ei, 'generating');
  clearInterval(cfTimer); cfFrames = []; cfHActs = []; cfAActs = [];
  updateVkbd(0, 0);
  document.getElementById('compareStage').classList.remove('visible');
  document.getElementById('practiceBtn').style.display = 'none';
  document.getElementById('fbSource').textContent = HAS_LLM ? '비교 리플레이 생성 및 코칭 요청 중...' : '비교 리플레이 생성 중...';
  document.getElementById('fbText').textContent = '';
  document.getElementById('centerFeedback').classList.add('visible');
  document.getElementById('fbLoading').classList.add('active');
  document.getElementById('llmStatus').textContent = HAS_LLM ? '현재: 외부 LLM 확인 중' : '현재: 로컬';
  if (status !== 'generating') {
    socket.emit(P + 'request_counterfactual', { entry_index: ei, horizon: REPLAY_HORIZON, session_id: sessionId });
  }
}

function showFeedback(data) {
  document.getElementById('centerFeedback').classList.add('visible');
  document.getElementById('fbLoading').classList.remove('active');
  document.getElementById('fbSource').textContent =
    data.feedback_source === 'llm' ? '외부 LLM 코칭 피드백'
    : (HAS_LLM ? '외부 LLM 지연으로 로컬 피드백 표시' : '로컬 데이터 기반 코칭 피드백');
  document.getElementById('llmStatus').textContent =
    `현재: ${data.feedback_route || (data.feedback_source === 'llm' ? '외부 LLM' : '로컬')}` +
    (data.feedback_model ? ` (${data.feedback_model})` : '');

  // 구조화 피드백 렌더링
  const fb = data.feedback_structured || {};
  const situation  = fb.situation  || '';
  const comparison = fb.comparison || '';
  const advice     = fb.advice     || '';
  const hasStructure = situation || comparison || advice;

  const sitEl  = document.getElementById('fbSituation');
  const cmpEl  = document.getElementById('fbComparison');
  const advEl  = document.getElementById('fbAdvice');
  const txtEl  = document.getElementById('fbText');

  if (hasStructure) {
    txtEl.style.display = 'none';
    sitEl.style.display  = situation  ? '' : 'none';
    cmpEl.style.display  = comparison ? '' : 'none';
    advEl.style.display  = advice     ? '' : 'none';
    document.getElementById('fbSituationText').textContent  = situation;
    document.getElementById('fbComparisonText').textContent = comparison;
    document.getElementById('fbAdviceText').textContent     = advice;
  } else {
    // 구조 없음: fallback 단일 텍스트
    sitEl.style.display = cmpEl.style.display = advEl.style.display = 'none';
    txtEl.style.display = 'block';
    txtEl.textContent   = data.feedback || '';
  }

  // 비교 요약 탭
  const s = data.summary || {};
  const c = candidates.find(x => parseInt(x.entry_index) === pendingEI);
  document.getElementById('sumStep').textContent   = c?.step ?? s.step ?? '—';
  document.getElementById('sumLoss').textContent   = c?.loss != null ? Number(c.loss).toFixed(3) : '—';
  document.getElementById('sumHAction').textContent = s.human_action_name || '—';
  document.getElementById('sumAAction').textContent = s.agent_action_name || '—';
  document.getElementById('sumHScore').textContent  = s.human_score_delta ?? '—';
  document.getElementById('sumAScore').textContent  = s.agent_score_delta ?? '—';
  document.getElementById('sumHFirst').textContent  = s.human_first_reward_step ?? '—';
  document.getElementById('sumAFirst').textContent  = s.agent_first_reward_step ?? '—';
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

socket.on(P + 'counterfactual_ready', data => {
  if (data.session_id !== undefined && data.session_id !== sessionId) return;
  const ei = parseInt(data.entry_index);
  cfCache.set(ei, data);
  setCandidateStatus(ei, 'ready');
  const firstEi = candidates.length ? parseInt(candidates[0].entry_index) : null;
  if ((pendingEI === null || pendingEI === undefined) && ei === firstEi) {
    pendingEI = ei;
    renderCandidateBar();
  }
  if (ei !== pendingEI) { maybePrefetch(); return; }
  playCfFrames(data);
  showFeedback(data);
  renderStepReview(data);
  maybePrefetch();
});

socket.on(P + 'counterfactual_error', data => {
  const ei = data.entry_index != null ? parseInt(data.entry_index) : null;
  if (ei != null && !Number.isNaN(ei)) {
    setCandidateStatus(ei, 'error');
    if (ei !== pendingEI) return;
  }
  document.getElementById('centerFeedback').classList.add('visible');
  document.getElementById('fbLoading').classList.remove('active');
  document.getElementById('fbSource').textContent = '비교 리플레이 생성 실패';
  document.getElementById('fbText').textContent   = data.message || '오류가 발생했습니다.';
  document.getElementById('compareStage').classList.remove('visible');
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

// ── 도전과제 ───────────────────────────────────────────────────────────────
const tierLabels = { bronze: '🥉 브론즈', silver: '🥈 실버', gold: '🥇 골드', platinum: '💎 플래티넘' };
const tierIcons = { bronze: '🥉', silver: '🥈', gold: '🥇', platinum: '💎' };
const unlockedAchievementIds = new Set();

socket.on(P + 'achievement', data => {
  // 실시간 토스트 (개별 도전과제 달성)
  if (data.id) unlockedAchievementIds.add(data.id);
  showAchievementToast(data);
});

function showAchievementToast(ach) {
  const container = document.getElementById('achToastContainer');
  const toast = document.createElement('div');
  toast.className = `ach-toast ach-toast-${ach.tier}`;
  toast.innerHTML = `
    <div class="ach-toast-icon">${tierIcons[ach.tier] || '🏆'}</div>
    <div class="ach-toast-body">
      <div class="ach-toast-label">${tierLabels[ach.tier] || ach.tier}</div>
      <div class="ach-toast-title">${ach.title}</div>
      <div class="ach-toast-desc">${ach.desc}</div>
    </div>
    <div class="ach-toast-bar"></div>
  `;
  container.appendChild(toast);
  while (container.children.length > 4) {
    const oldest = container.firstElementChild;
    oldest.classList.remove('show');
    oldest.classList.add('hide');
    setTimeout(() => oldest.remove(), 350);
  }
  setTimeout(() => toast.classList.add('show'), 50);
  setTimeout(() => {
    toast.classList.remove('show');
    toast.classList.add('hide');
    setTimeout(() => toast.remove(), 500);
  }, 5000);
}

function buildAchievementState(achievements) {
  const source = (achievements && achievements.length) ? achievements : ALL_ACHIEVEMENTS;
  return source.map(a => ({
    ...a,
    unlocked: Boolean(a.unlocked || unlockedAchievementIds.has(a.id)),
  }));
}

function showAchievementModal(achievements) {
  const modal = document.getElementById('achModalOverlay');
  const body = document.getElementById('achModalBody');
  const tiers = { bronze: [], silver: [], gold: [], platinum: [] };
  const list = buildAchievementState(achievements);
  list.forEach(a => {
    if (a.unlocked && a.id) unlockedAchievementIds.add(a.id);
  });

  list.forEach(a => {
    if (tiers[a.tier]) tiers[a.tier].push(a);
  });

  const unlocked = list.filter(a => a.unlocked).length;
  document.getElementById('achModalCount').textContent = `${unlocked} / ${list.length} 달성`;

  let html = '';
  ['bronze', 'silver', 'gold', 'platinum'].forEach(tier => {
    if (tiers[tier].length === 0) return;
    html += `<div><div class="ach-modal-section-label">${tierLabels[tier]}</div><div class="ach-modal-grid">`;
    tiers[tier].forEach(a => {
      const cls = a.unlocked ? `unlocked-${tier}` : 'locked';
      const icon = a.unlocked ? tierIcons[tier] : '🔒';
      html += `
        <div class="ach-modal-item ${cls}">
          <div class="ach-mi-icon">${icon}</div>
          <div class="ach-mi-body">
            <div class="ach-mi-title">${a.title}</div>
            <div class="ach-mi-desc">${a.desc}</div>
          </div>
        </div>
      `;
    });
    html += `</div></div>`;
  });

  body.innerHTML = html;
  modal.style.display = 'flex';
}

document.getElementById('achSummaryBtn')?.addEventListener('click', () => {
  showAchievementModal();
});

document.getElementById('achModalClose').addEventListener('click', () => {
  document.getElementById('achModalOverlay').style.display = 'none';
});
document.getElementById('achModalOverlay').addEventListener('click', (e) => {
  if (e.target.id === 'achModalOverlay') {
    document.getElementById('achModalOverlay').style.display = 'none';
  }
});

// ── 연습 모드 ───────────────────────────────────────────────────────────────
let practiceActive = false;
let practiceCanvasHuman, practiceCanvasAgent, practiceCtxH, practiceCtxA;
let compareTimerInterval = null;
let practiceReplayInterval = null;
let practiceActionInterval = null;
let practiceTimerInterval = null;
let practiceEntryIndex = null;
let practiceLastResult = null;
let practiceAgentFrameIdx = 0;
let practiceDoneFallbackTimer = null;
let latestPracticeScore = 0;
let practiceStopRequested = false;

// Practice 버튼 클릭
document.getElementById('practiceBtn')?.addEventListener('click', () => {
  const ei = pendingEI;
  if (ei === null || ei === undefined) return;

  // 3,2,1 카운트다운
  const countdown = document.getElementById('practiceCountdown');
  const countdownNum = document.getElementById('practiceCountdownNum');
  countdown.style.display = 'flex';

  let count = 3;
  countdownNum.textContent = count;

  const countInterval = setInterval(() => {
    count--;
    if (count > 0) {
      countdownNum.textContent = count;
    } else {
      clearInterval(countInterval);
      countdown.style.display = 'none';
      startPractice(ei);
    }
  }, 1000);
});

function startPractice(entry_index) {
  practiceActive = true;
  practiceEntryIndex = entry_index;
  practiceLastResult = null;
  practiceAgentFrameIdx = 0;
  latestPracticeScore = 0;
  practiceStopRequested = false;
  if (practiceDoneFallbackTimer) clearTimeout(practiceDoneFallbackTimer);
  practiceDoneFallbackTimer = null;
  Object.keys(keys).forEach(id => { keys[id] = false; });
  currentAction = getAction();

  // 비교 영상 숨기고 연습 뷰 표시
  document.getElementById('compareStage').style.display = 'none';
  document.getElementById('practiceView').style.display = 'block';
  document.getElementById('practiceExitBtn').style.display = 'inline-block';
  document.getElementById('practiceOverOverlay').style.display = 'none';
  document.getElementById('practiceResultCard').classList.remove('visible');
  document.getElementById('practiceLiveScore').textContent = 'SCORE 0';
  document.getElementById('practiceTimeLeft').textContent = '30s';
  document.getElementById('practiceTimeLeft').classList.remove('timer-pulse');

  // 캔버스 초기화
  practiceCanvasHuman = document.getElementById('practiceCanvasHuman');
  practiceCanvasAgent = document.getElementById('practiceCanvasAgent');
  practiceCtxH = practiceCanvasHuman.getContext('2d');
  practiceCtxA = practiceCanvasAgent.getContext('2d');
  practiceCtxH.clearRect(0, 0, practiceCanvasHuman.width, practiceCanvasHuman.height);
  practiceCtxA.clearRect(0, 0, practiceCanvasAgent.width, practiceCanvasAgent.height);

  stopCompareCountdown();
  startPracticeAgentReplay();
  startPracticeCountdown();

  socket.emit(P + 'start_practice', { entry_index, horizon: 1800 });
}

// Practice 종료 버튼들
document.getElementById('practiceExitBtn')?.addEventListener('click', exitPractice);
document.getElementById('practiceBackBtn')?.addEventListener('click', exitPractice);
document.getElementById('practiceInlineBackBtn')?.addEventListener('click', exitPractice);

document.getElementById('practiceRetryBtn')?.addEventListener('click', () => {
  document.getElementById('practiceOverOverlay').style.display = 'none';
  const ei = pendingEI;
  if (ei !== null && ei !== undefined) {
    startPractice(ei);
  }
});

function exitPractice() {
  practiceActive = false;
  document.getElementById('practiceView').style.display = 'none';
  document.getElementById('compareStage').style.display = '';
  document.getElementById('practiceExitBtn').style.display = 'none';
  document.getElementById('practiceOverOverlay').style.display = 'none';
  stopPracticeLoops();
  startCompareCountdown();
  if (practiceLastResult) {
    const card = document.getElementById('practiceResultCard');
    if (card && card.classList.contains('visible')) {
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }
}

function stopPracticeLoops() {
  if (practiceReplayInterval) clearInterval(practiceReplayInterval);
  if (practiceActionInterval) clearInterval(practiceActionInterval);
  if (practiceTimerInterval) clearInterval(practiceTimerInterval);
  practiceReplayInterval = null;
  practiceActionInterval = null;
  practiceTimerInterval = null;
}

function finishPracticeFromClient() {
  if (practiceStopRequested) return;
  practiceStopRequested = true;
  practiceActive = false;
  stopPracticeLoops();
  const fallback = {
    score: latestPracticeScore,
    agree_rate: 0,
    player_actions: {},
    original: cfCache.get(practiceEntryIndex)?.cumulative || {},
    entry_index: practiceEntryIndex,
  };
  document.getElementById('practiceOverOverlay').style.display = 'flex';
  document.getElementById('practiceOverScore').textContent = `SCORE: ${latestPracticeScore || 0}`;
  renderPracticeResult(fallback, { pending: true });
  socket.emit(P + 'stop_practice');

  if (practiceDoneFallbackTimer) clearTimeout(practiceDoneFallbackTimer);
  practiceDoneFallbackTimer = setTimeout(() => {
    if (practiceLastResult) return;
    renderPracticeResult(fallback);
  }, 1500);
}

function paintCountdownTimer(el, seconds) {
  if (!el) return;
  el.classList.toggle('timer-pulse', seconds <= 10);
  if (seconds > 10) {
    el.style.color = '';
    el.style.borderColor = '';
    el.style.background = '';
    return;
  }
  const t = Math.max(0, Math.min(1, (10 - seconds) / 10));
  const g = Math.round(255 - 204 * t);
  const b = Math.round(255 - 153 * t);
  el.style.color = `rgb(255, ${g}, ${b})`;
  el.style.borderColor = `rgba(255, 51, 102, ${0.35 + 0.45 * t})`;
  el.style.background = `rgba(255, 51, 102, ${0.06 + 0.12 * t})`;
}

function resetCountdownTimer(el) {
  if (!el) return;
  el.classList.remove('timer-pulse');
  el.style.color = '';
  el.style.borderColor = '';
  el.style.background = '';
}

function stopCompareCountdown() {
  if (compareTimerInterval) clearInterval(compareTimerInterval);
  compareTimerInterval = null;
  resetCountdownTimer(document.getElementById('replayTimer'));
}

function startCompareCountdown() {
  const timerEl = document.getElementById('replayTimer');
  if (!timerEl) return;
  stopCompareCountdown();
  let seconds = 30;
  let lastTick = performance.now();
  timerEl.style.display = 'block';
  timerEl.textContent = '30s';
  paintCountdownTimer(timerEl, seconds);
  compareTimerInterval = setInterval(() => {
    const now = performance.now();
    const elapsed = (now - lastTick) / 1000;
    lastTick = now;
    seconds = Math.max(0, seconds - elapsed * cfSpeed);
    timerEl.textContent = `${Math.ceil(seconds)}s`;
    paintCountdownTimer(timerEl, seconds);
    if (seconds <= 0) {
      clearInterval(compareTimerInterval);
      compareTimerInterval = null;
    }
  }, 100);
}

function drawPracticeAgentFrame(idx) {
  if (!practiceCtxA || !cfFrames.length) return;
  const img = new Image();
  img.src = b64src(cfFrames[idx % cfFrames.length]);
  img.onload = () => {
    const halfW = img.naturalWidth / 2;
    practiceCtxA.drawImage(
      img,
      halfW, 0, halfW, img.naturalHeight,
      0, 0, practiceCanvasAgent.width, practiceCanvasAgent.height
    );
  };
}

function startPracticeAgentReplay() {
  if (practiceReplayInterval) clearInterval(practiceReplayInterval);
  drawPracticeAgentFrame(0);
  practiceReplayInterval = setInterval(() => {
    if (!practiceActive) return;
    practiceAgentFrameIdx = (practiceAgentFrameIdx + 1) % Math.max(1, cfFrames.length);
    drawPracticeAgentFrame(practiceAgentFrameIdx);
  }, 1000 / (REPLAY_FPS * cfSpeed));
}

function startPracticeActionLoop() {
  if (practiceActionInterval) clearInterval(practiceActionInterval);
  practiceActionInterval = setInterval(() => {
    if (!practiceActive) return;
    socket.emit(P + 'practice_action', { action: currentAction });
  }, 1000 / 60);
}

function startPracticeCountdown() {
  if (practiceTimerInterval) clearInterval(practiceTimerInterval);
  let seconds = 30;
  let lastTick = performance.now();
  const timerEl = document.getElementById('practiceTimeLeft');
  timerEl.textContent = '30s';
  paintCountdownTimer(timerEl, seconds);
  practiceTimerInterval = setInterval(() => {
    const now = performance.now();
    const elapsed = (now - lastTick) / 1000;
    lastTick = now;
    seconds = Math.max(0, seconds - elapsed * cfSpeed);
    timerEl.textContent = `${Math.ceil(seconds)}s`;
    paintCountdownTimer(timerEl, seconds);
    if (seconds <= 0) {
      clearInterval(practiceTimerInterval);
      practiceTimerInterval = null;
      timerEl.textContent = '0s';
      if (practiceActive) {
        finishPracticeFromClient();
      }
    }
  }, 100);
}

// Socketio events
socket.on(P + 'practice_ready', data => {
  if (!practiceActive) return;
  // 첫 프레임 표시
  const img = new Image();
  img.src = 'data:image/jpeg;base64,' + data.first_frame;
  img.onload = () => {
    practiceCtxH.drawImage(img, 0, 0, practiceCanvasHuman.width, practiceCanvasHuman.height);
  };
  startPracticeActionLoop();
});

socket.on(P + 'practice_frame', data => {
  if (!practiceActive) return;

  const img = new Image();
  img.src = 'data:image/jpeg;base64,' + data.image;
  img.onload = () => {
    practiceCtxH.drawImage(img, 0, 0, practiceCanvasHuman.width, practiceCanvasHuman.height);
  };
  if (data.score != null) {
    latestPracticeScore = Number(data.score) || 0;
    document.getElementById('practiceLiveScore').textContent = `SCORE ${data.score}`;
  }

  if (data.done) {
    finishPracticeFromClient();
  }
});

socket.on(P + 'practice_done', data => {
  if (practiceDoneFallbackTimer) clearTimeout(practiceDoneFallbackTimer);
  practiceDoneFallbackTimer = null;
  practiceLastResult = data;
  stopPracticeLoops();
  document.getElementById('practiceOverOverlay').style.display = 'flex';
  document.getElementById('practiceOverScore').textContent = `SCORE: ${data.score || 0}`;
  renderPracticeResult(data);
});

function pctMap(counts) {
  const total = Object.values(counts || {}).reduce((sum, v) => sum + Number(v || 0), 0) || 1;
  const out = {};
  actionIds().forEach(id => {
    const name = ACTION_NAMES[id];
    out[name] = Math.round(((counts || {})[name] || 0) / total * 1000) / 10;
  });
  return out;
}

function renderPracticeResult(data, opts = {}) {
  const card = document.getElementById('practiceResultCard');
  const cf = practiceEntryIndex != null ? cfCache.get(practiceEntryIndex) : null;
  const originalAgree = Number(cf?.cumulative?.agree_rate ?? data?.original?.agree_rate ?? 0);
  const practiceAgree = Number(data?.agree_rate ?? 0);
  const originalScore = Number(cf?.summary?.human_score_delta ?? 0);
  const practiceScore = Number(data?.score ?? 0);
  const agreeDelta = Math.round((practiceAgree - originalAgree) * 10) / 10;
  const scoreDelta = Math.round((practiceScore - originalScore) * 10) / 10;
  const scoreComponent = originalScore > 0 ? Math.min(100, Math.max(0, practiceScore / originalScore * 100)) : (practiceScore > 0 ? 100 : 50);
  const composite = Math.round((scoreComponent * 0.6 + practiceAgree * 0.4));
  const level = composite >= 80 ? 'best' : composite >= 60 ? 'good' : composite >= 30 ? 'mid' : 'low';
  const message = composite >= 80
    ? '완벽해요! 이 순간만큼은 에이전트 수준입니다.'
    : composite >= 60
      ? '훌륭해요! 이 상황을 충분히 파악하고 있어요.'
      : composite >= 30
        ? '잘 하고 있어요! 피드백과 비교 영상을 조금 더 참고하면 더 나아질 거예요.'
        : '아직 이 상황이 익숙하지 않은 것 같아요. 비교 영상을 다시 보고 에이전트의 행동 패턴을 참고해보세요.';
  const origPct = pctMap(data?.original?.player_actions || {});
  const pracPct = pctMap(data?.player_actions || {});
  const rows = actionIds().map(id => {
    const name = ACTION_NAMES[id];
    return `
      <div class="practice-compare-row">
        <div class="practice-compare-cell orig">${name}</div>
        <div class="practice-compare-cell prac">${formatValue(origPct[name])}% → ${formatValue(pracPct[name])}%</div>
      </div>`;
  }).join('');
  card.className = `practice-result-card prac-composite--${level}`;
  card.innerHTML = `
    <div class="practice-result-header">🎮 PRACTICE RESULT</div>
    ${opts.pending ? '<div class="practice-result-pending">결과 집계 중...</div>' : ''}
    <div class="prac-composite">
      <div class="prac-composite-label">종합 점수</div>
      <div class="prac-composite-bar-wrap"><div class="prac-composite-bar" style="width:${composite}%"></div></div>
      <div class="prac-composite-val">${composite}<span class="prac-composite-unit">/100</span></div>
    </div>
    <div class="practice-compare-row"><div class="practice-compare-cell orig">원본 AI 일치율</div><div class="practice-compare-cell orig">${originalAgree}%</div></div>
    <div class="practice-compare-row"><div class="practice-compare-cell prac">연습 AI 일치율</div><div class="practice-compare-cell ${agreeDelta >= 0 ? 'improved' : 'worse'}">${practiceAgree}% (${agreeDelta >= 0 ? '+' : ''}${agreeDelta}%)</div></div>
    <div class="practice-compare-row"><div class="practice-compare-cell orig">원본 구간 점수</div><div class="practice-compare-cell orig">${originalScore}</div></div>
    <div class="practice-compare-row"><div class="practice-compare-cell prac">연습 점수</div><div class="practice-compare-cell ${scoreDelta >= 0 ? 'improved' : 'worse'}">${practiceScore} (${scoreDelta >= 0 ? '+' : ''}${scoreDelta})</div></div>
    <div class="practice-result-header">행동 비율 비교 (원본 → 연습)</div>
    ${rows}
    <div class="practice-feedback practice-feedback--${level}">${message}</div>
  `;
  card.classList.add('visible');
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
