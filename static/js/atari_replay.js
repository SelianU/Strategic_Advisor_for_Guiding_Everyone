// atari_replay.js — 소켓 초기화, Counterfactual 리플레이, Grad-CAM, 비교 타이머

const REPLAY_FPS    = 60;
const REPLAY_HORIZON = REPLAY_FPS * 30;
// STATUS_LABELS 는 common_utils.js 에서 공유

// ── 소켓 / 캔버스 ────────────────────────────────────────────────────────────
const socket  = createGameSocket();

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

// ── 비교 타이머 ───────────────────────────────────────────────────────────────
let compareTimerInterval = null;

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

// ── 비교 카운트다운 타이머 ────────────────────────────────────────────────────
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
  document.getElementById('llmStatus').textContent = `현재: ${feedbackRouteText(data)}`;

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

// ── 소켓 이벤트: 카운터팩추얼 ────────────────────────────────────────────────
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
