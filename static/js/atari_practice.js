// atari_practice.js — 연습 모드 전체 (startPractice, exitPractice, 소켓 이벤트 등)

let practiceActive = false;
let practiceCanvasHuman, practiceCanvasAgent, practiceCtxH, practiceCtxA;
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

// ── 소켓 이벤트: 연습 모드 ───────────────────────────────────────────────────
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
