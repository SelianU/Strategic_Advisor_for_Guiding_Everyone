const socket = createGameSocket();
const BOARD_W = 15, BOARD_H = 15;
const HAS_LLM = document.body.dataset.hasLlm === 'true';
let humanColor = 1;  // 1=흑, 2=백

function selectColor(color) {
  humanColor = color;
  document.getElementById('colorBlack').classList.toggle('active', color === 1);
  document.getElementById('colorWhite').classList.toggle('active', color === 2);
  if (color === 1) {
    document.getElementById('legendHuman').innerHTML = '<div class="dot dot-black"></div>흑(●) — 당신';
    document.getElementById('legendAI').innerHTML    = '<div class="dot dot-white"></div>백(○) — AI';
  } else {
    document.getElementById('legendHuman').innerHTML = '<div class="dot dot-white"></div>백(○) — 당신';
    document.getElementById('legendAI').innerHTML    = '<div class="dot dot-black"></div>흑(●) — AI';
  }
}

let gameActive = false, myMoveCount = 0, boardState = [];
let gomokuCompareFrames = [];   // agent frames
let gomokuHumanFrames = [];     // human frames (auto-play 동기화용)
let gomokuCompareTimer = null;
let gomokuCompareIdx = 0;
let gomokuCandidates = [];
const gomokuReplayCache = new Map();
let gomokuSessionId = 0;
let gomokuPendingMoveNum = null;
let gomokuPrecomputeStarted = false;
const gomokuCandidateStatus = new Map();
let gomokuSavedSessions = [];
let gomokuGameOverHoldUntil = 0;
let gomokuGameOverTimer = null;
let gomokuPendingAnalysisData = null;
let gomokuLastStateSeq = -1;
let gomokuRightTab = 'summary';
let gomokuFeedbackTab = 'feedback';
// 후보 상태 라벨은 common_utils.js 의 STATUS_LABELS 공유

function setGomokuRightTab(tab) {
  gomokuRightTab = tab;
  document.querySelectorAll('#gomokuRightTabs .right-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.gomokuTab === tab);
  });
  const summaryOpen = tab === 'summary';
  document.getElementById('scoreSection').classList.toggle('tab-open', summaryOpen);
  document.getElementById('tableSection').classList.toggle('tab-open', tab === 'qvalue');
}

function setGomokuFeedbackTab(tab) {
  gomokuFeedbackTab = tab;
  document.querySelectorAll('[data-gomoku-feedback-tab]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.gomokuFeedbackTab === tab);
  });
  document.getElementById('gomokuFeedbackTextCenter').style.display = tab === 'feedback' ? 'block' : 'none';
  document.getElementById('gomokuFeedbackSummaryBox').classList.toggle('visible', tab === 'summary');
}

function getGomokuCandidateByMove(moveNum) {
  return gomokuCandidates.find(candidate => parseInt(candidate.move_num) === parseInt(moveNum)) || null;
}

function updateGomokuSummaryFromPayload(moveNum, summary) {
  const candidate = getGomokuCandidateByMove(moveNum);
  document.getElementById('gomokuMoveVal').textContent = candidate?.move_num ?? summary?.move_num ?? '—';
  document.getElementById('gomokuLossVal').textContent =
    candidate?.loss !== undefined && candidate?.loss !== null
      ? Number(candidate.loss).toFixed(3)
      : (summary?.loss ?? '—');
  document.getElementById('gomokuHumanMoveVal').textContent = summary?.actual_row !== undefined ? `(${summary.actual_row}, ${summary.actual_col})` : '—';
  document.getElementById('gomokuAgentMoveVal').textContent = summary?.best_row !== undefined ? `(${summary.best_row}, ${summary.best_col})` : '—';
}

// 자동 재생/화살표 이동 공용: 인간+에이전트 프레임 동기 렌더
function renderGomokuFrame(idx) {
  if (!gomokuCompareFrames.length) return;
  gomokuCompareIdx = Math.max(0, Math.min(idx, gomokuCompareFrames.length - 1));
  document.getElementById('gomokuAgentImage').src = b64src(gomokuCompareFrames[gomokuCompareIdx]);
  if (gomokuHumanFrames[gomokuCompareIdx]) {
    document.getElementById('gomokuHumanImage').src = b64src(gomokuHumanFrames[gomokuCompareIdx]);
  }
  document.getElementById('gomokuFrameCounter').textContent =
    `${gomokuCompareIdx + 1} / ${gomokuCompareFrames.length}`;
  document.getElementById('gomokuPrevBtn').disabled = gomokuCompareIdx === 0;
  document.getElementById('gomokuNextBtn').disabled = gomokuCompareIdx === gomokuCompareFrames.length - 1;
}

function playGomokuFrames(humanFrames, agentFrames) {
  clearInterval(gomokuCompareTimer);
  gomokuCompareTimer = null;
  gomokuHumanFrames = humanFrames || [];
  gomokuCompareFrames = agentFrames || [];
  if (!gomokuCompareFrames.length) return;
  document.getElementById('gomokuCompareStage').classList.add('visible');
  // 처음엔 양쪽 동기화해서 재생
  renderGomokuFrame(0);
  gomokuCompareTimer = setInterval(() => {
    if (gomokuCompareIdx < gomokuCompareFrames.length - 1) {
      renderGomokuFrame(gomokuCompareIdx + 1);
    } else {
      // 1회 재생 완료: 첫 프레임으로 복귀 (양쪽 모두 화살표로 이동 가능)
      clearInterval(gomokuCompareTimer);
      gomokuCompareTimer = null;
      renderGomokuFrame(0);
    }
  }, 1800);
}

function renderGomokuSessionOptions(sessions) {
  gomokuSavedSessions = populateSessionSelect(
    document.getElementById('gomokuSessionSelect'), sessions);
}

function setGomokuCandidateStatus(moveNum, status) {
  gomokuCandidateStatus.set(moveNum, status);
  gomokuCandidates = [...gomokuCandidates];
  renderGomokuCandidateBar(gomokuCandidates);
}

function maybePrefetchGomokuCandidates() {
  if (gomokuPrecomputeStarted || !gomokuCandidates.length) return;
  const firstMoveNum = parseInt(gomokuCandidates[0].move_num);
  if (gomokuPendingMoveNum !== firstMoveNum) return;
  gomokuPrecomputeStarted = true;
  gomokuCandidates.slice(1).forEach((candidate, idx) => {
    const moveNum = parseInt(candidate.move_num);
    const cacheKey = `${gomokuSessionId}:${moveNum}`;
    if (gomokuReplayCache.has(cacheKey)) {
      setGomokuCandidateStatus(moveNum, 'ready');
      return;
    }
    setTimeout(() => prefetchGomokuReplay(moveNum), 150 * (idx + 1));
  });
}

function prefetchGomokuReplay(moveNum) {
  if (Number.isNaN(moveNum)) return;
  const cacheKey = `${gomokuSessionId}:${moveNum}`;
  if (gomokuReplayCache.has(cacheKey)) return;
  const currentStatus = gomokuCandidateStatus.get(moveNum);
  if (currentStatus === 'generating' || currentStatus === 'ready') return;
  setGomokuCandidateStatus(moveNum, 'generating');
  socket.emit('gomoku_request_counterfactual', { move_num: moveNum, session_id: gomokuSessionId });
}

// 리플레이 + 코칭 피드백 표시 (캐시 적중 / 신규 수신 공용)
function showGomokuCounterfactual(moveNum, data) {
  playGomokuFrames(data.human_frames || [], data.agent_frames || []);
  document.getElementById('gomokuCenterFeedback').classList.add('visible');
  document.getElementById('gomokuFeedbackLoading').classList.remove('active');
  document.getElementById('gomokuFeedbackSourceCenter').textContent =
    data.feedback_source === 'llm'
      ? '외부 LLM 코칭 피드백'
      : (HAS_LLM ? '외부 LLM 지연으로 로컬 데이터 기반 코칭 피드백을 표시합니다.' : '로컬 데이터 기반 코칭 피드백');
  document.getElementById('gomokuLlmCurrentStatus').textContent = `현재 선택: ${feedbackRouteText(data)}`;
  document.getElementById('gomokuFeedbackTextCenter').textContent = data.feedback || '';
  updateGomokuSummaryFromPayload(moveNum, data.summary || {});
}

function requestGomokuReplay(moveNum) {
  if (Number.isNaN(moveNum)) return;
  gomokuPendingMoveNum = moveNum;
  const cacheKey = `${gomokuSessionId}:${moveNum}`;
  if (gomokuReplayCache.has(cacheKey)) {
    setGomokuCandidateStatus(moveNum, 'ready');
    showGomokuCounterfactual(moveNum, gomokuReplayCache.get(cacheKey));
    return;
  }
  setGomokuCandidateStatus(moveNum, 'generating');
  clearInterval(gomokuCompareTimer);
  gomokuCompareFrames = [];
  gomokuHumanFrames = [];
  document.getElementById('gomokuCompareStage').classList.remove('visible');
  document.getElementById('gomokuHumanImage').src = '';
  document.getElementById('gomokuAgentImage').src = '';
  document.getElementById('gomokuCenterFeedback').classList.add('visible');
  document.getElementById('gomokuFeedbackSourceCenter').textContent = HAS_LLM ? '비교 리플레이 생성 및 코칭 요청 중...' : '비교 리플레이 생성 중...';
  document.getElementById('gomokuFeedbackTextCenter').textContent = '';
  document.getElementById('gomokuFeedbackLoading').classList.toggle('active', HAS_LLM);
  document.getElementById('gomokuLlmCurrentStatus').textContent = HAS_LLM ? '현재 선택: 외부 LLM 확인 중' : '현재 선택: 로컬';
  updateGomokuSummaryFromPayload(moveNum, {});
  socket.emit('gomoku_request_counterfactual', { move_num: moveNum, session_id: gomokuSessionId });
}

function renderGomokuCandidateBar(candidates) {
  const bar = document.getElementById('gomokuCandidateBar');
  bar.innerHTML = '';
  candidates.forEach((a, idx) => {
    const btn = document.createElement('button');
    const moveNum = parseInt(a.move_num);
    const status = gomokuCandidateStatus.get(moveNum) || 'waiting';
    btn.className = 'candidate-pill state-' + status + (gomokuPendingMoveNum === moveNum ? ' active' : '');
    btn.type = 'button';
    btn.dataset.moveNum = moveNum;
    btn.innerHTML =
      `<span class="candidate-rank">TOP ${idx + 1}</span>` +
      `<span class="candidate-main"><span class="candidate-step">${a.move_num}수</span><span class="candidate-loss">(LOSS ${a.loss.toFixed(3)})</span></span>` +
      `<span class="candidate-state">${STATUS_LABELS[status] || STATUS_LABELS.waiting}</span>`;
    btn.addEventListener('click', () => {
      document.querySelectorAll('#gomokuCandidateBar .candidate-pill').forEach(el => el.classList.remove('active'));
      btn.classList.add('active');
      requestGomokuReplay(parseInt(btn.dataset.moveNum));
    });
    bar.appendChild(btn);
  });
}

function updateEvalBar(winRate) {
  if (winRate == null) return;
  const fill = document.getElementById('evalBarFill');
  const topLabel = document.getElementById('evalBarTop');
  const bottomLabel = document.getElementById('evalBarBottom');
  
  // 인간이 흑이면 '아래(검정)'가 인간 점유, 백이면 반대
  const humanIsBlack = humanColor === 1;
  const blackPct = humanIsBlack ? winRate : (100 - winRate);
  
  // 검정 영역(아래쪽)이 차오르는 비율
  fill.style.height = blackPct + '%';
  
  // 라벨: 위(백)와 아래(흑) 영역에 각각 해당 측 승률 표시
  const blackWin = blackPct.toFixed(1);
  const whiteWin = (100 - blackPct).toFixed(1);
  topLabel.textContent = whiteWin + '%';
  bottomLabel.textContent = blackWin + '%';
}

// ── 보드 초기화 ──────────────────────────────────────────────
function initBoard() {
  const grid = document.getElementById('gomokuBoard');
  grid.style.gridTemplateColumns = `repeat(${BOARD_W}, 30px)`;
  grid.innerHTML = '';
  boardState = Array(BOARD_W * BOARD_H).fill(0);

  for (let r = BOARD_H - 1; r >= 0; r--) {
    for (let c = 0; c < BOARD_W; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.row = r;
      cell.dataset.col = c;
      cell.dataset.move = r * BOARD_W + c;
      cell.addEventListener('click', onCellClick);
      grid.appendChild(cell);
    }
  }
}

function clearGomokuTerminalOverlay() {
  document.getElementById('gomokuWinOverlay').innerHTML = '';
  const outcomeEl = document.getElementById('gomokuOutcomeOverlay');
  outcomeEl.textContent = '';
  outcomeEl.classList.remove('visible');
}

function drawGomokuWinningLine(winningLine) {
  const svg = document.getElementById('gomokuWinOverlay');
  svg.innerHTML = '';
  if (!Array.isArray(winningLine) || winningLine.length < 2) return;

  const overlayRect = document.getElementById('gomokuBoardOverlay').getBoundingClientRect();
  const normalized = winningLine.map(item => {
    if (Array.isArray(item)) return item;
    if (Number.isInteger(item)) return [Math.floor(item / BOARD_W), item % BOARD_W];
    return null;
  }).filter(Boolean);
  if (normalized.length < 2) return;
  const endpoints = [normalized[0], normalized[normalized.length - 1]].map(([row, col]) => {
    const cell = document.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
    if (!cell) return null;
    const rect = cell.getBoundingClientRect();
    return {
      x: rect.left - overlayRect.left + rect.width / 2,
      y: rect.top - overlayRect.top + rect.height / 2,
    };
  });
  if (!endpoints[0] || !endpoints[1]) return;
  svg.innerHTML = `<line x1="${endpoints[0].x}" y1="${endpoints[0].y}" x2="${endpoints[1].x}" y2="${endpoints[1].y}"></line>`;
}

function renderBoard(cells, lastMove, winningLine = null, outcomeText = '') {
  const allCells = document.querySelectorAll('.cell');
  const normalized = (winningLine || []).map(item => {
    if (Array.isArray(item)) return item;
    if (Number.isInteger(item)) return [Math.floor(item / BOARD_W), item % BOARD_W];
    return null;
  }).filter(Boolean);
  const winningSet = new Set(normalized.map(([row, col]) => `${row},${col}`));
  allCells.forEach(cell => {
    const move = parseInt(cell.dataset.move);
    const p = cells[move];
    cell.innerHTML = '';
    cell.classList.remove('last-move', 'winning-cell');

    if (p === 1) {
      const s = document.createElement('div');
      s.className = 'stone stone-black';
      cell.appendChild(s);
    } else if (p === 2) {
      const s = document.createElement('div');
      s.className = 'stone stone-white';
      cell.appendChild(s);
    }

    if (move === lastMove) cell.classList.add('last-move');
    if (winningSet.has(`${cell.dataset.row},${cell.dataset.col}`)) cell.classList.add('winning-cell');
  });
  drawGomokuWinningLine(normalized);
  const outcomeEl = document.getElementById('gomokuOutcomeOverlay');
  if (outcomeText) {
    outcomeEl.textContent = outcomeText;
    outcomeEl.classList.add('visible');
  } else {
    outcomeEl.textContent = '';
    outcomeEl.classList.remove('visible');
  }
}

function setMsg(text, color = 'var(--cyan)') {
  const el = document.getElementById('msgBox');
  el.textContent = text;
  el.style.color = color;
}

// ── 비교 영상 수동 탐색 (조작 시 자동 재생 중단) ───────────
function stopGomokuAutoPlay() {
  if (gomokuCompareTimer) {
    clearInterval(gomokuCompareTimer);
    gomokuCompareTimer = null;
  }
}

document.getElementById('gomokuPrevBtn').addEventListener('click', () => {
  stopGomokuAutoPlay();
  renderGomokuFrame(gomokuCompareIdx - 1);
});
document.getElementById('gomokuNextBtn').addEventListener('click', () => {
  stopGomokuAutoPlay();
  renderGomokuFrame(gomokuCompareIdx + 1);
});
document.addEventListener('keydown', (e) => {
  if (!document.getElementById('gomokuCompareStage').classList.contains('visible')) return;
  if (e.key === 'ArrowLeft')  { e.preventDefault(); stopGomokuAutoPlay(); renderGomokuFrame(gomokuCompareIdx - 1); }
  if (e.key === 'ArrowRight') { e.preventDefault(); stopGomokuAutoPlay(); renderGomokuFrame(gomokuCompareIdx + 1); }
});

// ── 이벤트 핸들러 ────────────────────────────────────────────
document.getElementById('startBtn').addEventListener('click', () => {
  const introPanel = document.getElementById('gameIntroPanel');
  if (introPanel) introPanel.classList.add('hidden');
  const hIcon = document.getElementById('coachHeaderIcon');
  const hLabel = document.getElementById('coachHeaderLabel');
  if (hIcon)  hIcon.textContent  = '🎓';
  if (hLabel) hLabel.textContent = 'SAGE · COACH';
  gameActive = false; myMoveCount = 0;
  gomokuUnlockedIds.clear();   // 새 게임마다 도전과제 초기화
  clearTimeout(gomokuGameOverTimer);
  gomokuGameOverTimer = null;
  gomokuGameOverHoldUntil = 0;
  gomokuPendingAnalysisData = null;
  gomokuLastStateSeq = -1;
  gomokuSessionId += 1;
  gomokuPendingMoveNum = null;
  gomokuPrecomputeStarted = false;
  gomokuCandidateStatus.clear();
  document.getElementById('moveCount').textContent = '0';
  document.getElementById('gomokuSessionLabel').textContent = 'LIVE';
  document.getElementById('statusBar').textContent = 'WAITING FOR SERVER';
  document.getElementById('gomokuBoardWrap').classList.remove('hidden');
  document.getElementById('evalBarContainer').style.display = '';
  document.getElementById('gomokuPostgamePanel').classList.remove('visible');
  document.getElementById('gomokuCompareStage').classList.remove('visible');
  document.getElementById('gomokuCenterFeedback').classList.remove('visible');
  document.getElementById('gomokuCandidateBar').innerHTML = '';
  document.getElementById('gomokuFeedbackSourceCenter').textContent = '';
  document.getElementById('gomokuFeedbackTextCenter').textContent = '착수를 고르면 코칭 피드백이 여기에 표시됩니다.';
  document.getElementById('gomokuFeedbackLoading').classList.remove('active');
  gomokuReplayCache.clear();
  clearGomokuTerminalOverlay();
  hideAnalysis();
  initBoard();
  socket.emit('gomoku_start', { human_color: humanColor });
  document.getElementById('startBtn').disabled = true;
  updateEvalBar(50);
});
document.getElementById('gomokuSaveBtn').addEventListener('click', () => {
  const title = document.getElementById('gomokuSessionTitle').value.trim();
  if (!title) {
    window.alert('저장할 기록 이름을 입력해주세요.');
    return;
  }
  socket.emit('gomoku_save_session', {
    title,
  });
});
document.getElementById('gomokuLoadBtn').addEventListener('click', () => {
  const sessionId = document.getElementById('gomokuSessionSelect').value;
  if (!sessionId) return;
  socket.emit('gomoku_load_session', { session_id: sessionId });
});
document.getElementById('gomokuDeleteBtn').addEventListener('click', () => {
  const sessionId = document.getElementById('gomokuSessionSelect').value;
  if (!sessionId) return;
  socket.emit('gomoku_delete_session', { session_id: sessionId });
});
socket.emit('gomoku_list_sessions');

function onCellClick(e) {
  if (!gameActive) return;
  const row = parseInt(this.dataset.row);
  const col = parseInt(this.dataset.col);
  const move = row * BOARD_W + col;
  if (boardState[move] !== 0) return;

  gameActive = false; // 응답 올 때까지 대기
  socket.emit('gomoku_move', { row, col });
}

// ── Socket 이벤트 ────────────────────────────────────────────
socket.on('gomoku_state', (data) => {
  console.log('gomoku_state received', {
    session_id: data.session_id,
    state_seq: data.state_seq,
    game_over: !!data.game_over,
    winner: data.winner,
    last_move: data.last_move,
    message: data.message,
  });
  if (data.session_id !== undefined) {
    gomokuSessionId = data.session_id;
  }
  if (data.state_seq !== undefined) {
    if (data.state_seq < gomokuLastStateSeq) return;
    gomokuLastStateSeq = data.state_seq;
  }
  if (data.human_color !== undefined) selectColor(data.human_color);
  if (data.win_rate !== undefined) updateEvalBar(data.win_rate);
  boardState = data.cells;
  renderBoard(data.cells, data.last_move, data.winning_line || null, data.outcome_text || '');
  // ── 생존 도전과제: 내 돌 수 기준으로 순차 체크 ──────────────
  const myStones = data.cells.filter(c => c === humanColor).length;
  checkGomokuSurviveAchievements(myStones);
  // ─────────────────────────────────────────────────────────
  if (data.game_over) {
    gameActive = false;
    document.getElementById('statusBar').textContent = 'GAME OVER';
    document.getElementById('startBtn').disabled = false;
    const humanWon = data.winner === humanColor;
    const color = humanWon ? 'var(--green)' : (data.winner === -1 ? 'var(--yellow)' : 'var(--red)');
    setMsg(data.message, color);
    // ── 승리 도전과제: 게임 종료 시에만 체크 ────────────────────
    checkGomokuWinAchievements(humanWon, humanColor);
    // ─────────────────────────────────────────────────────────
    clearTimeout(gomokuGameOverTimer);
    gomokuGameOverHoldUntil = Date.now() + 3200;
    gomokuPendingAnalysisData = null;
    gomokuGameOverTimer = setTimeout(() => {
      gomokuGameOverTimer = null;
      gomokuGameOverHoldUntil = 0;
      if (gomokuPendingAnalysisData) {
        const pending = gomokuPendingAnalysisData;
        gomokuPendingAnalysisData = null;
        showAnalysis(pending);
      }
    }, 3200);
    return;
  }
  setMsg(data.message);
  document.getElementById('statusBar').textContent =
    data.current_player === 1 ? '당신의 차례 (흑 ●)' : 'AI 생각 중...';
  if (data.current_player === humanColor) {
    gameActive = true;
    myMoveCount = data.cells.filter(c => c === humanColor).length;
    document.getElementById('moveCount').textContent = myMoveCount;
  }
});

socket.on('gomoku_error', (data) => {
  setMsg(data.message, 'var(--red)');
  gameActive = true;
});

// ── 분석 진행 ────────────────────────────────────────────────
socket.on('gomoku_analysis_start', (data) => {
  if (data.session_id !== undefined && data.session_id !== gomokuSessionId) return;
  if (Date.now() < gomokuGameOverHoldUntil) return;
  document.getElementById('progressSection').classList.add('visible');
  document.getElementById('progressText').textContent = `0 / ${data.total} 착수 분석 중...`;
  document.getElementById('progressFill').style.width = '0%';
});

socket.on('gomoku_analysis_progress', (data) => {
  if (data.session_id !== undefined && data.session_id !== gomokuSessionId) return;
  if (Date.now() < gomokuGameOverHoldUntil) return;
  const pct = (data.current / data.total * 100).toFixed(0);
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressText').textContent =
    `${data.current} / ${data.total} — 착수 ${data.move} 분석 중...`;
});

socket.on('gomoku_analysis_done', (data) => {
  if (data.session_id !== undefined && data.session_id !== gomokuSessionId) return;
  if (Date.now() < gomokuGameOverHoldUntil) {
    gomokuPendingAnalysisData = data;
    return;
  }
  document.getElementById('progressSection').classList.remove('visible');
  showAnalysis(data);
});

function hideAnalysis() {
  ['progressSection','scoreSection','tableSection'].forEach(id => {
    document.getElementById(id).classList.remove('visible');
  });
  document.getElementById('gomokuAnalysisZone').classList.remove('visible');
}

function showAnalysis(data) {
  clearGomokuTerminalOverlay();
  // 종합 점수
  const avgLoss = data.avg_loss;
  let grade, gradeColor;
  if (avgLoss < 0.05)      { grade = '🏆 탁월한 플레이';  gradeColor = 'var(--green)'; }
  else if (avgLoss < 0.12) { grade = '👍 좋은 플레이';    gradeColor = 'var(--cyan)'; }
  else if (avgLoss < 0.20) { grade = '⚖️ 평균적 플레이'; gradeColor = 'var(--yellow)'; }
  else                     { grade = '📖 개선 필요';       gradeColor = 'var(--red)'; }

  document.getElementById('avgLoss').textContent = avgLoss.toFixed(3);
  document.getElementById('avgLoss').style.color = gradeColor;
  document.getElementById('scoreGrade').textContent = grade;
  document.getElementById('gomokuAnalysisZone').classList.add('visible');
  document.getElementById('scoreSection').classList.add('visible');

  // 최악 착수
  if (data.worst) {
    const w = data.worst;
    const worstHtml =
      `<strong>${w.move_num}수</strong> → (${w.row}, ${w.col})<br>` +
      `실제 Q: <strong>${w.actual_q ?? 'N/A'}</strong><br>` +
      `최선 Q: <strong style="color:var(--green)">${w.best_q}</strong> → (${w.best_row}, ${w.best_col})<br>` +
      `<span style="color:var(--red)">손실: ${w.loss.toFixed(3)}</span>`;
    document.getElementById('worstDetail').innerHTML = worstHtml;
  }

  // 착수별 표
  const tbody = document.getElementById('moveTableBody');
  tbody.innerHTML = '';
  data.analyses.forEach(a => {
    const tr = document.createElement('tr');
    if (data.worst && a.move_num === data.worst.move_num) tr.classList.add('worst-row');

    const loss = a.loss;
    let lossClass = 'q-good';
    if (loss === null) lossClass = '';
    else if (loss > 0.20) lossClass = 'q-bad';
    else if (loss > 0.08) lossClass = 'q-ok';

    const qActual = a.actual_q !== null ? a.actual_q.toFixed(3) : 'N/A';
    const qBest   = a.best_q   !== null ? a.best_q.toFixed(3)   : 'N/A';
    const lossStr = loss        !== null ? loss.toFixed(3)       : 'N/A';
    const icon    = a.is_best ? '★' : '';

    tr.innerHTML =
      `<td>${a.move_num}${icon}</td>` +
      `<td>(${a.row},${a.col})</td>` +
      `<td class="${a.is_best ? 'q-best' : ''}">${qActual}</td>` +
      `<td>${qBest}</td>` +
      `<td class="${lossClass}">${lossStr}</td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('tableSection').classList.add('visible');

  gomokuCandidates = data.worst_10 || [];
  gomokuCandidateStatus.clear();
  gomokuPrecomputeStarted = false;
  gomokuPendingMoveNum = gomokuCandidates.length ? parseInt(gomokuCandidates[0].move_num) : null;
  gomokuCandidates.forEach(candidate => gomokuCandidateStatus.set(parseInt(candidate.move_num), 'waiting'));
  document.getElementById('gomokuBoardWrap').classList.add('hidden');
  document.getElementById('evalBarContainer').style.display = 'none';
  document.getElementById('gomokuPostgamePanel').classList.add('visible');
  renderGomokuCandidateBar(gomokuCandidates);
  setGomokuRightTab(gomokuRightTab);
  if (gomokuCandidates.length) {
    requestGomokuReplay(parseInt(gomokuCandidates[0].move_num));
  }
}

socket.on('gomoku_counterfactual_ready', (data) => {
  if (data.session_id !== undefined && data.session_id !== gomokuSessionId) return;
  const responseMoveNum = parseInt(data.move_num);
  gomokuReplayCache.set(`${gomokuSessionId}:${responseMoveNum}`, data);
  setGomokuCandidateStatus(responseMoveNum, 'ready');
  if (gomokuPendingMoveNum !== null && responseMoveNum !== gomokuPendingMoveNum) {
    maybePrefetchGomokuCandidates();
    return;
  }
  showGomokuCounterfactual(responseMoveNum, data);
  maybePrefetchGomokuCandidates();
});

socket.on('gomoku_counterfactual_error', (data) => {
  if (data.session_id !== undefined && data.session_id !== gomokuSessionId) return;
  const responseMoveNum = data.move_num !== undefined ? parseInt(data.move_num) : null;
  if (responseMoveNum !== null && !Number.isNaN(responseMoveNum)) {
    setGomokuCandidateStatus(responseMoveNum, 'error');
    if (gomokuPendingMoveNum !== null && responseMoveNum !== gomokuPendingMoveNum) return;
  }
  document.getElementById('gomokuCenterFeedback').classList.add('visible');
  document.getElementById('gomokuFeedbackLoading').classList.remove('active');
  document.getElementById('gomokuFeedbackSourceCenter').textContent = '비교 리플레이 생성 실패';
  document.getElementById('gomokuFeedbackTextCenter').textContent = data.message || '오류가 발생했습니다.';
  document.getElementById('gomokuCompareStage').classList.remove('visible');
});

socket.on('gomoku_sessions_list', (data) => {
  renderGomokuSessionOptions(data.sessions || []);
});

socket.on('gomoku_session_saved', (data) => {
  document.getElementById('gomokuSessionStatus').textContent = data.message || '저장 결과를 확인해주세요.';
  if (data.ok) {
    renderGomokuSessionOptions(data.sessions || []);
  }
});

socket.on('gomoku_session_deleted', (data) => {
  document.getElementById('gomokuSessionStatus').textContent = data.message || '삭제 결과를 확인해주세요.';
  renderGomokuSessionOptions(data.sessions || []);
});

socket.on('gomoku_session_loaded', (data) => {
  if (!data.ok) {
    document.getElementById('gomokuSessionStatus').textContent = data.message || '기록을 불러오지 못했습니다.';
    return;
  }
  gomokuSessionId = data.session_id;
  gomokuReplayCache.clear();
  gomokuCandidateStatus.clear();
  (data.cached_counterfactuals || []).forEach(item => {
    const moveNum = parseInt(item.move_num);
    gomokuReplayCache.set(`${gomokuSessionId}:${moveNum}`, item);
  });
  gameActive = false;
  boardState = data.board?.cells || [];
  renderBoard(boardState, data.board?.last_move ?? -1);
  document.getElementById('gomokuBoardWrap').classList.add('hidden');
  document.getElementById('evalBarContainer').style.display = 'none';
  document.getElementById('statusBar').textContent = '';
  document.getElementById('moveCount').textContent = data.analysis?.total_moves ?? '0';
  document.getElementById('gomokuSessionLabel').textContent = data.meta?.title || 'LOADED';
  document.getElementById('startBtn').disabled = false;
  showAnalysis(data.analysis || {});
  gomokuCandidates.forEach(candidate => {
    const moveNum = parseInt(candidate.move_num);
    gomokuCandidateStatus.set(moveNum, gomokuReplayCache.has(`${gomokuSessionId}:${moveNum}`) ? 'ready' : 'waiting');
  });
  gomokuPendingMoveNum = null;
  gomokuPrecomputeStarted = true;
  renderGomokuCandidateBar(gomokuCandidates);
  document.getElementById('gomokuSessionTitle').value = data.meta?.title || '';
  document.getElementById('gomokuSessionStatus').textContent = `불러온 기록: ${data.meta?.title || '이름 없음'}`;
  document.getElementById('gomokuCompareStage').classList.remove('visible');
  document.getElementById('gomokuHumanImage').src = '';
  document.getElementById('gomokuAgentImage').src = '';
  document.getElementById('gomokuCenterFeedback').classList.remove('visible');
  document.getElementById('gomokuFeedbackLoading').classList.remove('active');
  document.getElementById('gomokuFeedbackSourceCenter').textContent = '';
  document.getElementById('gomokuFeedbackTextCenter').textContent = '착수를 고르면 코칭 피드백이 여기에 표시됩니다.';
  setGomokuFeedbackTab('feedback');
  document.getElementById('gomokuLlmCurrentStatus').textContent = HAS_LLM ? '현재 선택: 대기 중' : '현재 선택: 로컬';
  updateGomokuSummaryFromPayload(null, {});
  clearGomokuTerminalOverlay();
});

// 초기화
initBoard();
document.querySelectorAll('#gomokuRightTabs .right-tab').forEach(btn => {
  btn.addEventListener('click', () => setGomokuRightTab(btn.dataset.gomokuTab));
});
document.querySelectorAll('[data-gomoku-feedback-tab]').forEach(btn => {
  btn.addEventListener('click', () => setGomokuFeedbackTab(btn.dataset.gomokuFeedbackTab));
});

// ── 도전과제 시스템 ───────────────────────────────────────────────────────────
// 티어 아이콘은 common_utils.js 의 ACH_TIER_ICONS 공유 (섹션 라벨만 gomoku 고유)
const ACH_TIER_LABEL = { bronze:'🥉 초급', silver:'🥈 중급', gold:'🥇 고급', platinum:'💎 최고급' };
const gomokuUnlockedIds = new Set();

function unlockGomokuAch(id) {
  if (gomokuUnlockedIds.has(id)) return;
  const ach = ALL_ACHIEVEMENTS.find(a => a.id === id);
  if (!ach) return;
  gomokuUnlockedIds.add(id);
  addGomokuAchToast(ach);
}

// 매 수마다 호출 — 임계값 통과 순간 한 번씩만 발동
function checkGomokuSurviveAchievements(totalMoves) {
  if (totalMoves >= 5)  unlockGomokuAch('survive_5');
  if (totalMoves >= 10) unlockGomokuAch('survive_10');
  if (totalMoves >= 15) unlockGomokuAch('survive_15');
  if (totalMoves >= 20) unlockGomokuAch('survive_20');
  if (totalMoves >= 30) unlockGomokuAch('survive_30');
}

// 게임 종료 시에만 호출
function checkGomokuWinAchievements(humanWon, color) {
  if (humanWon && color === 1) unlockGomokuAch('win_black');
  if (humanWon && color === 2) unlockGomokuAch('win_white');
}

function openAchModal() {
  renderAchievementModalContent({
    list: ALL_ACHIEVEMENTS.map(a => ({ ...a, unlocked: gomokuUnlockedIds.has(a.id) })),
    tierLabels: ACH_TIER_LABEL,
    countEl: document.getElementById('achModalCount'),
    bodyEl:  document.getElementById('achModalBody'),
  });
  document.getElementById('achModal').style.display = 'flex';
}

document.getElementById('achBtn').addEventListener('click', openAchModal);
document.getElementById('achModalClose').addEventListener('click', () => {
  document.getElementById('achModal').style.display = 'none';
});
document.getElementById('achModal').addEventListener('click', e => {
  if (e.target === e.currentTarget) e.currentTarget.style.display = 'none';
});

// 토스트 알림
const TOAST_TOP_START = 80, TOAST_STEP = 130, TOAST_DURATION = 5000;
let gomokuActiveToasts = [];

function addGomokuAchToast(ach) {
  const tier = ach.tier || 'bronze';
  gomokuActiveToasts.forEach(t => {
    const curTop = parseInt(t.style.top) || TOAST_TOP_START;
    t.style.top  = (curTop + TOAST_STEP) + 'px';
  });
  const toast = document.createElement('div');
  toast.className = `ach-toast ach-toast-${tier}`;
  toast.style.top = TOAST_TOP_START + 'px';
  toast.innerHTML = achToastInnerHTML(ach, '도전과제 달성!');
  document.body.appendChild(toast);
  gomokuActiveToasts.unshift(toast);
  requestAnimationFrame(() => { requestAnimationFrame(() => toast.classList.add('show')); });
  setTimeout(() => toast.classList.add('hide'), TOAST_DURATION);
  setTimeout(() => {
    toast.remove();
    gomokuActiveToasts = gomokuActiveToasts.filter(t => t !== toast);
  }, TOAST_DURATION + 600);
}