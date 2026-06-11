// common_utils.js — 게임 페이지 공통 유틸리티 모음.
//
// 로드 시 부수효과(소켓 연결, DOM 접근)가 없도록 순수 함수/상수만 정의한다.
// 모든 페이지에서 다른 스크립트보다 먼저 로드된다.

// ── 소켓 ─────────────────────────────────────────────────────────────────────
function createGameSocket() {
  return io({ transports: ['websocket'] });
}

// ── 인코딩 헬퍼 ──────────────────────────────────────────────────────────────
function b64src(b64) { return b64 ? 'data:image/jpeg;base64,' + b64 : ''; }

// ── API 요청 래퍼 ────────────────────────────────────────────────────────────
async function getJSON(url) {
  const res = await fetch(url);
  return res.json();
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  return res.json();
}

// ── 후보(counterfactual) 상태 라벨 ───────────────────────────────────────────
const STATUS_LABELS = { waiting: '대기 중', generating: '생성 중', ready: '준비됨', error: '실패' };

// ── 세션 select 옵션 렌더링 ──────────────────────────────────────────────────
// 기존 선택을 유지하면서 옵션을 다시 채우고, 세션 배열을 그대로 반환한다.
function populateSessionSelect(select, sessions) {
  const list = Array.isArray(sessions) ? sessions : [];
  const previous = select.value;
  select.innerHTML = '<option value="">저장된 기록 선택</option>';
  list.forEach(item => {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = `${item.title} · ${item.saved_at || ''}`;
    select.appendChild(option);
  });
  if (previous && list.some(item => item.id === previous)) select.value = previous;
  return list;
}

// ── LLM 피드백 라우트 표시 텍스트 ────────────────────────────────────────────
// "외부 LLM (model)" / "로컬" 등 — 접두어('현재:' 등)는 호출부에서 붙인다.
function feedbackRouteText(data) {
  return `${data.feedback_route || (data.feedback_source === 'llm' ? '외부 LLM' : '로컬')}` +
    (data.feedback_model ? ` (${data.feedback_model})` : '');
}

// ── 도전과제 공통 ────────────────────────────────────────────────────────────
const ACH_TIER_ICONS = { bronze: '🥉', silver: '🥈', gold: '🥇', platinum: '💎' };

// 도전과제 모달 내용 렌더링.
//   list       : [{id,title,desc,tier,unlocked}] (unlocked 필드 필수)
//   tierLabels : 티어별 섹션 라벨 (게임마다 표기가 다름)
//   countEl / bodyEl : 카운트·본문 DOM 요소
function renderAchievementModalContent({ list, tierLabels, countEl, bodyEl }) {
  const tiers = ['bronze', 'silver', 'gold', 'platinum'];
  const byTier = { bronze: [], silver: [], gold: [], platinum: [] };
  list.forEach(a => { if (byTier[a.tier]) byTier[a.tier].push(a); });

  const unlocked = list.filter(a => a.unlocked).length;
  countEl.textContent = `${unlocked} / ${list.length} 달성`;

  bodyEl.innerHTML = tiers.map(tier => {
    const items = byTier[tier];
    if (!items.length) return '';
    return `<div><div class="ach-modal-section-label">${tierLabels[tier]}</div><div class="ach-modal-grid">` +
      items.map(a => {
        const cls  = a.unlocked ? `unlocked-${a.tier}` : 'locked';
        const icon = a.unlocked ? ACH_TIER_ICONS[a.tier] : '🔒';
        return `<div class="ach-modal-item ${cls}">
          <span class="ach-mi-icon">${icon}</span>
          <div class="ach-mi-body">
            <div class="ach-mi-title">${a.title}</div>
            <div class="ach-mi-desc">${a.desc}</div>
          </div>
        </div>`;
      }).join('') +
      `</div></div>`;
  }).join('');
}

// 도전과제 토스트 내부 마크업 (배치/수명 관리는 페이지별 코드 담당)
function achToastInnerHTML(ach, labelText) {
  return `
    <div class="ach-toast-icon">${ACH_TIER_ICONS[ach.tier] || '🏆'}</div>
    <div class="ach-toast-body">
      <div class="ach-toast-label">${labelText}</div>
      <div class="ach-toast-title">${ach.title}</div>
      <div class="ach-toast-desc">${ach.desc}</div>
    </div>
    <div class="ach-toast-bar"></div>`;
}
