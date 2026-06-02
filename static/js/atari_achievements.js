// atari_achievements.js — 도전과제 시스템 (토스트 알림, 모달, 소켓 이벤트)

const tierLabels = { bronze: '🥉 브론즈', silver: '🥈 실버', gold: '🥇 골드', platinum: '💎 플래티넘' };
const tierIcons = { bronze: '🥉', silver: '🥈', gold: '🥇', platinum: '💎' };
const unlockedAchievementIds = new Set();

// ── 소켓 이벤트: 도전과제 달성 ───────────────────────────────────────────────
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
