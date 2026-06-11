const overlay       = document.getElementById('modalOverlay');
const settingsBtn   = document.getElementById('settingsBtn');
const cancelBtn     = document.getElementById('modalCancelBtn');
const saveBtn       = document.getElementById('saveBtn');
const testBtn       = document.getElementById('testBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const showKeyBtn    = document.getElementById('showKeyBtn');
const apiKeyInput   = document.getElementById('apiKeyInput');
const modelInput    = document.getElementById('modelInput');
const statusEl      = document.getElementById('modalStatus');
const keyHint       = document.getElementById('currentKeyHint');
const badgeDot      = document.getElementById('llmBadgeDot');
const badgeText     = document.getElementById('llmBadgeText');
const badgeText2    = document.getElementById('llmBadgeText2');

function setStatus(msg, type) { statusEl.textContent = msg; statusEl.className = 'modal-status visible ' + (type || ''); }
function clearStatus() { statusEl.className = 'modal-status'; }

function updateBadge(hasKey, model) {
  badgeDot.className = 'llm-badge-dot' + (hasKey ? ' on' : '');
  badgeText.textContent = hasKey ? 'LLM ON' : 'LLM OFF';
  badgeText.style.color = hasKey ? 'var(--green)' : 'var(--sub)';
  if (badgeText2) {
    badgeText2.textContent = hasKey ? (model || 'Connected') : '미설정';
    badgeText2.style.color = hasKey ? 'var(--cyan)' : 'var(--sub)';
  }
  disconnectBtn.disabled = !hasKey;
}

async function loadSettings() {
  try {
    const data = await getJSON('/api/settings');
    modelInput.value = data.model || '';
    keyHint.textContent = data.has_key ? `현재: ${data.masked_key}` : '현재: 미설정';
    updateBadge(data.has_key, data.model);
  } catch(e) {}
}

function openModal() { apiKeyInput.value = ''; clearStatus(); loadSettings(); overlay.classList.add('open'); }
function closeModal() { overlay.classList.remove('open'); }

settingsBtn.addEventListener('click', openModal);
cancelBtn.addEventListener('click', closeModal);
overlay.addEventListener('click', e => { if (e.target === overlay) closeModal(); });

disconnectBtn.addEventListener('click', async () => {
  setStatus('연결 해제 중...', 'loading'); disconnectBtn.disabled = true;
  try {
    const data = await postJSON('/api/settings', { clear_key:true, model:modelInput.value.trim() });
    if (data.ok) { setStatus('연결이 해제되었습니다.', 'err'); apiKeyInput.value = ''; await loadSettings(); }
  } catch(e) { setStatus('요청 오류: ' + e.message, 'err'); disconnectBtn.disabled = false; }
});

showKeyBtn.addEventListener('click', () => {
  if (apiKeyInput.type === 'password') { apiKeyInput.type = 'text'; showKeyBtn.textContent = '숨기기'; }
  else { apiKeyInput.type = 'password'; showKeyBtn.textContent = '보기'; }
});

saveBtn.addEventListener('click', async () => {
  const api_key = apiKeyInput.value.trim(), model = modelInput.value.trim();
  setStatus('저장 중...', 'loading');
  try {
    const data = await postJSON('/api/settings', { api_key, model });
    if (data.ok) { setStatus('저장되었습니다.', 'ok'); await loadSettings(); apiKeyInput.value = ''; }
    else setStatus(data.message || '저장에 실패했습니다.', 'err');
  } catch(e) { setStatus('요청 오류: ' + e.message, 'err'); }
});

testBtn.addEventListener('click', async () => {
  const api_key = apiKeyInput.value.trim(), model = modelInput.value.trim();
  setStatus('연결 테스트 중...', 'loading'); testBtn.disabled = true;
  try {
    if (api_key) await postJSON('/api/settings', { api_key, model });
    const data = await postJSON('/api/settings/test');
    setStatus(data.message || (data.ok ? '연결 성공' : '연결 실패'), data.ok ? 'ok' : 'err');
    await loadSettings();
  } catch(e) { setStatus('요청 오류: ' + e.message, 'err'); }
  finally { testBtn.disabled = false; }
});

loadSettings();