/* ═══════════════════════════════════════════════════════════════════
   REI — Client-side Application Logic
   PDF upload, chat with streaming, settings management
   ═══════════════════════════════════════════════════════════════════ */

(() => {
    'use strict';

    // ─── STATE ──────────────────────────────────────────────────────

    const state = {
        pdfs: [],                    // list of uploaded PDFs
        selectedPdfIds: new Set(),   // multi-select: checked PDF IDs
        scope: 'selected',           // 'selected' or 'all'
        chatHistory: [],             // conversation messages [{role, content}]
        isStreaming: false,           // true while receiving a response
        settings: {},                // loaded from API
        ollama: { running: false, models: [] },  // Ollama status
    };

    // ─── DOM REFERENCES ─────────────────────────────────────────────

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const els = {
        pdfList:        $('#pdf-list'),
        emptyState:     $('#empty-state'),
        uploadZone:     $('#upload-zone'),
        fileInput:      $('#file-input'),
        chatMessages:   $('#chat-messages'),
        chatWelcome:    $('#chat-welcome'),
        chatInput:      $('#chat-input'),
        sendBtn:        $('#send-btn'),
        settingsBtn:    $('#settings-btn'),
        settingsOverlay:$('#settings-overlay'),
        settingsClose:  $('#settings-close'),
        settingsSave:   $('#settings-save'),
        settingsStatus: $('#settings-status'),
        statusDot:      $('#status-indicator'),
        statusLabel:    $('#status-label'),
        sidebarInfo:    $('#sidebar-info'),
        uploadOverlay:  $('#upload-overlay'),
        uploadStep:     $('#upload-step'),
        progressFill:   $('#progress-fill'),
        scopeSingle:    $('#scope-single'),
        scopeAll:       $('#scope-all'),
    };

    // ─── INIT ───────────────────────────────────────────────────────

    async function init() {
        await loadSettings();
        await loadPdfs();
        await checkOllamaStatus();
        bindEvents();
        updateUI();

        // Periodically check Ollama status (every 30s)
        setInterval(checkOllamaStatus, 30000);
    }

    async function checkOllamaStatus() {
        try {
            state.ollama = await API.get('/ollama/status');
        } catch (e) {
            state.ollama = { running: false, models: [] };
        }
        renderStatusIndicator();
        renderSidebarInfo();
    }

    // ─── API HELPERS ────────────────────────────────────────────────

    const API = {
        _getHeaders(extraHeaders = {}) {
            const headers = { ...extraHeaders };
            if (state.settings) {
                headers['X-Client-Config'] = JSON.stringify(state.settings);
            }
            return headers;
        },
        async get(path) {
            const res = await fetch(`/api${path}`, {
                headers: this._getHeaders()
            });
            if (!res.ok) throw new Error(await res.text());
            return res.json();
        },
        async post(path, body) {
            const res = await fetch(`/api${path}`, {
                method: 'POST',
                headers: this._getHeaders({ 'Content-Type': 'application/json' }),
                body: JSON.stringify(body),
            });
            if (!res.ok) throw new Error(await res.text());
            return res.json();
        },
        async del(path) {
            const res = await fetch(`/api${path}`, {
                method: 'DELETE',
                headers: this._getHeaders()
            });
            if (!res.ok) throw new Error(await res.text());
            return res.json();
        },
        async upload(file) {
            const form = new FormData();
            form.append('file', file);
            const res = await fetch('/api/upload', {
                method: 'POST',
                headers: this._getHeaders(),
                body: form
            });
            if (!res.ok) throw new Error(await res.text());
            return res.json();
        },
    };

    // ─── DATA LOADING ───────────────────────────────────────────────

    async function loadSettings() {
        try {
            const saved = localStorage.getItem('rei_settings');
            if (saved) {
                state.settings = JSON.parse(saved);
            } else {
                state.settings = await API.get('/settings');
            }
        } catch (e) {
            console.error('Failed to load settings:', e);
            state.settings = {};
        }
    }

    async function loadPdfs() {
        try {
            state.pdfs = await API.get('/pdfs');
        } catch (e) {
            console.error('Failed to load PDFs:', e);
            state.pdfs = [];
        }
    }

    // ─── UI RENDERING ───────────────────────────────────────────────

    function updateUI() {
        renderPdfList();
        renderSidebarInfo();
        renderStatusIndicator();
        renderScopeToggle();
        updateSendBtn();
    }

    function renderPdfList() {
        // Clear existing items (keep empty state element)
        els.pdfList.querySelectorAll('.pdf-item').forEach(el => el.remove());

        if (state.pdfs.length === 0) {
            els.emptyState.style.display = 'block';
            return;
        }

        els.emptyState.style.display = 'none';

        state.pdfs.forEach(pdf => {
            const isSelected = state.selectedPdfIds.has(pdf.pdf_id);
            const item = document.createElement('div');
            item.className = `pdf-item${isSelected ? ' active' : ''}`;
            item.dataset.pdfId = pdf.pdf_id;
            item.innerHTML = `
                <label class="pdf-checkbox-wrapper">
                    <input type="checkbox" class="pdf-checkbox" data-pdf-id="${pdf.pdf_id}" ${isSelected ? 'checked' : ''}>
                    <span class="pdf-check-mark"></span>
                </label>
                <div class="pdf-info">
                    <div class="pdf-name" title="${escHtml(pdf.filename)}">${escHtml(pdf.filename)}</div>
                    <div class="pdf-meta">
                        <span>${pdf.page_count} pg</span>
                        <span>${pdf.chunk_count} chunks</span>
                        <span>${pdf.file_size}</span>
                    </div>
                </div>
                <button class="pdf-delete" title="Delete document" aria-label="Delete ${escHtml(pdf.filename)}">×</button>
            `;

            // Checkbox toggle
            const checkbox = item.querySelector('.pdf-checkbox');
            checkbox.addEventListener('change', (e) => {
                e.stopPropagation();
                togglePdfSelection(pdf.pdf_id, checkbox.checked);
            });

            // Click on item body (not checkbox, not delete) = toggle selection
            item.querySelector('.pdf-info').addEventListener('click', (e) => {
                checkbox.checked = !checkbox.checked;
                togglePdfSelection(pdf.pdf_id, checkbox.checked);
            });

            // Delete button
            item.querySelector('.pdf-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                e.preventDefault();
                deletePdf(pdf.pdf_id, pdf.filename);
            });

            els.pdfList.appendChild(item);
        });
    }

    function renderSidebarInfo() {
        const mode = state.settings.llm_mode || 'local';
        let modelName;
        if (mode === 'local') {
            modelName = state.settings.ollama_model || 'qwen3:4b';
        } else {
            const provider = state.settings.api_provider || 'openai';
            const modelKey = `${provider}_model`;
            modelName = state.settings[modelKey] || provider;
        }

        const selectedCount = state.selectedPdfIds.size;
        const scopeLabel = state.scope === 'all'
            ? 'all docs'
            : `${selectedCount} selected`;

        let ollamaRow = '';
        if (mode === 'local') {
            const ollamaStatus = state.ollama.running
                ? `<span class="info-value" style="color:var(--accent)">connected (${state.ollama.models.length} models)</span>`
                : `<span class="info-value" style="color:var(--error, #e55)">offline</span>`;
            ollamaRow = `
                <div class="info-row">
                    <span>ollama</span>
                    ${ollamaStatus}
                </div>
            `;
        }

        els.sidebarInfo.innerHTML = `
            <div class="info-row">
                <span>mode</span>
                <span class="info-value">${mode}</span>
            </div>
            <div class="info-row">
                <span>model</span>
                <span class="info-value">${escHtml(modelName)}</span>
            </div>
            <div class="info-row">
                <span>scope</span>
                <span class="info-value">${scopeLabel}</span>
            </div>
            ${ollamaRow}
        `;
    }

    function renderStatusIndicator() {
        const mode = state.settings.llm_mode || 'local';
        if (mode === 'local') {
            if (state.ollama.running) {
                els.statusDot.className = 'status-dot status-local';
                els.statusLabel.textContent = 'ollama ●';
            } else {
                els.statusDot.className = 'status-dot status-offline';
                els.statusLabel.textContent = 'ollama offline';
            }
        } else {
            els.statusDot.className = `status-dot status-${mode}`;
            els.statusLabel.textContent = mode;
        }
    }

    function renderScopeToggle() {
        const count = state.selectedPdfIds.size;
        els.scopeSingle.textContent = count > 0 ? `selected (${count})` : 'selected';
        els.scopeSingle.classList.toggle('active', state.scope === 'selected');
        els.scopeAll.classList.toggle('active', state.scope === 'all');
    }

    function updateSendBtn() {
        const hasText = els.chatInput.value.trim().length > 0;
        const hasTarget = state.scope === 'all' || state.selectedPdfIds.size > 0;
        els.sendBtn.disabled = !hasText || !hasTarget || state.isStreaming;
    }

    // ─── PDF MANAGEMENT ─────────────────────────────────────────────

    function togglePdfSelection(pdfId, selected) {
        if (selected) {
            state.selectedPdfIds.add(pdfId);
        } else {
            state.selectedPdfIds.delete(pdfId);
        }
        // Auto-switch to 'selected' scope when user toggles checkboxes
        if (state.selectedPdfIds.size > 0) {
            state.scope = 'selected';
        }
        updateUI();
    }

    async function deletePdf(pdfId, filename) {
        if (!confirm(`Delete "${filename || pdfId}"?`)) return;

        try {
            await API.del(`/pdfs/${pdfId}`);
            state.pdfs = state.pdfs.filter(p => p.pdf_id !== pdfId);
            state.selectedPdfIds.delete(pdfId);
            if (state.selectedPdfIds.size === 0 && state.pdfs.length === 0) {
                state.chatHistory = [];
                clearChat();
            }
            updateUI();
        } catch (e) {
            console.error('Delete failed:', e);
            addSystemMessage(`delete failed: ${extractError(e)}`);
        }
    }

    async function uploadPdf(file) {
        showUploadProgress('reading pdf...');

        try {
            setProgress(10);
            showUploadProgress('extracting text & chunking...');
            setProgress(30);

            const result = await API.upload(file);

            setProgress(80);
            showUploadProgress('storing embeddings...');

            if (result.success && result.pdf_info) {
                state.pdfs.push(result.pdf_info);
                // Auto-select the newly uploaded PDF
                state.selectedPdfIds.add(result.pdf_info.pdf_id);
                state.scope = 'selected';
            }

            setProgress(100);
            setTimeout(hideUploadProgress, 400);
            updateUI();

        } catch (e) {
            hideUploadProgress();
            const detail = extractError(e);
            addSystemMessage(`upload failed: ${detail}`);
        }
    }

    // ─── CHAT ───────────────────────────────────────────────────────

    function clearChat() {
        els.chatMessages.innerHTML = '';
        els.chatMessages.appendChild(els.chatWelcome);
        els.chatWelcome.style.display = 'flex';
    }

    function addUserMessage(text) {
        els.chatWelcome.style.display = 'none';

        const msg = document.createElement('div');
        msg.className = 'chat-msg';
        msg.innerHTML = `
            <div class="msg-header">
                <span class="msg-role role-user">user</span>
                <span class="msg-time">${timeNow()}</span>
            </div>
            <div class="msg-content user-msg">${escHtml(text)}</div>
        `;
        els.chatMessages.appendChild(msg);
        scrollToBottom();
    }

    function addAssistantMessage() {
        const msg = document.createElement('div');
        msg.className = 'chat-msg';
        msg.innerHTML = `
            <div class="msg-header">
                <span class="msg-role role-rei">rei</span>
                <span class="msg-time">${timeNow()}</span>
            </div>
            <div class="msg-content assistant-msg"></div>
        `;
        els.chatMessages.appendChild(msg);
        scrollToBottom();
        return msg.querySelector('.assistant-msg');
    }

    function addTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'chat-msg typing-msg';
        indicator.innerHTML = `
            <div class="msg-header">
                <span class="msg-role role-rei">rei</span>
            </div>
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        els.chatMessages.appendChild(indicator);
        scrollToBottom();
        return indicator;
    }

    function addSystemMessage(text) {
        els.chatWelcome.style.display = 'none';

        const msg = document.createElement('div');
        msg.className = 'chat-msg';
        msg.innerHTML = `
            <div class="msg-header">
                <span class="msg-role role-rei">sys</span>
            </div>
            <div class="msg-error">${escHtml(text)}</div>
        `;
        els.chatMessages.appendChild(msg);
        scrollToBottom();
    }

    async function sendMessage() {
        const text = els.chatInput.value.trim();
        if (!text || state.isStreaming) return;

        // Build pdf_ids based on scope
        let pdfIds;
        if (state.scope === 'all') {
            pdfIds = [];  // empty = all docs on backend
        } else {
            pdfIds = Array.from(state.selectedPdfIds);
            if (pdfIds.length === 0) {
                addSystemMessage('select at least one document, or switch scope to "all docs".');
                return;
            }
        }

        // Add user message to UI and history
        addUserMessage(text);
        state.chatHistory.push({ role: 'user', content: text });
        els.chatInput.value = '';
        autoResizeInput();
        updateSendBtn();

        // Show typing indicator
        state.isStreaming = true;
        updateSendBtn();
        const typing = addTypingIndicator();

        try {
            // SSE streaming request with custom configuration header
            const headers = { 'Content-Type': 'application/json' };
            if (state.settings) {
                headers['X-Client-Config'] = JSON.stringify(state.settings);
            }

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    question: text,
                    pdf_ids: pdfIds,
                    history: state.chatHistory.slice(-10),
                }),
            });

            // Remove typing indicator
            typing.remove();

            if (!response.ok) {
                throw new Error(await response.text());
            }

            // Create assistant message container
            const contentEl = addAssistantMessage();
            let fullResponse = '';

            // Read SSE stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Process SSE lines
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;

                    const data = line.slice(6);
                    if (data === '[DONE]') continue;

                    // Unescape newlines
                    const token = data.replace(/\\n/g, '\n');
                    fullResponse += token;

                    // Render markdown
                    contentEl.innerHTML = renderMarkdown(fullResponse);
                    scrollToBottom();
                }
            }

            // Add to history
            state.chatHistory.push({ role: 'assistant', content: fullResponse });

        } catch (e) {
            typing.remove();
            const detail = extractError(e);
            addSystemMessage(`error: ${detail}`);
        } finally {
            state.isStreaming = false;
            updateSendBtn();
        }
    }

    // ─── SETTINGS ───────────────────────────────────────────────────

    function openSettings() {
        loadSettingsIntoForm();
        els.settingsOverlay.classList.remove('hidden');
    }

    function closeSettings() {
        els.settingsOverlay.classList.add('hidden');
    }

    function loadSettingsIntoForm() {
        const s = state.settings;

        // LLM mode
        $$('.llm-mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === (s.llm_mode || 'local'));
        });

        // Show/hide sections
        const isLocal = (s.llm_mode || 'local') === 'local';
        $('#local-settings').classList.toggle('hidden', !isLocal);
        $('#api-settings').classList.toggle('hidden', isLocal);

        // Ollama model
        $('#ollama-model').value = s.ollama_model || 'qwen3:4b';

        // API provider
        $$('.provider-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.provider === (s.api_provider || 'openai'));
        });
        showProviderConfig(s.api_provider || 'openai');

        // API keys and models (show masked or actual)
        $('#openai-key').value = s.openai_api_key || '';
        $('#openai-model').value = s.openai_model || 'gpt-4o-mini';
        $('#gemini-key').value = s.gemini_api_key || '';
        $('#gemini-model').value = s.gemini_model || 'gemini-2.0-flash';
        $('#anthropic-key').value = s.anthropic_api_key || '';
        $('#anthropic-model').value = s.anthropic_model || 'claude-sonnet-4-20250514';
        $('#huggingface-key').value = s.huggingface_api_key || '';
        $('#huggingface-model').value = s.huggingface_model || 'meta-llama/Meta-Llama-3-8B-Instruct';

        // Embedding mode
        $$('.embed-mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === (s.embedding_mode || 'local'));
        });
    }

    function showProviderConfig(provider) {
        $$('.provider-config').forEach(el => el.classList.add('hidden'));
        const target = $(`#${provider}-config`);
        if (target) target.classList.remove('hidden');
    }

    async function saveSettings() {
        const settings = {
            llm_mode: $('.llm-mode-btn.active')?.dataset.mode || 'local',
            api_provider: $('.provider-btn.active')?.dataset.provider || 'openai',
            ollama_model: $('#ollama-model').value,
            openai_api_key: $('#openai-key').value,
            openai_model: $('#openai-model').value,
            gemini_api_key: $('#gemini-key').value,
            gemini_model: $('#gemini-model').value,
            anthropic_api_key: $('#anthropic-key').value,
            anthropic_model: $('#anthropic-model').value,
            huggingface_api_key: $('#huggingface-key').value,
            huggingface_model: $('#huggingface-model').value,
            embedding_mode: $('.embed-mode-btn.active')?.dataset.mode || 'local',
        };

        try {
            localStorage.setItem('rei_settings', JSON.stringify(settings));
            state.settings = settings;
            els.settingsStatus.textContent = 'saved ✓';
            els.settingsStatus.style.color = 'var(--accent)';
            setTimeout(() => { els.settingsStatus.textContent = ''; }, 2000);
            updateUI();
        } catch (e) {
            els.settingsStatus.textContent = 'save failed';
            els.settingsStatus.style.color = 'var(--error)';
            console.error('Settings save failed:', e);
        }
    }

    // ─── EVENTS ─────────────────────────────────────────────────────

    function bindEvents() {
        // Upload
        els.uploadZone.addEventListener('click', () => els.fileInput.click());
        els.uploadZone.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') els.fileInput.click();
        });
        els.fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) uploadPdf(e.target.files[0]);
            e.target.value = '';
        });

        // Drag and drop
        els.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            els.uploadZone.classList.add('drag-over');
        });
        els.uploadZone.addEventListener('dragleave', () => {
            els.uploadZone.classList.remove('drag-over');
        });
        els.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            els.uploadZone.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file && file.name.toLowerCase().endsWith('.pdf')) {
                uploadPdf(file);
            }
        });

        // Chat input
        els.chatInput.addEventListener('input', () => {
            autoResizeInput();
            updateSendBtn();
        });
        els.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        els.sendBtn.addEventListener('click', sendMessage);

        // Scope toggle
        els.scopeSingle.addEventListener('click', () => {
            state.scope = 'selected';
            updateUI();
        });
        els.scopeAll.addEventListener('click', () => {
            state.scope = 'all';
            updateUI();
        });

        // Settings
        els.settingsBtn.addEventListener('click', openSettings);
        els.settingsClose.addEventListener('click', closeSettings);
        els.settingsOverlay.addEventListener('click', (e) => {
            if (e.target === els.settingsOverlay) closeSettings();
        });
        els.settingsSave.addEventListener('click', saveSettings);

        // LLM mode toggle in settings
        $$('.llm-mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                $$('.llm-mode-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const isLocal = btn.dataset.mode === 'local';
                $('#local-settings').classList.toggle('hidden', !isLocal);
                $('#api-settings').classList.toggle('hidden', isLocal);
            });
        });

        // Provider toggle in settings
        $$('.provider-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                $$('.provider-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                showProviderConfig(btn.dataset.provider);
            });
        });

        // Embedding mode toggle
        $$('.embed-mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                $$('.embed-mode-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });

        // Keyboard shortcut: Escape closes settings
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !els.settingsOverlay.classList.contains('hidden')) {
                closeSettings();
            }
        });
    }

    // ─── HELPERS ────────────────────────────────────────────────────

    function scrollToBottom() {
        requestAnimationFrame(() => {
            els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
        });
    }

    function autoResizeInput() {
        els.chatInput.style.height = 'auto';
        els.chatInput.style.height = Math.min(els.chatInput.scrollHeight, 120) + 'px';
    }

    function timeNow() {
        return new Date().toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: false,
        });
    }

    function escHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function extractError(e) {
        if (!e) return 'unknown error';
        if (typeof e === 'string') return e;

        const message = e.message || e.toString() || '';
        try {
            const parsed = JSON.parse(message);
            if (parsed && typeof parsed === 'object') {
                if (parsed.detail) {
                    if (Array.isArray(parsed.detail)) {
                        return parsed.detail.map(err => {
                            const loc = err.loc ? err.loc.join('.') : '';
                            return loc ? `${loc}: ${err.msg}` : err.msg;
                        }).join(', ');
                    }
                    return parsed.detail;
                }
                return JSON.stringify(parsed);
            }
        } catch {
            // Not JSON (e.g. HTML 502/504 errors)
        }
        return message || 'unknown error';
    }

    function showUploadProgress(text) {
        els.uploadOverlay.classList.remove('hidden');
        els.uploadStep.textContent = text;
    }

    function hideUploadProgress() {
        els.uploadOverlay.classList.add('hidden');
        els.progressFill.style.width = '0%';
    }

    function setProgress(percent) {
        els.progressFill.style.width = `${percent}%`;
    }

    // ─── BASIC MARKDOWN RENDERER ────────────────────────────────────

    function renderMarkdown(text) {
        if (!text) return '';

        let html = escHtml(text);

        // Code blocks (``` ... ```)
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            return `<pre><code>${code.trim()}</code></pre>`;
        });

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Unordered lists
        html = html.replace(/^[\s]*[-*]\s+(.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

        // Ordered lists
        html = html.replace(/^[\s]*\d+\.\s+(.+)$/gm, '<li>$1</li>');

        // Blockquotes
        html = html.replace(/^&gt;\s+(.+)$/gm, '<blockquote>$1</blockquote>');

        // Headings
        html = html.replace(/^####\s+(.+)$/gm, '<strong>$1</strong>');
        html = html.replace(/^###\s+(.+)$/gm, '<strong>$1</strong>');
        html = html.replace(/^##\s+(.+)$/gm, '<strong>$1</strong>');
        html = html.replace(/^#\s+(.+)$/gm, '<strong>$1</strong>');

        // Paragraphs — convert double newlines
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');
        html = `<p>${html}</p>`;

        // Clean up empty paragraphs
        html = html.replace(/<p>\s*<\/p>/g, '');

        return html;
    }

    // ─── START ──────────────────────────────────────────────────────

    document.addEventListener('DOMContentLoaded', init);

})();
