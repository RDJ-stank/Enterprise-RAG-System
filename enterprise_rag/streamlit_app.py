import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page setup ──────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-header { font-size: 0.9rem; color: #888; margin-bottom: 1.5rem; }
    .chat-message { padding: 1rem 1.2rem; border-radius: 12px; margin-bottom: 0.8rem; max-width: 85%; }
    .chat-message.user { background: #e8f0fe; margin-left: auto; text-align: right; }
    .chat-message.assistant { background: #f5f5f5; margin-right: auto; }
    .chat-message .role { font-size: 0.8rem; font-weight: 600; color: #555; margin-bottom: 0.3rem; }
    .chat-message .content { font-size: 1rem; line-height: 1.6; color: #222; }
    .status-ok { color: #4caf50; font-weight: 600; }
    .status-err { color: #f44336; font-weight: 600; }
    .doc-card {
        border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem; font-size: 0.85rem;
    }
    .doc-card .name { font-weight: 600; }
    .doc-card .meta { color: #999; font-size: 0.78rem; }
    .doc-card .empty-warn { color: #e67e22; font-weight: 600; }
    hr { margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ──────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],
        "documents": [],
        "backend_ok": None,
        "upload_counter": 0,
        "last_upload_result": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ── API helpers ─────────────────────────────────────────────

def check_backend() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        st.session_state.backend_ok = r.status_code == 200
        return st.session_state.backend_ok
    except Exception:
        st.session_state.backend_ok = False
        return False


def refresh_documents():
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=10)
        if r.status_code == 200:
            st.session_state.documents = r.json()["documents"]
    except Exception:
        pass


def send_chat(question: str, top_k: int = 4) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/chat",
            json={"question": question, "top_k": top_k},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def upload_file(file) -> dict | None:
    """上传文件并返回完整的 API 响应 dict。失败返回 None。"""
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (file.name, file.getvalue(), file.type or "application/octet-stream")},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def delete_document(doc_id: str) -> bool:
    try:
        r = requests.delete(f"{API_BASE}/documents/{doc_id}", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def delete_all_documents() -> dict | None:
    try:
        r = requests.delete(f"{API_BASE}/documents", timeout=30)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def has_any_indexed_doc() -> bool:
    """检查是否存在至少一份有效入库的文档（chunk_count > 0）。"""
    return any(d.get("chunk_count", 0) > 0 for d in st.session_state.documents)


# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="main-header">🤖 Enterprise RAG</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">企业级智能知识库问答助手</div>',
        unsafe_allow_html=True,
    )

    # ── Backend status ───────────────────────────────────
    ok = check_backend()
    if ok:
        st.markdown(
            f'<span class="status-ok">●</span> 后端服务运行中 &nbsp;'
            f'<span style="font-size:0.8rem;color:#999;">({API_BASE})</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="status-err">●</span> 后端服务未连接 &nbsp;'
            f'<span style="font-size:0.8rem;color:#999;">({API_BASE})</span>',
            unsafe_allow_html=True,
        )
        st.warning("请先启动后端服务: `uvicorn main:app --port 8000`")

    st.divider()

    # ── Document upload ──────────────────────────────────
    st.subheader("📄 上传文档")

    # 动态 key：每次成功上传后 key 变化，强制 Streamlit 创建全新 widget
    uploaded = st.file_uploader(
        "拖拽文件到此处（PDF / TXT / DOCX / CSV / XLSX）",
        type=["pdf", "txt", "docx", "csv", "xlsx", "xls"],
        accept_multiple_files=False,
        key=f"file_uploader_{st.session_state.upload_counter}",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        if not ok:
            st.error("后端服务未连接，无法上传")
        else:
            with st.spinner(f"正在解析「{uploaded.name}」..."):
                result = upload_file(uploaded)

            if result is None:
                st.error(f"「{uploaded.name}」上传失败，请查看后端日志")
            elif result.get("chunk_count", 0) == 0:
                st.warning(
                    f"「{uploaded.name}」上传完成，但**未提取到任何有效文字**。\n\n"
                    "可能原因：\n"
                    "- PDF 为扫描件/图片（无文字层）\n"
                    "- 文件内容为空\n\n"
                    "请上传包含可提取文字的文档。"
                )
                refresh_documents()
                st.session_state.upload_counter += 1
                st.rerun()
            else:
                st.success(
                    f"「{uploaded.name}」上传成功！"
                    f"（{result['chunk_count']} 个文本块已入库）"
                )
                refresh_documents()
                st.session_state.upload_counter += 1
                st.rerun()

    st.divider()

    # ── Document list ────────────────────────────────────
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.subheader("📚 已入库文档")
    with col2:
        if st.button("🔄", help="刷新文档列表", use_container_width=True):
            refresh_documents()
    with col3:
        if st.button("🗑️🗑️", help="一键删除所有文档", use_container_width=True):
            if docs and st.session_state.backend_ok:
                result = delete_all_documents()
                if result:
                    st.success(f"已删除 {result['documents_removed']} 个文档")
                    refresh_documents()
                    st.rerun()
                else:
                    st.error("清空失败，请查看后端日志")

    if ok:
        refresh_documents()

    docs = st.session_state.documents
    if not docs:
        st.caption("暂无文档，请上传 PDF 或 TXT")
    else:
        for doc in docs:
            with st.container():
                c1, c2 = st.columns([20, 3])
                with c1:
                    chunk_info = (
                        f'{doc["chunk_count"]} 块'
                        if doc["chunk_count"] > 0
                        else '<span class="empty-warn">⚠ 0 块 — 无有效文字</span>'
                    )
                    st.markdown(
                        f'<div class="doc-card">'
                        f'<div class="name">{doc["filename"]}</div>'
                        f'<div class="meta">'
                        f'{chunk_info} · '
                        f'{doc["file_size"] // 1024} KB</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button(
                        "🗑️",
                        key=f"del_{doc['doc_id']}",
                        help=f"删除 {doc['filename']}",
                    ):
                        delete_document(doc["doc_id"])
                        refresh_documents()
                        st.rerun()

    st.divider()

    # ── Settings ─────────────────────────────────────────
    st.subheader("⚙️ 设置")
    top_k = st.slider("检索片段数 (top_k)", min_value=1, max_value=10, value=4)
    if st.button("🧹 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── MAIN CHAT AREA ───────────────────────────────────────────
st.markdown('<div class="main-header">💬 知识库问答</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">基于已入库文档的智能问答 — 回答严格依据上传的文档内容</div>',
    unsafe_allow_html=True,
)

# ── Welcome banner ──────────────────────────────────────
if not st.session_state.messages:
    st.info(
        "👋 **欢迎使用企业知识库助手！**\n\n"
        "1. 在左侧上传 PDF 或 TXT 文档\n"
        "2. 等待文档解析入库（状态栏显示 **● 后端服务运行中**）\n"
        "3. 在下方向知识库提问\n\n"
        "回答仅基于已入库文档生成，不会编造信息。"
    )

# ── Message history ─────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    css_class = "user" if role == "user" else "assistant"
    label = "你" if role == "user" else "🤖 助手"
    st.markdown(
        f'<div class="chat-message {css_class}">'
        f'<div class="role">{label}</div>'
        f'<div class="content">{msg["content"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if role == "assistant" and msg.get("sources"):
        with st.expander("📎 参考来源"):
            for i, src in enumerate(msg["sources"], 1):
                st.caption(
                    f"**[{i}] {src['filename']}** "
                    f"(相关度: {src['score']:.2%})"
                )
                st.text(src["chunk_text"][:300])
                st.divider()

# ── Chat input ───────────────────────────────────────────────
if prompt := st.chat_input("输入你的问题，按 Enter 发送..."):
    if not ok:
        st.error("后端服务未连接，请先启动后端再提问")
    elif not docs:
        st.warning("请先上传至少一份文档到知识库")
    elif not has_any_indexed_doc():
        st.warning(
            "知识库中所有文档的块数均为 0，无法检索。\n\n"
            "请确认上传的 PDF 包含可提取的文字层（非扫描件/图片），"
            "或先删除无效文档后重新上传。"
        )
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("正在检索文档并生成回答..."):
            result = send_chat(prompt, top_k=top_k)

        if result:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "抱歉，问答服务暂不可用，请检查后端日志。",
                "sources": [],
            })
        st.rerun()
