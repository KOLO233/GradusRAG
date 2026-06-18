"""GradusRAG Streamlit 前端 — SuperMew 风格。

居中卡片式布局，带背景纹理，类似聊天应用。
"""

import streamlit as st
import requests
import json

API_BASE = "http://localhost:8001"

# ===========================================================================
# 语言包
# ===========================================================================
L = {
    "zh": {
        "app_name": "GradusRAG",
        "subtitle": "多级查询理解与混合增强 RAG",
        "new_chat": "新建会话",
        "history": "历史记录",
        "docs": "文档管理",
        "graph": "知识图谱",
        "traces": "系统追踪",
        "clear": "清空当前对话",
        "connected": "已连接",
        "disconnected": "未连接",
        "input_ph": "输入您的问题...",
        "processing": "正在处理",
        "done": "完成",
        "failed": "失败",
        "citations": "引用 ({n})",
        "upload": "上传",
        "upload_hint": "PDF / Word / Excel / TXT / HTML / Markdown",
        "upload_btn": "上传入库",
        "doc_list": "文档列表",
        "refresh": "刷新",
        "no_docs": "暂无文档",
        "delete": "删除",
        "deleted": "已删除",
        "search": "搜索实体",
        "hops": "跳数",
        "entities": "实体",
        "relations": "关系",
        "types": "类型",
        "type_dist": "类型分布",
        "no_trace": "暂无记录",
        "legend": "图例",
        "graph_empty": "图谱为空",
        "install_pyvis": "需要安装 pyvis",
        "desc": "描述",
        "no_desc": "无描述",
        "found": "找到 {n} 个",
        "not_found": "未找到",
        "no_history": "暂无历史记录",
        "mode": "查询模式",
        "auto": "自动分类",
        "trace": "显示追踪",
        "topk": "检索数量",
    },
    "en": {
        "app_name": "GradusRAG",
        "subtitle": "Multi-level Query & Hybrid RAG",
        "new_chat": "New Chat",
        "history": "History",
        "docs": "Documents",
        "graph": "Knowledge Graph",
        "traces": "Traces",
        "clear": "Clear Chat",
        "connected": "Connected",
        "disconnected": "Disconnected",
        "input_ph": "Ask a question...",
        "processing": "Processing",
        "done": "Done",
        "failed": "Failed",
        "citations": "Citations ({n})",
        "upload": "Upload",
        "upload_hint": "PDF / Word / Excel / TXT / HTML / Markdown",
        "upload_btn": "Upload",
        "doc_list": "Documents",
        "refresh": "Refresh",
        "no_docs": "No documents",
        "delete": "Delete",
        "deleted": "Deleted",
        "search": "Search",
        "hops": "Hops",
        "entities": "Entities",
        "relations": "Relations",
        "types": "Types",
        "type_dist": "Distribution",
        "no_trace": "No records",
        "legend": "Legend",
        "graph_empty": "Graph empty",
        "install_pyvis": "Install pyvis",
        "desc": "Description",
        "no_desc": "N/A",
        "found": "Found {n}",
        "not_found": "Not found",
        "no_history": "No history",
        "mode": "Query Mode",
        "auto": "Auto Classify (Recommended)",
        "trace": "Show Trace",
        "topk": "Top-K (Retrieval Count)",
    },
}

# ===========================================================================
# 页面配置
# ===========================================================================
st.set_page_config(page_title="GradusRAG", page_icon="R", layout="wide", initial_sidebar_state="expanded")

if "lang" not in st.session_state:
    st.session_state.lang = "zh"
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "chat"

T = L[st.session_state.lang]

# ===========================================================================
# CSS — SuperMew 居中卡片 + 背景纹理
# ===========================================================================
st.markdown("""
<style>
    /* ===== 最小化 CSS，配合 config.toml 主题 ===== */

    /* 侧边栏常驻 */
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="stSidebar"] { min-width: 220px; }

    /* 主内容区圆角卡片 */
    .main .block-container {
        border-radius: 16px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
        padding: 24px 28px !important;
        margin: 12px auto !important;
        max-width: 1100px;
    }

    /* 级别标签 */
    .level-badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 10px; font-weight: 600; font-size: 12px;
    }
    .level-L1 { background: #bbf7d0; color: #14532d; }
    .level-L2 { background: #bfdbfe; color: #1e3a5f; }
    .level-L3 { background: #fef08a; color: #713f12; }
    .level-L4 { background: #fda4af; color: #881337; }

    /* 聊天气泡 */
    .stChatMessage { border-radius: 16px !important; }

    /* 引用卡片 */
    .cite-item {
        border-radius: 6px; padding: 6px 10px; margin: 3px 0;
        font-size: 12px; opacity: 0.85;
    }

    /* 统计数字 */
    .stat-num { font-size: 24px; font-weight: 700; }
    .stat-lbl { font-size: 11px; opacity: 0.6; }

    /* 侧边栏 Logo */
    .sidebar-logo h2 { font-size: 20px; font-weight: 700; margin: 0; }
    .sidebar-logo p { font-size: 12px; opacity: 0.5; margin: 4px 0 0 0; }

    /* 状态栏 */
    .status-bar { font-size: 12px; opacity: 0.5; padding: 8px 0; }

    /* 隐藏 footer，强制显示右上角菜单 */
    footer { display: none !important; }
    #MainMenu { display: block !important; visibility: visible !important; }
    [data-testid="stMainMenu"] { display: block !important; }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# 颜色常量
# ===========================================================================
TC = {
    "概念": "#10b981", "技术": "#3b82f6", "方法": "#f59e0b",
    "人物": "#8b5cf6", "组织": "#ef4444", "疾病": "#ec4899",
    "药物": "#06b6d4", "症状": "#f97316", "工具": "#78716c",
    "理论": "#6366f1",
}

# ===========================================================================
# 侧边栏 — SuperMew 风格
# ===========================================================================
with st.sidebar:
    # Logo 区域（用 Streamlit 原生组件，自动跟随主题）
    st.markdown(f"## {T['app_name']}")
    st.caption(T['subtitle'])

    # 状态
    try:
        requests.get(f"{API_BASE}/api/health", timeout=2).json()
        ok = True
    except Exception:
        ok = False

    try:
        gs = requests.get(f"{API_BASE}/api/graph/stats", timeout=2).json()
    except Exception:
        gs = {"entity_count": 0, "relation_count": 0}

    if ok:
        st.success(T['connected'])
    else:
        st.error(T['disconnected'])

    st.caption(f"{gs.get('entity_count',0)} {T['entities']} / {gs.get('graph_relations',0) if False else gs.get('relation_count',0)} {T['relations']}")

    st.divider()

    # 语言切换
    lang = st.selectbox("Language", ["CN", "EN"], index=0 if st.session_state.lang == "zh" else 1, label_visibility="collapsed")
    new_lang = "zh" if lang == "CN" else "en"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.divider()

    # 导航按钮
    nav_map = [
        ("chat", T["new_chat"]),
        ("docs", T["docs"]),
        ("graph", T["graph"]),
        ("traces", T["traces"]),
    ]
    for key, label in nav_map:
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                      type="primary" if st.session_state.active_nav == key else "secondary"):
            st.session_state.active_nav = key
            st.rerun()

    st.divider()

    # 清空对话
    if st.session_state.get("msgs") and st.session_state.active_nav == "chat":
        if st.button(T["clear"], use_container_width=True):
            st.session_state.msgs = []
            st.rerun()

# ===========================================================================
# 主内容区
# ===========================================================================
nav = st.session_state.active_nav

# ---- 聊天页面 ----
if nav == "chat":
    # 页面标题
    st.subheader(T["new_chat"])

    # 设置栏
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        mode = st.selectbox(T["mode"], [f"{T['auto']}（推荐）", "L1 显性事实", "L2 隐性事实", "L3 可解释原理", "L4 隐藏原理"], label_visibility="visible")
    with c2:
        topk = st.slider(f"{T['topk']}（检索数量）", 3, 20, 10, label_visibility="visible")
    with c3:
        trace_on = st.checkbox(f"{T['trace']}（显示中间步骤）", value=False)

    st.divider()

    # 会话
    if "msgs" not in st.session_state:
        st.session_state.msgs = []

    # 历史消息
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("level"):
                    st.markdown(f'<span class="level-badge level-{m["level"]}">{m["level"]}</span>', unsafe_allow_html=True)
                if m.get("cites"):
                    with st.expander(T["citations"].format(n=len(m["cites"]))):
                        for c in m["cites"]:
                            p = f" p.{c['page']}" if c.get("page") else ""
                            st.markdown(f'<div class="cite-item">[{c["index"]}] {c["source"]}{p} — {c["score"]:.3f}</div>', unsafe_allow_html=True)

    # 聊天输入框（底部）
    q = st.chat_input(T["input_ph"])

    # 附件上传（输入框下方）
    uf = st.file_uploader(
        T["upload"],
        type=["pdf","docx","doc","xlsx","xls","txt","html","htm","md"],
        label_visibility="visible",
        key="chat_up",
    )

    # 上传处理
    if uf and st.session_state.get("last_up") != uf.name:
        with st.spinner("..."):
            try:
                r = requests.post(f"{API_BASE}/api/documents/upload", files={"file": (uf.name, uf.getvalue())}, timeout=300)
                if r.status_code == 200:
                    st.success(r.json().get("message", "OK"))
                    st.session_state.last_up = uf.name
            except Exception as e:
                st.error(str(e))

    if q:
        st.session_state.msgs.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.status(T["processing"], expanded=True) as status:
                try:
                    r = requests.post(f"{API_BASE}/api/query", json={"question": q}, timeout=180)
                    d = r.json()
                    lv = d.get("query_level", "L1")
                    qt = d.get("query_type", "")
                    cf = d.get("confidence", 0)
                    st.write(f"[{lv}] {qt} ({cf:.0%})")
                    st.write(f"{len(d.get('citations',[]))} chunks")
                    status.update(label=T["done"], state="complete", expanded=False)
                except Exception as e:
                    status.update(label=T["failed"], state="error")
                    d = {"answer": "", "query_level": "", "citations": [], "retrieval_trace": {}}

            ans = d.get("answer", "")
            cites = d.get("citations", [])
            lv = d.get("query_level", "")

            if lv:
                st.markdown(f'<span class="level-badge level-{lv}">{lv}</span>', unsafe_allow_html=True)
            if ans:
                st.markdown(ans)
            if cites:
                with st.expander(T["citations"].format(n=len(cites))):
                    for c in cites:
                        p = f" p.{c['page']}" if c.get("page") else ""
                        st.markdown(f'<div class="cite-item">[{c["index"]}] {c["source"]}{p} — {c["score"]:.3f}</div>', unsafe_allow_html=True)

            st.session_state.msgs.append({"role": "assistant", "content": ans, "level": lv, "cites": cites})

# ---- 文档管理 ----
elif nav == "docs":
    st.subheader(T["docs"])

    c_up, c_list = st.columns([1, 2])
    with c_up:
        st.markdown(f"**{T['upload']}**")
        st.caption(T["upload_hint"])
        uf = st.file_uploader("f", type=["pdf","docx","doc","xlsx","xls","txt","html","htm","md"], label_visibility="collapsed", key="doc_up")
        if uf:
            st.info(f"{uf.name} ({uf.size/1024:.1f} KB)")
            if st.button(T["upload_btn"], type="primary", use_container_width=True):
                with st.spinner("..."):
                    try:
                        r = requests.post(f"{API_BASE}/api/documents/upload", files={"file": (uf.name, uf.getvalue())}, timeout=300)
                        st.success(r.json().get("message", "OK") if r.status_code == 200 else r.text)
                    except Exception as e:
                        st.error(str(e))

    with c_list:
        try:
            s = requests.get(f"{API_BASE}/api/documents/stats", timeout=5).json()
            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, val, lbl in [(sc1, s.get("file_count",0), T["docs"]),
                                   (sc2, f"{s.get('total_size_kb',0):.0f}KB", "Size"),
                                   (sc3, s.get("graph_entities",0), T["entities"]),
                                   (sc4, s.get("graph_relations",0), T["relations"])]:
                with col:
                    st.markdown(f'<div style="text-align:center"><div class="stat-num">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)
        except Exception:
            pass

        st.divider()
        if st.button(T["refresh"]):
            st.rerun()

        try:
            docs = requests.get(f"{API_BASE}/api/documents", timeout=5).json().get("documents", [])
            if not docs:
                st.info(T["no_docs"])
            for doc in docs:
                c_n, c_s, c_d = st.columns([5, 1, 1])
                with c_n:
                    st.text(f"[{doc['suffix'].strip('.').upper()}] {doc['filename']}")
                with c_s:
                    st.caption(f"{doc['size_kb']} KB")
                with c_d:
                    confirm_key = f"confirm_{doc['filename']}"
                    if st.session_state.get(confirm_key):
                        # 二次确认状态
                        c_y, c_x = st.columns(2)
                        with c_y:
                            if st.button("确认", key=f"yes_{doc['filename']}", type="primary"):
                                try:
                                    dr = requests.delete(f"{API_BASE}/api/documents/{doc['filename']}", timeout=10)
                                    if dr.status_code == 200:
                                        st.success(T["deleted"])
                                        st.session_state.pop(confirm_key, None)
                                        st.rerun()
                                except Exception as e:
                                    st.error(str(e))
                        with c_x:
                            if st.button("取消", key=f"no_{doc['filename']}"):
                                st.session_state.pop(confirm_key, None)
                                st.rerun()
                    else:
                        if st.button(T["delete"], key=f"d_{doc['filename']}"):
                            st.session_state[confirm_key] = True
                            st.rerun()
        except Exception as e:
            st.error(str(e))

# ---- 知识图谱 ----
elif nav == "graph":
    st.subheader(T["graph"])

    t1, t2, t3 = st.tabs(["Overview", "Explorer", "Stats"])

    with t1:
        try:
            gd = requests.get(f"{API_BASE}/api/graph/data", timeout=10).json()
            nodes = gd.get("nodes", [])
            edges = gd.get("edges", [])
            if not nodes:
                st.info(T["graph_empty"])
            else:
                types = sorted(set(n.get("type", "") for n in nodes))
                fcols = st.columns(min(len(types), 6))
                sel = []
                for i, t in enumerate(types):
                    with fcols[i % len(fcols)]:
                        cnt = sum(1 for n in nodes if n.get("type") == t)
                        if st.checkbox(f"{t} ({cnt})", value=True, key=f"t_{t}"):
                            sel.append(t)

                fn = [n for n in nodes if n.get("type", "") in sel]
                fi = {n["id"] for n in fn}
                fe = [e for e in edges if e["source"] in fi and e["target"] in fi]

                mc1, mc2 = st.columns(2)
                mc1.metric(T["entities"], len(fn))
                mc2.metric(T["relations"], len(fe))

                try:
                    from pyvis.network import Network
                    import tempfile, streamlit.components.v1 as components

                    net = Network(height="500px", width="100%", directed=True, bgcolor="#ffffff", font_color="#333")
                    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200)

                    deg = {}
                    for e in fe:
                        deg[e["source"]] = deg.get(e["source"], 0) + 1
                        deg[e["target"]] = deg.get(e["target"], 0) + 1

                    for nd in fn:
                        c = TC.get(nd.get("type", ""), "#6b7280")
                        d = deg.get(nd["id"], 0)
                        net.add_node(nd["id"], label=nd["label"], color=c, size=max(14, min(36, 14 + d*3)))
                    for eg in fe:
                        net.add_edge(eg["source"], eg["target"], label=eg.get("label", ""))

                    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                        net.save_graph(f.name)
                        html = open(f.name, "r", encoding="utf-8").read()
                        components.html(html, height=520)

                    leg = " | ".join(f'<span style="color:{TC.get(t,"#6b7280")};font-weight:600">{t}</span>' for t in sel)
                    st.caption(f"{T['legend']}: {leg}", unsafe_allow_html=True)
                except ImportError:
                    st.warning(T["install_pyvis"])
        except Exception as e:
            st.error(str(e))

    with t2:
        cs, ch = st.columns([3, 1])
        with cs:
            kw = st.text_input(T["search"], placeholder=T["search"], label_visibility="collapsed")
        with ch:
            hp = st.selectbox(T["hops"], [1, 2, 3], index=1, label_visibility="collapsed")

        if kw:
            try:
                ents = requests.get(f"{API_BASE}/api/graph/entities", params={"keyword": kw}, timeout=5).json().get("entities", [])
                if not ents:
                    st.info(f"{T['not_found']}: {kw}")
                else:
                    st.success(T["found"].format(n=len(ents)))
                    for ent in ents:
                        with st.expander(f"{ent['name']} — {ent.get('entity_type','')}", expanded=True):
                            st.markdown(f"**{T['desc']}:** {ent.get('description', T['no_desc'])}")
                            nb = requests.get(f"{API_BASE}/api/graph/neighbors/{ent['name']}", params={"hops": hp}, timeout=5).json()
                            for rel in nb.get("relations", []):
                                hl = f" (h{rel.get('hop',1)})" if rel.get('hop',1) > 1 else ""
                                st.caption(f"  {rel['source']} ->[{rel.get('relation_type','')}]-> {rel['target']}{hl}")
            except Exception as e:
                st.error(str(e))

    with t3:
        try:
            gd = requests.get(f"{API_BASE}/api/graph/data", timeout=10).json()
            nodes = gd.get("nodes", [])
            edges = gd.get("edges", [])
            if nodes:
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric(T["entities"], len(nodes))
                sc2.metric(T["relations"], len(edges))
                sc3.metric(T["types"], len(set(n.get("type","") for n in nodes)))
                st.divider()
                tc = {}
                for n in nodes:
                    tc[n.get("type", "")] = tc.get(n.get("type", ""), 0) + 1
                st.subheader(T["type_dist"])
                try:
                    import pandas as pd
                    df = pd.DataFrame(list(tc.items()), columns=[T["types"], "Count"])
                    st.bar_chart(df.set_index(T["types"]))
                except ImportError:
                    for t, c in sorted(tc.items(), key=lambda x: -x[1]):
                        st.text(f"  {t}: {c}")
        except Exception as e:
            st.error(str(e))

# ---- 系统追踪 ----
elif nav == "traces":
    st.subheader(T["traces"])
    try:
        traces = requests.get(f"{API_BASE}/api/traces", params={"limit": 20}, timeout=5).json().get("traces", [])
        if traces:
            for t in reversed(traces):
                with st.expander(f"{t.get('query','')[:50]}  |  {t.get('query_level','?')}  |  {t.get('total_elapsed_ms',0):.0f}ms"):
                    st.json(t)
        else:
            st.info(T["no_trace"])
    except Exception as e:
        st.error(str(e))
