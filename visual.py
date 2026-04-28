"""
dashboard_st.py — URG-IS Streamlit Dashboard
=============================================
Run:
    pip install streamlit plotly
    streamlit run dashboard_st.py

Shows:
  - Live camera feed (MJPEG from run_live.py)
  - Live D3/Plotly relationship graph
  - Per-person details with all metrics
  - Incident breakdown
  - Decay/pruning view
"""

import json, time, requests, streamlit as st
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime

BASE = "http://localhost:8765"

st.set_page_config(
    page_title="URG-IS · Relationship Intelligence",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded",
)

# ── Dark theme ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#07090c}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #1e2d3d}
.stMetric{background:#111820;border:1px solid #1e2d3d;border-radius:6px;padding:12px}
.stMetric label{color:#5a7a9a!important;font-size:10px!important;letter-spacing:2px!important;text-transform:uppercase!important}
.stMetric [data-testid="metric-container"] div{color:#e8f4ff!important}
h1,h2,h3{color:#00d4ff!important;font-family:'SF Mono',monospace!important}
p,span,div{color:#c8d8e8}
.rel-card{background:#111820;border:1px solid #1e2d3d;border-radius:6px;padding:12px;margin-bottom:8px}
</style>
""", unsafe_allow_html=True)

# ── Fetch data ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=2)
def fetch_graph():
    try:
        r = requests.get(f"{BASE}/graph", timeout=2)
        return r.json()
    except:
        return {"nodes": [], "edges": [], "stats": {}}

@st.cache_data(ttl=5)
def fetch_cameras():
    try:
        r = requests.get(f"{BASE}/cameras", timeout=2)
        return r.json()
    except:
        return []

def rel_color(rel):
    return {
        "significant":     "#c0392b",
        "close_associate": "#d4850a",
        "associate":       "#2ea84e",
        "acquaintance":    "#1a7abf",
        "stranger":        "#444e5c",
    }.get(rel, "#444e5c")

# ── Sidebar: camera + controls ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 URG-IS")
    st.markdown("---")

    data = fetch_graph()
    cams = fetch_cameras()
    stats = data.get("stats", {})
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # Status
    if stats.get("pipeline_running"):
        st.success("● PIPELINE RUNNING", icon=None)
    else:
        st.warning("● Waiting for run_live.py...")

    st.markdown("---")

    # Stats
    col1, col2 = st.columns(2)
    col1.metric("People", stats.get("total_people", len(nodes)))
    col2.metric("Relations", stats.get("total_relations", len(edges)))
    col1.metric("Non-stranger", stats.get("non_stranger_relations", 0))
    col2.metric("Cameras", len(cams))

    st.markdown("---")

    # Camera feed
    st.markdown("### 📷 Live CCTV")
    if cams:
        cam = st.selectbox("Camera", cams, key="cam_select")
        st.image(
            f"{BASE}/video/{cam}",
            caption=f"{cam.upper()} — LIVE",
            use_container_width=True,
        )
    else:
        st.info("No cameras yet. Start run_live.py")

    st.markdown("---")
    # Auto-refresh
    refresh = st.slider("Refresh (seconds)", 1, 10, 2)
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────
tab_graph, tab_person, tab_edge, tab_decay = st.tabs([
    "📊 Relationship Graph",
    "👤 Person Details",
    "🔗 Edge Details",
    "⏳ Decay Monitor",
])

# ── TAB 1: Graph ───────────────────────────────────────────────────────────
with tab_graph:
    st.markdown("### Live Relationship Graph")
    st.caption("Nodes coloured by strongest relationship · Size = number of connections · Strangers shown as grey")

    if not nodes:
        st.info("No data yet. Run: `python run_live.py data/ --fresh`")
    else:
        # Build networkx graph for layout
        G = nx.Graph()
        for n in nodes:
            G.add_node(n["id"], degree=n.get("degree", 0))
        for e in edges:
            G.add_edge(e["person_id_a"], e["person_id_b"],
                       confidence=e["confidence"],
                       relationship=e["relationship"])

        # Layout
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, k=2.5, seed=42)
        else:
            pos = {}

        # Build plotly figure
        fig = go.Figure()

        # Draw edges
        for e in edges:
            a, b = e["person_id_a"], e["person_id_b"]
            if a not in pos or b not in pos:
                continue
            x0,y0 = pos[a]; x1,y1 = pos[b]
            color = rel_color(e["relationship"])
            width = 1 + e["confidence"] * 6
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="text",
                text=f"P{a} ↔ P{b}<br>{e['relationship']}<br>Confidence: {e['confidence']:.3f}<br>Meetings: {e['total_meetings']}",
                name="",
                showlegend=False,
            ))

        # Draw nodes
        for n in nodes:
            pid = n["id"]
            if pid not in pos:
                continue
            x, y = pos[pid]
            deg = n.get("degree", 0)
            # Node colour = strongest relationship
            node_edges = [e for e in edges if e["person_id_a"]==pid or e["person_id_b"]==pid]
            if node_edges:
                best = max(node_edges, key=lambda e: e["confidence"])
                color = rel_color(best["relationship"])
                rel_label = best["relationship"].replace("_", " ").title()
            else:
                color = "#444e5c"
                rel_label = "Stranger"

            size = 14 + min(deg * 3, 20)
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=size, color=color,
                            line=dict(color=color, width=2)),
                text=[str(pid)],
                textposition="top center",
                textfont=dict(size=9, color="rgba(200,216,232,0.7)"),
                hoverinfo="text",
                hovertext=f"Person {pid}<br>Connections: {deg}<br>Best relation: {rel_label}",
                name=f"P{pid}",
                showlegend=False,
            ))

        # Legend
        for rel, col in [("Significant","#c0392b"),("Close Associate","#d4850a"),
                          ("Associate","#2ea84e"),("Acquaintance","#1a7abf"),("Stranger","#444e5c")]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=col),
                name=rel, showlegend=True,
            ))

        fig.update_layout(
            paper_bgcolor="#07090c",
            plot_bgcolor="#07090c",
            font=dict(color="#c8d8e8", family="SF Mono, monospace", size=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(bgcolor="#0d1117", bordercolor="#1e2d3d", borderwidth=1,
                        font=dict(size=10, color="#c8d8e8")),
            margin=dict(l=10, r=10, t=10, b=10),
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.markdown("### All Relationships")
        if edges:
            sorted_edges = sorted(edges, key=lambda e: e["confidence"], reverse=True)
            cols = st.columns([1,1,2,2,1,2])
            cols[0].markdown("**Person A**")
            cols[1].markdown("**Person B**")
            cols[2].markdown("**Relationship**")
            cols[3].markdown("**Confidence**")
            cols[4].markdown("**Meetings**")
            cols[5].markdown("**Cameras**")
            st.markdown("---")
            for e in sorted_edges:
                col = rel_color(e["relationship"])
                c = st.columns([1,1,2,2,1,2])
                c[0].markdown(f"**{e['person_id_a']}**")
                c[1].markdown(f"**{e['person_id_b']}**")
                c[2].markdown(f'<span style="color:{col}">{e["relationship"].replace("_"," ").upper()}</span>', unsafe_allow_html=True)
                pct = int(e["confidence"] * 100)
                c[3].markdown(f'<div style="background:#1e2d3d;border-radius:3px;height:8px;overflow:hidden"><div style="width:{pct}%;height:100%;background:{col}"></div></div><small>{e["confidence"]:.4f}</small>', unsafe_allow_html=True)
                c[4].write(e["total_meetings"])
                c[5].write(", ".join(e.get("cameras", [])) or "—")

# ── TAB 2: Person details ──────────────────────────────────────────────────
with tab_person:
    st.markdown("### Person Details")

    if not nodes:
        st.info("No people identified yet.")
    else:
        person_ids = sorted([n["id"] for n in nodes], key=lambda x: int(x) if str(x).isdigit() else 999)
        selected_pid = st.selectbox("Select Person", person_ids, key="person_select")

        if selected_pid:
            person_edges = [e for e in edges
                           if e["person_id_a"] == selected_pid or e["person_id_b"] == selected_pid]
            cameras = list(set(c for e in person_edges for c in e.get("cameras", [])))
            non_stranger = [e for e in person_edges if e["relationship"] != "stranger"]

            # Header metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Person ID", selected_pid)
            c2.metric("Total Connections", len(person_edges))
            c3.metric("Meaningful Relations", len(non_stranger))
            c4.metric("Cameras Seen On", len(cameras))

            if cameras:
                st.markdown(f"**Cameras:** " + " ".join([f"`{c}`" for c in cameras]))

            st.markdown("---")

            if not person_edges:
                st.info("No connections yet for this person.")
            else:
                st.markdown("#### Connections (sorted by confidence)")

                for e in sorted(person_edges, key=lambda x: x["confidence"], reverse=True):
                    other = e["person_id_b"] if e["person_id_a"] == selected_pid else e["person_id_a"]
                    col = rel_color(e["relationship"])
                    pct = int(e["confidence"] * 100)

                    with st.expander(
                        f"Person {other} — {e['relationship'].replace('_',' ').upper()} ({e['confidence']:.3f})",
                        expanded=(e["confidence"] > 0.3)
                    ):
                        cc1, cc2, cc3 = st.columns(3)
                        cc1.metric("Confidence", f"{e['confidence']:.4f}")
                        cc2.metric("Meetings", e["total_meetings"])
                        cc3.metric("Avg Duration", f"{e.get('avg_duration_s', 0)}s")

                        # Confidence bar
                        st.markdown(
                            f'<div style="background:#1e2d3d;border-radius:4px;height:10px;overflow:hidden;margin:8px 0">'
                            f'<div style="width:{pct}%;height:100%;background:{col};border-radius:4px"></div></div>',
                            unsafe_allow_html=True
                        )

                        # Incident breakdown
                        inc = e.get("incident_counts", {})
                        if inc:
                            st.markdown("**Incident Breakdown:**")
                            ic1, ic2, ic3, ic4, ic5 = st.columns(5)
                            ic1.metric("Close Contact", inc.get("CLOSE_CONTACT", 0))
                            ic2.metric("Conversation", inc.get("CONVERSATION", 0))
                            ic3.metric("Group", inc.get("GROUP_GATHERING", 0))
                            ic4.metric("Extended", inc.get("EXTENDED_MEETING", 0))
                            ic5.metric("Proximity", inc.get("PROXIMITY", 0))

                        cams = e.get("cameras", [])
                        if cams:
                            st.markdown(f"**Seen together on:** " + " ".join([f"`{c}`" for c in cams]))

# ── TAB 3: Edge details ────────────────────────────────────────────────────
with tab_edge:
    st.markdown("### Edge Details — Full Metrics")

    if not edges:
        st.info("No edges yet.")
    else:
        edge_labels = [
            f"P{e['person_id_a']} ↔ P{e['person_id_b']} [{e['relationship']}] {e['confidence']:.3f}"
            for e in sorted(edges, key=lambda x: x["confidence"], reverse=True)
        ]
        selected_edge_idx = st.selectbox("Select Edge", range(len(edge_labels)),
                                          format_func=lambda i: edge_labels[i])
        e = sorted(edges, key=lambda x: x["confidence"], reverse=True)[selected_edge_idx]
        col = rel_color(e["relationship"])
        pct = int(e["confidence"] * 100)

        # Header
        st.markdown(f"## Person {e['person_id_a']} ↔ Person {e['person_id_b']}")
        st.markdown(
            f'<span style="color:{col};background:rgba(0,0,0,.3);border:1px solid {col};padding:4px 12px;border-radius:4px;font-size:12px">'
            f'{e["relationship"].replace("_"," ").upper()}</span>',
            unsafe_allow_html=True
        )
        st.markdown("")

        # Confidence bar
        st.markdown(f"**Confidence: {e['confidence']:.4f} ({pct}%)**")
        st.markdown(
            f'<div style="background:#1e2d3d;border-radius:5px;height:14px;overflow:hidden;margin-bottom:16px">'
            f'<div style="width:{pct}%;height:100%;background:{col};border-radius:5px"></div></div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Meetings", e["total_meetings"])
        col2.metric("Avg Duration", f"{e.get('avg_duration_s', 0)}s")
        col3.metric("Meetings Today", e.get("meetings_today", 0))
        col4.metric("Cameras", len(e.get("cameras", [])))

        st.markdown("---")

        # Incident breakdown with chart
        st.markdown("#### Incident Breakdown")
        inc = e.get("incident_counts", {})
        TYPES = ["CLOSE_CONTACT", "EXTENDED_MEETING", "CONVERSATION", "GROUP_GATHERING", "PROXIMITY"]
        TYPE_COLORS = ["#c0392b", "#d4850a", "#2ea84e", "#9b59b6", "#1a7abf"]

        fig_inc = go.Figure(go.Bar(
            x=[t.replace("_", " ") for t in TYPES],
            y=[inc.get(t, 0) for t in TYPES],
            marker_color=TYPE_COLORS,
            text=[inc.get(t, 0) for t in TYPES],
            textposition="outside",
        ))
        fig_inc.update_layout(
            paper_bgcolor="#07090c", plot_bgcolor="#111820",
            font=dict(color="#c8d8e8", size=10),
            xaxis=dict(tickfont=dict(color="#c8d8e8")),
            yaxis=dict(gridcolor="#1e2d3d", tickfont=dict(color="#c8d8e8")),
            margin=dict(l=10, r=10, t=10, b=10),
            height=220,
            showlegend=False,
        )
        st.plotly_chart(fig_inc, use_container_width=True)

        # Boost modifiers explanation
        st.markdown("#### Confidence Modifiers")
        st.markdown("""
| Modifier | Effect | Why |
|---|---|---|
| **Base boost** | Depends on incident type | CLOSE_CONTACT=0.20, CONVERSATION=0.12, PROXIMITY=0.03 |
| **Distance mod** | Closer = higher | `1 + (1.5 - dist_m) / 1.5` → max 2× |
| **Location mod** | Same spot repeatedly = higher | Up to 1.3× bonus |
| **Privacy mod** | 1-on-1 stronger than group | 1-on-1 = 1.20×, group = 0.80× |
| **Diminishing** | More meetings = lower boost | `1 / (1 + meetings × 0.3)` |
| **Decay** | Fades over time | ×0.998 every 10 minutes |
""")

        cams = e.get("cameras", [])
        if cams:
            st.markdown(f"**Seen together on:** " + " ".join([f"`{c}`" for c in cams]))

# ── TAB 4: Decay monitor ───────────────────────────────────────────────────
with tab_decay:
    st.markdown("### Decay & Pruning Monitor")
    st.markdown("Each edge loses **0.2% confidence every 10 minutes**. Edges below 0.01 are deleted.")

    if not edges:
        st.info("No edges to monitor yet.")
    else:
        sorted_e = sorted(edges, key=lambda x: x["confidence"])

        # Decay chart — all edges as horizontal bars
        fig_decay = go.Figure()
        for e in sorted_e:
            col = rel_color(e["relationship"])
            label = f"P{e['person_id_a']}↔P{e['person_id_b']}"
            conf = e["confidence"]
            # Show how many 10-min ticks until gone
            if conf > 0:
                import math
                ticks_to_die = math.log(0.01/conf) / math.log(0.998) if conf > 0.01 else 0
                hours = ticks_to_die * 10 / 60
            else:
                hours = 0

            fig_decay.add_trace(go.Bar(
                x=[conf], y=[label],
                orientation='h',
                marker_color=col,
                text=f"{conf:.3f} ({hours:.1f}h until pruned)",
                textposition="outside",
                name=label,
                showlegend=False,
                hovertext=f"{label}<br>Confidence: {conf:.4f}<br>Relationship: {e['relationship']}<br>Meetings: {e['total_meetings']}<br>~{hours:.1f}h until pruned at current rate",
                hoverinfo="text",
            ))

        # Threshold line
        fig_decay.add_vline(x=0.01, line_dash="dash", line_color="#c0392b",
                             annotation_text="Prune threshold (0.01)",
                             annotation_font_color="#c0392b")
        fig_decay.add_vline(x=0.20, line_dash="dot", line_color="#5a7a9a",
                             annotation_text="Acquaintance")
        fig_decay.add_vline(x=0.40, line_dash="dot", line_color="#1a7abf",
                             annotation_text="Associate")
        fig_decay.add_vline(x=0.60, line_dash="dot", line_color="#d4850a",
                             annotation_text="Close Associate")

        fig_decay.update_layout(
            paper_bgcolor="#07090c", plot_bgcolor="#111820",
            font=dict(color="#c8d8e8", size=9),
            xaxis=dict(title="Confidence", range=[0, 1.1],
                       gridcolor="#1e2d3d", tickfont=dict(color="#c8d8e8")),
            yaxis=dict(tickfont=dict(color="#c8d8e8", size=8)),
            margin=dict(l=10, r=150, t=10, b=30),
            height=max(300, len(sorted_e) * 28 + 60),
            barmode='overlay',
        )
        st.plotly_chart(fig_decay, use_container_width=True)

        # Table
        st.markdown("#### Edge Status Table")
        import math
        rows = []
        for e in sorted_e:
            conf = e["confidence"]
            ticks = math.log(0.01/conf) / math.log(0.998) if conf > 0.01 else 0
            hours = ticks * 10 / 60
            rows.append({
                "Pair": f"P{e['person_id_a']} ↔ P{e['person_id_b']}",
                "Relationship": e["relationship"].replace("_"," ").title(),
                "Confidence": f"{conf:.4f}",
                "Meetings": e["total_meetings"],
                "Cameras": ", ".join(e.get("cameras",[])) or "—",
                "Time until pruned": f"{hours:.1f}h" if conf > 0.01 else "⚠️ PRUNED",
            })
        import pandas as pd
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

# ── Auto-refresh ───────────────────────────────────────────────────────────
time.sleep(refresh)
st.rerun()