import streamlit as st
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import sys
import os

# Ensure we can import the core modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.graph.graph_db import GraphDB

st.set_page_config(layout="wide", page_title="Spatial Heatmap Test")

st.markdown("""
<style>
.stApp { background-color: #050c1a; color: #e2e8f0; font-family: 'Inter', sans-serif; }
.card { background: #081229; border: 1px solid #0f2242; border-radius: 12px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='color: #60c3ff;'>🗺️ Spatial Activity Heatmap</h2>", unsafe_allow_html=True)
st.markdown("This visualization maps physical tracking coordinates onto a floorplan to identify activity hotspots and physical relationship paths.")

# Use the absolute path to the generated artifact image
bg_path = "/Users/bhargavi/.gemini/antigravity/brain/730532f5-e21b-4d62-88d5-0cb775416b7b/.tempmediaStorage/media_730532f5-e21b-4d62-88d5-0cb775416b7b_1777198445565.png"

@st.cache_data
def get_graph_data():
    try:
        db = GraphDB("data/snapshots/prod_graph.json")
    except Exception as e:
        st.error(f"Could not load graph database: {e}")
        return [], [], [], {}
        
    edges = db.get_all_edges()
    
    # 1. Gather all locations for the heatmap
    all_x = []
    all_y = []
    
    # 2. Calculate person centroids
    person_locs = {} # pid -> list of (x,y)
    
    # We will normalize coordinates to 0-1000 range for the generic floorplan
    # Assume original image coords were around 1920x1080 (HD Cameras)
    max_x, max_y = 1920, 1080
    
    for e in edges:
        for x, y in e.locations:
            # normalize to map
            nx = (x / max_x) * 1000
            ny = 1000 - ((y / max_y) * 1000) # invert Y for Plotly
            
            all_x.append(nx)
            all_y.append(ny)
            
            if e.person_id_a not in person_locs: person_locs[e.person_id_a] = []
            if e.person_id_b not in person_locs: person_locs[e.person_id_b] = []
            person_locs[e.person_id_a].append((nx, ny))
            person_locs[e.person_id_b].append((nx, ny))
            
    # If the database has edges but no location data, generate synthetic ones for this demo
    if edges and not all_x:
        np.random.seed(42)
        for e in edges:
            for _ in range(e.total_meetings):
                # People interact around the same spots
                nx = np.random.normal(500, 200)
                ny = np.random.normal(500, 200)
                nx = max(100, min(900, nx))
                ny = max(100, min(900, ny))
                all_x.append(nx); all_y.append(ny)
                if e.person_id_a not in person_locs: person_locs[e.person_id_a] = []
                if e.person_id_b not in person_locs: person_locs[e.person_id_b] = []
                person_locs[e.person_id_a].append((nx, ny))
                person_locs[e.person_id_b].append((nx, ny))

    centroids = {}
    for pid, locs in person_locs.items():
        if locs:
            centroids[pid] = (sum(l[0] for l in locs)/len(locs), sum(l[1] for l in locs)/len(locs))
            
    return edges, all_x, all_y, centroids

edges, all_x, all_y, centroids = get_graph_data()

c1, c2 = st.columns([1, 4])
with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎛️ Map Controls")
    
    pids = sorted(list(centroids.keys()), key=lambda x: int(x) if x.isdigit() else 999)
    selected_p = st.selectbox("Highlight Person's Relationships", ["None"] + pids)
    
    st.markdown("---")
    show_bg = st.checkbox("Show Floorplan Background", value=True)
    show_heatmap = st.checkbox("Show Density Heatmap", value=True)
    show_nodes = st.checkbox("Show People Markers", value=True)
    
    st.markdown("---")
    st.metric("Total Edges Displayed", len(edges))
    st.metric("Total Data Points", len(all_x))
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    fig = go.Figure()
    
    # Layer 0: Background Image
    if show_bg and os.path.exists(bg_path):
        try:
            img = Image.open(bg_path)
            fig.add_layout_image(
                dict(
                    source=img,
                    xref="x", yref="y",
                    x=0, y=1000,
                    sizex=1000, sizey=1000,
                    sizing="stretch",
                    opacity=1.0,
                    layer="below"
                )
            )
        except Exception as e:
            st.error(f"Could not load background image: {e}")

    # Layer 1: Heatmap Contour Overlay
    if show_heatmap and all_x:
        fig.add_trace(go.Histogram2dContour(
            x=all_x, y=all_y,
            colorscale="Jet",
            reversescale=False,
            opacity=0.45,
            ncontours=15,
            contours=dict(
                coloring='heatmap',
                showlines=True,
            ),
            line=dict(width=1.5, color='rgba(255,255,255,0.6)'),
            showscale=False,
            hoverinfo="skip"
        ))

    # Layer 2: Relationship Lines
    if selected_p != "None" and selected_p in centroids:
        px, py = centroids[selected_p]
        # find all edges for this person
        for e in edges:
            if e.person_id_a == selected_p or e.person_id_b == selected_p:
                other_p = e.person_id_b if e.person_id_a == selected_p else e.person_id_a
                if other_p in centroids:
                    ox, oy = centroids[other_p]
                    width = max(1.0, e.confidence * 6)
                    fig.add_trace(go.Scatter(
                        x=[px, ox], y=[py, oy],
                        mode="lines",
                        line=dict(color="white", width=width, dash='dash'),
                        opacity=0.9,
                        hoverinfo="text",
                        text=f"Confidence: {e.confidence:.2f} ({e.relationship})"
                    ))

    # Layer 3: Persons Marker (Icons)
    if show_nodes and centroids:
        nx = [c[0] for c in centroids.values()]
        ny = [c[1] for c in centroids.values()]
        n_ids = list(centroids.keys())
        
        colors = ["#ef4444" if p == selected_p else "#10b981" for p in n_ids]
        sizes = [35 if p == selected_p else 24 for p in n_ids]
        
        fig.add_trace(go.Scatter(
            x=nx, y=ny,
            mode="markers+text",
            marker=dict(size=sizes, symbol='star', color=colors, line=dict(color="#ffffff", width=2)),
            text=[f"P{p}" for p in n_ids],
            textposition="top center",
            textfont=dict(color="#ffffff", size=14, weight="bold"),
            hoverinfo="text",
            hovertext=[f"Person {p}" for p in n_ids]
        ))

    fig.update_layout(
        xaxis=dict(range=[0, 1000], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 1000], showgrid=False, zeroline=False, visible=False),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0),
        height=750,
        showlegend=False,
        plot_bgcolor="#050c1a",
        paper_bgcolor="#050c1a"
    )
    
    st.plotly_chart(fig, use_container_width=True)
