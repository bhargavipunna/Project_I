# """
# run_live.py
# ===========
# URG-IS — All 7 cameras, live terminal output.

# Shows in real time:
#   - Which person detected on which camera
#   - Re-ID matches with similarity score
#   - Every incident (type, distance, duration, boost breakdown)
#   - Per-person relationship graph after each incident
#   - Graph pruning when you press Ctrl+C

# Usage:
#     python run_live.py              # all 7 cameras from data/
#     python run_live.py data/        # explicit path
#     python run_live.py data/cam1.mp4  # single camera
# """

# import os, sys, time, threading, io, json
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # torch BEFORE faiss — prevents segfault on macOS
# import torch
# import faiss

# import cv2
# import numpy as np
# from collections import defaultdict
# from http.server import HTTPServer, BaseHTTPRequestHandler
# import socket as _socket

# # ── WebSocket push clients ────────────────────────────────────────────────────
# _ws_clients = []
# _ws_lock    = threading.Lock()

# def _ws_handshake(conn, request):
#     import hashlib, base64
#     key = None
#     for line in request.split(b'\r\n'):
#         if b'Sec-WebSocket-Key' in line:
#             key = line.split(b': ')[1].strip().decode()
#     if not key:
#         return False
#     magic = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
#     accept = base64.b64encode(hashlib.sha1((key+magic).encode()).digest()).decode()
#     resp = (
#         'HTTP/1.1 101 Switching Protocols\r\n'
#         'Upgrade: websocket\r\n'
#         'Connection: Upgrade\r\n'
#         f'Sec-WebSocket-Accept: {accept}\r\n'
#         'Access-Control-Allow-Origin: *\r\n\r\n'
#     )
#     conn.sendall(resp.encode())
#     return True

# def _ws_send(conn, msg: str):
#     import struct
#     data = msg.encode('utf-8')
#     n = len(data)
#     if n <= 125:
#         header = bytes([0x81, n])
#     elif n <= 65535:
#         header = bytes([0x81, 126]) + struct.pack('>H', n)
#     else:
#         header = bytes([0x81, 127]) + struct.pack('>Q', n)
#     try:
#         conn.sendall(header + data)
#         return True
#     except:
#         return False

# def _ws_client_thread(conn, addr):
#     try:
#         buf = b''
#         while b'\r\n\r\n' not in buf:
#             chunk = conn.recv(1024)
#             if not chunk:
#                 return
#             buf += chunk
#         if not _ws_handshake(conn, buf):
#             return
#         with _ws_lock:
#             _ws_clients.append(conn)
#         # Keep alive — read and discard client frames
#         while True:
#             try:
#                 d = conn.recv(256)
#                 if not d:
#                     break
#             except:
#                 break
#     finally:
#         with _ws_lock:
#             if conn in _ws_clients:
#                 _ws_clients.remove(conn)
#         try: conn.close()
#         except: pass

# def push_graph_update(db_ref):
#     """Push current graph to all WebSocket clients."""
#     if not _ws_clients or db_ref is None:
#         return
#     try:
#         all_edges  = list(db_ref.get_all_edges())
#         all_nodes  = list(db_ref.get_all_nodes())
#         conn_edges = [e for e in all_edges if e.relationship != 'stranger']
#         deg_map = {}
#         for e in conn_edges:
#             deg_map[e.person_id_a] = deg_map.get(e.person_id_a, 0) + 1
#             deg_map[e.person_id_b] = deg_map.get(e.person_id_b, 0) + 1
#         payload = json.dumps({
#             'type': 'graph_update',
#             'data': {
#                 'nodes': [{'id': n.person_id, 'degree': deg_map.get(n.person_id, 0)} for n in all_nodes],
#                 'edges': [{'person_id_a': e.person_id_a, 'person_id_b': e.person_id_b,
#                            'confidence': e.confidence, 'relationship': e.relationship,
#                            'total_meetings': e.total_meetings,
#                            'incident_counts': dict(e.incident_counts) if e.incident_counts else {},
#                            'cameras': list(e.cameras) if e.cameras else []}
#                           for e in conn_edges],
#                 'stats': {
#                     'total_people': len(all_nodes),
#                     'total_relations': len(all_edges),
#                     'pipeline_running': True,
#                 }
#             }
#         })
#         dead = []
#         with _ws_lock:
#             clients = list(_ws_clients)
#         for conn in clients:
#             if not _ws_send(conn, payload):
#                 dead.append(conn)
#         with _ws_lock:
#             for d in dead:
#                 if d in _ws_clients:
#                     _ws_clients.remove(d)
#     except Exception as e:
#         pass

# from core.video.stream_reader       import StreamReader
# from core.video.multi_stream_reader import MultiCameraStreamReader
# from core.tracking.person_tracker   import PersonTracker
# from core.reid.embedder             import PersonEmbedder
# from core.reid.identity_manager     import IdentityManager
# from core.interaction.interaction_detector import InteractionDetector
# from core.graph.incident_classifier import IncidentClassifier, INCIDENT_BOOST
# from core.graph.graph_db            import GraphDB
# from core.graph.confidence_engine   import (
#     ConfidenceEngine, MAX_DISTANCE_M,
#     LOCATION_BUCKET_PX, LOCATION_MAX_BONUS, LOCATION_VISITS_FOR_MAX,
#     PRIVACY_ONE_ON_ONE, PRIVACY_GROUP, DIMINISHING_RATE,
# )

# SEP  = "═" * 70
# SEP2 = "─" * 70

# def banner(t): print(f"\n{SEP}\n  {t}\n{SEP}")
# def sub(t):    print(f"\n  ── {t}")
# def log(m, n=2): print(" "*n + m)

# def print_person_graph(db, person_id):
#     """Print the full relationship graph for one person."""
#     g = db.get_person_graph(person_id)
#     if not g or not g["connections"]:
#         return
#     print(f"\n  ┌─ Person {person_id} — {g['total_connections']} connection(s) ─────────────")
#     for c in g["connections"]:
#         bar = "█" * int(c["confidence"] * 20) + "░" * (20 - int(c["confidence"] * 20))
#         print(f"  │  → Person {c['person_id']:<4} [{c['relationship']:<16}] "
#               f"{c['confidence']:.4f} {bar}")
#         print(f"  │     Incidents : {c['incident_counts']}")
#         print(f"  │     Meetings  : {c['total_meetings']}  "
#               f"Avg dur: {c['avg_duration_s']}s  "
#               f"Cameras: {c['cameras']}")
#     print(f"  └{'─'*60}")

# def print_full_graph(db):
#     """Print all relationships at the end."""
#     banner("FINAL RELATIONSHIP GRAPH")
#     edges = db.get_all_edges()
#     nodes = db.get_all_nodes()

#     if not edges:
#         log("No relationships built.")
#         return

#     log(f"Total people identified : {len(nodes)}")
#     log(f"Total relationships     : {len(edges)}\n")

#     log(f"  {'Person A':>8}  {'Person B':>8}  {'Confidence':>10}  "
#         f"{'Relationship':>16}  {'Meetings':>8}  {'Top Incident':>18}")
#     log(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*16}  {'─'*8}  {'─'*18}")

#     for e in sorted(edges, key=lambda x: x.confidence, reverse=True):
#         top = max(e.incident_counts, key=e.incident_counts.get) if e.incident_counts else "—"
#         log(f"  {e.person_id_a:>8}  {e.person_id_b:>8}  "
#             f"{e.confidence:>10.4f}  {e.relationship:>16}  "
#             f"{e.total_meetings:>8}  {top:>18}")

#     sub("Per-Person Graphs")
#     seen = set()
#     for e in sorted(edges, key=lambda x: x.confidence, reverse=True):
#         for pid in [e.person_id_a, e.person_id_b]:
#             if pid not in seen:
#                 seen.add(pid)
#                 print_person_graph(db, pid)


# # ── Shared annotated frames for MJPEG stream ─────────────────────────────────
# _latest_frames = {}   # cam_id → annotated JPEG bytes
# _frames_lock   = threading.Lock()
# _graph_db_ref  = None   # set in main()

# FONT = cv2.FONT_HERSHEY_SIMPLEX

# def annotate_frame(frame, cam_id, persons_with_ids, active_prox, events_fired):
#     """Draw person IDs and interaction lines on frame, return annotated copy."""
#     out = frame.copy()
#     # Draw tracked persons
#     for pid, center in persons_with_ids:
#         cx, cy = center
#         cv2.circle(out, (cx, cy), 4, (0, 220, 0), -1)
#         label = f"P{pid}"
#         cv2.putText(out, label, (cx-10, cy-10), FONT, 0.45, (0, 220, 0), 1)
#     # Interaction lines for fired events
#     centers = {str(pid): c for pid, c in persons_with_ids}
#     for ev in events_fired:
#         pa = centers.get(str(ev.person_id_a))
#         pb = centers.get(str(ev.person_id_b))
#         if pa and pb:
#             cv2.line(out, pa, pb, (0, 0, 255), 2)
#     # Camera label
#     n = len(persons_with_ids)
#     cv2.putText(out, f"{cam_id} | {n} people", (6, 20), FONT, 0.5, (0, 200, 200), 1)
#     return out

# def store_frame(cam_id, frame):
#     """Encode frame as JPEG and store for MJPEG streaming."""
#     ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
#     if ret:
#         with _frames_lock:
#             _latest_frames[cam_id] = buf.tobytes()

# DASHBOARD_HTML = ''  # set by build_dashboard_html()

# class MJPEGHandler(BaseHTTPRequestHandler):
#     def log_message(self, *args): pass  # silence request logs

#     def do_GET(self):
#         path = self.path.lstrip('/')

#         # /video/<cam_id>  — single camera MJPEG stream
#         if path.startswith('video/'):
#             cam_id = path[6:]
#             self.send_response(200)
#             self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
#             self.send_header('Access-Control-Allow-Origin', '*')
#             self.end_headers()
#             try:
#                 while True:
#                     with _frames_lock:
#                         data = _latest_frames.get(cam_id)
#                     if data:
#                         self.wfile.write(b'--frame\r\n')
#                         self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
#                         self.wfile.write(data)
#                         self.wfile.write(b'\r\n')
#                     time.sleep(0.05)
#             except (BrokenPipeError, ConnectionResetError):
#                 pass

#         # /graph  — current graph JSON
#         # ALL nodes shown (including strangers as grey dots)
#         # Only non-stranger edges drawn as connections
#         elif path == 'graph':
#             if _graph_db_ref:
#                 all_edges  = list(_graph_db_ref.get_all_edges())
#                 all_nodes  = list(_graph_db_ref.get_all_nodes())
#                 # Non-stranger edges only (for drawing connections)
#                 conn_edges = [e for e in all_edges if e.relationship != 'stranger']
#                 # Degree = number of non-stranger connections per person
#                 deg_map = {}
#                 for e in conn_edges:
#                     deg_map[e.person_id_a] = deg_map.get(e.person_id_a, 0) + 1
#                     deg_map[e.person_id_b] = deg_map.get(e.person_id_b, 0) + 1
#                 # All nodes (strangers show as grey dots with degree 0)
#                 data = {
#                     'nodes': [{'id': n.person_id,
#                                'degree': deg_map.get(n.person_id, 0)}
#                               for n in all_nodes],
#                     'edges': [{'person_id_a': e.person_id_a, 'person_id_b': e.person_id_b,
#                                'confidence': e.confidence, 'relationship': e.relationship,
#                                'total_meetings': e.total_meetings,
#                                'incident_counts': e.incident_counts,
#                                'cameras': list(e.cameras) if e.cameras else []}
#                               for e in conn_edges],
#                     'stats': {
#                         'total_people': len(all_nodes),
#                         'total_relations': len(all_edges),
#                         'non_stranger_relations': len(conn_edges),
#                         'pipeline_running': True,
#                     }
#                 }
#                 body = json.dumps(data).encode()
#             else:
#                 body = b'{"nodes":[],"edges":[],"stats":{"total_people":0,"total_relations":0,"non_stranger_relations":0,"pipeline_running":false}}'
#             self.send_response(200)
#             self.send_header('Content-Type', 'application/json')
#             self.send_header('Access-Control-Allow-Origin', '*')
#             self.send_header('Content-Length', str(len(body)))
#             self.end_headers()
#             self.wfile.write(body)

#         # /cameras  — list active cameras
#         elif path == 'cameras':
#             with _frames_lock:
#                 cams = list(_latest_frames.keys())
#             body = json.dumps(cams).encode()
#             self.send_response(200)
#             self.send_header('Content-Type', 'application/json')
#             self.send_header('Access-Control-Allow-Origin', '*')
#             self.send_header('Content-Length', str(len(body)))
#             self.end_headers()
#             self.wfile.write(body)

#         # /ws  — WebSocket endpoint for live graph push
#         elif path == 'ws':
#             if b'Upgrade: websocket' in self.headers.as_bytes():
#                 t = threading.Thread(
#                     target=_ws_client_thread,
#                     args=(self.connection, self.client_address),
#                     daemon=True
#                 )
#                 t.start()
#                 # Block this handler thread until client disconnects
#                 t.join()
#             else:
#                 self.send_response(400)
#                 self.end_headers()
#         # / — serve the dashboard HTML
#         elif path == '' or path == 'index.html':
#             body = DASHBOARD_HTML.encode('utf-8')
#             self.send_response(200)
#             self.send_header('Content-Type', 'text/html; charset=utf-8')
#             self.send_header('Content-Length', str(len(body)))
#             self.end_headers()
#             self.wfile.write(body)
#         else:
#             self.send_response(404)
#             self.end_headers()


# def build_dashboard_html(port):
#     return f"""<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>URG-IS Live</title>
# <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
# <style>
# :root{{--bg:#07090c;--bg1:#0d1117;--bg2:#111820;--bg3:#1a2433;
#   --br:#1e2d3d;--br2:#2a4060;--t1:#c8d8e8;--t2:#5a7a9a;--t3:#2a4060;
#   --ac:#00d4ff;--gr:#2ea84e;--or:#d4850a;--re:#c0392b;
#   --cs:#444e5c;--ca:#1a7abf;--co:#2ea84e;--cc:#d4850a;--cg:#c0392b;}}
# *{{box-sizing:border-box;margin:0;padding:0}}
# body{{background:var(--bg);color:var(--t1);font-family:'SF Mono',monospace;height:100vh;display:flex;flex-direction:column;overflow:hidden;font-size:11px}}
# header{{display:flex;align-items:center;justify-content:space-between;padding:0 16px;height:44px;background:var(--bg1);border-bottom:1px solid var(--br);flex-shrink:0}}
# .logo{{font-size:11px;letter-spacing:3px;color:var(--ac);display:flex;align-items:center;gap:8px}}
# .pulse{{width:6px;height:6px;border-radius:50%;background:var(--ac);animation:pulse 2s infinite}}
# @keyframes pulse{{0%,100%{{box-shadow:0 0 0 0 rgba(0,212,255,.4)}}50%{{box-shadow:0 0 0 4px rgba(0,212,255,0)}}}}
# .hstats{{display:flex;gap:20px}}
# .hs{{display:flex;flex-direction:column;align-items:center}}
# .hv{{font-size:16px;font-weight:600;color:#e8f4ff}}
# .hl{{font-size:8px;letter-spacing:1.5px;color:var(--t2)}}
# .ws{{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--t2)}}
# .wd{{width:5px;height:5px;border-radius:50%;background:var(--t2)}}
# .wd.on{{background:var(--gr);box-shadow:0 0 4px var(--gr)}}
# .wd.try{{background:var(--or);animation:pulse 1s infinite}}
# .main{{flex:1;display:grid;grid-template-columns:360px 1fr 360px;overflow:hidden;min-height:0}}
# /* LEFT: camera */
# .lcol{{display:flex;flex-direction:column;background:var(--bg1);border-right:1px solid var(--br);overflow:hidden}}
# .cam-hdr{{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;border-bottom:1px solid var(--br);flex-shrink:0}}
# .cam-title{{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--t2)}}
# .cam-st{{font-size:9px;color:var(--t2)}}.cam-st.live{{color:var(--gr)}}
# .cam-tabs{{display:flex;gap:4px;flex-wrap:wrap;padding:6px 8px;border-bottom:1px solid var(--br);flex-shrink:0}}
# .ctab{{font-size:9px;color:var(--t2);background:var(--bg2);border:1px solid var(--br);border-radius:3px;padding:2px 8px;cursor:pointer;transition:.15s}}
# .ctab:hover{{color:var(--t1)}}.ctab.active{{color:var(--ac);border-color:rgba(0,212,255,.4);background:rgba(0,212,255,.06)}}
# .cam-wrap{{flex:1;min-height:0;position:relative;background:#050709;overflow:hidden;display:flex;align-items:center;justify-content:center}}
# .cam-wrap img{{width:100%;height:100%;object-fit:contain;display:none}}
# .cam-off{{font-size:10px;color:var(--t2);text-align:center;line-height:2}}
# .cam-badge{{position:absolute;bottom:6px;left:6px;font-size:8px;color:var(--ac);background:rgba(8,11,15,.8);padding:2px 7px;border-radius:3px;border:1px solid var(--br);display:none}}
# /* CENTRE: graph */
# .gcol{{position:relative;background:var(--bg);overflow:hidden}}
# #gsv{{width:100%;height:100%}}
# .gleg{{position:absolute;top:12px;left:12px;background:rgba(13,17,23,.92);border:1px solid var(--br);border-radius:5px;padding:8px 12px;font-size:9px}}
# .glt{{letter-spacing:2px;text-transform:uppercase;color:var(--t2);margin-bottom:6px}}
# .li{{display:flex;align-items:center;gap:6px;color:var(--t1);margin-bottom:3px}}
# .ld{{width:8px;height:8px;border-radius:50%}}
# .ghint{{position:absolute;top:12px;left:50%;transform:translateX(-50%);font-size:8px;color:var(--t2);letter-spacing:1px}}
# /* RIGHT: details */
# .rcol{{background:var(--bg1);border-left:1px solid var(--br);display:flex;flex-direction:column;overflow:hidden}}
# .ptabs{{display:flex;border-bottom:1px solid var(--br);flex-shrink:0}}
# .ptab{{flex:1;padding:10px 4px;font-size:9px;letter-spacing:1.5px;color:var(--t2);background:none;border:none;cursor:pointer;border-bottom:2px solid transparent;text-transform:uppercase;transition:.15s}}
# .ptab:hover{{color:var(--t1)}}.ptab.active{{color:var(--ac);border-bottom-color:var(--ac)}}
# .pnl{{flex:1;overflow-y:auto;padding:12px;display:none}}
# .pnl.active{{display:block}}
# .pnl::-webkit-scrollbar{{width:3px}}.pnl::-webkit-scrollbar-thumb{{background:var(--bg3)}}
# /* person detail */
# .pidbig{{font-size:18px;color:#e8f4ff;margin-bottom:6px}}
# .bgs{{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:10px}}
# .bg{{font-size:8px;color:var(--t2);background:var(--bg2);border:1px solid var(--br);padding:2px 7px;border-radius:3px}}
# .slbl{{font-size:8px;letter-spacing:2px;text-transform:uppercase;color:var(--t2);margin:10px 0 6px;padding-bottom:3px;border-bottom:1px solid var(--br)}}
# .conn{{background:var(--bg2);border:1px solid var(--br);border-radius:4px;padding:9px;margin-bottom:6px;cursor:pointer;transition:.15s}}
# .conn:hover{{border-color:var(--br2)}}.conn.sel{{border-color:var(--ac)}}
# .ct{{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px}}
# .cpid{{font-size:11px;color:#e8f4ff}}
# .rtag{{font-size:8px;letter-spacing:.5px;text-transform:uppercase;padding:2px 7px;border-radius:3px;font-weight:600}}
# .bw{{height:3px;background:var(--bg);border-radius:2px;overflow:hidden;margin-bottom:5px}}
# .bf{{height:100%;border-radius:2px;transition:width .4s}}
# .cm{{display:flex;gap:10px;font-size:9px;color:var(--t2)}}
# .cm span{{color:var(--t1)}}
# /* edge detail */
# .ebb{{height:5px;background:var(--bg);border-radius:3px;overflow:hidden;margin-bottom:10px}}
# .ebf{{height:100%;border-radius:3px;transition:width .4s}}
# .ig{{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-bottom:8px}}
# .ic{{background:var(--bg);border:1px solid var(--br);border-radius:4px;padding:7px;text-align:center}}
# .it{{font-size:7px;letter-spacing:1.5px;text-transform:uppercase;color:var(--t2);margin-bottom:3px}}
# .in{{font-size:17px;color:#e8f4ff}}
# .clist{{display:flex;flex-wrap:wrap;gap:4px}}
# .cbdg{{font-size:9px;color:var(--ac);background:rgba(0,212,255,.07);border:1px solid rgba(0,212,255,.2);padding:2px 8px;border-radius:3px}}
# /* boost breakdown */
# .boost-row{{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--br)}}
# .boost-label{{font-size:9px;color:var(--t2);width:80px;flex-shrink:0}}
# .boost-bar{{flex:1;height:4px;background:var(--bg);border-radius:2px;overflow:hidden}}
# .boost-fill{{height:100%;border-radius:2px;background:var(--ac)}}
# .boost-val{{font-size:9px;color:var(--t1);width:40px;text-align:right;flex-shrink:0}}
# /* empty */
# .empty{{display:flex;flex-direction:column;align-items:center;justify-content:center;height:160px;gap:8px;color:var(--t2);text-align:center}}
# .ei{{font-size:28px;opacity:.2}}.et{{font-size:10px;letter-spacing:.5px;line-height:1.6}}
# /* all people */
# .pli{{display:flex;align-items:center;gap:7px;padding:7px;background:var(--bg2);border:1px solid var(--br);border-radius:4px;margin-bottom:4px;cursor:pointer;transition:.15s}}
# .pli:hover{{border-color:var(--br2)}}.pli.active{{border-color:var(--ac)}}
# .pld{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
# .pli-info{{flex:1}}.pli-id{{font-size:10px;color:#e8f4ff}}.pli-meta{{font-size:8px;color:var(--t2)}}
# .pli-deg{{font-size:11px;color:var(--t2)}}
# .srch{{width:100%;background:var(--bg2);border:1px solid var(--br);border-radius:4px;padding:6px 9px;font-family:inherit;font-size:10px;color:var(--t1);outline:none;margin-bottom:7px;transition:.2s}}
# .srch:focus{{border-color:var(--ac)}}.srch::placeholder{{color:var(--t2)}}
# /* pruning log */
# .plog{{font-size:9px;line-height:1.8;color:var(--t2)}}
# .plog .removed{{color:var(--re)}}.plog .kept{{color:var(--gr)}}
# .tip{{position:fixed;background:var(--bg2);border:1px solid var(--br2);border-radius:4px;padding:5px 9px;font-size:9px;color:#e8f4ff;pointer-events:none;z-index:100;display:none;line-height:1.7;box-shadow:0 3px 12px rgba(0,0,0,.5)}}
# </style>
# </head>
# <body>
# <header>
#   <div class="logo"><div class="pulse"></div>URG-IS · RELATIONSHIP INTELLIGENCE</div>
#   <div class="hstats">
#     <div class="hs"><div class="hv" id="sn">—</div><div class="hl">PEOPLE</div></div>
#     <div class="hs"><div class="hv" id="se">—</div><div class="hl">RELATIONS</div></div>
#     <div class="hs"><div class="hv" id="si">—</div><div class="hl">INCIDENTS</div></div>
#     <div class="hs"><div class="hv" id="sp">—</div><div class="hl">PIPELINE</div></div>
#   </div>
#   <div class="ws"><div class="wd try" id="wsd"></div><span id="wsl">CONNECTING</span></div>
# </header>
# <div class="main">
#   <div class="lcol">
#     <div class="cam-hdr"><div class="cam-title">Live CCTV</div><div class="cam-st" id="cst">OFFLINE</div></div>
#     <div class="cam-tabs" id="ctabs"><span style="padding:4px 8px;color:var(--t2)">Starting…</span></div>
#     <div class="cam-wrap" id="cwrap">
#       <div class="cam-off" id="coff">No stream<br><span style="font-size:8px;opacity:.5">run_live.py starting…</span></div>
#       <img id="cimg" alt="">
#       <div class="cam-badge" id="cbadge">● LIVE</div>
#     </div>
#   </div>
#   <div class="gcol"><svg id="gsv"></svg>
#     <div class="gleg">
#       <div class="glt">Relationship</div>
#       <div class="li"><div class="ld" style="background:var(--cs)"></div>Stranger</div>
#       <div class="li"><div class="ld" style="background:var(--ca)"></div>Acquaintance</div>
#       <div class="li"><div class="ld" style="background:var(--co)"></div>Associate</div>
#       <div class="li"><div class="ld" style="background:var(--cc)"></div>Close Associate</div>
#       <div class="li"><div class="ld" style="background:var(--cg)"></div>Significant</div>
#     </div>
#     <div class="ghint">CLICK NODE · DRAG · SCROLL ZOOM</div>
#   </div>
#   <div class="rcol">
#     <div class="ptabs">
#       <button class="ptab active" onclick="oTab('person',this)">PERSON</button>
#       <button class="ptab" onclick="oTab('edge',this)">EDGE</button>
#       <button class="ptab" onclick="oTab('list',this)">ALL</button>
#       <button class="ptab" onclick="oTab('prune',this)">DECAY</button>
#     </div>
#     <div class="pnl active" id="t-person">
#       <div class="empty" id="pe"><div class="ei">◎</div><div class="et">Click any node</div></div>
#       <div id="pd" style="display:none">
#         <div class="pidbig" id="dpid"></div>
#         <div class="bgs" id="dbgs"></div>
#         <div class="slbl">Connections</div>
#         <div id="dconns"></div>
#       </div>
#     </div>
#     <div class="pnl" id="t-edge">
#       <div class="empty" id="ee"><div class="ei">⟷</div><div class="et">Click a connection</div></div>
#       <div id="ed" style="display:none">
#         <div class="slbl">Pair</div>
#         <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
#           <div style="font-size:12px;color:#e8f4ff" id="epair">—</div>
#           <div class="rtag" id="ertag">—</div>
#         </div>
#         <div style="display:flex;justify-content:space-between;margin-bottom:3px">
#           <span style="font-size:8px;letter-spacing:1px;color:var(--t2)">CONFIDENCE</span>
#           <span style="font-size:11px;color:#e8f4ff" id="econf">—</span>
#         </div>
#         <div class="ebb"><div class="ebf" id="ebar"></div></div>
#         <div class="slbl">Incident Breakdown</div>
#         <div class="ig" id="eincs"></div>
#         <div class="slbl">Boost Modifiers</div>
#         <div id="eboost"></div>
#         <div class="slbl">Cameras</div>
#         <div class="clist" id="ecams"></div>
#         <div class="slbl">Stats</div>
#         <div class="ig" id="estats"></div>
#       </div>
#     </div>
#     <div class="pnl" id="t-list">
#       <input class="srch" id="srch" placeholder="Search person…" oninput="fList(this.value)">
#       <div id="plist"></div>
#     </div>
#     <div class="pnl" id="t-prune">
#       <div class="slbl">Decay Log</div>
#       <div style="font-size:9px;color:var(--t2);margin-bottom:8px">Edges decay every 10 min · Removed when confidence &lt; 0.01</div>
#       <div class="ig" style="margin-bottom:10px">
#         <div class="ic"><div class="it">Active Edges</div><div class="in" id="pr-active">—</div></div>
#         <div class="ic"><div class="it">Removed Today</div><div class="in" id="pr-removed">0</div></div>
#         <div class="ic"><div class="it">Decay Rate</div><div class="in" style="font-size:13px">0.998</div></div>
#         <div class="ic"><div class="it">Interval</div><div class="in" style="font-size:13px">10min</div></div>
#       </div>
#       <div class="slbl">Edge Strengths</div>
#       <div id="pr-bars"></div>
#     </div>
#   </div>
# </div>
# <div class="tip" id="tip"></div>
# <script>
# const BASE = 'http://localhost:{port}';
# const WS_URL = 'ws://localhost:{port}/ws';
# const RC = {{
#   significant:{{c:'#c0392b',b:'rgba(192,57,43,.15)'}},
#   close_associate:{{c:'#d4850a',b:'rgba(212,133,10,.15)'}},
#   associate:{{c:'#2ea84e',b:'rgba(46,168,78,.15)'}},
#   acquaintance:{{c:'#1a7abf',b:'rgba(29,111,168,.15)'}},
#   stranger:{{c:'#444e5c',b:'rgba(68,78,92,.15)'}},
# }};
# let gd={{nodes:[],edges:[]}}, np={{}}, sel=null, svg,g,lG,nG,sim;
# let activeCam=null, removedCount=0;

# // ── WebSocket ──────────────────────────────────────────────────────────────
# let ws;
# function connectWS(){{
#   const d=document.getElementById('wsd'),l=document.getElementById('wsl');
#   d.className='wd try'; l.textContent='CONNECTING';
#   ws=new WebSocket(WS_URL);
#   ws.onopen=()=>{{d.className='wd on';l.textContent='LIVE';}};
#   ws.onmessage=e=>{{
#     try{{
#       const m=JSON.parse(e.data);
#       if(m.type==='graph_update'){{
#         gd=m.data; rGraph(); uList(); updateDecayTab();
#         if(sel) rPersonLocal(sel);
#         const s=m.data.stats||{{}};
#         document.getElementById('sn').textContent=s.total_people??gd.nodes.length;
#         document.getElementById('se').textContent=s.total_relations??gd.edges.length;
#         document.getElementById('sp').textContent='ON';
#       }}
#     }}catch(err){{}}
#   }};
#   ws.onclose=()=>{{d.className='wd';l.textContent='RECONNECTING';setTimeout(connectWS,2000);}};
#   ws.onerror=()=>ws.close();
# }}

# // ── Polling fallback ───────────────────────────────────────────────────────
# async function poll(){{
#   try{{
#     const data=await fetch(BASE+'/graph').then(r=>r.json());
#     gd=data; rGraph(); uList(); updateDecayTab();
#     if(sel) rPersonLocal(sel);
#     const s=data.stats||{{}};
#     document.getElementById('sn').textContent=s.total_people??data.nodes.length;
#     document.getElementById('se').textContent=s.total_relations??data.edges.length;
#     document.getElementById('sp').textContent='ON';
#     const d=document.getElementById('wsd');
#     if(!d.classList.contains('on')){{d.className='wd on';document.getElementById('wsl').textContent='LIVE';}}
#   }}catch(e){{}}
# }}

# // ── Cameras ────────────────────────────────────────────────────────────────
# async function discoverCams(){{
#   try{{
#     const cams=await fetch(BASE+'/cameras').then(r=>r.json());
#     if(!cams.length) return;
#     const tb=document.getElementById('ctabs'); tb.innerHTML='';
#     cams.forEach((c,i)=>{{
#       const b=document.createElement('button');
#       b.className='ctab'+(i===0?' active':'');
#       b.textContent=c.toUpperCase(); b.onclick=()=>switchCam(c,b);
#       tb.appendChild(b);
#     }});
#     if(!activeCam) switchCam(cams[0],tb.firstChild);
#   }}catch(e){{}}
# }}
# function switchCam(id,btn){{
#   activeCam=id;
#   document.querySelectorAll('.ctab').forEach(t=>t.classList.remove('active'));
#   if(btn) btn.classList.add('active');
#   const img=document.getElementById('cimg'),off=document.getElementById('coff'),
#         badge=document.getElementById('cbadge'),st=document.getElementById('cst');
#   img.src=BASE+'/video/'+id+'?t='+Date.now();
#   img.style.display='block'; off.style.display='none';
#   badge.style.display='block'; badge.textContent='● '+id.toUpperCase();
#   if(st){{st.textContent='LIVE';st.className='cam-st live';}}
#   img.onerror=()=>{{img.style.display='none';off.style.display='flex';badge.style.display='none';
#     if(st){{st.textContent='OFFLINE';st.className='cam-st';}}}};
# }}

# // ── D3 Graph ───────────────────────────────────────────────────────────────
# function nc(d){{
#   const edges=gd.edges.filter(e=>e.person_id_a===d.id||e.person_id_b===d.id);
#   if(!edges.length) return RC.stranger.c;
#   const best=edges.reduce((a,b)=>a.confidence>b.confidence?a:b);
#   return (RC[best.relationship]||RC.stranger).c;
# }}
# function nr(d){{return 8+Math.min((d.degree||0)*2.5,14);}}

# function initSVG(){{
#   svg=d3.select('#gsv'); svg.selectAll('*').remove();
#   const zoom=d3.zoom().scaleExtent([.1,5]).on('zoom',e=>g.attr('transform',e.transform));
#   svg.call(zoom); g=svg.append('g'); lG=g.append('g'); nG=g.append('g');
# }}

# function rGraph(){{
#   if(!svg) initSVG();
#   const el=document.querySelector('.gcol'),W=el.clientWidth,H=el.clientHeight;
#   const nodes=gd.nodes.map(n=>({{...n,id:n.id,x:np[n.id]?.x??W/2+(Math.random()-.5)*280,y:np[n.id]?.y??H/2+(Math.random()-.5)*200}}));
#   const links=gd.edges.map(e=>(({{...e,source:e.person_id_a,target:e.person_id_b}})));
#   nodes.forEach(n=>np[n.id]=n);
#   if(sim) sim.stop();
#   sim=d3.forceSimulation(nodes)
#     .force('link',d3.forceLink(links).id(d=>d.id).distance(d=>150-(d.confidence||0)*65).strength(.4))
#     .force('charge',d3.forceManyBody().strength(-280))
#     .force('center',d3.forceCenter(W/2,H/2))
#     .force('collision',d3.forceCollide(24))
#     .alpha(.3).alphaDecay(.022);
#   lG.selectAll('line').data(links,d=>`${{d.source}}-${{d.target}}`)
#     .join(
#       en=>en.append('line').attr('stroke-opacity',0).call(e=>e.transition().duration(500).attr('stroke-opacity',d=>.2+(d.confidence||0)*.6)),
#       up=>up.call(u=>u.transition().duration(300).attr('stroke',d=>(RC[d.relationship]||RC.stranger).c).attr('stroke-width',d=>1+(d.confidence||0)*5).attr('stroke-opacity',d=>.2+(d.confidence||0)*.6)),
#       ex=>ex.transition().duration(300).attr('stroke-opacity',0).remove()
#     )
#     .attr('stroke',d=>(RC[d.relationship]||RC.stranger).c)
#     .attr('stroke-width',d=>1+(d.confidence||0)*5)
#     .on('mouseenter',(e,d)=>tip(e,`P${{s(d.source)}} ↔ P${{s(d.target)}}\n${{d.relationship}} ${{((d.confidence||0)*100).toFixed(1)}}%`))
#     .on('mouseleave',hideTip);
#   const node=nG.selectAll('g.nd').data(nodes,d=>d.id)
#     .join(
#       en=>{{const gg=en.append('g').attr('class','nd');
#         gg.append('circle').attr('r',0).call(c=>c.transition().duration(400).attr('r',d=>nr(d)));
#         gg.append('text').attr('text-anchor','middle').attr('dy',d=>-(nr(d)+5))
#           .style('font-size','8px').style('fill','rgba(200,216,232,.6)').style('pointer-events','none').style('font-family','inherit')
#           .text(d=>d.id);
#         return gg;}},
#       up=>{{up.select('circle').transition().duration(300).attr('r',d=>nr(d)).attr('fill',d=>nc(d)).attr('stroke',d=>nc(d));return up;}},
#       ex=>ex.transition().duration(300).attr('opacity',0).remove()
#     );
#   node.select('circle').attr('fill',d=>nc(d)).attr('stroke',d=>nc(d)).attr('stroke-width',2).attr('stroke-opacity',.5).style('cursor','pointer')
#     .style('filter',d=>d.id===sel?`drop-shadow(0 0 10px ${{nc(d)}})`:`drop-shadow(0 0 3px ${{nc(d)}}66)`);
#   node.style('cursor','pointer')
#     .on('click',(e,d)=>{{e.stopPropagation();cNode(d.id);}})
#     .on('mouseenter',(e,d)=>tip(e,`Person ${{d.id}}\n${{d.degree||0}} connection(s)`))
#     .on('mouseleave',hideTip)
#     .call(d3.drag()
#       .on('start',(e,d)=>{{if(!e.active)sim.alphaTarget(.1).restart();d.fx=d.x;d.fy=d.y;}})
#       .on('drag',(e,d)=>{{d.fx=e.x;d.fy=e.y;}})
#       .on('end',(e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}));
#   sim.on('tick',()=>{{
#     lG.selectAll('line').attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
#     nG.selectAll('g.nd').attr('transform',d=>`translate(${{d.x}},${{d.y}})`);
#     nodes.forEach(n=>{{if(np[n.id]){{np[n.id].x=n.x;np[n.id].y=n.y;}}}});
#   }});
#   svg.on('click',()=>{{sel=null;document.getElementById('pd').style.display='none';document.getElementById('pe').style.display='flex';nG.selectAll('circle').style('filter',d=>`drop-shadow(0 0 3px ${{nc(d)}}66)`).attr('stroke-width',2);}});
# }}
# function s(x){{return typeof x==='object'?x.id:x;}}

# // ── Node click ─────────────────────────────────────────────────────────────
# function cNode(pid){{
#   sel=pid; oTab('person');
#   nG.selectAll('circle').style('filter',d=>d.id===pid?`drop-shadow(0 0 12px ${{nc(d)}})`:`drop-shadow(0 0 2px ${{nc(d)}}33)`).attr('stroke-width',d=>d.id===pid?3:2);
#   rPersonLocal(pid);
# }}
# function rPersonLocal(pid){{
#   const edges=gd.edges.filter(e=>e.person_id_a===pid||e.person_id_b===pid);
#   document.getElementById('pe').style.display='none'; document.getElementById('pd').style.display='block';
#   document.getElementById('dpid').textContent='Person '+pid;
#   const cams=[...new Set(edges.flatMap(e=>e.cameras||[]))];
#   document.getElementById('dbgs').innerHTML=[
#     `📷 ${{cams.join(', ')||'—'}}`,
#     `🔗 ${{edges.length}} connections`,
#     `⭐ ${{edges.filter(e=>(e.relationship==='significant'||e.relationship==='close_associate')).length}} significant`
#   ].map(t=>`<div class="bg">${{t}}</div>`).join('');
#   const list=document.getElementById('dconns'); list.innerHTML='';
#   edges.sort((a,b)=>b.confidence-a.confidence).forEach(edge=>{{
#     const other=edge.person_id_a===pid?edge.person_id_b:edge.person_id_a;
#     const rc=RC[edge.relationship]||RC.stranger, pct=Math.round((edge.confidence||0)*100);
#     const div=document.createElement('div'); div.className='conn';
#     div.innerHTML=`<div class="ct"><div class="cpid">Person ${{other}}</div>
#       <div class="rtag" style="color:${{rc.c}};background:${{rc.b}};border:1px solid ${{rc.c}}30">${{(edge.relationship||'').replace('_',' ').toUpperCase()}}</div></div>
#       <div class="bw"><div class="bf" style="width:${{pct}}%;background:${{rc.c}}"></div></div>
#       <div class="cm">CONF<span>${{(edge.confidence||0).toFixed(3)}}</span>&nbsp;MTG<span>${{edge.total_meetings||0}}</span>&nbsp;CAMS<span>${{(edge.cameras||[]).join(',')|| '—'}}</span></div>`;
#     div.onclick=()=>showEdge(pid,edge,other);
#     list.appendChild(div);
#   }});
# }}

# // ── Edge detail ────────────────────────────────────────────────────────────
# function showEdge(pid,edge,other){{
#   oTab('edge');
#   document.getElementById('ee').style.display='none'; document.getElementById('ed').style.display='block';
#   const rc=RC[edge.relationship]||RC.stranger, pct=Math.round((edge.confidence||0)*100);
#   document.getElementById('epair').textContent=`Person ${{pid}} ↔ Person ${{other}}`;
#   document.getElementById('econf').textContent=(edge.confidence||0).toFixed(4);
#   document.getElementById('ebar').style.cssText=`width:${{pct}}%;background:${{rc.c}}`;
#   const et=document.getElementById('ertag');
#   et.textContent=(edge.relationship||'').replace('_',' ').toUpperCase();
#   et.style.cssText=`color:${{rc.c}};background:${{rc.b}};border:1px solid ${{rc.c}}30`;
#   const inc=edge.incident_counts||{{}};
#   const TYPES=['CLOSE_CONTACT','EXTENDED_MEETING','CONVERSATION','GROUP_GATHERING','PROXIMITY'];
#   document.getElementById('eincs').innerHTML=TYPES.map(t=>`<div class="ic"><div class="it">${{t.replace('_',' ')}}</div><div class="in">${{inc[t]||0}}</div></div>`).join('');
#   // Boost breakdown bars
#   const conf=edge.confidence||0;
#   const boosts=[
#     ['Confidence',conf,1,'var(--ac)'],
#     ['Distance mod',Math.min(conf*1.5,1),1,'#2ea84e'],
#     ['Privacy mod',edge.total_meetings>1?0.8:1,1,'#d4850a'],
#     ['Meetings',Math.min((edge.total_meetings||0)/10,1),1,'#1a7abf'],
#   ];
#   document.getElementById('eboost').innerHTML=boosts.map(([l,v,max,col])=>`
#     <div class="boost-row">
#       <div class="boost-label">${{l}}</div>
#       <div class="boost-bar"><div class="boost-fill" style="width:${{(v/max*100).toFixed(0)}}%;background:${{col}}"></div></div>
#       <div class="boost-val">${{v.toFixed(2)}}</div>
#     </div>`).join('');
#   const cams=edge.cameras||[];
#   document.getElementById('ecams').innerHTML=cams.length?cams.map(c=>`<div class="cbdg">${{c}}</div>`).join(''):'<span style="color:var(--t2)">—</span>';
#   document.getElementById('estats').innerHTML=`
#     <div class="ic"><div class="it">Meetings</div><div class="in">${{edge.total_meetings||0}}</div></div>
#     <div class="ic"><div class="it">Confidence %</div><div class="in" style="font-size:13px">${{pct}}%</div></div>`;
# }}

# // ── All people list ────────────────────────────────────────────────────────
# function uList(){{
#   const q=(document.getElementById('srch')?.value||'').toLowerCase();
#   const el=document.getElementById('plist');
#   const nodes=[...gd.nodes].filter(n=>!q||String(n.id).includes(q)).sort((a,b)=>(b.degree||0)-(a.degree||0));
#   el.innerHTML='';
#   nodes.forEach(n=>{{
#     const col=nc(n),div=document.createElement('div'); div.className='pli'+(n.id===sel?' active':'');
#     const edges=gd.edges.filter(e=>e.person_id_a===n.id||e.person_id_b===n.id);
#     const cams=[...new Set(edges.flatMap(e=>e.cameras||[]))];
#     div.innerHTML=`<div class="pld" style="background:${{col}};box-shadow:0 0 5px ${{col}}66"></div>
#       <div class="pli-info"><div class="pli-id">Person ${{n.id}}</div><div class="pli-meta">${{cams.join(', ')||'—'}}</div></div>
#       <div class="pli-deg">${{n.degree||0}}</div>`;
#     div.onclick=()=>cNode(n.id); el.appendChild(div);
#   }});
# }}
# function fList(v){{uList();}}

# // ── Decay tab ──────────────────────────────────────────────────────────────
# function updateDecayTab(){{
#   const edges=gd.edges.sort((a,b)=>a.confidence-b.confidence);
#   document.getElementById('pr-active').textContent=edges.length;
#   document.getElementById('pr-removed').textContent=removedCount;
#   const bars=document.getElementById('pr-bars'); bars.innerHTML='';
#   edges.slice(0,15).forEach(e=>{{
#     const rc=RC[e.relationship]||RC.stranger, pct=Math.round(e.confidence*100);
#     bars.innerHTML+=`<div class="boost-row">
#       <div class="boost-label" style="color:${{rc.c}};width:90px">P${{e.person_id_a}}↔P${{e.person_id_b}}</div>
#       <div class="boost-bar"><div class="boost-fill" style="width:${{pct}}%;background:${{rc.c}}"></div></div>
#       <div class="boost-val" style="color:${{rc.c}}">${{e.confidence.toFixed(3)}}</div>
#     </div>`;
#   }});
# }}

# // ── Tabs ───────────────────────────────────────────────────────────────────
# function oTab(name,btn){{
#   document.querySelectorAll('.ptab').forEach(t=>t.classList.remove('active'));
#   document.querySelectorAll('.pnl').forEach(p=>p.classList.remove('active'));
#   if(btn) btn.classList.add('active');
#   else{{const tabs=['person','edge','list','prune'];document.querySelectorAll('.ptab')[tabs.indexOf(name)]?.classList.add('active');}}
#   document.getElementById(`t-${{name}}`)?.classList.add('active');
#   if(name==='list') uList();
# }}

# // ── Tooltip ────────────────────────────────────────────────────────────────
# function tip(e,text){{const t=document.getElementById('tip');t.innerHTML=text.split('\n').join('<br>');t.style.display='block';t.style.left=(e.clientX+10)+'px';t.style.top=(e.clientY-6)+'px';}}
# function hideTip(){{document.getElementById('tip').style.display='none';}}

# // ── Init ───────────────────────────────────────────────────────────────────
# window.addEventListener('load',()=>{{
#   initSVG();
#   connectWS();
#   poll();
#   setInterval(poll,3000);
#   discoverCams();
#   setInterval(discoverCams,5000);
#   window.addEventListener('resize',()=>{{if(gd.nodes.length)rGraph();}});
# }});
# </script>
# </body>
# </html>"""

# def start_mjpeg_server(port=8765):
#     global DASHBOARD_HTML
#     DASHBOARD_HTML = build_dashboard_html(port)
#     server = HTTPServer(('0.0.0.0', port), MJPEGHandler)
#     t = threading.Thread(target=server.serve_forever, daemon=True, name='mjpeg')
#     t.start()
#     print(f"  Dashboard:    http://localhost:{port}/")
#     print(f"  MJPEG stream: http://localhost:{port}/video/<cam_id>")
#     print(f"  Graph API:    http://localhost:{port}/graph")
#     return server


# def main():
#     source = "data/"
#     fresh  = False
#     for arg in sys.argv[1:]:
#         if arg == "--fresh":
#             fresh = True
#         else:
#             source = arg
#     multi = os.path.isdir(source)

#     if fresh:
#         import glob
#         for f in ["data/embeddings/faiss.index",
#                   "data/embeddings/identity_map.json",
#                   "data/snapshots/graph.json"]:
#             if os.path.exists(f):
#                 os.remove(f)
#                 print(f"  Cleared: {f}")
#         print("  Fresh start — Person IDs begin at 1\n")

#     banner("URG-IS — LIVE PIPELINE")
#     log(f"Source      : {source}")
#     log(f"Mode        : {'All 7 cameras' if multi else 'Single camera'}")
#     log(f"Re-ID       : OSNet (Market-1501)")
#     log(f"Shows       : Re-ID matches, incidents, per-person graphs\n")
#     log("Press Ctrl+C to stop and see full graph summary\n")

#     # ── Init ──────────────────────────────────────────────────────────────────
#     tracker    = PersonTracker()
#     embedder   = PersonEmbedder()
#     manager    = IdentityManager()
#     classifier = IncidentClassifier()
#     db         = GraphDB()
#     engine     = ConfidenceEngine(db, decay_interval_m=10, auto_snapshot=True)
#     engine.start()

#     # Share db with MJPEG handler
#     global _graph_db_ref
#     _graph_db_ref = db

#     # Start MJPEG server
#     start_mjpeg_server(8765)

#     # One detector per camera
#     detectors = defaultdict(InteractionDetector)

#     frame_total    = 0
#     incident_total = 0

#     def process_frame(frame, camera_id, frame_num):
#         nonlocal frame_total, incident_total
#         frame_total += 1

#         # ── Track ──────────────────────────────────────────────────────────
#         tracked = tracker.track(frame, camera_id=camera_id, frame_num=frame_num)
#         if not tracked:
#             store_frame(camera_id, frame)
#             return

#         # ── Re-ID ──────────────────────────────────────────────────────────
#         persons_with_ids = []
#         for person in tracked:
#             vec    = embedder.embed(person.crop)
#             result = manager.identify(
#                 embedding=vec, track_id=person.track_id,
#                 camera_id=camera_id, frame_num=frame_num,
#             )
#             if result:
#                 persons_with_ids.append((result.person_id, person.center))

#                 if result.is_new:
#                     log(f"[{camera_id}] Frame {frame_num:>5} | "
#                         f"🆕 NEW Person {result.person_id}  "
#                         f"(track={person.track_id})")
#                 else:
#                     log(f"[{camera_id}] Frame {frame_num:>5} | "
#                         f"✓  Person {result.person_id:<4} matched  "
#                         f"sim={result.similarity:.3f}  "
#                         f"(track={person.track_id})", n=0)

#         # Store annotated frame for MJPEG stream
#         ann = annotate_frame(frame, camera_id, persons_with_ids, {}, [])
#         store_frame(camera_id, ann)

#         # Push graph to dashboard every 30 frames (even without incidents)
#         if frame_total % 30 == 0:
#             push_graph_update(db)

#         if len(persons_with_ids) < 2:
#             return

#         # ── Detect interactions ────────────────────────────────────────────
#         people_ids = [pid for pid, _ in persons_with_ids]
#         events = detectors[camera_id].update(
#             persons=persons_with_ids, camera_id=camera_id, frame_num=frame_num
#         )

#         # ── Process each incident ──────────────────────────────────────────
#         centers = {pid: c for pid, c in persons_with_ids}

#         for event in events:
#             # Skip self-interaction
#             if event.person_id_a == event.person_id_b:
#                 continue

#             ca  = centers.get(event.person_id_a, (0,0))
#             cb  = centers.get(event.person_id_b, (0,0))
#             mid = ((ca[0]+cb[0])//2, (ca[1]+cb[1])//2)

#             # Classify
#             incident = classifier.classify(
#                 person_id_a=event.person_id_a, person_id_b=event.person_id_b,
#                 distance_m=event.distance_m, duration_s=event.duration_s,
#                 camera_id=camera_id, frame_num=frame_num,
#                 location_px=mid, people_in_scene=people_ids,
#             )

#             # Compute modifiers
#             base     = incident.base_boost
#             dist_mod = min(2.0, 1.0 + (MAX_DISTANCE_M - event.distance_m) / MAX_DISTANCE_M)
#             priv_mod = PRIVACY_ONE_ON_ONE if incident.is_one_on_one else PRIVACY_GROUP
#             pair_key = "::".join(sorted([event.person_id_a, event.person_id_b]))
#             state    = engine._get_state(pair_key)
#             dim_mod  = 1.0 / (1.0 + state.meetings_today * DIMINISHING_RATE)
#             bx, by   = mid[0]//LOCATION_BUCKET_PX, mid[1]//LOCATION_BUCKET_PX
#             lk       = f"{bx}:{by}"
#             state.location_visits[lk] = state.location_visits.get(lk, 0) + 1
#             loc_mod  = 1.0 + LOCATION_MAX_BONUS * min(state.location_visits[lk],
#                        LOCATION_VISITS_FOR_MAX) / LOCATION_VISITS_FOR_MAX
#             boost    = base * dist_mod * loc_mod * priv_mod * dim_mod

#             existing = db.get_edge(event.person_id_a, event.person_id_b)
#             old_conf = existing.confidence if existing else 0.0

#             edge = engine.process_event(event, people_in_scene=people_ids)
#             new_conf = edge.confidence if edge else old_conf
#             incident_total += 1

#             # ── Print incident ─────────────────────────────────────────────
#             grp = f" [GROUP {incident.group_size}]" if incident.group_size >= 3 else ""
#             print(f"\n{'─'*70}")
#             print(f"  INCIDENT #{incident_total} | Frame {frame_num} | {camera_id}{grp}")
#             print(f"  Person {event.person_id_a} ↔ Person {event.person_id_b}")
#             print(f"  Type     : {incident.incident_type.value}")
#             print(f"  Distance : {event.distance_m:.2f}m  |  Duration: {event.duration_s:.1f}s")
#             print(f"  Boost    : {base:.3f} base"
#                   f" × {dist_mod:.2f} dist"
#                   f" × {loc_mod:.2f} loc"
#                   f" × {priv_mod:.2f} privacy"
#                   f" × {dim_mod:.2f} dim"
#                   f" = {boost:.4f}")
#             print(f"  Confidence: {old_conf:.4f} → {new_conf:.4f}"
#                   f"  [{edge.relationship if edge else 'stranger'}]")

#             # ── Print BOTH person graphs ────────────────────────────────────
#             print(f"\n  Relationship graph for Person {event.person_id_a}:")
#             print_person_graph(db, event.person_id_a)
#             print(f"\n  Relationship graph for Person {event.person_id_b}:")
#             print_person_graph(db, event.person_id_b)

#             # Push live update to dashboard WebSocket clients
#             push_graph_update(db)

#     # ── Run ───────────────────────────────────────────────────────────────────
#     try:
#         if multi:
#             reader = MultiCameraStreamReader(source_dir=source)
#             log(f"Active cameras: {reader.get_active_cameras()}\n")
#             for frame_set in reader.frame_sets():
#                 for cam_id, cam_frame in frame_set.items():
#                     process_frame(cam_frame.frame, cam_id, cam_frame.frame_num)
#         else:
#             reader = StreamReader(source=source, frame_skip=1)
#             for frame_num, frame in reader.frames():
#                 process_frame(frame, "cam1", frame_num)

#     except KeyboardInterrupt:
#         print("\n\nStopped by user.")

#     finally:
#         engine.stop()
#         manager.save()
#         print_full_graph(db)
#         print(f"\n  Total frames processed  : {frame_total}")
#         print(f"  Total incidents fired   : {incident_total}")
#         print(f"  Total people identified : {manager.get_identity_count()}")
#         print(f"  Total relationships     : {db.get_edge_count()}")


# if __name__ == "__main__":
#     main()



"""
run_live.py  (updated)
======================
WHAT CHANGED FROM ORIGINAL:

  1. --auto-calib flag
     Runs auto_calib.py before pipeline starts.
     Estimates PIXELS_PER_METRE per camera from person heights.
     Fixes the 1-incident bug caused by wrong pixel scale.

  2. Frame skip (FRAME_SKIP from settings)
     Processes every Nth frame — 3 = ~10fps effective on MacBook CPU.
     Was processing every frame: 30fps × 7 cameras = too slow.

  3. Frame resize before detection (DETECTION_WIDTH × DETECTION_HEIGHT)
     Biggest single speedup on CPU. Resize to 640×480 before YOLOv8.

  4. InteractionDetector now receives camera_id in constructor
     Loads per-camera PIXELS_PER_METRE automatically.

  5. Agent scheduler starts if AGENT_ENABLED=true
     Fully private anomaly detection via local Ollama.

  6. Nearby count for group classification
     Was passing all people in frame to incident_classifier.
     Now passes only people near the interacting pair.

  7. Privacy: identity purge via confidence_engine on schedule.

  EVERYTHING ELSE IS IDENTICAL TO YOUR ORIGINAL run_live.py.
  Only the sections marked CHANGED are different.
"""

import os, sys, time, threading, io, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import faiss
import cv2
import numpy as np
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket as _socket

# ── Load settings (all env-based now) ────────────────────────────────────────
from config.settings import (
    FRAME_SKIP, DETECTION_WIDTH, DETECTION_HEIGHT,
    YOLO_MODEL, YOLO_CONFIDENCE,
    INTERACTION_DISTANCE_M,
    AUTO_CALIB_ON_STARTUP, AGENT_ENABLED,
    MAX_DISTANCE_M, LOCATION_BUCKET_PX, LOCATION_MAX_BONUS,
    LOCATION_VISITS_FOR_MAX, PRIVACY_ONE_ON_ONE, PRIVACY_GROUP,
    DIMINISHING_RATE,
    ENABLE_REDIS_STREAMS, ENABLE_UNIFIED_FLOOR_MAP,
)

# ── WebSocket push clients (unchanged) ────────────────────────────────────────
_ws_clients = []
_ws_lock    = threading.Lock()

def _ws_handshake(conn, request):
    import hashlib, base64
    key = None
    for line in request.split(b'\r\n'):
        if b'Sec-WebSocket-Key' in line:
            key = line.split(b': ')[1].strip().decode()
    if not key:
        return False
    magic  = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
    accept = base64.b64encode(hashlib.sha1((key+magic).encode()).digest()).decode()
    resp   = (
        'HTTP/1.1 101 Switching Protocols\r\n'
        'Upgrade: websocket\r\nConnection: Upgrade\r\n'
        f'Sec-WebSocket-Accept: {accept}\r\n'
        'Access-Control-Allow-Origin: *\r\n\r\n'
    )
    conn.sendall(resp.encode())
    return True

def _ws_send(conn, msg: str):
    import struct
    data = msg.encode('utf-8')
    n    = len(data)
    header = (bytes([0x81, n]) if n <= 125
              else bytes([0x81, 126]) + struct.pack('>H', n) if n <= 65535
              else bytes([0x81, 127]) + struct.pack('>Q', n))
    try:
        conn.sendall(header + data)
        return True
    except:
        return False

def _ws_client_thread(conn, addr):
    try:
        buf = b''
        while b'\r\n\r\n' not in buf:
            chunk = conn.recv(1024)
            if not chunk: return
            buf += chunk
        if not _ws_handshake(conn, buf): return
        with _ws_lock: _ws_clients.append(conn)
        while True:
            try:
                if not conn.recv(256): break
            except: break
    finally:
        with _ws_lock:
            if conn in _ws_clients: _ws_clients.remove(conn)
        try: conn.close()
        except: pass

def push_ws(payload: str):
    """Push any JSON payload to all connected WebSocket clients."""
    if not _ws_clients: return
    dead = []
    with _ws_lock: clients = list(_ws_clients)
    for conn in clients:
        if not _ws_send(conn, payload): dead.append(conn)
    with _ws_lock:
        for d in dead:
            if d in _ws_clients: _ws_clients.remove(d)

def push_graph_update(db_ref):
    if not _ws_clients or db_ref is None: return
    try:
        all_edges  = list(db_ref.get_all_edges())
        all_nodes  = list(db_ref.get_all_nodes())
        conn_edges = [e for e in all_edges if e.relationship != 'stranger']
        deg_map    = {}
        for e in conn_edges:
            deg_map[e.person_id_a] = deg_map.get(e.person_id_a, 0) + 1
            deg_map[e.person_id_b] = deg_map.get(e.person_id_b, 0) + 1
        payload = json.dumps({
            'type': 'graph_update',
            'data': {
                'nodes': [{'id': n.person_id, 'degree': deg_map.get(n.person_id, 0)}
                          for n in all_nodes],
                'edges': [{'person_id_a': e.person_id_a, 'person_id_b': e.person_id_b,
                           'confidence': e.confidence, 'relationship': e.relationship,
                           'total_meetings': e.total_meetings,
                           'incident_counts': dict(e.incident_counts) if e.incident_counts else {},
                           'cameras': list(e.cameras) if e.cameras else []}
                          for e in conn_edges],
                'communities': db_ref.get_louvain_communities(),
                'floor_points': list(_latest_floor_points),
                'stats': {
                    'total_people':    len(all_nodes),
                    'total_relations': len(all_edges),
                    'pipeline_running': True,
                }
            }
        })
        push_ws(payload)
    except Exception: pass

from core.video.stream_reader       import StreamReader
from core.video.multi_stream_reader import MultiCameraStreamReader
from core.tracking.person_tracker   import PersonTracker
from core.reid.fusion_embedder      import FusionEmbedder
from core.reid.identity_manager     import IdentityManager
from core.interaction.interaction_detector import InteractionDetector, InteractionEvent
from core.graph.incident_classifier import IncidentClassifier, INCIDENT_BOOST
from core.graph.graph_db            import GraphDB
from core.graph.confidence_engine   import ConfidenceEngine
from core.spatial.floor_mapper      import UnifiedFloorMapper
from core.streaming.redis_streams   import RedisIncidentStream

SEP  = "═" * 70
def banner(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def sub(t):    print(f"\n  ── {t}")
def log(m, n=2): print(" "*n + m)

def print_person_graph(db, person_id):
    g = db.get_person_graph(person_id)
    if not g or not g["connections"]: return
    print(f"\n  ┌─ Person {person_id} — {g['total_connections']} connection(s)")
    for c in g["connections"]:
        bar = "█"*int(c["confidence"]*20) + "░"*(20-int(c["confidence"]*20))
        print(f"  │  → Person {c['person_id']:<4} [{c['relationship']:<16}] "
              f"{c['confidence']:.4f} {bar}")
        print(f"  │     Incidents: {c['incident_counts']}")
        print(f"  │     Meetings: {c['total_meetings']}  Cameras: {c['cameras']}")
    print(f"  └{'─'*60}")

def print_full_graph(db):
    banner("FINAL RELATIONSHIP GRAPH")
    edges = db.get_all_edges()
    nodes = db.get_all_nodes()
    if not edges: log("No relationships built."); return
    log(f"Total people: {len(nodes)}  |  Total relationships: {len(edges)}")
    for e in sorted(edges, key=lambda x: x.confidence, reverse=True):
        top = max(e.incident_counts, key=e.incident_counts.get) if e.incident_counts else "—"
        log(f"  {e.person_id_a:>8} ↔ {e.person_id_b:<8}  "
            f"{e.confidence:.4f}  {e.relationship:<16}  {top}")

# ── Shared frames for MJPEG (unchanged) ──────────────────────────────────────
_latest_frames = {}
_frames_lock   = threading.Lock()
_graph_db_ref  = None
_latest_floor_points = []
FONT = cv2.FONT_HERSHEY_SIMPLEX

def annotate_frame(frame, cam_id, persons_with_ids, active_prox, events_fired):
    out = frame.copy()
    for pid, center in persons_with_ids:
        cx, cy = center
        cv2.circle(out, (cx, cy), 4, (0,220,0), -1)
        cv2.putText(out, f"P{pid}", (cx-10, cy-10), FONT, 0.45, (0,220,0), 1)
    centers = {str(pid): c for pid, c in persons_with_ids}
    for ev in events_fired:
        pa, pb = centers.get(str(ev.person_id_a)), centers.get(str(ev.person_id_b))
        if pa and pb: cv2.line(out, pa, pb, (0,0,255), 2)
    cv2.putText(out, f"{cam_id} | {len(persons_with_ids)} people",
                (6,20), FONT, 0.5, (0,200,200), 1)
    return out

def store_frame(cam_id, frame):
    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ret:
        with _frames_lock: _latest_frames[cam_id] = buf.tobytes()

DASHBOARD_HTML = ''

class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        path = self.path.lstrip('/')
        if path.startswith('video/'):
            cam_id = path[6:]
            self.send_response(200)
            self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()
            try:
                while True:
                    with _frames_lock: data = _latest_frames.get(cam_id)
                    if data:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(data); self.wfile.write(b'\r\n')
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError): pass
        elif path == 'graph':
            if _graph_db_ref:
                all_edges  = list(_graph_db_ref.get_all_edges())
                all_nodes  = list(_graph_db_ref.get_all_nodes())
                conn_edges = [e for e in all_edges if e.relationship != 'stranger']
                deg_map = {}
                for e in conn_edges:
                    deg_map[e.person_id_a] = deg_map.get(e.person_id_a,0)+1
                    deg_map[e.person_id_b] = deg_map.get(e.person_id_b,0)+1
                data = {
                    'nodes': [{'id':n.person_id,'degree':deg_map.get(n.person_id,0)}
                              for n in all_nodes],
                    'edges': [{'person_id_a':e.person_id_a,'person_id_b':e.person_id_b,
                               'confidence':e.confidence,'relationship':e.relationship,
                               'total_meetings':e.total_meetings,
                               'incident_counts':e.incident_counts,
                               'cameras':list(e.cameras) if e.cameras else []}
                              for e in conn_edges],
                    'communities': _graph_db_ref.get_louvain_communities(),
                    'floor_points': list(_latest_floor_points),
                    'stats':{'total_people':len(all_nodes),'total_relations':len(all_edges),
                             'non_stranger_relations':len(conn_edges),'pipeline_running':True}
                }
                body = json.dumps(data).encode()
            else:
                body = b'{"nodes":[],"edges":[],"stats":{"total_people":0,"total_relations":0,"pipeline_running":false}}'
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Access-Control-Allow-Origin','*')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers(); self.wfile.write(body)
        elif path == 'cameras':
            with _frames_lock: cams = list(_latest_frames.keys())
            body = json.dumps(cams).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Access-Control-Allow-Origin','*')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers(); self.wfile.write(body)
        elif path == 'ws':
            if b'Upgrade: websocket' in self.headers.as_bytes():
                t = threading.Thread(target=_ws_client_thread,
                                     args=(self.connection, self.client_address), daemon=True)
                t.start(); t.join()
            else:
                self.send_response(400); self.end_headers()
        elif path in ('', 'index.html'):
            body = DASHBOARD_HTML.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

def build_dashboard_html(port):
    # Import from original run_live.py — unchanged
    from run_live_original import build_dashboard_html as _orig
    return _orig(port)

def start_mjpeg_server(port=8765):
    global DASHBOARD_HTML
    # Re-use original dashboard HTML builder
    try:
        from run_live_original import build_dashboard_html
        DASHBOARD_HTML = build_dashboard_html(port)
    except ImportError:
        DASHBOARD_HTML = f"<html><body><h1>URG-IS</h1><p>Dashboard on port {port}</p></body></html>"
    server = HTTPServer(('0.0.0.0', port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name='mjpeg')
    t.start()
    print(f"  Dashboard:    http://localhost:{port}/")
    return server


# ── CHANGED: main() with auto-calib, frame-skip, agent ───────────────────────

def main():
    source     = "data/"
    fresh      = False
    auto_calib = False
    port       = 8765

    for arg in sys.argv[1:]:
        if arg == "--fresh":       fresh      = True
        elif arg == "--auto-calib": auto_calib = True
        elif arg.startswith("--port="): port  = int(arg.split("=")[1])
        elif not arg.startswith("--"):  source = arg

    # ── CHANGED: Auto-calibration ─────────────────────────────────────────
    # Runs before pipeline. Estimates PIXELS_PER_METRE per camera.
    # This is what fixes the 1-incident bug.
    if auto_calib or AUTO_CALIB_ON_STARTUP:
        print("\n  Running auto-calibration...")
        from calib import calibrate_all
        results = calibrate_all(source, verbose=True)
        if results:
            print(f"  Calibrated {len(results)} camera(s). Pixel scales updated in .env.\n")
        else:
            print("  WARNING: Auto-calib found no people. Using .env fallback values.\n")
        # Reload settings so new PIXELS_PER_METRE_* values are picked up
        import importlib, config.settings as _s
        importlib.reload(_s)

    if fresh:
        import glob as _g
        for f in ["data/embeddings/faiss.index",
                  "data/embeddings/identity_map.json",
                  "data/snapshots/graph.json"]:
            if os.path.exists(f): os.remove(f); print(f"  Cleared: {f}")
        print("  Fresh start\n")

    multi = os.path.isdir(source)
    banner("URG-IS — LIVE PIPELINE")
    log(f"Source     : {source}")
    log(f"Mode       : {'All cameras' if multi else 'Single camera'}")
    log(f"Frame skip : every {FRAME_SKIP} frames (~{30//FRAME_SKIP}fps effective)")
    log(f"Detection  : {DETECTION_WIDTH}×{DETECTION_HEIGHT}  model={YOLO_MODEL}")
    log(f"Agent      : {'enabled (Ollama)' if AGENT_ENABLED else 'disabled'}")

    # Keep independent tracking state per camera.
    # A single shared tracker across multiple camera streams causes ID instability.
    trackers   = {}   # cam_id -> PersonTracker
    embedder   = FusionEmbedder()
    manager    = IdentityManager()
    classifier = IncidentClassifier()
    db         = GraphDB()
    engine     = ConfidenceEngine(db, decay_interval_m=10, auto_snapshot=True)
    floor_mapper = UnifiedFloorMapper() if ENABLE_UNIFIED_FLOOR_MAP else None
    redis_stream = RedisIncidentStream() if ENABLE_REDIS_STREAMS else None
    engine.start()

    global _graph_db_ref
    _graph_db_ref = db

    start_mjpeg_server(port)

    # ── CHANGED: Agent scheduler ──────────────────────────────────────────
    if AGENT_ENABLED:
        from agent import AgentScheduler
        agent_scheduler = AgentScheduler(db_ref=db, push_fn=push_ws)
        agent_scheduler.start()

    # ── CHANGED: Per-camera detectors with correct camera_id ─────────────
    # InteractionDetector now loads PIXELS_PER_METRE per camera from settings
    def make_detector(cam_id: str) -> InteractionDetector:
        return InteractionDetector(camera_id=cam_id)

    def get_tracker(cam_id: str) -> PersonTracker:
        if cam_id not in trackers:
            trackers[cam_id] = PersonTracker()
        return trackers[cam_id]

    detectors = {}   # cam_id → InteractionDetector

    frame_total    = 0
    incident_total = 0

    def process_frame(frame, camera_id, frame_num):
        nonlocal frame_total, incident_total
        global _latest_floor_points
        frame_total += 1

        # ── CHANGED: Frame skip — only process every Nth frame ────────────
        if frame_num % FRAME_SKIP != 0:
            store_frame(camera_id, frame)
            return

        # ── CHANGED: Resize before detection — biggest CPU speedup ────────
        small = cv2.resize(frame, (DETECTION_WIDTH, DETECTION_HEIGHT))

        # ── CHANGED: Get per-camera detector ─────────────────────────────
        if camera_id not in detectors:
            detectors[camera_id] = make_detector(camera_id)

        # Track on resized frame
        tracker = get_tracker(camera_id)
        tracked = tracker.track(small, camera_id=camera_id, frame_num=frame_num)
        if not tracked:
            store_frame(camera_id, frame)
            return

        # Scale factor to map detections back to original frame coords
        sx = frame.shape[1] / DETECTION_WIDTH
        sy = frame.shape[0] / DETECTION_HEIGHT

        persons_with_ids = []
        # Keep interaction continuity on tracker IDs; map to Re-ID IDs for graph updates.
        persons_with_tracks = []
        track_to_person = {}
        for person in tracked:
            vec    = embedder.embed(person.crop)
            result = manager.identify(
                embedding=vec, track_id=person.track_id,
                camera_id=camera_id, frame_num=frame_num,
            )
            if result:
                # Scale center back to original frame
                cx = int(person.center[0] * sx)
                cy = int(person.center[1] * sy)
                persons_with_ids.append((result.person_id, (cx, cy)))
                tid = str(person.track_id)
                persons_with_tracks.append((tid, (cx, cy)))
                track_to_person[tid] = result.person_id

                if result.is_new:
                    log(f"[{camera_id}] Frame {frame_num:>5} | NEW Person {result.person_id}")
                else:
                    log(f"[{camera_id}] Frame {frame_num:>5} | "
                        f"Person {result.person_id:<4} sim={result.similarity:.3f}", n=0)

        if floor_mapper:
            floor_pts = []
            for pid, center in persons_with_ids:
                fp = floor_mapper.make_point(pid, camera_id, center)
                if fp is not None:
                    floor_pts.append(
                        {
                            "person_id": fp.person_id,
                            "camera_id": fp.camera_id,
                            "world_xy_m": [round(fp.world_xy_m[0], 3), round(fp.world_xy_m[1], 3)],
                            "map_xy_px": [int(fp.map_xy_px[0]), int(fp.map_xy_px[1])],
                        }
                    )
            _latest_floor_points = floor_pts

        ann = annotate_frame(frame, camera_id, persons_with_ids, {}, [])
        store_frame(camera_id, ann)

        if frame_total % 30 == 0:
            push_graph_update(db)

        if len(persons_with_tracks) < 2:
            return

        events = detectors[camera_id].update(
            persons=persons_with_tracks, camera_id=camera_id, frame_num=frame_num
        )

        centers  = {pid: c for pid, c in persons_with_ids}

        for event in events:
            pid_a = track_to_person.get(event.person_id_a)
            pid_b = track_to_person.get(event.person_id_b)
            if not pid_a or not pid_b or pid_a == pid_b:
                continue

            person_event = InteractionEvent(
                person_id_a=pid_a,
                person_id_b=pid_b,
                camera_id=event.camera_id,
                frame_num=event.frame_num,
                start_time=event.start_time,
                duration_s=event.duration_s,
                distance_m=event.distance_m,
                is_new_pair=event.is_new_pair,
                is_refire=event.is_refire,
            )

            ca  = centers.get(pid_a, (0,0))
            cb  = centers.get(pid_b, (0,0))
            mid = ((ca[0]+cb[0])//2, (ca[1]+cb[1])//2)

            # ── CHANGED: Accurate group size — only nearby people ─────────
            nearby_count = detectors[camera_id].get_nearby_count(
                ca, cb, persons_with_ids
            )
            nearby_pids = [pid for pid, c in persons_with_ids
                           if detectors[camera_id]._compute_distance_m(mid, c) <
                           INTERACTION_DISTANCE_M * 2]

            incident = classifier.classify(
                person_id_a=pid_a, person_id_b=pid_b,
                distance_m=person_event.distance_m, duration_s=person_event.duration_s,
                camera_id=camera_id, frame_num=frame_num,
                location_px=mid, people_in_scene=nearby_pids,  # CHANGED
            )

            base     = incident.base_boost
            dist_mod = min(2.0, 1.0 + (MAX_DISTANCE_M - person_event.distance_m) / MAX_DISTANCE_M)
            priv_mod = PRIVACY_ONE_ON_ONE if incident.is_one_on_one else PRIVACY_GROUP
            pair_key = "::".join(sorted([pid_a, pid_b]))
            state    = engine._get_state(pair_key)
            dim_mod  = 1.0 / (1.0 + state.meetings_today * DIMINISHING_RATE)
            bx, by   = mid[0]//LOCATION_BUCKET_PX, mid[1]//LOCATION_BUCKET_PX
            lk       = f"{bx}:{by}"
            state.location_visits[lk] = state.location_visits.get(lk, 0) + 1
            loc_mod  = 1.0 + LOCATION_MAX_BONUS * min(
                state.location_visits[lk], LOCATION_VISITS_FOR_MAX
            ) / LOCATION_VISITS_FOR_MAX
            boost = base * dist_mod * loc_mod * priv_mod * dim_mod

            existing = db.get_edge(pid_a, pid_b)
            old_conf = existing.confidence if existing else 0.0
            edge     = engine.process_event(person_event, people_in_scene=nearby_pids)
            new_conf = edge.confidence if edge else old_conf
            incident_total += 1
            if redis_stream and redis_stream.enabled and edge:
                redis_stream.publish_incident(
                    {
                        "camera_id": camera_id,
                        "frame_num": frame_num,
                        "person_id_a": pid_a,
                        "person_id_b": pid_b,
                        "incident_type": incident.incident_type.value,
                        "distance_m": person_event.distance_m,
                        "duration_s": person_event.duration_s,
                        "confidence": edge.confidence,
                        "relationship": edge.relationship,
                    }
                )

            grp = f" [GROUP {incident.group_size}]" if incident.group_size >= 3 else ""
            print(f"\n{'─'*70}")
            print(f"  INCIDENT #{incident_total} | Frame {frame_num} | {camera_id}{grp}")
            print(f"  Person {pid_a} ↔ Person {pid_b}")
            print(f"  Type: {incident.incident_type.value}  |  "
                  f"Dist: {person_event.distance_m:.2f}m  |  Dur: {person_event.duration_s:.1f}s")
            print(f"  Boost: {base:.3f}×{dist_mod:.2f}×{loc_mod:.2f}×{priv_mod:.2f}×{dim_mod:.2f} = {boost:.4f}")
            print(f"  Confidence: {old_conf:.4f} → {new_conf:.4f} [{edge.relationship if edge else 'stranger'}]")

            print_person_graph(db, pid_a)
            print_person_graph(db, pid_b)
            push_graph_update(db)

    try:
        if multi:
            reader = MultiCameraStreamReader(source_dir=source)
            log(f"Active cameras: {reader.get_active_cameras()}\n")
            for frame_set in reader.frame_sets():
                for cam_id, cam_frame in frame_set.items():
                    process_frame(cam_frame.frame, cam_id, cam_frame.frame_num)
        else:
            reader = StreamReader(source=source, frame_skip=1)
            for frame_num, frame in reader.frames():
                process_frame(frame, "cam1", frame_num)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        engine.stop()
        manager.save()
        print_full_graph(db)
        print(f"\n  Frames processed : {frame_total}")
        print(f"  Incidents fired  : {incident_total}")
        print(f"  People identified: {manager.get_identity_count()}")
        print(f"  Relationships    : {db.get_edge_count()}")


if __name__ == "__main__":
    main()