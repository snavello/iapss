# padel_tournament_pro_multiuser_v3_3.py — v3.3.30
# Novedades:
# - Configuración: checkbox "use_seeds" (cabezas de serie), default False.
# - Parejas: marcar exactamente N seeds (N = num_zones). Se puede marcar al dar de alta o luego.
# - Sorteo de zonas: con seeds → 1 cabeza por zona, resto al azar. Sin seeds → reparto con mínimo por zona = top_per_zone.
# - Reparto respeta casos como 14 parejas en 4 zonas con top=2 (p.ej. 4-4-4-2).
# - Pie: "Iapps Padel Tournament · iAPPs Pádel — v3.3.30".
# - Correcciones previas de f-strings y defensas de estado.

import streamlit as st
import pandas as pd
from itertools import combinations
import random
from datetime import datetime, date
import json
from pathlib import Path
import hashlib
import uuid
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO
import base64, requests

# ====== PDF opcional ======
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

st.set_page_config(page_title="Torneo de Pádel — v3.3.30", layout="wide")

# ====== Rutas/Persistencia ======
DATA_DIR = Path("data")
APP_CONFIG_PATH = DATA_DIR / "app_config.json"   # Config global (logo_url)
USERS_PATH = DATA_DIR / "users.json"
TOURN_DIR = DATA_DIR / "tournaments"
SNAP_ROOT = TOURN_DIR / "snapshots"
TOURN_INDEX = TOURN_DIR / "index.json"
KEEP_SNAPSHOTS = 20

DATA_DIR.mkdir(exist_ok=True)
TOURN_DIR.mkdir(exist_ok=True)
SNAP_ROOT.mkdir(parents=True, exist_ok=True)

sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
now_iso = lambda: datetime.now().isoformat()

# ====== Config de App global (logo por URL RAW) ======
DEFAULT_APP_CONFIG = {
    # Cambiá esta URL si subís otro logo RAW a tu repo
    "app_logo_url": "https://raw.githubusercontent.com/snavello/iapss/main/1000138052.png"
}

def load_app_config() -> Dict[str, Any]:
    if not APP_CONFIG_PATH.exists():
        APP_CONFIG_PATH.write_text(json.dumps(DEFAULT_APP_CONFIG, indent=2), encoding="utf-8")
        return DEFAULT_APP_CONFIG.copy()
    try:
        data = json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
        if "app_logo_url" not in data:
            data["app_logo_url"] = DEFAULT_APP_CONFIG["app_logo_url"]
            save_app_config(data)
        return data
    except Exception:
        return DEFAULT_APP_CONFIG.copy()

def save_app_config(cfg: Dict[str, Any]):
    APP_CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

# ====== Logo como data URI (cacheado) ======
@st.cache_data(show_spinner=False)
def fetch_image_as_data_uri(url: str, bust: str = "") -> str:
    try:
        u = url.strip()
        if not u:
            return ""
        if bust:
            sep = "&" if "?" in u else "?"
            u = f"{u}{sep}v={bust}"
        resp = requests.get(u, timeout=8)
        resp.raise_for_status()
        content = resp.content
        mime = "image/png"
        ct = resp.headers.get("Content-Type","").lower()
        if "svg" in ct: mime = "image/svg+xml"
        elif "jpeg" in ct or "jpg" in ct: mime = "image/jpeg"
        elif "webp" in ct: mime = "image/webp"
        b64 = base64.b64encode(content).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""

# ====== Branding / layout ======
PRIMARY_BLUE = "#0D47A1"
LIME_GREEN  = "#AEEA00"
DARK_BLUE   = "#082D63"

def brand_svg(width_px: int = 220) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" viewBox="0 0 660 200" role="img" aria-label="iAPPs PADEL TOURNAMENT">
  <defs>
    <linearGradient id="g1" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{PRIMARY_BLUE}" />
      <stop offset="100%" stop-color="{DARK_BLUE}" />
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="660" height="200" fill="transparent"/>
  <text x="8" y="65" font-family="Inter, Segoe UI, Roboto, Arial, sans-serif" font-weight="800"
        font-size="74" fill="url(#g1)" letter-spacing="2">iAPP</text>
  <text x="445" y="65" font-family="Inter, Segoe UI, Roboto, Arial, sans-serif" font-weight="900"
        font-size="72" fill="{LIME_GREEN}">s</text>
  <text x="8" y="125" font-family="Inter, Segoe UI, Roboto, Arial, sans-serif" font-weight="800"
        font-size="76" fill="{PRIMARY_BLUE}" letter-spacing="4">PADEL</text>
  <text x="8" y="182" font-family="Inter, Segoe UI, Roboto, Arial, sans-serif" font-weight="700"
        font-size="58" fill="{PRIMARY_BLUE}" letter-spacing="6">TOURNAMENT</text>
</svg>"""

def inject_global_layout(user_info_text: str):
    app_cfg = load_app_config()
    url = (app_cfg or {}).get("app_logo_url", "").strip() or None

    data_uri = fetch_image_as_data_uri(url, bust="v3_3_30") if url else ""
    if data_uri:
        logo_html = f'<img src="{data_uri}" alt="logo" style="display:block;max-width:20vw;max-height:64px;width:auto;height:auto;object-fit:contain;" />'
    else:
        logo_html = brand_svg(220)

    st.markdown(f"""
    <style>
      .topbar {{
        position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
        background: white; border-bottom: 1px solid #e5e5e5;
        padding: 6px 12px; display: flex; align-items: center; gap: 12px;
      }}
      .topbar .left {{ display:flex; align-items:center; gap:10px; overflow:visible; }}
      .topbar .right {{ margin-left:auto; display:flex; align-items:center; gap:12px; font-size:.92rem; color:#333; }}
      .content-offset {{ padding-top: 92px; }}
      .stTabs [data-baseweb="tab-list"] {{
        position: sticky; top: 92px; z-index: 9998; background: white; border-bottom:1px solid #e5e5e5;
      }}
      .dark-header th {{ background-color: #2f3b52 !important; color:#fff !important; }}
      .zebra tr:nth-child(even) td {{ background-color: #f5f7fa !important; }}
      .zebra tr:nth-child(odd) td  {{ background-color: #ffffff !important; }}
      .winner-badge {{
        display:inline-block; padding:4px 8px; border-radius:8px;
        background:#e8f5e9; color:#1b5e20; font-weight:600; margin-left:8px;
      }}
      .champion-banner {{
        padding:14px 18px; border-radius:10px; background:#fff9c4; border:1px solid #ffeb3b;
        font-size:1.1rem; font-weight:700; color:#795548; margin:8px 0;
      }}
    </style>
    <div class="topbar">
      <div class="left">{logo_html}</div>
      <div class="right">{user_info_text}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="content-offset"></div>', unsafe_allow_html=True)

# ====== Usuarios ======
DEFAULT_SUPER = {
    "username": "ADMIN", "pin_hash": sha("199601"), "role": "SUPER_ADMIN",
    "assigned_admin": None, "created_at": now_iso(), "active": True
}

def load_users() -> List[Dict[str, Any]]:
    if not USERS_PATH.exists():
        USERS_PATH.write_text(json.dumps([DEFAULT_SUPER], indent=2), encoding="utf-8")
        return [DEFAULT_SUPER]
    return json.loads(USERS_PATH.read_text(encoding="utf-8"))

def save_users(users: List[Dict[str, Any]]):
    USERS_PATH.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")

def get_user(username: str) -> Optional[Dict[str, Any]]:
    for u in load_users():
        if u["username"].lower() == username.lower():
            return u
    return None

def set_user(user: Dict[str, Any]):
    users = load_users()
    for i, u in enumerate(users):
        if u["username"].lower() == user["username"].lower():
            users[i] = user
            save_users(users)
            return
    users.append(user)
    save_users(users)

# ====== Torneos (guardar/cargar) ======
def load_index() -> List[Dict[str, Any]]:
    if not TOURN_INDEX.exists():
        TOURN_INDEX.write_text("[]", encoding="utf-8")
        return []
    return json.loads(TOURN_INDEX.read_text(encoding="utf-8"))

def save_index(idx: List[Dict[str, Any]]):
    TOURN_INDEX.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

def tourn_path(tid: str) -> Path:
    return TOURN_DIR / f"{tid}.json"

def snap_dir_for(tid: str) -> Path:
    p = SNAP_ROOT / tid
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_tournament(tid: str) -> Dict[str, Any]:
    p = tourn_path(tid)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def save_tournament(tid: str, obj: Dict[str, Any], make_snapshot: bool=True):
    p = tourn_path(tid)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    if make_snapshot:
        sd = snap_dir_for(tid)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (sd / f"snapshot_{ts}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        snaps = sorted([x for x in sd.glob("snapshot_*.json")], reverse=True)
        for old in snaps[KEEP_SNAPSHOTS:]:
            try:
                old.unlink()
            except Exception:
                pass

# ====== Config / reglas ======
DEFAULT_CONFIG = {
    "t_name": "Open Pádel",
    "num_pairs": 16,
    "num_zones": 4,
    "top_per_zone": 2,
    "points_win": 2,
    "points_loss": 0,
    "seed": 42,
    "format": "best_of_3",  # one_set | best_of_3 | best_of_5
    "use_seeds": False      # NUEVO: cabezas de serie
}

rng = lambda off, seed: random.Random(int(seed) + int(off))

def rr_schedule(group):
    return list(combinations(group, 2))

def build_fixtures(groups):
    rows = []
    if not groups:
        return []
    for zi, group in enumerate(groups, start=1):
        zone = f"Z{zi}"
        for a, b in rr_schedule(group):
            rows.append({
                "zone": zone, "pair1": a, "pair2": b,
                "sets": [], "golden1": 0, "golden2": 0
            })
    return rows

def validate_sets(fmt: str, sets: List[Dict[str,int]]) -> Tuple[bool, str]:
    n = len(sets)
    if fmt == "one_set":
        if n != 1:
            return False, "Formato a 1 set: debe haber exactamente 1 set."
    elif fmt == "best_of_3":
        if n < 2 or n > 3:
            return False, "Formato al mejor de 3: debe haber 2 o 3 sets."
    elif fmt == "best_of_5":
        if n < 3 or n > 5:
            return False, "Formato al mejor de 5: debe haber entre 3 y 5 sets."
    return True, ""

def compute_sets_stats(sets: List[Dict[str,int]]) -> Dict[str,int]:
    g1=g2=s1=s2=0
    for s in sets:
        a = int(s.get("s1",0)); b = int(s.get("s2",0))
        g1 += a; g2 += b
        if a>b: s1 += 1
        elif b>a: s2 += 1
    return {"games1": g1, "games2": g2, "sets1": s1, "sets2": s2}

def match_has_winner(sets: List[Dict[str,int]]) -> bool:
    stats = compute_sets_stats(sets)
    return stats["sets1"] != stats["sets2"]

def zone_complete(zone_name: str, results_list: List[Dict[str,Any]], fmt: str) -> bool:
    ms = [m for m in results_list if m["zone"]==zone_name]
    if not ms:
        return False
    for m in ms:
        ok,_ = validate_sets(fmt, m.get("sets", []))
        if not ok or not match_has_winner(m.get("sets", [])):
            return False
    return True

def standings_from_results(zone_name, group_pairs, results_list, cfg):
    rows = [{"pair": p, "PJ": 0, "PG": 0, "PP": 0, "GF": 0, "GC": 0, "GP": 0, "PTS": 0} for p in group_pairs]
    table = pd.DataFrame(rows).set_index("pair")
    fmt = cfg.get("format","best_of_3")
    for m in results_list:
        if m["zone"] != zone_name:
            continue
        sets = m.get("sets", [])
        ok, _ = validate_sets(fmt, sets)
        if not ok or not match_has_winner(sets):
            continue
        stats = compute_sets_stats(sets)
        p1, p2 = m["pair1"], m["pair2"]
        g1,g2 = stats["games1"], stats["games2"]
        s1,s2 = stats["sets1"], stats["sets2"]
        for p, gf, gc in [(p1,g1,g2),(p2,g2,g1)]:
            table.at[p, "PJ"] += 1
            table.at[p, "GF"] += gf
            table.at[p, "GC"] += gc
        table.at[p1, "GP"] += int(m.get("golden1",0))
        table.at[p2, "GP"] += int(m.get("golden2",0))
        if s1>s2:
            table.at[p1, "PG"] += 1; table.at[p2, "PP"] += 1
            table.at[p1, "PTS"] += cfg["points_win"]
            table.at[p2, "PTS"] += cfg["points_loss"]
        elif s2>s1:
            table.at[p2, "PG"] += 1; table.at[p1, "PP"] += 1
            table.at[p2, "PTS"] += cfg["points_win"]
            table.at[p1, "PTS"] += cfg["points_loss"]
    table["DG"] = table["GF"] - table["GC"]
    r = rng(0, cfg["seed"])
    randmap = {p: r.random() for p in table.index}
    table["RND"] = table.index.map(randmap.get)
    table = table.sort_values(by=["PTS","DG","GP","RND"], ascending=[False,False,False,False]).reset_index()
    table.insert(0, "Zona", zone_name)
    table.insert(1, "Pos", range(1, len(table)+1))
    table = table.drop(columns=["RND"])
    return table

def qualified_from_tables(zone_tables, k):
    qualified = []
    for table in zone_tables:
        if table.empty:
            continue
        z = table.iloc[0]["Zona"]
        q = table.head(int(k))
        for _, row in q.iterrows():
            qualified.append((z, int(row["Pos"]), row["pair"]))
    return qualified

# ====== Playoffs helpers ======
def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def starting_round_name(n_slots: int) -> str:
    if n_slots <= 2: return "FN"
    if n_slots <= 4: return "SF"
    return "QF"

def seed_pairs(winners: List[Tuple[str,int,str]], runners: List[Tuple[str,int,str]]) -> List[Tuple[str,str]]:
    """Empareja ganadores con segundos de otra zona, rotando los segundos."""
    if not winners or not runners:
        return []
    if len(winners) != len(runners):
        m = min(len(winners), len(runners))
        winners = winners[:m]; runners = runners[:m]
    rr = runners[1:] + runners[:1] if len(runners) > 1 else runners
    return list(zip([w for _,_,w in winners], [r for _,_,r in rr]))

def build_initial_ko(qualified: List[Tuple[str,int,str]]) -> List[Dict[str,Any]]:
    """
    qualified: lista de (zona, pos, pareja).
    Devuelve partidos iniciales sin BYE cuando N es 2, 4 u 8.
    Si N no es potencia de 2, reparte BYEs en la ronda inicial para seeds altos.
    """
    N = len(qualified)
    if N == 0:
        return []
    winners = [q for q in qualified if q[1]==1]
    runners = [q for q in qualified if q[1]==2]
    slots = next_pow2(N)  # 2,4,8,16...
    start_round = starting_round_name(slots)

    if N == slots:
        if N == 2:
            a = qualified[0][2]; b = qualified[1][2]
            return [{"round":"FN","label":"FINAL","a":a,"b":b,"sets":[],"goldenA":0,"goldenB":0}]
        if N == 4:
            pairs = seed_pairs(winners, runners)
            out=[]
            labels=["SF1","SF2"]
            for i,(a,b) in enumerate(pairs):
                out.append({"round":"SF","label":labels[i],"a":a,"b":b,"sets":[],"goldenA":0,"goldenB":0})
            if not out:  # fallback
                names = [q[2] for q in qualified]
                out = [
                    {"round":"SF","label":"SF1","a":names[0],"b":names[3],"sets":[],"goldenA":0,"goldenB":0},
                    {"round":"SF","label":"SF2","a":names[1],"b":names[2],"sets":[],"goldenA":0,"goldenB":0},
                ]
            return out
        if N == 8:
            pairs = seed_pairs(winners, runners)
            if len(pairs) != 4:
                names = [q[2] for q in qualified]
                pairs = [(names[i], names[-(i+1)]) for i in range(4)]
            labels=[f"QF{i}" for i in range(1,5)]
            return [{"round":"QF","label":labels[i],"a":a,"b":b,"sets":[],"goldenA":0,"goldenB":0} for i,(a,b) in enumerate(pairs)]

    # N no potencia de 2 → BYEs
    names = [q[2] for q in sorted(qualified, key=lambda x:(x[1], x[0]))]  # pos1 antes que pos2
    byes = slots - N
    start_matches = []
    labels_map = {"FN":["FINAL"], "SF":["SF1","SF2"], "QF":["QF1","QF2","QF3","QF4"]}
    labels = labels_map[start_round]
    i=0; li=0
    while i < len(names):
        a = names[i]; b = None
        if i+1 < len(names):
            b = names[i+1]; i += 2
        else:
            i += 1
        lab = labels[li] if li < len(labels) else f"{start_round}{li+1}"
        if byes>0 and b is None:
            b = "BYE"; byes -= 1
        start_matches.append({"round":start_round,"label":lab,"a":a,"b":b or "BYE","sets":[],"goldenA":0,"goldenB":0})
        li += 1
    return start_matches

def advance_pairs_from_round(matches_round: List[Dict[str,Any]]) -> List[str]:
    winners=[]
    for m in matches_round:
        sets = m.get("sets", [])
        if not sets or not match_has_winner(sets):
            return []
        stats = compute_sets_stats(sets)
        winners.append(m['a'] if stats["sets1"]>stats["sets2"] else m['b'])
    return winners

def make_next_round_name(current: str) -> Optional[str]:
    order=["QF","SF","FN"]
    if current=="FN": return None
    try:
        i=order.index(current)
    except ValueError:
        return None
    return order[i+1]

def pairs_to_matches(pairs: List[Tuple[str, Optional[str]]], round_name: str) -> List[Dict[str,Any]]:
    labels = {"SF":["SF1","SF2"], "FN":["FINAL"]}
    out=[]
    for i,(a,b) in enumerate(pairs, start=1):
        lab = labels.get(round_name, [f"{round_name}{i}"]*len(pairs))
        lab = lab[i-1] if i-1 < len(lab) else f"{round_name}{i}"
        out.append({"round":round_name,"label":lab,"a":a,"b":b or "BYE","sets":[],"goldenA":0,"goldenB":0})
    return out

def next_round(slots: List[str]):
    out=[]; i=0
    while i < len(slots):
        if i+1 < len(slots): out.append((slots[i], slots[i+1])); i+=2
        else: out.append((slots[i], None)); i+=1
    return out

# ====== Sesión ======
def init_session():
    st.session_state.setdefault("auth_user", None)
    st.session_state.setdefault("current_tid", None)
    st.session_state.setdefault("autosave", True)
    st.session_state.setdefault("last_hash", "")
    st.session_state.setdefault("pdf_fixture_bytes", None)
    st.session_state.setdefault("pdf_playoffs_bytes", None)
    st.session_state.setdefault("pdf_generated_at", None)
    st.session_state.setdefault("suspend_autosave_runs", 0)

def compute_state_hash(state: Dict[str,Any]) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def tournament_state_template(admin_username: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    cfg["t_name"] = meta.get("t_name") or cfg["t_name"]
    return {
        "meta": {
            "tournament_id": meta["tournament_id"],
            "t_name": cfg["t_name"],
            "place": meta.get("place",""),
            "date": meta.get("date",""),
            "gender": meta.get("gender","mixto"),
            "admin_username": admin_username,
            "created_at": now_iso(),
        },
        "config": cfg,
        "pairs": [],
        "groups": None,
        "results": [],
        "ko": {"matches": []},
        "seeded_pairs": []   # asegurar existencia
    }

# ====== Utilidades parejas ======
def parse_pair_number(label: str) -> Optional[int]:
    try:
        left = label.split("—", 1)[0].strip()
        return int(left)
    except Exception:
        return None

def next_available_number(pairs: List[str], max_pairs: int) -> Optional[int]:
    used = set()
    for p in pairs:
        n = parse_pair_number(p)
        if n is not None:
            used.add(n)
    for n in range(1, max_pairs+1):
        if n not in used:
            return n
    return None

def format_pair_label(n: int, j1: str, j2: str) -> str:
    return f"{n:02d} — {j1.strip()} / {j2.strip()}"

def remove_pair_by_number(pairs: List[str], n: int) -> List[str]:
    out = []
    for p in pairs:
        pn = parse_pair_number(p)
        if pn != n:
            out.append(p)
    return out

# ====== Sorteo de zonas (con y sin seeds) ======
def create_groups_seeded(pairs: List[str], num_groups: int, top_per_zone: int, seed: int, seeded_pairs: List[str]) -> List[List[str]]:
    """
    - seeded_pairs: exactamente num_groups parejas (o error).
    - Coloca 1 seed por zona al azar.
    - Reparte el resto al azar procurando:
        * mínimo por zona = top_per_zone (si total alcanza),
        * luego balancear lo más parejo posible.
    """
    r = random.Random(int(seed))
    if len(pairs) < num_groups:
        # el caller validará y mostrará error
        pass
    seeded = [p for p in pairs if p in set(seeded_pairs)]
    if len(seeded) != num_groups:
        raise ValueError(f"Debes marcar exactamente {num_groups} cabezas de serie (actual: {len(seeded)}).")

    r.shuffle(seeded)
    groups = [[s] for s in seeded]  # 1 seed por zona

    rest = [p for p in pairs if p not in set(seeded)]
    r.shuffle(rest)

    min_per_zone = max(1, int(top_per_zone))
    total = len(pairs)
    desired_min_total = num_groups * min_per_zone

    current_sizes = [1] * num_groups  # ya hay 1 seed
    gi = 0
    while rest and sum(current_sizes) < min(total, desired_min_total):
        if current_sizes[gi] < min_per_zone:
            groups[gi].append(rest.pop())
            current_sizes[gi] += 1
        gi = (gi + 1) % num_groups

    gi = 0
    while rest:
        groups[gi].append(rest.pop())
        current_sizes[gi] += 1
        gi = (gi + 1) % num_groups

    return groups

def create_groups_unseeded(pairs: List[str], num_groups: int, top_per_zone: int, seed: int) -> List[List[str]]:
    """
    Sin cabezas de serie:
    - Reparte al azar procurando mínimo por zona = top_per_zone (si total alcanza).
    - Luego balancea lo demás de forma round-robin.
    """
    r = random.Random(int(seed))
    pool = pairs[:]
    r.shuffle(pool)

    groups = [[] for _ in range(num_groups)]
    min_per_zone = max(1, int(top_per_zone))
    total = len(pool)
    desired_min_total = num_groups * min_per_zone

    gi = 0
    while pool and sum(len(g) for g in groups) < min(total, desired_min_total):
        if len(groups[gi]) < min_per_zone:
            groups[gi].append(pool.pop())
        gi = (gi + 1) % num_groups

    gi = 0
    while pool:
        groups[gi].append(pool.pop())
        gi = (gi + 1) % num_groups

    return groups

# ====== Login ======
def login_form():
    st.markdown("### Ingreso — Usuario + PIN (6 dígitos)")
    with st.form("login", clear_on_submit=True):
        username = st.text_input("Usuario").strip()
        pin = st.text_input("PIN (6 dígitos)", type="password").strip()
        submitted = st.form_submit_button("Ingresar", type="primary")
    if submitted:
        user = get_user(username)
        if not user or not user.get("active", True):
            st.error("Usuario inexistente o inactivo.")
            return
        if len(pin)!=6 or not pin.isdigit():
            st.error("PIN inválido.")
            return
        if sha(pin) != user["pin_hash"]:
            st.error("PIN incorrecto.")
            return
        st.session_state.auth_user = user
        st.success(f"Bienvenido {user['username']} ({user['role']})")
        st.rerun()

# ====== Super Admin / Admin ======
def load_index_for_admin(admin_username: str) -> List[Dict[str, Any]]:
    idx = load_index()
    my = [t for t in idx if t.get("admin_username")==admin_username]
    def keyf(t):
        try:
            return datetime.fromisoformat(t.get("date"))
        except Exception:
            return datetime.min
    return sorted(my, key=keyf, reverse=True)

def create_tournament(admin_username: str, t_name: str, place: str, tdate: str, gender: str) -> str:
    tid = str(uuid.uuid4())[:8]
    meta = {"tournament_id":tid,"t_name":t_name,"place":place,"date":tdate,"gender":gender}
    state = tournament_state_template(admin_username, meta)
    save_tournament(tid, state)
    idx = load_index()
    idx.append({
        "tournament_id":tid,"t_name":t_name,"place":place,"date":tdate,
        "gender":gender,"admin_username":admin_username,"created_at":now_iso()
    })
    save_index(idx)
    return tid

def delete_tournament(admin_username: str, tid: str):
    idx = load_index()
    idx = [t for t in idx if not (t["tournament_id"]==tid and t["admin_username"]==admin_username)]
    save_index(idx)
    p = tourn_path(tid)
    if p.exists():
        p.unlink()
    for f in (snap_dir_for(tid)).glob("*.json"):
        try:
            f.unlink()
        except Exception:
            pass

def admin_dashboard(admin_user: Dict[str, Any]):
    user_text = f"Usuario: <b>{admin_user['username']}</b> &nbsp;|&nbsp; Rol: <code>{admin_user['role']}</code> &nbsp;&nbsp;<a href='#' onclick='window.location.reload()'>Cerrar sesión</a>"
    inject_global_layout(user_text)

    st.markdown("### Mis Torneos")

    index = load_index_for_admin(admin_user["username"])
    if not index:
        st.info("Aún no tienes torneos.")
    
    col_sel, col_new = st.columns([1,1])
    with col_sel:
        current_tid = st.selectbox(
            "Seleccionar Torneo",
            [""] + [tid for tid in (t["tournament_id"] for t in index)],
            format_func=lambda tid: load_tournament(tid)["meta"]["t_name"] if tid else "— Nuevo Torneo —"
        )
        if current_tid != st.session_state.current_tid:
            st.session_state.current_tid = current_tid
            st.session_state.last_hash = ""
            st.rerun()

    with col_new:
        with st.form("new_tourn_form", clear_on_submit=True):
            t_name = st.text_input("Nombre del nuevo torneo").strip()
            t_id_suf = st.text_input("Identificador URL (opcional)", help="Si no lo pones, se generará uno aleatorio.").strip()
            date_col, place_col = st.columns([1,2])
            with date_col:
                t_date = st.date_input("Fecha del torneo", value=date.today())
            with place_col:
                t_place = st.text_input("Lugar / Club", value="Mi Club").strip()
            gen = st.selectbox("Género", ["masculino","femenino","mixto"], index=2)
            submitted = st.form_submit_button("Crear torneo", type="primary")
        if submitted:
            tid = create_tournament(admin_user["username"], t_name or "Nuevo Torneo", t_place, t_date.isoformat(), gen)
            st.session_state.current_tid = tid
            st.success(f"Torneo creado: {t_name} ({tid})")
            st.rerun()

    if st.session_state.current_tid:
        tourn_tid = st.session_state.current_tid
        tourn_state = load_tournament(tourn_tid)
        tourn_state.setdefault("seeded_pairs", [])  # compatibilidad
        st.session_state.last_hash = compute_state_hash(tourn_state)

        t_name = tourn_state["meta"]["t_name"]
        st.subheader(f"🛠️ {t_name}")

        with st.container(border=True):
            st.markdown(f"**Lugar:** {tourn_state['meta']['place']} &nbsp;&nbsp; **Fecha:** {tourn_state['meta']['date']} &nbsp;&nbsp; **Género:** {tourn_state['meta']['gender']}")

        tabs = st.tabs(["⚙️ Configuración", "👥 Parejas", "📝 Resultados", "📊 Tablas", "🗂️ Playoffs", "💾 Persistencia"])

        # ========== CONFIG ==========
        with tabs[0]:
            cfg = tourn_state.get("config", DEFAULT_CONFIG.copy())
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                cfg["t_name"] = st.text_input("Nombre para mostrar", value=cfg.get("t_name","Open Pádel"))
                cfg["num_pairs"] = st.number_input("Cantidad máxima de parejas", 2, 256, int(cfg.get("num_pairs",16)), step=1)
            with c2:
                cfg["num_zones"] = st.number_input("Cantidad de zonas", 2, 32, int(cfg.get("num_zones",4)), step=1)
                cfg["top_per_zone"] = st.number_input("Clasifican por zona (Top N)", 1, 8, int(cfg.get("top_per_zone",2)), step=1)
            with c3:
                cfg["points_win"] = st.number_input("Puntos por victoria", 1, 10, int(cfg.get("points_win",2)), step=1)
                cfg["points_loss"] = st.number_input("Puntos por derrota", 0, 5, int(cfg.get("points_loss",0)), step=1)
            with c4:
                cfg["seed"] = st.number_input("Semilla (sorteo zonas)", 1, 999999, int(cfg.get("seed",42)), step=1)
            fmt = st.selectbox(
                "Formato de partido",
                ["one_set","best_of_3","best_of_5"],
                index=["one_set","best_of_3","best_of_5"].index(cfg.get("format","best_of_3"))
            )
            cfg["format"] = fmt

            # NUEVO: cabezas de serie
            cfg["use_seeds"] = st.checkbox("Usar cabezas de serie (1 por zona)", value=bool(cfg.get("use_seeds", False)))
            st.caption("Si está activo, podrás marcar exactamente N parejas como cabeza de serie (N = cantidad de zonas). Cada zona recibirá 1 cabeza; el resto se sortea al azar.")

            cA,cB,cC = st.columns(3)
            with cA:
                if st.button("💾 Guardar configuración", type="primary"):
                    tourn_state["config"] = {
                        k:int(v) if isinstance(v,(int,float)) and k not in ["t_name","format","use_seeds"] else v
                        for k,v in cfg.items()
                    }
                    save_tournament(tourn_tid, tourn_state)
                    st.success("Configuración guardada.")
            with cB:
                if st.button("🎲 Sortear zonas (crear/rehacer fixture)"):
                    pairs = tourn_state.get("pairs", [])
                    if len(pairs) < cfg["num_zones"]:
                        st.error("Debe haber al menos tantas parejas como zonas.")
                    else:
                        try:
                            if cfg.get("use_seeds", False):
                                seeded_pairs = tourn_state.get("seeded_pairs", [])
                                groups = create_groups_seeded(
                                    pairs, int(cfg["num_zones"]),
                                    int(cfg["top_per_zone"]), int(cfg["seed"]),
                                    seeded_pairs
                                )
                            else:
                                groups = create_groups_unseeded(
                                    pairs, int(cfg["num_zones"]),
                                    int(cfg["top_per_zone"]), int(cfg["seed"])
                                )
                        except ValueError as e:
                            st.error(str(e))
                            st.stop()

                        tourn_state["groups"] = groups
                        tourn_state["results"] = build_fixtures(groups)
                        tourn_state["ko"] = {"matches": []}  # limpiar KO si rehaces
                        save_tournament(tourn_tid, tourn_state)
                        st.success("Zonas + fixture generados.")
            with cC:
                if REPORTLAB_OK and st.button("🧾 Generar PDFs"):
                    with st.spinner("Generando PDFs..."):
                        buf1 = export_fixture_pdf(tourn_state)
                        buf2 = export_playoffs_pdf(tourn_state)
                    if buf1: st.session_state.pdf_fixture_bytes = buf1.getvalue()
                    if buf2: st.session_state.pdf_playoffs_bytes = buf2.getvalue()
                    st.session_state.pdf_generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("PDFs generados. Descarga abajo.")
                elif not REPORTLAB_OK:
                    st.info("Para PDF: pip install reportlab")

            if st.session_state.pdf_fixture_bytes or st.session_state.pdf_playoffs_bytes:
                st.markdown("#### Descargas de PDF")
                st.caption(f"Generado: {st.session_state.pdf_generated_at or '-'}")
                if st.session_state.pdf_fixture_bytes:
                    st.download_button("⬇️ Fixture (PDF)", data=st.session_state.pdf_fixture_bytes,
                                    file_name=f"fixture_{tourn_state['meta']['tournament_id']}.pdf", mime="application/pdf")
                if st.session_state.pdf_playoffs_bytes:
                    st.download_button("⬇️ Playoffs (PDF)", data=st.session_state.pdf_playoffs_bytes,
                                    file_name=f"playoffs_{tourn_state['meta']['tournament_id']}.pdf", mime="application/pdf")
                if st.button("🧹 Limpiar PDFs generados"):
                    st.session_state.pdf_fixture_bytes = None
                    st.session_state.pdf_playoffs_bytes = None
                    st.session_state.pdf_generated_at = None
                    st.success("Limpio.")

        # ========== PAREJAS ==========
        with tabs[1]:
            st.subheader("Parejas")
            pairs = tourn_state.get("pairs", [])
            max_pairs = int(tourn_state.get("config", {}).get("num_pairs", 16))

            use_seeds = bool(tourn_state.get("config", {}).get("use_seeds", False))
            num_zones = int(tourn_state.get("config", {}).get("num_zones", 4))
            seeded = tourn_state.get("seeded_pairs", [])
            seeded_set = set(seeded)
            seeds_needed = num_zones
            seeds_missing = max(0, seeds_needed - len(seeded_set))

            st.markdown("**Alta manual — una pareja por vez**")
            next_n = next_available_number(pairs, max_pairs)

            with st.form(f"add_pair_form_{tourn_tid}", clear_on_submit=True):
                c1,c2,c3,c4,c5 = st.columns([1,3,3,2,2])
                with c1:
                    st.text_input("N° pareja", value=(str(next_n) if next_n else "—"), disabled=True, key=f"num_auto_{tourn_tid}")
                with c2:
                    p1 = st.text_input("Jugador 1")
                with c3:
                    p2 = st.text_input("Jugador 2")
                mark_seed = False
                with c5:
                    if use_seeds:
                        can_mark = seeds_missing > 0
                        mark_seed = st.checkbox("Cabeza de serie", value=False, disabled=not can_mark)
                        if not can_mark:
                            st.caption(f"({len(seeded_set)}/{seeds_needed} marcadas)")
                submitted = st.form_submit_button("➕ Agregar pareja", disabled=(next_n is None), type="primary")

            if submitted:
                p1c, p2c = (p1 or "").strip(), (p2 or "").strip()
                if not p1c or not p2c:
                    st.error("Completá ambos nombres.")
                else:
                    label = format_pair_label(next_n, p1c, p2c)
                    pairs.append(label)
                    tourn_state["pairs"] = pairs
                    if use_seeds and mark_seed:
                        if label not in seeded_set and len(seeded_set) < seeds_needed:
                            seeded_set.add(label)
                            tourn_state["seeded_pairs"] = list(seeded_set)
                    save_tournament(tourn_tid, tourn_state)
                    st.success(f"Agregada: {label}{' (Cabeza de serie)' if use_seeds and label in seeded_set else ''}")
                    st.rerun()

            if next_n is None:
                st.warning(f"Se alcanzó el máximo de parejas configurado ({max_pairs}).")

            st.divider()

            # Importar CSV (opcional)
            st.markdown("**Importar CSV (opcional)**")
            st.caption("Formato: columnas `numero, jugador1, jugador2`.")
            up = st.file_uploader("Seleccionar CSV", type=["csv"], key=f"csv_{tourn_tid}")
            if up is not None:
                try:
                    df = pd.read_csv(up, header=0)
                except Exception:
                    up.seek(0)
                    df = pd.read_csv(up, header=None, names=["numero","jugador1","jugador2"])
                cols = [c.strip().lower() for c in df.columns.tolist()]
                df.columns = cols
                if "numero" not in df.columns or "jugador1" not in df.columns or "jugador2" not in df.columns:
                    st.error("El CSV debe contener columnas: numero, jugador1, jugador2")
                else:
                    parsed = []
                    for _, row in df.iterrows():
                        try:
                            num = int(row["numero"])
                        except Exception:
                            continue
                        j1 = str(row["jugador1"]).strip()
                        j2 = str(row["jugador2"]).strip()
                        if j1 and j2 and num >= 1:
                            parsed.append((num, j1, j2))
                    if parsed:
                        parsed.sort(key=lambda x: x[0])
                        new_list = []
                        used = set()
                        for num, j1, j2 in parsed:
                            if len(new_list) >= max_pairs:
                                break
                            if 1 <= num <= max_pairs and num not in used:
                                used.add(num)
                                new_list.append(format_pair_label(num, j1, j2))
                        if not new_list:
                            st.warning("El CSV no contenía filas válidas dentro del rango permitido.")
                        else:
                            tourn_state["pairs"] = new_list
                            # no asignamos seeds por CSV automáticamente
                            save_tournament(tourn_tid, tourn_state)
                            st.success(f"Importadas {len(new_list)} parejas (máximo {max_pairs}).")
                            st.rerun()
                    else:
                        st.warning("No se encontraron filas válidas en el CSV.")

            st.divider()

            # Listado + borrar
            if pairs:
                st.markdown("### Listado de parejas")
                df_pairs = pd.DataFrame({"Pareja": pairs})
                # marcar seeds en la tabla
                if use_seeds:
                    df_pairs["Cabeza"] = df_pairs["Pareja"].apply(lambda x: "Sí" if x in seeded_set else "")
                st.markdown(df_pairs.to_html(index=False, classes=["zebra","dark-header"]), unsafe_allow_html=True)

                st.markdown("**Borrar pareja:**")
                cols = st.columns(4)
                per_row = 4
                for i, label in enumerate(pairs):
                    n = parse_pair_number(label) or (i+1)
                    col = cols[i % per_row]
                    with col:
                        if st.button(f"🗑️ Nº {n}", key=f"del_{tourn_tid}_{n}"):
                            # si la pareja era seed, quitarla
                            if label in seeded_set:
                                seeded_set.remove(label)
                                tourn_state["seeded_pairs"] = list(seeded_set)
                            tourn_state["pairs"] = remove_pair_by_number(pairs, n)
                            save_tournament(tourn_tid, tourn_state)
                            st.success(f"Eliminada pareja Nº {n}.")
                            st.rerun()
            else:
                st.info("Aún no hay parejas cargadas.")

            # Panel marcar/desmarcar seeds
            if use_seeds and pairs:
                st.divider()
                st.markdown("#### Marcar / Desmarcar cabezas de serie")
                st.caption(f"Debes marcar exactamente {seeds_needed} parejas. Actual: {len(seeded_set)}.")
                for i, label in enumerate(pairs):
                    is_seed = label in seeded_set
                    cols = st.columns([6,1])
                    cols[0].write(label + (" — **(Cabeza de serie)**" if is_seed else ""))
                    with cols[1]:
                        if st.button(("Quitar" if is_seed else "Marcar"), key=f"seed_toggle_{tourn_tid}_{i}"):
                            if is_seed:
                                seeded_set.remove(label)
                            else:
                                if len(seeded_set) >= seeds_needed:
                                    st.error("Ya alcanzaste el máximo de cabezas de serie.")
                                    st.stop()
                                seeded_set.add(label)
                            tourn_state["seeded_pairs"] = list(seeded_set)
                            save_tournament(tourn_tid, tourn_state)
                            st.rerun()

            if tourn_state.get("groups"):
                st.divider()
                st.markdown("### Zonas")
                for zi, group in enumerate(tourn_state["groups"], start=1):
                    st.write(f"**Z{zi}**")
                    df_g = pd.DataFrame({"Parejas": group})
                    st.markdown(df_g.to_html(index=False, classes=["zebra","dark-header"]), unsafe_allow_html=True)

        # ========== RESULTADOS ==========
        with tabs[2]:
            st.subheader("Resultados — fase de grupos (sets + puntos de oro)")
            if not tourn_state.get("groups"):
                st.info("Primero crea/sortea zonas en Configuración o en Parejas.")
            else:
                fmt = tourn_state["config"].get("format","best_of_3")
                zones = sorted({m["zone"] for m in tourn_state["results"]})
                z_filter = st.selectbox("Filtrar por zona", ["(todas)"] + zones)
                pnames = sorted(set([m["pair1"] for m in tourn_state["results"]] + [m["pair2"] for m in tourn_state["results"]]))
                p_filter = st.selectbox("Filtrar por pareja", ["(todas)"] + pnames)

                listing = tourn_state["results"]
                if z_filter != "(todas)":
                    listing = [m for m in listing if m["zone"]==z_filter]
                if p_filter != "(todas)":
                    listing = [m for m in listing if m["pair1"]==p_filter or m["pair2"]==p_filter]

                for m in listing:
                    idx = tourn_state["results"].index(m)
                    with st.container(border=True):
                        title = f"**{m['zone']}** — {m['pair1']} vs {m['pair2']}"
                        stats_now = compute_sets_stats(m.get("sets", [])) if m.get("sets") else {"sets1":0,"sets2":0}
                        if m.get("sets") and match_has_winner(m["sets"]):
                            winner = m['pair1'] if stats_now["sets1"]>stats_now["sets2"] else m['pair2']
                            title += f"  <span class='winner-badge'>🏆 {winner}</span>"
                        else:
                            title += "  <span style='color:#999'>(A definir)</span>"
                        st.markdown(title, unsafe_allow_html=True)

                        cur_sets = m.get("sets", [])
                        n_min, n_max = (1,1) if fmt=="one_set" else ((2,3) if fmt=="best_of_3" else (3,5))
                        n_sets = st.number_input(
                            "Sets jugados", min_value=n_min, max_value=n_max,
                            value=min(max(len(cur_sets), n_min), n_max),
                            key=f"ns_{tourn_tid}_{idx}"
                        )
                        new_sets = []
                        for si in range(n_sets):
                            cA,cB = st.columns(2)
                            with cA:
                                s1 = st.number_input(
                                    f"Set {si+1} — games {m['pair1']}", 0, 20,
                                    int(cur_sets[si]["s1"]) if si<len(cur_sets) and "s1" in cur_sets[si] else 0,
                                    key=f"s1_{tourn_tid}_{idx}_{si}"
                                )
                            with cB:
                                s2 = st.number_input(
                                    f"Set {si+1} — games {m['pair2']}", 0, 20,
                                    int(cur_sets[si]["s2"]) if si<len(cur_sets) and "s2" in cur_sets[si] else 0,
                                    key=f"s2_{tourn_tid}_{idx}_{si}"
                                )
                            new_sets.append({"s1":int(s1),"s2":int(s2)})
                        ok, msg = validate_sets(fmt, new_sets)
                        if not ok:
                            st.error(msg)
                        gC,gD = st.columns(2)
                        with gC:
                            g1 = st.number_input(f"Puntos de oro {m['pair1']}", 0, 200, int(m.get("golden1",0)), key=f"g1_{tourn_tid}_{idx}")
                        with gD:
                            g2 = st.number_input(f"Puntos de oro {m['pair2']}", 0, 200, int(m.get("golden2",0)), key=f"g2_{tourn_tid}_{idx}")

                        if st.button("Guardar partido", key=f"sv_{tourn_tid}_{idx}"):
                            stats = compute_sets_stats(new_sets)
                            if stats["sets1"] == stats["sets2"]:
                                st.error("Debe haber un ganador (no se permiten empates). Ajustá los sets.")
                            else:
                                tourn_state["results"][idx]["sets"] = new_sets
                                tourn_state["results"][idx]["golden1"] = int(g1)
                                tourn_state["results"][idx]["golden2"] = int(g2)
                                save_tournament(tourn_tid, tourn_state)
                                winner = m['pair1'] if stats["sets1"]>stats["sets2"] else m['pair2']
                                st.success(f"Partido guardado. 🏆 Ganó {winner}")
                                st.rerun()

        # ========== TABLAS ==========
        with tabs[3]:
            st.subheader("Tablas por zona y clasificados")
            if not tourn_state.get("groups") or not tourn_state.get("results"):
                st.info("Aún no hay fixture o resultados.")
            else:
                cfg = tourn_state["config"]
                fmt = cfg.get("format","best_of_3")
                zone_tables = []
                all_complete = True
                for zi, group in enumerate(tourn_state["groups"], start=1):
                    zone_name = f"Z{zi}"
                    complete = zone_complete(zone_name, tourn_state["results"], fmt)
                    status = "✅ Completa" if complete else "⏳ A definir"
                    if not complete:
                        all_complete = False
                    st.markdown(f"#### Tabla {zone_name} — {status}")
                    table = standings_from_results(zone_name, group, tourn_state["results"], cfg)
                    zone_tables.append(table)
                    if table.empty:
                        st.info("Sin datos para mostrar todavía.")
                    else:
                        st.markdown(table.to_html(index=False, classes=["zebra","dark-header"]), unsafe_allow_html=True)

                st.markdown("### Clasificados a Playoffs")
                if not all_complete:
                    st.info("⏳ A definir — Deben completarse todos los partidos de las zonas.")
                else:
                    qualified = qualified_from_tables(zone_tables, cfg["top_per_zone"])
                    if not qualified:
                        st.info("Sin clasificados aún.")
                    else:
                        dfq = pd.DataFrame([{"Zona":z,"Pos":pos,"Pareja":p} for (z,pos,p) in qualified])
                        st.markdown(dfq.to_html(index=False, classes=["zebra","dark-header"]), unsafe_allow_html=True)

        # ========== PLAYOFFS ==========
        with tabs[4]:
            st.subheader("Playoffs (por sets + puntos de oro)")
            if not tourn_state.get("groups") or not tourn_state.get("results"):
                st.info("Necesitas tener zonas y resultados para definir clasificados.")
            else:
                cfg = tourn_state["config"]
                fmt = cfg.get("format","best_of_3")
                all_complete = all(zone_complete(f"Z{zi}", tourn_state["results"], fmt) for zi in range(1, len(tourn_state["groups"])+1))
                if not all_complete:
                    st.info("⏳ A definir — Completa la fase de grupos para habilitar los playoffs.")
                else:
                    zone_tables = []
                    for zi, group in enumerate(tourn_state["groups"], start=1):
                        zone_name = f"Z{zi}"
                        table = standings_from_results(zone_name, group, tourn_state["results"], cfg)
                        zone_tables.append(table)
                    qualified = qualified_from_tables(zone_tables, cfg["top_per_zone"])

                    c1,c2 = st.columns(2)
                    with c1:
                        if st.button("🔄 Regenerar Playoffs (desde clasificados)"):
                            tourn_state["ko"]["matches"] = build_initial_ko(qualified)
                            save_tournament(tourn_tid, tourn_state)
                            st.success("Playoffs regenerados.")
                            st.rerun()
                    with c2:
                        st.caption("Usa esto si cambiaste resultados de zonas y querés rehacer la llave.")

                    if not tourn_state["ko"]["matches"]:
                        tourn_state["ko"]["matches"] = build_initial_ko(qualified)
                        save_tournament(tourn_tid, tourn_state)

                    round_order = ["QF","SF","FN"]
                    can_progress = True
                    final_champion = None

                    for rname in round_order:
                        ms = [m for m in tourn_state["ko"]["matches"] if m.get("round")==rname]
                        if not ms:
                            continue
                        st.markdown(f"### {rname}")
                        advancing = []
                        for idx, m in enumerate(ms, start=1):
                            with st.container(border=True):
                                title = f"**{m['label']}** — {m['a']} vs {m['b']}"
                                stats_now = compute_sets_stats(m.get("sets", [])) if m.get("sets") else {"sets1":0,"sets2":0}
                                if m.get("sets") and match_has_winner(m["sets"]):
                                    winner = m['a'] if stats_now["sets1"]>stats_now["sets2"] else m['b']
                                    title += f"  <span class='winner-badge'>🏆 {winner}</span>"
                                else:
                                    title += "  <span style='color:#999'>(A definir)</span>"
                                st.markdown(title, unsafe_allow_html=True)

                                cur_sets = m.get("sets", [])
                                n_min, n_max = (1,1) if fmt=="one_set" else ((2,3) if fmt=="best_of_3" else (3,5))
                                n_sets = st.number_input(
                                    "Sets jugados", min_value=n_min, max_value=n_max,
                                    value=min(max(len(cur_sets), n_min), n_max),
                                    key=f"ko_ns_{tourn_tid}_{rname}_{idx}"
                                )
                                new_sets = []
                                for si in range(n_sets):
                                    cA,cB = st.columns(2)
                                    with cA:
                                        s1 = st.number_input(
                                            f"Set {si+1} — games {m['a']}", 0, 20,
                                            int(cur_sets[si]["s1"]) if si<len(cur_sets) and "s1" in cur_sets[si] else 0,
                                            key=f"ko_s1_{tourn_tid}_{rname}_{idx}_{si}"
                                        )
                                    with cB:
                                        s2 = st.number_input(
                                            f"Set {si+1} — games {m['b']}", 0, 20,
                                            int(cur_sets[si]["s2"]) if si<len(cur_sets) and "s2" in cur_sets[si] else 0,
                                            key=f"ko_s2_{tourn_tid}_{rname}_{idx}_{si}"
                                        )
                                    new_sets.append({"s1":int(s1),"s2":int(s2)})
                                ok, msg = validate_sets(fmt, new_sets)
                                if not ok:
                                    st.error(msg)
                                gC,gD = st.columns(2)
                                with gC:
                                    g1 = st.number_input(f"Puntos de oro {m['a']}", 0, 200, int(m.get("goldenA",0)), key=f"ko_g1_{tourn_tid}_{rname}_{idx}")
                                with gD:
                                    g2 = st.number_input(f"Puntos de oro {m['b']}", 0, 200, int(m.get("goldenB",0)), key=f"ko_g2_{tourn_tid}_{rname}_{idx}")

                                if st.button("Guardar partido KO", key=f"ko_sv_{tourn_tid}_{rname}_{idx}"):
                                    stats = compute_sets_stats(new_sets)
                                    if stats["sets1"] == stats["sets2"]:
                                        st.error("Debe haber un ganador. Ajustá los sets.")
                                    else:
                                        m["sets"] = new_sets
                                        m["goldenA"] = int(g1)
                                        m["goldenB"] = int(g2)
                                        save_tournament(tourn_tid, tourn_state)
                                        winner = m['a'] if stats["sets1"]>stats["sets2"] else m['b']
                                        st.success(f"KO guardado. 🏆 Ganó {winner}")
                                        st.rerun()

                                if m.get("sets") and match_has_winner(m["sets"]):
                                    stats = compute_sets_stats(m["sets"])
                                    winner = m['a'] if stats["sets1"]>stats["sets2"] else m['b']
                                    advancing.append(winner)
                                else:
                                    can_progress = False

                        if rname=="FN":
                            if advancing and len(advancing)==1:
                                final_champion = advancing[0]
                                st.markdown(f"<div class='champion-banner'>🏆 CAMPEÓN: {final_champion}</div>", unsafe_allow_html=True)
                                st.balloons()
                            continue

                        if can_progress and advancing:
                            next_rname = make_next_round_name(rname)
                            if next_rname:
                                pairs = next_round(advancing)
                                tourn_state["ko"]["matches"] = [m for m in tourn_state["ko"]["matches"] if m.get("round") not in (next_rname,)]
                                tourn_state["ko"]["matches"].extend(pairs_to_matches(pairs, next_rname))
                                save_tournament(tourn_tid, tourn_state)
                                st.info(f"Ronda {next_rname} preparada. Completa todos para llegar a la FINAL.")
                                st.rerun()
                        else:
                            st.info("⏳ A definir — Falta completar partidos de esta fase para avanzar.")

        # ========== PERSISTENCIA ==========
        with tabs[5]:
            st.subheader("Persistencia (autosave + snapshots)")
            def sanitize_filename(s: str) -> str:
                return "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in s).strip("_")
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                st.session_state.autosave = st.checkbox("Autosave", value=st.session_state.autosave)
            with c2:
                if st.button("💾 Guardar ahora"):
                    save_tournament(tourn_tid, tourn_state)
                    st.success("Guardado")
            with c3:
                meta = tourn_state.get("meta", {})
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{meta.get('tournament_id','')}_{sanitize_filename(meta.get('t_name',''))}_{meta.get('date','')}_{ts}.json"
                st.download_button(
                    "⬇️ Descargar estado (JSON)",
                    data=json.dumps(tourn_state, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=fname,
                    mime="application/json",
                    key="dl_state_json"
                )
            with c4:
                up = st.file_uploader("⬆️ Cargar estado", type=["json"], key=f"up_{tourn_tid}")
                if up is not None:
                    st.warning("⚠️ Vas a restaurar un estado completo. Se desactiva el autosave temporalmente para acelerar la importación.")
                    if st.button("Confirmar restauración", key=f"confirm_restore_{tourn_tid}", type="primary"):
                        try:
                            new_state = json.load(up)
                            st.session_state["suspend_autosave_runs"] = 2
                            save_tournament(tourn_tid, new_state)
                            st.success("Cargado y guardado. (Autosave reactivado automáticamente en unos segundos)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al cargar: {e}")

        # Autosave con suspensión temporal
        current_hash = compute_state_hash(tourn_state)
        if st.session_state.get("suspend_autosave_runs", 0) > 0:
            st.session_state["suspend_autosave_runs"] -= 1
        else:
            if st.session_state.autosave and current_hash != st.session_state.last_hash:
                save_tournament(tourn_tid, tourn_state)
                st.toast("💾 Autosaved", icon="💾")
                st.session_state.last_hash = current_hash
            elif not st.session_state.autosave:
                st.session_state.last_hash = current_hash

# ====== Viewer ======
def viewer_dashboard(user: Dict[str, Any]):
    user_text = f"Usuario: <b>{user['username']}</b> &nbsp;|&nbsp; Rol: <code>{user['role']}</code>"
    inject_global_layout(user_text)
    st.header(f"Vista de consulta — {user['username']}")
    if not user.get("assigned_admin"):
        st.warning("No asignado a un admin.")
        return
    my = load_index_for_admin(user["assigned_admin"])
    if not my:
        st.info("El admin asignado no tiene torneos.")
        return
    names = [f"{t['date']} — {t['t_name']} ({t['gender']}) — {t['place']} — ID:{t['tournament_id']}" for t in my]
    selected = st.selectbox("Selecciona un torneo para ver", names, index=0)
    sel = my[names.index(selected)]
    viewer_tournament(sel["tournament_id"])

def viewer_tournament(tid: str, public: bool=False):
    user_text = "Modo público" if public else "Modo consulta"
    inject_global_layout(user_text)
    state = load_tournament(tid)
    if not state:
        st.error("No se encontró el torneo.")
        return
    st.subheader(f"{state['meta'].get('t_name')} — {state['meta'].get('place')} — {state['meta'].get('date')} — {state['meta'].get('gender')}")
    tab_over, tab_tables, tab_ko = st.tabs(["👀 General","📊 Tablas","🏁 Playoffs"])
    with tab_over:
        st.write("Parejas")
        dfp = pd.DataFrame({"Parejas": state.get("pairs", [])})
        st.markdown(dfp.to_html(index=False, classes=["zebra","dark-header"]), unsafe_allow_html=True)
        if state.get("groups"):
            st.write("Zonas")
            for zi, group in enumerate(state["groups"], start=1):
                st.write(f"**Z{zi}**")
                df_g = pd.DataFrame({"Parejas": group})
                st.markdown(df_g.to_html(index=False, classes=["zebra","dark-header"]), unsafe_allow_html=True)
    with tab_tables:
        if not state.get("groups") or not state.get("results"):
            st.info("Sin fixture/resultados aún.")
        else:
            cfg = state["config"]
            fmt = cfg.get("format","best_of_3")
            for zi, group in enumerate(state["groups"], start=1):
                zone_name = f"Z{zi}"
                status = "✅ Completa" if zone_complete(zone_name, state["results"], fmt) else "⏳ A definir"
                st.markdown(f"#### Tabla {zone_name} — {status}")
                table = standings_from_results(zone_name, group, state["results"], cfg)
                if table.empty:
                    st.info("Sin datos para mostrar todavía.")
                else:
                    st.markdown(table.to_html(index=False, classes=["zebra","dark-header"]), unsafe_allow_html=True)
    with tab_ko:
        ko = state.get("ko", {"matches": []})
        if not ko.get("matches"):
            st.info("Aún no hay partidos de playoffs.")
        else:
            rows = []
            for m in ko["matches"]:
                stats = compute_sets_stats(m.get("sets", [])) if m.get("sets") else {"sets1":0,"sets2":0}
                res = "A definir"
                if m.get("sets") and match_has_winner(m["sets"]):
                    res = f"{stats['sets1']}-{stats['sets2']}"
                rows.append({"Ronda": m.get("round",""), "Clave": m.get("label",""), "A": m.get("a",""), "B": m.get("b",""), "Resultado": res})
            dfo = pd.DataFrame(rows)
            st.markdown(dfo.to_html(index=False, classes=["zebra","dark-header"]), unsafe_allow_html=True)
    if public:
        st.info("Modo público (solo lectura)")

# ====== PDF (fixture / playoffs) ======
def export_fixture_pdf(state: Dict[str,Any]) -> Optional[BytesIO]:
    if not REPORTLAB_OK:
        return None
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    elems = []
    title = f"Fixture — {state['meta'].get('t_name')} — {state['meta'].get('place')} — {state['meta'].get('date')}"
    elems.append(Paragraph(title, styles['Title']))
    elems.append(Spacer(1, 12))
    if not state.get("groups"):
        elems.append(Paragraph("Sin zonas generadas.", styles['Normal']))
    else:
        for zi, group in enumerate(state["groups"], start=1):
            elems.append(Paragraph(f"Zona Z{zi}", styles['Heading2']))
            data = [["Parejas"]] + [[p] for p in group]
            t = Table(data, colWidths=[16*cm])
            t.setStyle(TableStyle([
                ('GRID',(0,0),(-1,-1),0.5,colors.grey),
                ('BACKGROUND',(0,0),(-1,0),colors.lightgrey)
            ]))
            elems.append(t)
            elems.append(Spacer(1,8))
        rows = [["Zona","Pareja 1","Pareja 2"]]
        for m in state["results"]:
            rows.append([m["zone"], m["pair1"], m["pair2"]])
        elems.append(Paragraph("Partidos (fase de grupos)", styles['Heading2']))
        t2 = Table(rows, colWidths=[2*cm, 7*cm, 7*cm])
        t2.setStyle(TableStyle([
            ('GRID',(0,0),(-1,-1),0.5,colors.grey),
            ('BACKGROUND',(0,0),(-1,0),colors.lightgrey)
        ]))
        elems.append(t2)
    doc.build(elems)
    buf.seek(0)
    return buf

def export_playoffs_pdf(state: Dict[str,Any]) -> Optional[BytesIO]:
    if not REPORTLAB_OK:
        return None
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.0*cm, rightMargin=1.0*cm, topMargin=1.0*cm, bottomMargin=1.0*cm)
    styles = getSampleStyleSheet()
    elems = []
    elems.append(Paragraph(f"Playoffs — {state['meta'].get('t_name')}", styles['Title']))
    elems.append(Spacer(1, 10))
    ko = state.get("ko", {"matches": []})
    rounds = ["QF","SF","FN"]
    for r in rounds:
        ms = [m for m in ko.get("matches", []) if m.get("round")==r]
        if not ms:
            continue
        elems.append(Paragraph(r, styles['Heading2']))
        rows = [["Clave","A","B","Sets A-B","Ptos Oro A-B"]]
        for m in ms:
            stats = compute_sets_stats(m.get("sets", []))
            sets_str = f"{stats['sets1']}-{stats['sets2']}" if m.get("sets") and match_has_winner(m["sets"]) else "A definir"
            gp_str = f"{m.get('goldenA',0)}-{m.get('goldenB',0)}"
            rows.append([m.get("label",""), m.get("a",""), m.get("b",""), sets_str, gp_str])
        t = Table(rows, colWidths=[3*cm, 5*cm, 5*cm, 3*cm, 3*cm])
        t.setStyle(TableStyle([
            ('GRID',(0,0),(-1,-1),0.5,colors.grey),
            ('BACKGROUND',(0,0),(-1,0),colors.lightgrey)
        ]))
        elems.append(t)
        elems.append(Spacer(1,8))
    doc.build(elems)
    buf.seek(0)
    return buf

# ====== Entrada ======
def main():
    try:
        params = st.query_params
    except Exception:
        params = st.experimental_get_query_params()

    init_session()

    mode = params.get("mode", [""])
    mode = mode[0] if isinstance(mode, list) else mode
    _tid = params.get("tid", [""])
    _tid = _tid[0] if isinstance(_tid, list) else _tid

    if mode=="public" and _tid:
        viewer_tournament(_tid, public=True)
        st.caption("Iapps Padel Tournament · iAPPs Pádel — v3.3.30")
        return

    if not st.session_state.get("auth_user"):
        inject_global_layout("No autenticado")
        login_form()
        st.caption("Iapps Padel Tournament · iAPPs Pádel — v3.3.30")
        return

    user = st.session_state["auth_user"]

    user_text = f"Usuario: <b>{user['username']}</b> &nbsp;|&nbsp; Rol: <code>{user['role']}</code> &nbsp;&nbsp;<a href='#' onclick='window.location.reload()'>Cerrar sesión</a>"
    inject_global_layout(user_text)

    top = st.columns([4,3,3,1])
    with top[0]:
        st.markdown(f"**Usuario:** {user['username']} · Rol: `{user['role']}`")
    with top[1]:
        st.link_button("Abrir Super Admin", url="?mode=super")
    with top[2]:
        st.button("Cerrar sesión", on_click=lambda: st.session_state.update({"auth_user":None,"current_tid":None}))
    st.divider()

    if user["role"]=="SUPER_ADMIN" and (mode=="super"):
        admin_dashboard(user)  # reutilizamos dashboard como superadmin view simple
    elif user["role"]=="SUPER_ADMIN":
        admin_dashboard(user)
    elif user["role"]=="TOURNAMENT_ADMIN":
        admin_dashboard(user)
    elif user["role"]=="VIEWER":
        viewer_dashboard(user)
    else:
        st.error("Rol desconocido.")

    st.caption("Iapps Padel Tournament · iAPPs Pádel — v3.3.30")

# Ejecutar
if __name__ == "__main__":
    main()
