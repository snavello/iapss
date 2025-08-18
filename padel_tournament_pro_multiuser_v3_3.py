# app.py — v3.3.19
# - Fix logo visibility issue: removed fixed topbar and red line.
# - Replaced st.experimental_get_query_params with st.query_params.
# - Playoffs according to N qualifiers (2→FN; 4→SF+FN; 8→QF+SF+FN)
# - "Regenerate Playoffs" button
# - Champion highlighted in FINAL
# - Warning + quick JSON restoration (autosave suspended)
# - Fix NameError: init_app()

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

st.set_page_config(page_title="Torneo de Pádel — v3.3.19", layout="wide")

# ====== Estilos / colores ======
PRIMARY_BLUE = "#0D47A1"
LIME_GREEN  = "#AEEA00"
DARK_BLUE   = "#082D63"

# ====== Persistencia local ======
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

# ====== Usuarios ======
DEFAULT_SUPER = {
    "username": "ADMIN", "pin_hash": sha("199601"), "role": "SUPER_ADMIN",
    "assigned_admin": None, "created_at": now_iso(), "active": True
}

def load_users() -> List[Dict[str, Any]]:
    if not USERS_PATH.exists():
        USERS_PATH.write_text(json.dumps([DEFAULT_SUPER], indent=2), encoding="utf-8")
        return [DEFAULT_SUPER]
    try:
        return json.loads(USERS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return [DEFAULT_SUPER]

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

# ====== Reglas / utilidades ======
DEFAULT_CONFIG = {
    "t_name": "Open Pádel",
    "num_pairs": 16,
    "num_zones": 4,
    "top_per_zone": 2,
    "points_win": 2,
    "points_loss": 0,
    "seed": 42,
    "format": "best_of_3"  # one_set | best_of_3 | best_of_5
}

rng = lambda off, seed: random.Random(int(seed) + int(off))

def create_groups(pairs, num_groups, seed=42):
    r = random.Random(int(seed))
    shuffled = pairs[:]
    r.shuffle(shuffled)
    groups = [[] for _ in range(num_groups)]
    for i, p in enumerate(shuffled):
        groups[i % num_groups].append(p)
    return groups

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
        table.at[p, "GP"] += int(m.get("golden1",0))
        table.at[p, "GP"] += int(m.get("golden2",0))
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

# ====== Bracket helpers ======
def next_pow2(n: int) -> int:
    # siguiente potencia de 2 >= n
    p = 1
    while p < n:
        p <<= 1
    return p

def starting_round_name(n_slots: int) -> str:
    # 2→FN ; 4→SF ; 8→QF ; 16→R16 ; etc. (solo usamos QF/SF/FN aquí)
    if n_slots <= 2: return "FN"
    if n_slots <= 4: return "SF"
    return "QF"

def seed_pairs(winners: List[str], runners: List[str]) -> List[Tuple[str,str]]:
    """Empareja ganadores con segundos de otra zona, rotando los segundos."""
    if not winners or not runners:
        return []
    if len(winners) != len(runners):
        # si N no coincide, emparejamos hasta min
        m = min(len(winners), len(runners))
        winners = winners[:m]; runners = runners[:m]
    rr = runners[1:] + runners[:1] if len(runners) > 1 else runners
    return list(zip([w for _,_,w in winners], [r for _,_,r in rr]))

def build_initial_ko(qualified: List[Tuple[str,int,str]]) -> List[Dict[str,Any]]:
    """
    qualified: lista de (zona, pos, pareja).
    Devuelve lista de partidos iniciales con round: FN/SF/QF según N clasificados, evitando BYE cuando N=2 o 4 u 8 exacto.
    Si N no es potencia de 2, reparte BYEs.
    """
    N = len(qualified)
    if N == 0:
        return []
    # dividir por posición
    winners = [q for q in qualified if q[1]==1]
    runners = [q for q in qualified if q[1]==2]
    # slots necesarios
    slots = next_pow2(N)  # 2,4,8,16...
    start_round = starting_round_name(slots)

    # Caso potencia de 2 exacta: emparejamientos sin BYE
    if N == slots:
        # 2 → FINAL directa entre los dos
        if N == 2:
            a = qualified[0][2]; b = qualified[1][2]
            return [{"round":"FN","label":"FINAL","a":a,"b":b,"sets":[],"golden1":0,"golden2":0}]
        # 4 → Semifinales por ganadores/segundos cruzados
        if N == 4:
            pairs = seed_pairs(winners, runners)
            # por si el orden no vino exacto: si seed_pairs devolvió 2 cruces
            out=[]
            labels=["SF1","SF2"]
            for i,(a,b) in enumerate(pairs):
                out.append({"round":"SF","label":labels[i],"a":a,"b":b,"sets":[],"goldenA":0,"goldenB":0})
            # fallback si no se pudo seedear (por posiciones no uniformes)
            if not out:
                names = [q[2] for q in qualified]
                out = [
                    {"round":"SF","label":"SF1","a":names[0],"b":names[3],"sets":[],"goldenA":0,"goldenB":0},
                    {"round":"SF","label":"SF2","a":names[1],"b":names[2],"sets":[],"goldenA":0,"goldenB":0},
                ]
            return out
        # 8 → Cuartos cruzando W vs R
        if N == 8:
            # winners y runners deben tener 4 cada uno
            pairs = seed_pairs(winners, runners)
            # si no coincide por alguna razón, rellenamos secuencial
            if len(pairs) != 4:
                names = [q[2] for q in qualified]
                pairs = [(names[i], names[-(i+1)]) for i in range(4)]
            labels=[f"QF{i}" for i in range(1,5)]
            return [{"round":"QF","label":labels[i],"a":a,"b":b,"sets":[],"goldenA":0,"goldenB":0} for i,(a,b) in enumerate(pairs)]

    # Caso no potencia de 2: BYEs necesarios
    # estrategia: ordenar clasificados (W primero, luego R), y asignar BYEs a los mejores seeds
    names = [q[2] for q in sorted(qualified, key=lambda x:(x[1], x[0]))]  # pos 1 antes que 2
    byes = slots - N
    # construir emparejamientos de la ronda inicial
    start_matches = []
    labels_map = {
        "FN":["FINAL"],
        "SF":["SF1","SF2"],
        "QF":["QF1","QF2","QF3","QF4"]
    }
    labels = labels_map[start_round]
    # convertir lista en slots, insertando BYEs alternando desde el final
    # ejemplo N=6 → slots=8 → 2 byes para seeds más altos
    seeded = names[:]  # ya prioriza pos1
    # crear pares
    i=0; li=0
    while i < len(seeded):
        a = seeded[i]; b = None
        if i+1 < len(seeded):
            b = seeded[i+1]
            i += 2
        else:
            i += 1
        lab = labels[li] if li < len(labels) else f"{start_round}{li+1}"
        # si faltan BYEs, aplicarlos a b
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
            return []  # no se puede avanzar aun
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

# ====== Branding / layout ======
def brand_text_logo() -> str:
    """Genera el logo de texto con ancho uniforme."""
    return f"""
    <div style="font-family: 'Inter', 'Segoe UI', 'Roboto', 'Arial', sans-serif; font-weight: 800; line-height: 1.1; margin-bottom: 0px; padding: 0.5rem 0;">
        <div style="font-size: 1.6rem; color: {DARK_BLUE}; letter-spacing: 0.5rem; white-space: nowrap;">iAPPS</div>
        <div style="font-size: 1.6rem; color: {LIME_GREEN}; letter-spacing: 0.5rem; white-space: nowrap;">PADEL</div>
        <div style="font-size: 1.3rem; color: {DARK_BLUE}; letter-spacing: 0.1rem; white-space: nowrap;">TOURNAMENTS</div>
    </div>
    """

def inject_global_layout(user_info_text: str):
    # App logo is now always the text logo, regardless of config
    logo_html = brand_text_logo()

    st.markdown(f"""
    <style>
      .top-header-container {{
        display: flex; align-items: center; justify-content: space-between; gap: 12px;
        padding: 0.5rem 1rem;
        border-bottom: 1px solid #e5e5e5;
      }}
      .top-header-left {{ display:flex; align-items:center; gap:10px; overflow:visible; }}
      .top-header-right {{ display:flex; align-items:center; gap:12px; font-size:.92rem; color:#333; }}
      .stTabs [data-baseweb="tab-list"] {{
        position: sticky; top: 0px; z-index: 9998; background: white; border-bottom:1px solid #e5e5e5;
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
    <div class="top-header-container">
      <div class="top-header-left">{logo_html}</div>
      <div class="top-header-right">{user_info_text}</div>
    </div>
    """, unsafe_allow_html=True)


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
# ====== Login ======
def login_form():
    st.markdown("### Ingreso — Usuario + PIN (6 dígitos)")
    with st.form("login"):
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

# ====== Super Admin ======
def super_admin_panel():
    user = st.session_state["auth_user"]
    user_text = f"Usuario: <b>{user['username']}</b> &nbsp;|&nbsp; Rol: <code>{user['role']}</code> &nbsp;&nbsp;<a href='#' onclick='window.location.reload()'>Cerrar sesión</a>"
    inject_global_layout(user_text)

    st.header("Panel de ADMIN (Super Admin)")

    with st.expander("🎨 Apariencia (Logo global de la app)", expanded=True):
        app_cfg = load_app_config()
        url = st.text_input(
            "URL pública del logotipo (no por torneo)",
            value=app_cfg.get("app_logo_url", DEFAULT_APP_CONFIG["app_logo_url"])
        ).strip()
        if st.button("Guardar logo global", type="primary"):
            app_cfg["app_logo_url"] = url
            save_app_config(app_cfg)
            st.success("Logo global guardado.")

    st.subheader("➕ Crear usuario")

    if st.session_state.get("sa_clear_form", False):
        st.session_state["sa_new_user"] = ""
        st.session_state["sa_new_pin"] = ""
        st.session_state["sa_clear_form"] = False

    c1,c2,c3,c4 = st.columns([3,3,2,4])
    with c1:
        u = st.text_input("Username nuevo", key="sa_new_user").strip()
    with c2:
        role = st.selectbox("Rol", ["TOURNAMENT_ADMIN","VIEWER"], key="sa_new_role")
    with c3:
        pin = st.text_input("PIN inicial (6)", max_chars=6, key="sa_new_pin").strip()

    assigned_admin=None
    if role=="VIEWER":
        users_all = load_users()
        admins=[x["username"] for x in users_all if x["role"]=="TOURNAMENT_ADMIN" and x.get("active",True)]
        assigned_admin = st.selectbox("Asignar a admin", admins if admins else [""], key="sa_new_assigned")
        if assigned_admin == "": assigned_admin = None

    if st.button("Crear usuario", type="primary", key="sa_create_user_btn"):
        if not u:
            st.error("Username requerido.")
        elif get_user(u):
            st.error("Ya existe.")
        elif len(pin)!=6 or not pin.isdigit():
            st.error("PIN inválido.")
        else:
            set_user({
                "username":u,"pin_hash":sha(pin),"role":role,
                "assigned_admin":assigned_admin,"created_at":now_iso(),"active":True
            })
            st.success(f"Usuario {u} creado.")
            st.session_state["sa_clear_form"] = True
            st.rerun()

    st.subheader("Usuarios existentes")
    users = load_users()
    for usr in users:
        with st.container(border=True):
            st.write(f"**{usr['username']}** — rol `{usr['role']}` — activo `{usr.get("active",True)}`")
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                new_pin = st.text_input(f"Nuevo PIN para {usr['username']}", key=f"np_{usr['username']}", max_chars=6)
                if st.button(f"Guardar PIN {usr['username']}", key=f"rst_{usr['username']}"):
                    if new_pin and new_pin.isdigit() and len(new_pin)==6:
                        usr["pin_hash"] = sha(new_pin); set_user(usr); st.success("PIN actualizado."); st.rerun()
                    else:
                        st.error("PIN inválido (6 dígitos).")
            with c2:
                if usr["role"]=="VIEWER":
                    admins=[x["username"] for x in users if x["role"]=="TOURNAMENT_ADMIN" and x.get("active",True)]
                    new_admin = st.selectbox(f"Admin de {usr['username']}", admins+[None], key=f"adm_{usr['username']}")
                    if st.button(f"Guardar admin {usr['username']}", key=f"sadm_{usr['username']}"):
                        usr["assigned_admin"]=new_admin; set_user(usr); st.success("Asignado."); st.rerun()
                else:
                    st.caption("—")
            with c3:
                active_toggle = st.checkbox("Activo", value=usr.get("active",True), key=f"act_{usr['username']}")
                if st.button(f"Guardar estado {usr['username']}", key=f"sact_{usr['username']}"):
                    usr["active"]=active_toggle; set_user(usr); st.success("Estado guardado."); st.rerun()
            with c4:
                if usr["username"]!="ADMIN" and st.button(f"Inactivar {usr['username']}", key=f"del_{usr['username']}"):
                    usr["active"] = False; set_user(usr); st.success("Inactivado."); st.rerun()

# ====== Admin de torneo ======
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
    try:
        for f in (snap_dir_for(tid)).glob("*.json"):
            try:
                f.unlink()
            except Exception:
                pass
        snap_dir_for(tid).rmdir()
    except Exception:
        pass

# ====== Admin Dashboard ======
def admin_dashboard(user):
    st.header("Panel de Administración de Torneos")
    st.caption(f"Administrador: {user['username']}")

    with st.expander("➕ Crear nuevo torneo", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            t_name = st.text_input("Nombre del torneo", key="new_t_name")
            place = st.text_input("Lugar", key="new_t_place")
        with c2:
            t_date = st.date_input("Fecha", key="new_t_date")
            gender = st.selectbox("Categoría", ["masculino","femenino","mixto"], key="new_t_gender")

        if st.button("Crear Torneo", type="primary", key="create_tournament_btn"):
            if not t_name:
                st.error("El nombre del torneo es obligatorio.")
            else:
                tid = create_tournament(user["username"], t_name, place, t_date.isoformat(), gender)
                st.success(f"Torneo '{t_name}' creado con ID: {tid}")
                st.session_state.current_tid = tid
                st.rerun()

    my_tournaments = load_index_for_admin(user["username"])
    if not my_tournaments:
        st.info("Aún no tienes torneos. Crea uno para empezar.")
        return

    st.subheader("Mis torneos")
    cols = st.columns([1,1,1,1,1,1,1,1,1,1,1,1])
    for i, t in enumerate(my_tournaments):
        with st.container(border=True):
            name_and_date = f"**{t['t_name']}** (`{t['tournament_id']}`)"
            st.markdown(name_and_date)
            st.caption(f"{t['place']} — {t['date']} — Cat: {t['gender']}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Abrir", key=f"open_{t['tournament_id']}", use_container_width=True, type="primary"):
                    st.session_state.current_tid = t["tournament_id"]
                    st.rerun()
            with c2:
                if st.button("Eliminar", key=f"del_{t['tournament_id']}", use_container_width=True):
                    delete_tournament(user["username"], t["tournament_id"])
                    if st.session_state.current_tid == t["tournament_id"]:
                        st.session_state.current_tid = None
                    st.success("Torneo eliminado.")
                    st.rerun()

    if st.session_state.current_tid:
        st.divider()
        st.header(f"Torneo Abierto (`{st.session_state.current_tid}`)")
        try:
            tourn_state = load_tournament(st.session_state.current_tid)
            if not tourn_state:
                st.session_state.current_tid = None
                st.error("Torneo no encontrado. Selecciona otro o crea uno nuevo.")
                st.rerun()

            main_tabs = st.tabs(["📝 Configurar", "🤼 Parejas", "🗓️ Fixture (Grupos)", "🏆 Playoffs"])

            with main_tabs[0]:
                st.markdown("### Configuración General")
                st.caption("Configura los detalles y reglas del torneo.")
                with st.form("tourn_config_form"):
                    cfg = tourn_state.get("config", {})
                    meta = tourn_state.get("meta", {})
                    c1,c2 = st.columns(2)
                    with c1:
                        meta["t_name"] = st.text_input("Nombre del Torneo", value=meta.get("t_name"), key="cfg_t_name")
                        meta["place"] = st.text_input("Lugar", value=meta.get("place",""), key="cfg_place")
                        meta["date"] = st.date_input("Fecha", value=datetime.fromisoformat(meta["date"]), key="cfg_date").isoformat()
                    with c2:
                        cfg["num_pairs"] = st.number_input("Número de Parejas", min_value=2, value=cfg.get("num_pairs"), step=2)
                        cfg["num_zones"] = st.number_input("Número de Zonas (grupos)", min_value=1, value=cfg.get("num_zones"), step=1)
                        cfg["top_per_zone"] = st.number_input("Clasifican por Zona", min_value=1, value=cfg.get("top_per_zone"), step=1)
                        cfg["points_win"] = st.number_input("Puntos por Ganar", min_value=0, value=cfg.get("points_win"), step=1)
                        cfg["points_loss"] = st.number_input("Puntos por Perder", min_value=0, value=cfg.get("points_loss"), step=1)
                        cfg["seed"] = st.number_input("Seed (semilla) para sorteo", min_value=1, value=cfg.get("seed"), step=1)
                        cfg["format"] = st.selectbox("Formato de partidos", ["one_set","best_of_3","best_of_5"], index=["one_set","best_of_3","best_of_5"].index(cfg.get("format","best_of_3")))

                    if st.form_submit_button("Guardar Configuración"):
                        tourn_state["config"] = cfg
                        tourn_state["meta"] = meta
                        save_tournament(st.session_state.current_tid, tourn_state)
                        st.success("Configuración guardada.")
                        st.rerun()

            with main_tabs[1]:
                st.markdown("### Gestión de Parejas")
                st.caption("Agrega, edita o elimina las parejas del torneo.")
                cfg = tourn_state.get("config", {})
                pairs = tourn_state.get("pairs", [])
                
                c1, c2, c3 = st.columns([1,1,1])
                with c1: j1 = st.text_input("Jugador 1", key="new_j1")
                with c2: j2 = st.text_input("Jugador 2", key="new_j2")
                with c3:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    next_n = next_available_number(pairs, cfg.get("num_pairs",16))
                    if st.button("Añadir pareja", use_container_width=True, type="primary"):
                        if j1 and j2 and next_n is not None:
                            new_pair_label = format_pair_label(next_n, j1, j2)
                            pairs.append(new_pair_label)
                            tourn_state["pairs"] = sorted(pairs, key=parse_pair_number)
                            save_tournament(st.session_state.current_tid, tourn_state)
                            st.success(f"Pareja '{new_pair_label}' añadida.")
                            st.rerun()
                        else:
                            st.error("Se necesitan 2 jugadores y el número de parejas no debe superar la capacidad.")

                if pairs:
                    st.divider()
                    st.markdown("#### Parejas Registradas")
                    for p in pairs:
                        c1, c2 = st.columns([4,1])
                        with c1:
                            st.write(p)
                        with c2:
                            n = parse_pair_number(p)
                            if st.button("Eliminar", key=f"del_pair_{n}", use_container_width=True):
                                new_pairs = remove_pair_by_number(pairs, n)
                                tourn_state["pairs"] = new_pairs
                                save_tournament(st.session_state.current_tid, tourn_state)
                                st.success(f"Pareja {p} eliminada.")
                                st.rerun()
            
            with main_tabs[2]:
                st.markdown("### Fixture y Resultados de Grupos")
                st.caption("Aquí se gestionan los partidos de la fase de grupos y sus resultados.")
                cfg = tourn_state.get("config", {})
                pairs = tourn_state.get("pairs", [])
                
                if len(pairs) < cfg["num_pairs"]:
                    st.warning(f"Se necesitan **{cfg['num_pairs']}** parejas para el sorteo. Solo hay **{len(pairs)}**.")
                    if st.button("Volver a parejas", key="back_to_pairs"):
                        st.session_state.update({"current_tab":"parejas"}); st.rerun()
                    st.stop()
                
                if st.button("Generar Grupos y Fixture", type="primary", key="gen_groups"):
                    groups = create_groups(pairs, cfg["num_zones"], cfg["seed"])
                    fixtures = build_fixtures(groups)
                    tourn_state["groups"] = groups
                    tourn_state["results"] = fixtures
                    tourn_state["ko"]["matches"] = [] # reset playoffs
                    tourn_state["ko"]["qualified"] = []
                    save_tournament(st.session_state.current_tid, tourn_state)
                    st.success("Grupos y fixture generados.")
                    st.rerun()

                groups = tourn_state.get("groups", [])
                if not groups:
                    st.info("Genera los grupos para comenzar.")
                else:
                    st.markdown("#### Grupos del Torneo")
                    for i, group in enumerate(groups, start=1):
                        st.subheader(f"Zona {i}")
                        st.markdown(f"**Parejas:** {', '.join(group)}")

                    st.divider()
                    
                    st.markdown("#### Resultados de Partidos")
                    results = tourn_state.get("results", [])
                    st.info("Completa los sets de cada partido para ver el avance. Usa `0` si un jugador no obtuvo sets.")
                    
                    if REPORTLAB_OK and st.session_state.pdf_fixture_bytes:
                        st.download_button(
                            label="Descargar PDF Fixture",
                            data=st.session_state.pdf_fixture_bytes,
                            file_name="fixture_grupos.pdf",
                            mime="application/pdf"
                        )
                    
                    for match in results:
                        zone = match["zone"]
                        pair1 = match["pair1"]
                        pair2 = match["pair2"]
                        
                        st.markdown(f"**{zone}**: {pair1} vs {pair2}")
                        with st.container(border=True):
                            sets = match.get("sets", [])
                            cols = st.columns(3)
                            
                            with cols[0]:
                                golden1 = st.number_input(f"Puntos Golden {pair1}", min_value=0, step=1, value=match.get("golden1",0), key=f"g1_{pair1}_{pair2}")
                            with cols[1]:
                                golden2 = st.number_input(f"Puntos Golden {pair2}", min_value=0, step=1, value=match.get("golden2",0), key=f"g2_{pair1}_{pair2}")

                            st.caption("Sets")
                            
                            num_sets_needed = 3 if cfg["format"]=="best_of_3" else (5 if cfg["format"]=="best_of_5" else 1)
                            st.write(f"Formato: **{cfg['format']}** ({num_sets_needed} sets)")
                            
                            set_cols = st.columns(num_sets_needed*2)
                            
                            new_sets = []
                            for i in range(num_sets_needed):
                                with set_cols[2*i]:
                                    s1 = st.number_input(f"Set {i+1} - {pair1}", min_value=0, step=1, value=sets[i].get("s1",0) if len(sets)>i else 0, key=f"s1_{pair1}_{pair2}_{i}")
                                with set_cols[2*i+1]:
                                    s2 = st.number_input(f"Set {i+1} - {pair2}", min_value=0, step=1, value=sets[i].get("s2",0) if len(sets)>i else 0, key=f"s2_{pair1}_{pair2}_{i}")
                                new_sets.append({"s1":s1, "s2":s2})
                            
                            if st.button("Guardar resultado", key=f"save_{pair1}_{pair2}"):
                                match["sets"] = new_sets
                                match["golden1"] = golden1
                                match["golden2"] = golden2
                                ok, reason = validate_sets(cfg["format"], new_sets)
                                if not ok:
                                    st.warning(f"Error en sets: {reason}")
                                else:
                                    save_tournament(st.session_state.current_tid, tourn_state)
                                    st.success("Resultado guardado.")
                                    st.rerun()

                    st.divider()

                    st.markdown("#### Posiciones de Zonas")
                    all_zones_complete = True
                    qualified_pairs = []
                    for i, group in enumerate(groups, start=1):
                        zone_name = f"Z{i}"
                        if not zone_complete(zone_name, results, cfg["format"]):
                            all_zones_complete = False
                        
                        standings = standings_from_results(zone_name, group, results, cfg)
                        if not standings.empty:
                            st.subheader(f"Posiciones Zona {i}")
                            st.dataframe(standings.set_index("Pos"))
                            
                            if zone_complete(zone_name, results, cfg["format"]):
                                st.success("Zona completa.")
                            else:
                                st.warning("Zona incompleta. Completa todos los partidos.")
                            
                            qualified = qualified_from_tables([standings], cfg["top_per_zone"])
                            qualified_pairs.extend(qualified)

            with main_tabs[3]:
                st.markdown("### Llaves de Playoffs")
                st.caption("Se generan automáticamente una vez que todas las zonas están completas.")

                if not tourn_state.get("groups"):
                    st.info("Primero debes generar los grupos en la pestaña 'Fixture'.")
                    st.stop()

                all_zones_complete = True
                for i in range(cfg["num_zones"]):
                    zone_name = f"Z{i+1}"
                    if not zone_complete(zone_name, tourn_state.get("results",[]), cfg["format"]):
                        all_zones_complete = False
                        break

                if not all_zones_complete:
                    st.warning("Para generar los playoffs, todos los partidos de la fase de grupos deben estar completos.")
                    st.stop()

                ko_state = tourn_state.get("ko", {})
                
                # Regenerate button
                if st.button("Regenerar Playoffs", key="regen_ko_btn"):
                    tourn_state["ko"] = {"matches": []}
                    ko_state = tourn_state.get("ko", {})
                    save_tournament(st.session_state.current_tid, tourn_state)
                    st.rerun()

                if not ko_state["matches"]:
                    st.info("Generando primera ronda de playoffs...")
                    qualified_pairs = []
                    for i, group in enumerate(tourn_state["groups"], start=1):
                        standings = standings_from_results(f"Z{i}", group, tourn_state["results"], cfg)
                        qualified = qualified_from_tables([standings], cfg["top_per_zone"])
                        qualified_pairs.extend(qualified)
                    
                    ko_state["qualified"] = qualified_pairs
                    ko_state["matches"] = build_initial_ko(qualified_pairs)
                    save_tournament(st.session_state.current_tid, tourn_state)
                    st.rerun()

                # Display playoffs
                st.markdown("#### Cuartos de final / Semifinales / Final")
                
                if REPORTLAB_OK and st.session_state.pdf_playoffs_bytes:
                    st.download_button(
                        label="Descargar PDF Playoffs",
                        data=st.session_state.pdf_playoffs_bytes,
                        file_name="playoffs.pdf",
                        mime="application/pdf"
                    )

                round_names = ["QF", "SF", "FN"]
                
                for r_name in round_names:
                    round_matches = [m for m in ko_state["matches"] if m["round"] == r_name]
                    if not round_matches:
                        continue
                    
                    st.markdown(f"##### {r_name} - {len(round_matches)} partidos")
                    
                    all_matches_complete = True
                    for i, match in enumerate(round_matches):
                        if not match_has_winner(match.get("sets",[])):
                            all_matches_complete = False
                            
                        with st.container(border=True):
                            st.write(f"**{match['label']}**: {match['a']} vs {match['b']}")
                            
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                goldenA = st.number_input(f"Puntos Golden {match['a']}", min_value=0, step=1, value=match.get("goldenA",0), key=f"gA_{r_name}_{i}")
                            with c2:
                                goldenB = st.number_input(f"Puntos Golden {match['b']}", min_value=0, step=1, value=match.get("goldenB",0), key=f"gB_{r_name}_{i}")
                            
                            num_sets_needed = 3 if cfg["format"]=="best_of_3" else (5 if cfg["format"]=="best_of_5" else 1)
                            set_cols = st.columns(num_sets_needed*2)
                            
                            new_sets = []
                            sets = match.get("sets", [])
                            for j in range(num_sets_needed):
                                with set_cols[2*j]:
                                    sA = st.number_input(f"Set {j+1} - {match['a']}", min_value=0, step=1, value=sets[j].get("s1",0) if len(sets)>j else 0, key=f"sA_{r_name}_{i}_{j}")
                                with set_cols[2*j+1]:
                                    sB = st.number_input(f"Set {j+1} - {match['b']}", min_value=0, step=1, value=sets[j].get("s2",0) if len(sets)>j else 0, key=f"sB_{r_name}_{i}_{j}")
                                new_sets.append({"s1":sA, "s2":sB})
                                
                            if st.button("Guardar resultado KO", key=f"save_ko_{r_name}_{i}"):
                                ok, reason = validate_sets(cfg["format"], new_sets)
                                if not ok:
                                    st.warning(f"Error en sets: {reason}")
                                else:
                                    match["sets"] = new_sets
                                    match["goldenA"] = goldenA
                                    match["goldenB"] = goldenB
                                    save_tournament(st.session_state.current_tid, tourn_state)
                                    st.success("Resultado guardado.")
                                    st.rerun()
                                
                    if all_matches_complete:
                        st.divider()
                        next_r = make_next_round_name(r_name)
                        if next_r:
                            winners = advance_pairs_from_round(round_matches)
                            st.markdown(f"**Ganadores de {r_name}:** {', '.join(winners)}")
                            
                            existing_next_round = [m for m in ko_state["matches"] if m["round"]==next_r]
                            if not existing_next_round:
                                new_pairs = next_round(winners)
                                new_matches = pairs_to_matches(new_pairs, next_r)
                                ko_state["matches"].extend(new_matches)
                                save_tournament(st.session_state.current_tid, tourn_state)
                                st.success(f"Se generó la siguiente ronda: {next_r}")
                                st.rerun()

                        if r_name=="FN":
                            winner = advance_pairs_from_round(round_matches)
                            if winner:
                                st.balloons()
                                st.markdown(f"#### ¡¡ CAMPEÓN: {winner[0]} !! 🎉")
                                tourn_state["meta"]["champion"] = winner[0]
                                save_tournament(st.session_state.current_tid, tourn_state)
        
        except Exception as e:
            st.error(f"Ocurrió un error al cargar el torneo: {e}")
            st.caption("Volviendo al panel principal...")
            st.session_state.current_tid = None
            st.rerun()

# ====== Public Viewer ======
def viewer_tournament(tid, public=False):
    tourn_state = load_tournament(tid)
    if not tourn_state:
        st.error("Torneo no encontrado.")
        st.stop()
    
    inject_global_layout(f"Visualizando torneo público")
    
    st.header(f"{tourn_state['meta']['t_name']} — {tourn_state['meta']['place']}")
    st.caption(f"Fecha: {tourn_state['meta']['date']} · Categoría: {tourn_state['meta']['gender']}")
    st.divider()
    
    if st.button("Volver", type="secondary"):
        st.session_state.current_tid = None
        st.rerun()

    main_tabs = st.tabs(["🗓️ Grupos", "🏆 Playoffs"])
    
    with main_tabs[0]:
        st.markdown("### Fixture y Posiciones")
        
        cfg = tourn_state.get("config", {})
        groups = tourn_state.get("groups", [])
        if not groups:
            st.info("Fixture no disponible.")
        else:
            all_zones_complete = True
            for i, group in enumerate(groups, start=1):
                st.subheader(f"Zona {i}")
                
                results_in_zone = [m for m in tourn_state.get("results",[]) if m["zone"] == f"Z{i}"]
                
                standings = standings_from_results(f"Z{i}", group, results_in_zone, cfg)
                if not standings.empty:
                    st.dataframe(standings.set_index("Pos"))
                
                if not zone_complete(f"Z{i}", tourn_state.get("results",[]), cfg["format"]):
                    st.warning("Zona incompleta.")
                    all_zones_complete = False
                else:
                    st.success("Zona completa.")
                
    with main_tabs[1]:
        st.markdown("### Llaves de Playoffs")
        ko_state = tourn_state.get("ko", {})
        
        if not ko_state["matches"]:
            st.info("Playoffs aún no disponibles o no generados.")
        else:
            st.subheader("Cuartos de final / Semifinales / Final")
            for r_name in ["QF", "SF", "FN"]:
                round_matches = [m for m in ko_state["matches"] if m["round"] == r_name]
                if not round_matches:
                    continue
                
                st.markdown(f"##### Ronda: {r_name}")
                for match in round_matches:
                    with st.container(border=True):
                        st.markdown(f"**{match['label']}**: **{match['a']}** vs **{match['b']}**")
                        st.markdown(f"Sets: {match.get('sets',[])}")
                        if match_has_winner(match.get('sets',[])):
                            stats = compute_sets_stats(match.get('sets',[]))
                            winner = match['a'] if stats['sets1']>stats['sets2'] else match['b']
                            st.success(f"Ganador: {winner}")
                        else:
                            st.info("Resultado pendiente.")
            
            champion = tourn_state['meta'].get("champion")
            if champion:
                st.balloons()
                st.markdown(f"### ¡¡ CAMPEÓN: {champion} !! 🎉")

# ====== Main App ======
def main():
    init_session()

    # Query params for direct links
    params = st.query_params
    mode = params.get("mode", "")
    _tid = params.get("tid", "")

    if mode=="public" and _tid:
        viewer_tournament(_tid, public=True)
        st.caption("iAPPs Pádel — v3.3.19")
        return

    # Check auth
    if not st.session_state.get("auth_user"):
        inject_global_layout("No autenticado")
        login_form()
        st.caption("iAPPs Pádel — v3.3.19")
        return

    user = st.session_state["auth_user"]

    user_text = f"Usuario: <b>{user['username']}</b> &nbsp;|&nbsp; Rol: <code>{user['role']}</code> &nbsp;&nbsp;<a href='#' onclick='window.location.reload()'>Cerrar sesión</a>"
    inject_global_layout(user_text)

    # st.sidebar.title("Navegación")
    # if st.sidebar.button("Cerrar sesión"):
    #     st.session_state.auth_user = None
    #     st.session_state.current_tid = None
    #     st.rerun()

    if user["role"]=="SUPER_ADMIN":
        super_admin_panel()
    elif user["role"]=="TOURNAMENT_ADMIN":
        admin_dashboard(user)
    elif user["role"]=="VIEWER":
        st.info("Modo solo lectura. Puedes ver los torneos de tu administrador asignado.")
        admin = get_user(user["assigned_admin"])
        if not admin:
            st.error("Administrador asignado no encontrado. Contacta al super admin.")
        else:
            admin_dashboard(admin)

if __name__ == "__main__":
    main()
