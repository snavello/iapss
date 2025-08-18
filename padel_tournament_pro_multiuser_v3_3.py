# app.py — v3.3.22
# - Fix logo visibility issue: removed fixed topbar and red line.
# - Replaced st.experimental_get_query_params with st.query_params.
# - Playoffs according to N qualifiers (2→FN; 4→SF+FN; 8→QF+SF+FN)
# - "Regenerate Playoffs" button
# - Champion highlighted in FINAL
# - Warning + quick JSON restoration (autosave suspended)
# - Fix NameError: init_app()
# - Reworked delete_tournament for better file cleanup robustness.
# - Reworked Playoffs/Tables logic to check for 'groups' to prevent KeyError.

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

st.set_page_config(page_title="Torneo de Pádel — v3.3.22", layout="wide")

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
            return [{"round":"FN","label":"FINAL","a":a,"b":b,"sets":[],"goldenA":0,"goldenB":0}]
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
        <div style="font-size: 0.7rem; color: {DARK_BLUE}; letter-spacing: 0.08rem; white-space: nowrap;">TOURNAMENTS</div>
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
    st.session_state.setdefault("j1_input", "")
    st.session_state.setdefault("j2_input", "")

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

    # st.header("Panel de ADMIN (Super Admin)") # Removed double header

    st.subheader("Configuración de la Aplicación")

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
    try:
        p = tourn_path(tid)
        if p.exists():
            p.unlink()
        for f in (snap_dir_for(tid)).glob("*.json"):
            f.unlink()
        (snap_dir_for(tid)).rmdir()
    except Exception:
        # Si algo falla en la eliminación de archivos, no detiene el programa.
        # El índice ya se actualizó.
        pass

def admin_dashboard(user: Dict[str, Any]):
    user_text = f"Usuario: <b>{user['username']}</b> &nbsp;|&nbsp; Rol: <code>{user['role']}</code> &nbsp;&nbsp;<a href='#' onclick='window.location.reload()'>Cerrar sesión</a>"
    inject_global_layout(user_text)

    st.header(f"Torneos de {user['username']}")

    with st.expander("➕ Crear torneo nuevo", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            t_name = st.text_input("Nombre del torneo", value="Open Pádel")
        with c2:
            place = st.text_input("Lugar / Club", value="Mi Club")
        with c3:
            tdate = st.date_input("Fecha", value=date.today()).isoformat()
        with c4:
            gender = st.selectbox("Género", ["masculino","femenino","mixto"], index=2)
        if st.button("Crear torneo", type="primary"):
            tid = create_tournament(user["username"], t_name, place, tdate, gender)
            st.session_state.current_tid = tid
            st.success(f"Torneo creado: {t_name} ({tid})")
            st.rerun()

    my = load_index_for_admin(user["username"])
    if not my:
        st.info("Aún no tienes torneos.")
    else:
        st.subheader("Abrir / eliminar torneo")
        names = [f"{t['date']} — {t['t_name']} ({t['gender']}) — {t['place']} — ID:{t['tournament_id']}" for t in my]
        # set default index to the one that matches st.session_state.current_tid if it exists
        try:
            default_index = [t['tournament_id'] for t in my].index(st.session_state.get('current_tid'))
        except ValueError:
            default_index = 0
        selected = st.selectbox("Selecciona un torneo", names, index=default_index)
        sel = my[names.index(selected)]
        c1,c2,c3 = st.columns(3)
        with c1:
            if st.button("Abrir torneo"):
                st.session_state.current_tid = sel["tournament_id"]
                st.rerun()
        with c2:
            if st.button("Eliminar torneo", type="secondary"):
                delete_tournament(user["username"], sel["tournament_id"])
                st.success("Torneo eliminado.")
                if st.session_state.get("current_tid")==sel["tournament_id"]:
                    st.session_state.current_tid=None
                st.rerun()
        with c3:
            tid = sel["tournament_id"]
            st.caption("Link público (solo lectura):")
            st.code(f"?mode=public&tid={tid}")

    if st.session_state.get("current_tid"):
        tournament_manager(user, st.session_state["current_tid"])

# ====== Gestor del Torneo ======
def tournament_manager(user: Dict[str,Any], tid: str):
    user_text = f"Usuario: <b>{user['username']}</b> &nbsp;|&nbsp; Rol: <code>{user['role']}</code> &nbsp;&nbsp;<a href='#' onclick='window.location.reload()'>Cerrar sesión</a>"
    inject_global_layout(user_text)

    state = load_tournament(tid)
    state["tournament_id"] = tid # para que no se pierda en la carga
    if not state.get("meta"):
        st.warning("El torneo no se pudo cargar. Puede que haya sido eliminado o el archivo esté dañado.")
        if st.button("Volver al dashboard"):
            st.session_state.current_tid = None
            st.rerun()
        return

    # Autoguardado periódico
    if st.session_state.autosave and st.session_state.get("suspend_autosave_runs", 0) <= 0:
        current_hash = compute_state_hash(state)
        if current_hash != st.session_state.last_hash:
            save_tournament(tid, state, make_snapshot=True)
            st.session_state.last_hash = current_hash
            st.toast("Autoguardado: Estado del torneo guardado.")
    elif st.session_state.get("suspend_autosave_runs", 0) > 0:
        st.session_state.suspend_autosave_runs -= 1


    tabs = ["Config", "Parejas", "Tablas", "Resultados", "Playoffs", "Persistencia"]
    tab_config, tab_pairs, tab_tables, tab_results, tab_ko, tab_persist = st.tabs(tabs)

    # ====== Tab Config ======
    with tab_config:
        st.subheader("Configuración del Torneo")
        with st.form("config_form"):
            new_t_name = st.text_input("Nombre del Torneo", value=state["config"]["t_name"])
            c1,c2,c3,c4 = st.columns(4)
            with c1: num_pairs = st.number_input("Máx. Parejas", min_value=2, max_value=32, value=state["config"]["num_pairs"])
            with c2: num_zones = st.number_input("Nº de Zonas", min_value=1, max_value=8, value=state["config"]["num_zones"])
            with c3: top_per_zone = st.number_input("Clasifican por Zona", min_value=1, value=state["config"]["top_per_zone"])
            with c4: fmt = st.selectbox("Formato de Sets", ["one_set", "best_of_3", "best_of_5"], index=["one_set", "best_of_3", "best_of_5"].index(state["config"]["format"]))
            c1,c2,c3,c4 = st.columns(4)
            with c1: points_win = st.number_input("Puntos por victoria", min_value=0, value=state["config"]["points_win"])
            with c2: points_loss = st.number_input("Puntos por derrota", min_value=0, value=state["config"]["points_loss"])
            with c3: seed = st.number_input("Seed de Sorteo", value=state["config"]["seed"])
            with c4: # Placeholder to keep alignment
                 st.markdown(" ")
            submitted = st.form_submit_button("Guardar configuración", type="primary")
            if submitted:
                state["config"]["t_name"] = new_t_name
                state["config"]["num_pairs"] = int(num_pairs)
                state["config"]["num_zones"] = int(num_zones)
                state["config"]["top_per_zone"] = int(top_per_zone)
                state["config"]["format"] = fmt
                state["config"]["points_win"] = int(points_win)
                state["config"]["points_loss"] = int(points_loss)
                state["config"]["seed"] = int(seed)
                save_tournament(tid, state)
                st.success("Configuración actualizada.")
                st.rerun()

    # ====== Tab Parejas ======
    with tab_pairs:
        st.subheader("Administración de Parejas")
        
        col_list, col_add = st.columns([1,2])
        
        with col_add:
            st.markdown("##### Agregar nueva pareja")
            with st.form("add_pair_form", clear_on_submit=True):
                next_num = next_available_number(state["pairs"], state["config"]["num_pairs"])
                st.text_input(f"Número de pareja (auto): {next_num if next_num else ''}", disabled=True)
                j1_name = st.text_input("Nombre Jugador 1", value=st.session_state.j1_input, key="j1")
                j2_name = st.text_input("Nombre Jugador 2", value=st.session_state.j2_input, key="j2")
                add_pair_button = st.form_submit_button("Agregar pareja", type="primary")

            if add_pair_button:
                if not j1_name or not j2_name:
                    st.error("Por favor, introduce los nombres de ambos jugadores.")
                elif not next_num:
                    st.warning("Se ha alcanzado el número máximo de parejas configurado.")
                else:
                    state["pairs"].append(format_pair_label(next_num, j1_name, j2_name))
                    state["pairs"].sort()
                    save_tournament(tid, state)
                    st.session_state.j1_input = ""
                    st.session_state.j2_input = ""
                    st.rerun()

        with col_list:
            st.markdown("##### Listado de Parejas")
            for i, p in enumerate(state["pairs"]):
                c_pair, c_del = st.columns([4,1])
                with c_pair:
                    st.text_input("Pareja", p, key=f"pair_show_{i}", disabled=True)
                with c_del:
                    if st.button("❌ Eliminar", key=f"del_pair_{i}"):
                        num_to_del = parse_pair_number(p)
                        state["pairs"] = remove_pair_by_number(state["pairs"], num_to_del)
                        state["pairs"].sort()
                        save_tournament(tid, state)
                        st.rerun()

    # ====== Tab Tablas ======
    with tab_tables:
        st.subheader("Tablas de Posiciones")
        if not state["groups"]:
            st.info("No se han generado los grupos. Ve a la pestaña 'Resultados' para generarlos.")
        else:
            st.markdown("""<style>
                .dataframe.dark-header th { background-color: #2f3b52 !important; color:#fff !important; }
                .dataframe.zebra tr:nth-child(even) td { background-color: #f5f7fa !important; }
                .dataframe.zebra tr:nth-child(odd) td  {{ background-color: #ffffff !important; }
            </style>""", unsafe_allow_html=True)
            all_tables = [standings_from_results(f"Z{i+1}", group, state["results"], state["config"]) for i, group in enumerate(state["groups"])]
            for table in all_tables:
                if not table.empty:
                    st.markdown(f"**Tabla de Posiciones - {table.iloc[0]['Zona']}**")
                    st.dataframe(table.style, hide_index=True)
            qualified = qualified_from_tables(all_tables, state["config"]["top_per_zone"])
            st.markdown(f"##### Clasificados a la Fase Final:")
            st.dataframe(pd.DataFrame(qualified, columns=["Zona", "Pos", "Pareja"]))

    # ====== Tab Resultados ======
    with tab_results:
        st.subheader("Carga de Resultados y Fixture")
        if st.session_state.get("pdf_fixture_bytes"):
            st.download_button(
                label="Descargar PDF del Fixture y Zonas",
                data=st.session_state.pdf_fixture_bytes,
                file_name=f"fixture_grupos_{state['meta']['t_name']}_{state['meta']['date']}.pdf",
                mime="application/pdf"
            )

        if not state["groups"]:
            if st.button("Generar Grupos y Fixture", type="primary"):
                if len(state["pairs"]) < state["config"]["num_pairs"]:
                    st.warning("No se ha alcanzado el número mínimo de parejas para el torneo. Asegúrese de que el número de parejas agregadas sea igual al número máximo de parejas para generar los grupos.")
                else:
                    state["groups"] = create_groups(state["pairs"], state["config"]["num_zones"], state["config"]["seed"])
                    state["results"] = build_fixtures(state["groups"])
                    state["ko"]["matches"] = []
                    save_tournament(tid, state, make_snapshot=False)
                    st.success("Grupos y fixture generados.")
                    st.rerun()
        else:
            current_zone = st.selectbox("Selecciona una Zona", [f"Z{i+1}" for i in range(len(state["groups"]))])
            st.info("Para registrar un resultado, marca los sets ganados por cada pareja.")
            with st.container(border=True):
                st.markdown(f"#### Partidos de la {current_zone}")
                
                # Check for completed zone
                is_zone_complete = zone_complete(current_zone, state["results"], state["config"]["format"])
                if is_zone_complete:
                    st.info(f"Todos los partidos de la {current_zone} están completos.")

                for i, match in enumerate(state["results"]):
                    if match["zone"] != current_zone: continue
                    with st.expander(f"{match['pair1']} vs {match['pair2']}", expanded=False):
                        if match_has_winner(match.get("sets",[])):
                            stats = compute_sets_stats(match["sets"])
                            winner_name = match['pair1'] if stats['sets1']>stats['sets2'] else match['pair2']
                            st.markdown(f"**Ganador:** **{winner_name}** <span class='winner-badge'>¡Ganador!</span>", unsafe_allow_html=True)
                        
                        col_p1, col_p2 = st.columns(2)
                        with col_p1: st.markdown(f"**{match['pair1']}**")
                        with col_p2: st.markdown(f"**{match['pair2']}**")
                        
                        num_sets = 1
                        if state["config"]["format"] == "best_of_3": num_sets = 3
                        elif state["config"]["format"] == "best_of_5": num_sets = 5

                        current_sets = match.get("sets", [])
                        new_sets = []
                        valid_scores = True
                        
                        for s_i in range(num_sets):
                            cols = st.columns(2)
                            with cols[0]:
                                g1 = st.number_input(f"Puntos Set {s_i+1}", min_value=0, value=current_sets[s_i]['s1'] if len(current_sets)>s_i else 0, key=f"s1_{i}_{s_i}")
                            with cols[1]:
                                g2 = st.number_input(f"Puntos Set {s_i+1}", min_value=0, value=current_sets[s_i]['s2'] if len(current_sets)>s_i else 0, key=f"s2_{i}_{s_i}")
                            new_sets.append({"s1":g1, "s2":g2})
                            if g1 == g2 and g1 > 0: valid_scores = False
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            golden1 = st.number_input(f"Puntos de oro {match['pair1']}", min_value=0, value=match.get('golden1', 0), key=f"g1_{i}")
                        with c2:
                            golden2 = st.number_input(f"Puntos de oro {match['pair2']}", min_value=0, value=match.get('golden2', 0), key=f"g2_{i}")
                        
                        if st.button("Guardar resultado", key=f"save_{i}"):
                            if not valid_scores:
                                st.error("Los puntos de un set no pueden ser iguales.")
                            else:
                                match["sets"] = new_sets
                                match["golden1"] = golden1
                                match["golden2"] = golden2
                                save_tournament(tid, state, make_snapshot=False)
                                st.success("Resultado guardado.")
                                st.rerun()

    # ====== Tab Playoffs ======
    with tab_ko:
        st.subheader("Fase Final (Playoffs)")
        if not state["groups"]:
            st.info("Aún no se han generado los grupos. Ve a la pestaña 'Resultados' para generarlos.")
        else:
            if st.session_state.get("pdf_playoffs_bytes"):
                st.download_button(
                    label="Descargar PDF de los Playoffs",
                    data=st.session_state.pdf_playoffs_bytes,
                    file_name=f"playoffs_{state['meta']['t_name']}_{state['meta']['date']}.pdf",
                    mime="application/pdf"
                )
            
            qualified_list = qualified_from_tables([standings_from_results(f"Z{i+1}", g, state["results"], state["config"]) for i, g in enumerate(state["groups"])], state["config"]["top_per_zone"])
            
            if not qualified_list:
                st.info("Aún no hay parejas clasificadas a los Playoffs. Asegúrate de que los resultados de los grupos estén completos.")
            else:
                if not state["ko"]["matches"]:
                    if st.button("Generar Playoffs", type="primary"):
                        state["ko"]["matches"] = build_initial_ko(qualified_list)
                        save_tournament(tid, state)
                        st.success("Playoffs generados.")
                        st.rerun()
                else:
                    st.info("Los partidos se irán mostrando a medida que se completen los resultados.")
                    current_round = state["ko"]["matches"][0]["round"]
                    
                    all_matches_complete = True
                    
                    while current_round:
                        round_matches = [m for m in state["ko"]["matches"] if m["round"]==current_round]
                        
                        st.markdown(f"### Ronda: **{current_round}**")
                        cols_in_row = st.columns(2)
                        col_idx = 0
                        
                        for match in round_matches:
                            with cols_in_row[col_idx % 2]:
                                with st.container(border=True):
                                    st.markdown(f"**Partido: {match['label']}**")
                                    c1,c2 = st.columns(2)
                                    with c1: st.markdown(f"**{match['a']}**")
                                    with c2: st.markdown(f"**{match['b']}**")
                                    
                                    # Input for sets
                                    current_sets = match.get("sets", [])
                                    new_sets = []
                                    num_sets = 1
                                    if state["config"]["format"] == "best_of_3": num_sets = 3
                                    elif state["config"]["format"] == "best_of_5": num_sets = 5

                                    for s_i in range(num_sets):
                                        cols_s = st.columns(2)
                                        with cols_s[0]:
                                            g1 = st.number_input(f"Puntos Set {s_i+1}", min_value=0, value=current_sets[s_i]['s1'] if len(current_sets)>s_i else 0, key=f"s1_ko_{match['label']}_{s_i}")
                                        with cols_s[1]:
                                            g2 = st.number_input(f"Puntos Set {s_i+1}", min_value=0, value=current_sets[s_i]['s2'] if len(current_sets)>s_i else 0, key=f"s2_ko_{match['label']}_{s_i}")
                                        new_sets.append({"s1":g1, "s2":g2})

                                    if st.button("Guardar Resultado", key=f"save_ko_{match['label']}"):
                                        match["sets"] = new_sets
                                        match["sets"] = new_sets
                                        save_tournament(tid, state, make_snapshot=False)
                                        st.success("Resultado de playoffs guardado.")
                                        st.rerun()

                                    if match_has_winner(match.get("sets",[])):
                                        stats = compute_sets_stats(match["sets"])
                                        winner_name = match['a'] if stats['sets1']>stats['sets2'] else match['b']
                                        st.markdown(f"**Ganador:** **{winner_name}** <span class='winner-badge'>¡Ganador!</span>", unsafe_allow_html=True)
                                    else:
                                        all_matches_complete = False

                            col_idx += 1
                        
                        if all_matches_complete and current_round != "FN":
                            next_round_name = make_next_round_name(current_round)
                            winners = advance_pairs_from_round(round_matches)
                            
                            if st.button(f"Generar siguiente ronda ({next_round_name})", key=f"next_round_btn_{current_round}"):
                                new_matches = pairs_to_matches(next_round(winners), next_round_name)
                                state["ko"]["matches"].extend(new_matches)
                                save_tournament(tid, state)
                                st.success(f"Ronda de {next_round_name} generada.")
                                st.rerun()
                                
                        current_round = make_next_round_name(current_round) if all_matches_complete else None

    # ====== Tab Persistencia ======
    with tab_persist:
        st.subheader("Persistencia y Copias de Seguridad")

        # Restaurar desde JSON
        uploaded_file = st.file_uploader("Cargar estado del torneo desde un archivo JSON", type="json")
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                state.update(data)
                save_tournament(tid, state, make_snapshot=False)
                st.session_state.autosave = False
                st.session_state.suspend_autosave_runs = 5
                st.success("Estado del torneo restaurado exitosamente. El autoguardado está temporalmente suspendido para evitar sobrescribir la carga.")
                st.rerun()
            except Exception as e:
                st.error(f"Error al cargar el archivo: {e}")

        # Descargar JSON
        state_json = json.dumps(state, indent=2, ensure_ascii=False)
        st.download_button(
            label="Descargar JSON con el estado del torneo",
            data=state_json,
            file_name=f"torneo_{state['meta']['t_name']}_{state['meta']['date']}.json",
            mime="application/json"
        )
        
        st.markdown("---")
        
        st.markdown("#### Configuración de Autoguardado")
        if st.checkbox("Activar autoguardado", value=st.session_state.get("autosave", True), help="Guarda el estado del torneo automáticamente con cada cambio."):
            st.session_state.autosave = True
            st.info("Autoguardado activado.")
        else:
            st.session_state.autosave = False
            st.info("Autoguardado desactivado.")


# ====== PDF Generation Functions ======
def generate_groups_pdf(state):
    if not REPORTLAB_OK: return None
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=cm, leftMargin=cm, topMargin=cm, bottomMargin=cm)
    styles = getSampleStyleSheet()
    Story = []
    Story.append(Paragraph(f"**{state['meta']['t_name']}** - Torneo de Pádel", styles['Title']))
    Story.append(Paragraph(f"Fixture y Zonas - {state['meta']['date']}", styles['Normal']))
    Story.append(Spacer(1, 0.5*cm))

    # Groups
    Story.append(Paragraph("--- Zonas y Parejas ---", styles['h2']))
    for i, group in enumerate(state["groups"]):
        data = [["Pos", "Pareja"]]
        for j, p in enumerate(group, start=1):
            data.append([j, p])
        table = Table(data, colWidths=[1.5*cm, None])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(DARK_BLUE)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        Story.append(Paragraph(f"**Zona {i+1}**", styles['h3']))
        Story.append(table)
        Story.append(Spacer(1, 0.5*cm))

    # Fixture
    Story.append(Paragraph("--- Fixture de Partidos ---", styles['h2']))
    results_df = pd.DataFrame(state["results"])
    results_df = results_df[['zone', 'pair1', 'pair2']]
    data = [results_df.columns.tolist()] + results_df.values.tolist()
    table = Table(data, colWidths=[2*cm, None, None])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(DARK_BLUE)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    Story.append(table)
    doc.build(Story)
    buffer.seek(0)
    return buffer

def generate_ko_pdf(state):
    if not REPORTLAB_OK: return None
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=cm, leftMargin=cm, topMargin=cm, bottomMargin=cm)
    styles = getSampleStyleSheet()
    Story = []
    Story.append(Paragraph(f"**{state['meta']['t_name']}** - Cuadro de Playoffs", styles['Title']))
    Story.append(Paragraph(f"Fase Final - {state['meta']['date']}", styles['Normal']))
    Story.append(Spacer(1, 0.5*cm))

    if not state["ko"]["matches"]:
        Story.append(Paragraph("No hay partidos de playoffs generados.", styles['Normal']))
    else:
        rounds_data = {}
        for m in state["ko"]["matches"]:
            rounds_data.setdefault(m["round"], []).append(m)
        
        for round_name, matches in rounds_data.items():
            Story.append(Paragraph(f"--- Ronda: {round_name} ---", styles['h2']))
            data = [["Partido", "Pareja 1", "Pareja 2", "Resultado"]]
            for m in matches:
                sets_str = ""
                stats = compute_sets_stats(m["sets"])
                for s in m["sets"]:
                    sets_str += f"({s['s1']}-{s['s2']}) "
                winner_text = ""
                if match_has_winner(m["sets"]):
                    winner_text = f"Ganador: {m['a'] if stats['sets1']>stats['sets2'] else m['b']}"
                data.append([m["label"], m["a"], m["b"], f"{sets_str} {winner_text}"])

            table = Table(data, colWidths=[3*cm, 5*cm, 5*cm, None])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(DARK_BLUE)),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            Story.append(table)
            Story.append(Spacer(1, 0.5*cm))

    doc.build(Story)
    buffer.seek(0)
    return buffer


def viewer_tournament(tid: str, public: bool = False):
    st.header("Modo solo lectura")
    if public:
        st.info("Estás viendo este torneo en modo público. No necesitas un usuario para acceder, pero no puedes realizar cambios.")
    
    state = load_tournament(tid)
    if not state.get("meta"):
        st.error("Torneo no encontrado.")
        return
    
    st.title(state["meta"]["t_name"])
    st.markdown(f"**Lugar:** {state['meta']['place']} | **Fecha:** {state['meta']['date']}")

    tabs = ["Parejas", "Tablas", "Playoffs"]
    tab_pairs, tab_tables, tab_ko = st.tabs(tabs)
    
    with tab_pairs:
        st.subheader("Parejas Registradas")
        st.dataframe(pd.DataFrame({"Parejas": state["pairs"]}), hide_index=True)
    
    with tab_tables:
        if not state["groups"]:
            st.info("No se han generado los grupos para este torneo.")
        else:
            st.subheader("Tablas de Posiciones")
            for i, group in enumerate(state["groups"]):
                table = standings_from_results(f"Z{i+1}", group, state["results"], state["config"])
                st.markdown(f"**Zona {i+1}**")
                st.dataframe(table, hide_index=True)
                
    with tab_ko:
        if not state["groups"]:
            st.info("No se han generado los grupos. Ve a la pestaña 'Resultados' para generarlos.")
        else:
            if not state["ko"]["matches"]:
                st.info("No se han generado los playoffs para este torneo.")
            else:
                st.subheader("Cuadro de Playoffs")
                current_round = state["ko"]["matches"][0]["round"]
                while current_round:
                    st.markdown(f"#### Ronda: {current_round}")
                    for m in state["ko"]["matches"]:
                        if m["round"] != current_round: continue
                        with st.container(border=True):
                            winner = "Ganador aún no definido"
                            if match_has_winner(m.get("sets",[])):
                                stats = compute_sets_stats(m["sets"])
                                winner = m['a'] if stats['sets1']>stats['sets2'] else m['b']
                                winner = f"**Ganador:** {winner}"
                            
                            st.markdown(f"**{m['a']}** vs **{m['b']}**")
                            st.markdown(winner)

                    current_round = make_next_round_name(current_round)


def main():
    init_session()
    
    if st.session_state.get("pdf_fixture_bytes") is None and st.session_state.current_tid:
        state = load_tournament(st.session_state.current_tid)
        if state and state["groups"] and REPORTLAB_OK:
            pdf_data = generate_groups_pdf(state)
            st.session_state.pdf_fixture_bytes = pdf_data.getvalue() if pdf_data else None

    if st.session_state.get("pdf_playoffs_bytes") is None and st.session_state.current_tid:
        state = load_tournament(st.session_state.current_tid)
        if state and state["ko"]["matches"] and REPORTLAB_OK:
            pdf_data = generate_ko_pdf(state)
            st.session_state.pdf_playoffs_bytes = pdf_data.getvalue() if pdf_data else None

    if st.query_params.get("mode") == "super":
        if st.session_state.get("auth_user") and st.session_state.auth_user["role"] == "SUPER_ADMIN":
            super_admin_panel()
        else:
            st.warning("Acceso denegado. Solo Super Admin.")
            login_form()
        return

    params = st.query_params
    mode = params.get("mode", [""])[0]
    _tid = params.get("tid", [""])[0]

    if mode=="public" and _tid:
        viewer_tournament(_tid, public=True)
        st.caption("iAPPs Pádel — v3.3.22")
        return

    if not st.session_state.get("auth_user"):
        inject_global_layout("No autenticado")
        login_form()
        st.caption("iAPPs Pádel — v3.3.22")
        return

    user = st.session_state["auth_user"]
    
    if user["role"]=="SUPER_ADMIN":
        super_admin_panel()
    elif user["role"]=="TOURNAMENT_ADMIN":
        admin_dashboard(user)
    elif user["role"]=="VIEWER":
        st.info("Modo solo lectura. Puedes ver los torneos de tu administrador asignado.")
        admin = get_user(user["assigned_admin"])
        if admin:
            st.session_state.current_tid = st.selectbox("Torneo", [t["tournament_id"] for t in load_index_for_admin(admin["username"])], format_func=lambda tid: load_tournament(tid)["meta"]["t_name"])
            if st.session_state.current_tid:
                viewer_tournament(st.session_state.current_tid)
        else:
            st.warning("No tienes un administrador asignado o no se encontró el torneo.")


if __name__ == "__main__":
    main()
