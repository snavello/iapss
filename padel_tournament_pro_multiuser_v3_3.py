# app.py — v3.3.23
# - Exponer link publico completo + icono copiar al portapapeles.
# - Agregar checkbox 'Usar cabezas de serie'.
# - Sistema de cabezas de serie con 1 por zona.
# - Administracion de parejas: form a la izq, lista a la der, icono de basura para eliminar.
# - Tablas: encabezados en gris, alternancia de colores, icono de check para los clasificados.
# - Corregido el estilo del texto 'TOURNAMENTS'.

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
from urllib.parse import urlparse, urlunparse

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

st.set_page_config(page_title="Torneo de Pádel — v3.3.23", layout="wide")

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
        for old in snaps[KEEP_SNAPSHOTs:]:
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
    "format": "best_of_3",  # one_set | best_of_3 | best_of_5
    "seeded_mode": False
}

rng = lambda off, seed: random.Random(int(seed) + int(off))

def create_groups(pairs, num_groups, seed=42, seeded_mode=False, seeded_pairs=[]):
    r = random.Random(int(seed))
    
    if seeded_mode and len(seeded_pairs) != num_groups:
        st.error(f"Error: Debes seleccionar exactamente {num_groups} parejas como cabezas de serie.")
        return None

    if seeded_mode:
        groups = [[] for _ in range(num_groups)]
        # Asignar un cabeza de serie a cada grupo
        shuffled_seeded = seeded_pairs[:]
        r.shuffle(shuffled_seeded)
        for i, p in enumerate(shuffled_seeded):
            groups[i].append(p)
        
        # Distribuir el resto de las parejas
        unseeded_pairs = [p for p in pairs if p not in seeded_pairs]
        r.shuffle(unseeded_pairs)
        for i, p in enumerate(unseeded_pairs):
            groups[i % num_groups].append(p)

    else:
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
        <div style="font-size: 0.8rem; color: {DARK_BLUE}; letter-spacing: 0.09rem; white-space: nowrap;">TOURNAMENTS</div>
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
      table.dataframe th, table.dataframe td {{ padding: 6px 10px; }}
      .stButton>button {{ height: 100%; }}
      .copy-btn-container {{ display:flex; align-items: center; gap: 5px; }}
      .st-emotion-cache-1r7r32t {{ margin-top: 0; }}
      .compact-table td, .compact-table th { padding: 4px 8px; }
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
    st.session_state.setdefault("current_url", "")
    st.session_state.setdefault("seeded_pairs", [])

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
        "seeded_pairs": [],
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
    # st.header("Panel de ADMIN (Super Admin)")
    # Removed double header
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
            st.write(f"**{usr['username']}** — rol `{usr['role']}` — activo `{usr.get('active',True)}`")
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                new_pin = st.text_input(f"Nuevo PIN para {usr['username']}", key=f"np_{usr['username']}", max_chars=6)
                if st.button(f"Guardar PIN {usr['username']}", key=f"rst_{usr['username']}"):
                    if new_pin and new_pin.isdigit() and len(new_pin)==6:
                        usr["pin_hash"] = sha(new_pin); set_user(usr); st.success("PIN actualizado."); st.rerun()
                    else: st.error("PIN inválido (6 dígitos).")
            with c2:
                if usr["role"] in ["TOURNAMENT_ADMIN","VIEWER"]:
                    if st.button(f"Borrar {usr['username']}", key=f"del_{usr['username']}"):
                        if st.session_state.get(f"confirm_del_{usr['username']}"):
                            users.remove(usr)
                            save_users(users)
                            st.success(f"Usuario {usr['username']} eliminado.")
                            st.session_state[f"confirm_del_{usr['username']}"] = False
                            st.rerun()
                        else:
                            st.session_state[f"confirm_del_{usr['username']}"] = True
                            st.warning("Confirma para borrar: haz clic de nuevo.")
            with c3:
                if usr["username"] != "ADMIN" and usr["role"]!="SUPER_ADMIN":
                    if st.button(f"{'Activar' if not usr.get('active',True) else 'Desactivar'} {usr['username']}", key=f"act_{usr['username']}"):
                        usr["active"] = not usr.get("active", True); set_user(usr); st.success("Estado actualizado."); st.rerun()
            with c4:
                st.write(f"Creado: {usr['created_at'][:10]} por {usr.get('assigned_admin') or 'n/a'}")

def load_index_for_admin(username: str) -> List[Dict[str,Any]]:
    all_tourns = load_index()
    return [t for t in all_tourns if t.get("admin_username")==username]

# ====== Admin Dashboard ======
def admin_dashboard(user: Dict[str, Any]):
    user_text = f"Usuario: <b>{user['username']}</b> &nbsp;|&nbsp; Rol: <code>{user['role']}</code> &nbsp;&nbsp;<a href='#' onclick='window.location.reload()'>Cerrar sesión</a>"
    inject_global_layout(user_text)

    # Sidebar for tournament selection/creation
    st.sidebar.markdown("### Torneos")
    tourn_list = load_index_for_admin(user["username"])
    # If no tournament is selected, default to the first one, if available
    if not st.session_state.current_tid and tourn_list:
        st.session_state.current_tid = tourn_list[0]["tournament_id"]

    tids_with_names = {t["tournament_id"]: t["t_name"] for t in tourn_list}
    selected_tid = st.sidebar.selectbox(
        "Seleccionar Torneo",
        options=list(tids_with_names.keys()),
        format_func=lambda tid: tids_with_names[tid],
        key="tourn_select"
    )

    st.sidebar.divider()
    st.sidebar.markdown("### ➕ Crear nuevo torneo")
    new_t_name = st.sidebar.text_input("Nombre del Torneo", key="new_t_name")
    new_t_place = st.sidebar.text_input("Lugar", key="new_t_place")
    new_t_date = st.sidebar.date_input("Fecha", value=date.today(), key="new_t_date")
    new_t_gender = st.sidebar.selectbox("Género", ["masculino","femenino","mixto"], key="new_t_gender")

    if st.sidebar.button("Crear torneo", type="primary", key="create_tourn_btn"):
        if not new_t_name:
            st.sidebar.error("Nombre requerido.")
        else:
            new_tid = str(uuid.uuid4())
            new_meta = {
                "tournament_id": new_tid,
                "t_name": new_t_name,
                "place": new_t_place,
                "date": new_t_date.isoformat(),
                "gender": new_t_gender,
            }
            new_state = tournament_state_template(user["username"], new_meta)
            save_tournament(new_tid, new_state)
            index = load_index()
            index.append(new_state["meta"])
            save_index(index)
            st.session_state.current_tid = new_tid
            st.sidebar.success(f"Torneo '{new_t_name}' creado.")
            st.rerun()

    # Main content of the dashboard
    if selected_tid:
        st.session_state.current_tid = selected_tid
        tourn_state = load_tournament(selected_tid)
        st.title(tourn_state["meta"]["t_name"])

        # Create tabs
        tab_pairs, tab_config, tab_groups, tab_ko = st.tabs(["Parejas", "Configuración", "Grupos", "Playoffs"])

        with tab_pairs:
            st.subheader("Parejas del Torneo")
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("##### ➕ Agregar/Eliminar")
                j1 = st.text_input("Jugador 1", key="j1_input")
                j2 = st.text_input("Jugador 2", key="j2_input")
                if st.button("Agregar Pareja", type="primary"):
                    if not j1 or not j2:
                        st.error("Jugador 1 y 2 son requeridos.")
                    else:
                        pairs = tourn_state.get("pairs", [])
                        next_n = next_available_number(pairs, tourn_state["config"]["num_pairs"])
                        if next_n is None:
                            st.warning("Número máximo de parejas alcanzado.")
                        else:
                            new_label = format_pair_label(next_n, j1, j2)
                            tourn_state["pairs"].append(new_label)
                            save_tournament(selected_tid, tourn_state)
                            st.session_state.j1_input = ""
                            st.session_state.j2_input = ""
                            st.rerun()

                st.markdown("##### 🗑️ Eliminar")
                pair_to_delete = st.selectbox("Seleccionar pareja a eliminar", options=tourn_state.get("pairs",[]), key="del_pair")
                if st.button("Eliminar pareja seleccionada", type="secondary"):
                    pairs = tourn_state.get("pairs", [])
                    if pair_to_delete in pairs:
                        tourn_state["pairs"].remove(pair_to_delete)
                        save_tournament(selected_tid, tourn_state)
                        st.rerun()
                    else:
                        st.warning("Pareja no encontrada.")

            with col2:
                st.markdown("##### Lista de Parejas")
                pairs_df = pd.DataFrame(tourn_state.get("pairs",[]), columns=["Pareja"])
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)


        with tab_config:
            st.subheader("Configuración del Torneo")
            with st.form("tourn_config_form"):
                cfg = tourn_state["config"]
                t_name = st.text_input("Nombre del Torneo", value=tourn_state["meta"]["t_name"], key="conf_t_name")
                num_pairs = st.number_input("Nº total de parejas", min_value=2, max_value=64, value=cfg.get("num_pairs",16), step=2)
                num_zones = st.number_input("Nº de zonas (grupos)", min_value=1, max_value=8, value=cfg.get("num_zones",4))
                top_per_zone = st.number_input("Clasifican por zona", min_value=1, max_value=num_pairs, value=cfg.get("top_per_zone",2))
                points_win = st.number_input("Puntos por victoria", value=cfg.get("points_win",2))
                points_loss = st.number_input("Puntos por derrota", value=cfg.get("points_loss",0))
                st.checkbox("Usar cabezas de serie", value=cfg.get("seeded_mode",False), key="conf_seeded_mode")
                st.selectbox("Formato de partidos de grupo", options=["one_set","best_of_3","best_of_5"], index=["one_set","best_of_3","best_of_5"].index(cfg.get("format","best_of_3")), key="conf_format")
                seed_value = st.number_input("Semilla para sorteo", value=cfg.get("seed",42), min_value=1)
                submitted = st.form_submit_button("Guardar Configuración", type="primary")

            if submitted:
                # Actualizar state
                tourn_state["meta"]["t_name"] = t_name
                tourn_state["config"].update({
                    "num_pairs": num_pairs,
                    "num_zones": num_zones,
                    "top_per_zone": top_per_zone,
                    "points_win": points_win,
                    "points_loss": points_loss,
                    "seeded_mode": st.session_state["conf_seeded_mode"],
                    "format": st.session_state["conf_format"],
                    "seed": seed_value
                })
                # Re-sortear grupos si cambian las parejas o el número de grupos/semilla
                pairs_changed = len(tourn_state.get("pairs",[])) != num_pairs
                if pairs_changed or tourn_state["groups"] is None:
                    #tourn_state["groups"] = None
                    pass
                save_tournament(selected_tid, tourn_state)
                st.success("Configuración guardada.")
                st.rerun()

            if tourn_state["config"].get("seeded_mode"):
                st.markdown("##### Selección de cabezas de serie")
                # Solo se pueden seleccionar N_grupos parejas
                max_seeds = tourn_state["config"].get("num_zones", 4)
                all_pairs = sorted(tourn_state.get("pairs",[]))
                if not all_pairs:
                    st.info("Agregue parejas para seleccionar cabezas de serie.")
                else:
                    selected_pairs = st.multiselect(
                        f"Selecciona {max_seeds} parejas",
                        options=all_pairs,
                        default=tourn_state["seeded_pairs"],
                        max_selections=max_seeds,
                        key="seeded_select"
                    )
                    tourn_state["seeded_pairs"] = selected_pairs
                    save_tournament(selected_tid, tourn_state)

        with tab_groups:
            st.subheader("Fase de Grupos")
            st.write(f"Jugando a '{tourn_state['config']['format']}'")
            if st.button("Sortear/Regenerar Grupos", type="primary"):
                # Check for minimum pairs for the selected config
                n_pairs_needed = tourn_state["config"]["num_zones"] * (tourn_state["config"]["num_zones"] -1) / 2
                if len(tourn_state.get("pairs",[])) < n_pairs_needed:
                     st.warning(f"Necesitas al menos {n_pairs_needed} parejas para esta configuración.")
                else:
                    groups = create_groups(
                        tourn_state.get("pairs",[]),
                        tourn_state["config"]["num_zones"],
                        tourn_state["config"]["seed"],
                        tourn_state["config"].get("seeded_mode", False),
                        tourn_state["seeded_pairs"]
                    )
                    if groups is not None:
                        tourn_state["groups"] = groups
                        tourn_state["results"] = build_fixtures(groups)
                        save_tournament(selected_tid, tourn_state)
                        st.success("Grupos y fixture generados.")
                        st.rerun()
            if not tourn_state["groups"]:
                st.info("Presiona 'Sortear/Regenerar Grupos' para comenzar.")
            else:
                st.markdown("##### Grupos")
                st.json(tourn_state["groups"])
                st.markdown("##### Fixture")
                with st.container(border=True):
                    for i, m in enumerate(tourn_state["results"]):
                        zone = m["zone"]
                        st.markdown(f"**{zone}**")
                        c1, c2, c3, c4 = st.columns([1,4,4,2])
                        with c1: st.write("Partido:")
                        with c2: st.write(f"**{m['pair1']}**")
                        with c3: st.write(f"vs **{m['pair2']}**")
                        with c4: st.write("Resultado:")
                        
                        num_sets = 1 if tourn_state["config"]["format"]=="one_set" else 3
                        set_cols = st.columns(num_sets*2)
                        
                        sets_in_state = m.get("sets", [])
                        
                        new_sets = []
                        valid_scores = True
                        for s_i in range(num_sets):
                            s1 = set_cols[s_i*2].text_input(f"Set {s_i+1} - {m['pair1']}", value=sets_in_state[s_i].get("s1","") if s_i<len(sets_in_state) else "", key=f"s{i}_{s_i}_p1")
                            s2 = set_cols[s_i*2+1].text_input(f"Set {s_i+1} - {m['pair2']}", value=sets_in_state[s_i].get("s2","") if s_i<len(sets_in_state) else "", key=f"s{i}_{s_i}_p2")
                            
                            try:
                                s1_val = int(s1) if s1.strip() else 0
                                s2_val = int(s2) if s2.strip() else 0
                                new_sets.append({"s1":s1_val, "s2":s2_val})
                            except ValueError:
                                valid_scores = False
                                
                        if st.button("Guardar resultado", key=f"save_match_{i}"):
                            if not valid_scores:
                                st.error("Los scores deben ser números.")
                            else:
                                tourn_state["results"][i]["sets"] = new_sets
                                save_tournament(selected_tid, tourn_state)
                                st.success("Resultado guardado.")
                                st.rerun()
                
                st.markdown("##### Tablas de Posiciones")
                zone_tables = []
                for zi, group in enumerate(tourn_state["groups"], start=1):
                    zone_name = f"Z{zi}"
                    st.markdown(f"**Zona {zi}**")
                    if zone_complete(zone_name, tourn_state["results"], tourn_state["config"]["format"]):
                        table = standings_from_results(zone_name, group, tourn_state["results"], tourn_state["config"])
                        zone_tables.append(table)
                        st.dataframe(table, use_container_width=True, hide_index=True)
                        st.markdown("---")
                    else:
                        st.info("Completa los resultados para ver la tabla.")
                st.session_state.zone_tables = zone_tables
                st.session_state.qualified_pairs = qualified_from_tables(zone_tables, tourn_state["config"]["top_per_zone"])
                
        with tab_ko:
            st.subheader("Playoffs (Eliminatoria)")
            if not st.session_state.get("qualified_pairs"):
                st.warning("Completa la fase de grupos para generar los playoffs.")
            else:
                qualified = st.session_state.qualified_pairs
                
                if st.button("Regenerar Playoffs", type="primary"):
                    st.info("Generando nuevos emparejamientos de playoffs...")
                    tourn_state["ko"]["matches"] = build_initial_ko(qualified)
                    save_tournament(selected_tid, tourn_state)
                    st.rerun()

                # Display KO matches
                ko = tourn_state.get("ko",{})
                matches_by_round = {}
                for m in ko.get("matches",[]):
                    matches_by_round.setdefault(m["round"], []).append(m)

                if matches_by_round:
                    sorted_rounds = sorted(matches_by_round.keys(), key=lambda r: ["QF","SF","FN"].index(r))
                    
                    st.markdown("##### Partidos")
                    for round_name in sorted_rounds:
                        st.subheader(round_name)
                        for i, m in enumerate(matches_by_round[round_name]):
                            
                            if m["a"] == "BYE" or m["b"] == "BYE":
                                st.info(f"**{m['label']}:** {m['a']} vs {m['b']}")
                            else:
                                with st.container(border=True):
                                    c1, c2, c3, c4 = st.columns([1,4,4,2])
                                    with c1: st.write(f"**{m['label']}**")
                                    with c2: st.write(f"**{m['a']}**")
                                    with c3: st.write(f"vs **{m['b']}**")
                                    with c4: st.write("Resultado:")
                                    
                                    num_sets = 1 if tourn_state["config"]["format"]=="one_set" else 3
                                    set_cols = st.columns(num_sets*2)
                                    
                                    sets_in_state = m.get("sets", [])
                                    
                                    new_sets = []
                                    valid_scores = True
                                    for s_i in range(num_sets):
                                        s1 = set_cols[s_i*2].text_input(f"Set {s_i+1} - {m['a']}", value=sets_in_state[s_i].get("s1","") if s_i<len(sets_in_state) else "", key=f"ko_s{round_name}{i}_{s_i}_a")
                                        s2 = set_cols[s_i*2+1].text_input(f"Set {s_i+1} - {m['b']}", value=sets_in_state[s_i].get("s2","") if s_i<len(sets_in_state) else "", key=f"ko_s{round_name}{i}_{s_i}_b")
                                        
                                        try:
                                            s1_val = int(s1) if s1.strip() else 0
                                            s2_val = int(s2) if s2.strip() else 0
                                            new_sets.append({"s1":s1_val, "s2":s2_val})
                                        except ValueError:
                                            valid_scores = False
                                    
                                    if st.button("Guardar resultado", key=f"ko_save_match_{round_name}_{i}"):
                                        if not valid_scores:
                                            st.error("Los scores deben ser números.")
                                        else:
                                            tourn_state["ko"]["matches"][i]["sets"] = new_sets
                                            save_tournament(selected_tid, tourn_state)
                                            st.success("Resultado guardado.")
                                            st.rerun()

                    # Check if round is complete to advance
                    if st.button("Avanzar a la siguiente ronda", type="secondary"):
                        current_round_matches = matches_by_round[sorted_rounds[-1]]
                        if all(match_has_winner(m.get("sets",[])) or m["a"] == "BYE" or m["b"] == "BYE" for m in current_round_matches):
                            winners = advance_pairs_from_round(current_round_matches)
                            if winners:
                                next_r = make_next_round_name(sorted_rounds[-1])
                                if next_r:
                                    next_pairs = next_round(winners)
                                    next_matches = pairs_to_matches(next_pairs, next_r)
                                    tourn_state["ko"]["matches"].extend(next_matches)
                                    save_tournament(selected_tid, tourn_state)
                                    st.success(f"Avanzado a la ronda {next_r}")
                                    st.rerun()
                                else:
                                    st.success(f"¡Torneo finalizado! El campeón es: {winners[0]} 🎉")
                            else:
                                st.warning("No se puede avanzar: resultados de la ronda actual incompletos.")
                        else:
                            st.warning("Completa todos los partidos de la ronda actual antes de avanzar.")

# ====== Viewer ======
def viewer_tournament(tid: str, public: bool=False):
    tourn = load_tournament(tid)
    if not tourn:
        st.error("Torneo no encontrado.")
        return
    
    # if not public:
    #     user = st.session_state["auth_user"]
    #     user_text = f"Usuario: <b>{user['username']}</b> &nbsp;|&nbsp; Rol: <code>{user['role']}</code>"
    #     inject_global_layout(user_text)
    
    st.title(tourn["meta"]["t_name"])
    st.markdown(f"**Lugar:** {tourn['meta']['place']} | **Fecha:** {tourn['meta']['date']}")
    
    tab_groups, tab_ko = st.tabs(["Fase de Grupos", "Playoffs"])
    
    with tab_groups:
        st.subheader("Tablas de Posiciones")
        if tourn.get("groups"):
            for zi, group in enumerate(tourn["groups"], start=1):
                zone_name = f"Z{zi}"
                st.markdown(f"**Zona {zi}**")
                table = standings_from_results(zone_name, group, tourn["results"], tourn["config"])
                st.dataframe(table, use_container_width=True, hide_index=True)
                st.markdown("---")
        else:
            st.info("Aún no se han generado los grupos.")
            
    with tab_ko:
        st.subheader("Playoffs")
        ko = tourn.get("ko", {})
        if ko and ko.get("matches"):
            matches_by_round = {}
            for m in ko.get("matches", []):
                matches_by_round.setdefault(m["round"], []).append(m)
            
            sorted_rounds = sorted(matches_by_round.keys(), key=lambda r: ["QF","SF","FN"].index(r))
            
            for round_name in sorted_rounds:
                st.subheader(round_name)
                for m in matches_by_round[round_name]:
                    match_str = f"**{m['a']}** vs **{m['b']}**"
                    sets = m.get("sets", [])
                    if sets:
                        stats = compute_sets_stats(sets)
                        score_str = " ".join([f"{s['s1']}-{s['s2']}" for s in sets])
                        winner = None
                        if stats["sets1"] > stats["sets2"]:
                            winner = m["a"]
                        elif stats["sets2"] > stats["sets1"]:
                            winner = m["b"]
                        match_str += f" | Score: {score_str}"
                        if winner:
                            match_str += f' <span class="winner-badge">Ganador: {winner}</span>'
                            if round_name=="FN":
                                match_str = f'<div class="champion-banner">¡CAMPEÓN: {winner} 🎉!</div>'
                    
                    st.markdown(f"<p>{match_str}</p>", unsafe_allow_html=True)

        else:
            st.info("Los playoffs no han sido generados todavía.")

# ====== Main ======
def main():
    init_session()
    
    # Manejar query params para acceso público y super admin
    params = st.query_params
    mode = params.get("mode", [""])[0]
    _tid = params.get("tid", [""])[0]

    if mode=="public" and _tid:
        viewer_tournament(_tid, public=True)
        st.caption("iAPPs Pádel — v3.3.23")
        return

    if not st.session_state.get("auth_user"):
        inject_global_layout("No autenticado")
        login_form()
        st.caption("iAPPs Pádel — v3.3.23")
        return

    user = st.session_state["auth_user"]
    
    # En versiones anteriores, esta línea se eliminó. Ahora se mantiene.
    user_text = f"Usuario: <b>{user['username']}</b> &nbsp;|&nbsp; Rol: <code>{user['role']}</code> &nbsp;&nbsp;<a href='#' onclick='window.location.reload()'>Cerrar sesión</a>"
    inject_global_layout(user_text)
    
    if user["role"]=="SUPER_ADMIN":
        super_admin_panel()
    elif user["role"]=="TOURNAMENT_ADMIN":
        admin_dashboard(user)
    elif user["role"]=="VIEWER":
        st.info("Modo solo lectura. Puedes ver los torneos de tu administrador asignado.")
        admin = get_user(user["assigned_admin"])
        if admin:
            st.session_state.current_tid = st.selectbox("Torneo", [t["tournament_id"] for t in load_index_for_admin(admin["username"])], format_func=lambda tid: load_tournament(tid)["meta"]["t_name"])
            viewer_tournament(st.session_state.current_tid)
        else:
            st.warning("No tienes torneos asignados para ver.")

if __name__ == "__main__":
    main()
