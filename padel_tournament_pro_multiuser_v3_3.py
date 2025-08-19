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
            st.caption("—")

def main_dashboard():
    user = st.session_state["auth_user"]
    if user["role"]=="TOURNAMENT_ADMIN":
        if st.session_state.get("current_tid"):
            tourn_editor(user)
        else:
            admin_dashboard(user)
    elif user["role"]=="SUPER_ADMIN":
        super_admin_panel()
    elif user["role"]=="VIEWER":
        viewer_dashboard(user)

def tourn_editor(user):
    tid = st.session_state.current_tid
    state = load_tournament(tid)
    is_admin = user["username"] == state["meta"]["admin_username"] or user["role"]=="SUPER_ADMIN"
    if not is_admin:
        st.warning("No tienes permisos para editar este torneo.")
        st.session_state.current_tid=None
        st.rerun()
    
    tourn_name = state["meta"]["t_name"]
    user_text = f"Usuario: <b>{user['username']}</b> &nbsp;|&nbsp; Rol: <code>{user['role']}</code> &nbsp;&nbsp;<a href='#' onclick='window.location.reload()'>Cerrar sesión</a>"
    inject_global_layout(user_text)

    # st.header(f"Editor de torneo: {tourn_name}")

    if st.button("← Volver al panel", key="back_to_panel"):
        st.session_state.current_tid = None
        st.rerun()

    tabs = st.tabs(["📝 Configurar", "🗂️ Grupos", "🏆 Playoffs", "🔗 Público"])

    with tabs[0]:
        st.subheader("Configuración del Torneo")
        with st.expander("🛠️ Parámetros básicos", expanded=True):
            c1,c2,c3 = st.columns(3)
            with c1:
                tname = st.text_input("Nombre del torneo", value=state["meta"]["t_name"], key="t_name_edit").strip()
                place = st.text_input("Lugar / Club", value=state["meta"]["place"], key="t_place_edit").strip()
            with c2:
                tdate = st.date_input("Fecha", value=datetime.fromisoformat(state["meta"]["date"]).date(), key="t_date_edit").isoformat()
                gender = st.selectbox("Género", ["masculino","femenino","mixto"], index=["masculino","femenino","mixto"].index(state["meta"]["gender"]), key="t_gender_edit")
            with c3:
                num_pairs_old = state["config"]["num_pairs"]
                num_pairs = st.number_input("N° de parejas", min_value=2, max_value=32, value=num_pairs_old, step=2, key="num_pairs_edit")
                num_zones = st.number_input("N° de zonas", min_value=1, max_value=int(num_pairs/2), value=state["config"]["num_zones"], step=1, key="num_zones_edit")
            
            with st.container(border=True):
                st.caption("Configuración de Puntuación")
                c1,c2,c3 = st.columns(3)
                with c1:
                    win = st.number_input("Puntos por victoria", value=state["config"]["points_win"], step=1, key="pts_win_edit")
                with c2:
                    loss = st.number_input("Puntos por derrota", value=state["config"]["points_loss"], step=1, key="pts_loss_edit")
                with c3:
                    top_per_zone = st.number_input("Clasifican por zona", min_value=1, value=state["config"]["top_per_zone"], step=1, key="top_per_zone_edit")
            
            c1,c2 = st.columns([1,1])
            with c1:
                seeded_mode = st.checkbox("Usar cabezas de serie (1 por zona)", value=state["config"]["seeded_mode"], key="seeded_mode_edit")
            with c2:
                seed_val = st.number_input("Seed (semilla)", value=state["config"]["seed"], step=1, key="seed_edit")

            if st.button("Guardar configuración", type="primary"):
                if num_pairs < len(state["pairs"]):
                    st.error(f"Error: No puedes reducir el número de parejas por debajo de las actuales ({len(state['pairs'])}).")
                elif num_zones > num_pairs:
                    st.error("Error: El número de zonas no puede ser mayor al de parejas.")
                else:
                    state["meta"]["t_name"] = tname
                    state["meta"]["place"] = place
                    state["meta"]["date"] = tdate
                    state["meta"]["gender"] = gender
                    state["config"]["num_pairs"] = num_pairs
                    state["config"]["num_zones"] = num_zones
                    state["config"]["top_per_zone"] = top_per_zone
                    state["config"]["points_win"] = win
                    state["config"]["points_loss"] = loss
                    state["config"]["seed"] = seed_val
                    state["config"]["seeded_mode"] = seeded_mode
                    save_tournament(tid, state, make_snapshot=False)
                    st.success("Configuración actualizada.")
                    st.rerun()

    with tabs[1]:
        st.subheader("Parejas e Instancia de Grupos")
        c1,c2 = st.columns([1,2])
        with c1:
            with st.expander("➕ Agregar / Eliminar parejas", expanded=True):
                j1 = st.text_input("Jugador 1", value=st.session_state.j1_input)
                j2 = st.text_input("Jugador 2", value=st.session_state.j2_input)
                
                next_n = next_available_number(state["pairs"], state["config"]["num_pairs"])
                if next_n is None:
                    st.warning("No se pueden agregar más parejas. Máximo alcanzado.")
                else:
                    if st.button(f"Agregar pareja ({next_n})", type="primary", use_container_width=True):
                        if j1.strip() and j2.strip():
                            pair_label = format_pair_label(next_n, j1, j2)
                            state["pairs"].append(pair_label)
                            state["pairs"].sort() # Ordenar alfabéticamente
                            save_tournament(tid, state)
                            st.session_state.j1_input = ""
                            st.session_state.j2_input = ""
                            st.rerun()
                        else:
                            st.error("Nombres de jugadores requeridos.")
                
                st.markdown("---")
                st.caption("Eliminar pareja por N°")
                pairs_to_remove = [p for p in state["pairs"]]
                to_remove = st.selectbox("Selecciona la pareja a eliminar", options=[""] + pairs_to_remove, key="remove_pair_select")
                
                if st.button("Eliminar pareja", type="secondary", use_container_width=True):
                    if to_remove:
                        n_to_remove = parse_pair_number(to_remove)
                        state["pairs"] = remove_pair_by_number(state["pairs"], n_to_remove)
                        # También eliminar de cabezas de serie si está
                        state["seeded_pairs"] = [p for p in state["seeded_pairs"] if parse_pair_number(p) != n_to_remove]
                        save_tournament(tid, state)
                        st.success(f"Pareja {to_remove} eliminada.")
                        st.rerun()
                        
        with c2:
            st.markdown(f"**Total parejas:** {len(state['pairs'])} de {state['config']['num_pairs']}")
            st.markdown(f"**Parejas por zona:** {int(len(state['pairs'])/state['config']['num_zones']) if state['config']['num_zones']>0 else 0}")
            
            if state["config"]["seeded_mode"]:
                st.caption("Selecciona 1 pareja cabeza de serie por zona:")
                all_pairs = sorted(state["pairs"])
                num_zones = state["config"]["num_zones"]
                new_seeded = st.multiselect(
                    f"Seleccionar {num_zones} cabezas de serie",
                    options=all_pairs,
                    default=state["seeded_pairs"],
                    key="seeded_pairs_select"
                )
                if st.button("Guardar cabezas de serie", use_container_width=True):
                    state["seeded_pairs"] = new_seeded
                    save_tournament(tid, state)
                    st.rerun()

        st.markdown("---")
        if st.button("🔄 Generar Zonas y Fixture", type="primary", use_container_width=True, disabled=len(state["pairs"])<state["config"]["num_zones"]):
            if len(state["pairs"]) < state["config"]["num_zones"]:
                st.warning("Debes tener al menos tantas parejas como zonas para generar el fixture.")
            else:
                groups = create_groups(state["pairs"], state["config"]["num_zones"], state["config"]["seed"], state["config"]["seeded_mode"], state["seeded_pairs"])
                if groups is not None:
                    state["groups"] = groups
                    state["results"] = build_fixtures(groups)
                    state["ko"]["matches"] = [] # Reset KO
                    save_tournament(tid, state)
                    st.success("Zonas y fixture generados.")
                    st.rerun()

        if state["groups"]:
            st.subheader("Fixture y Resultados por Zona")
            cols = st.columns(state["config"]["num_zones"])
            zone_tables=[]
            for zi, group in enumerate(state["groups"], start=0):
                with cols[zi]:
                    st.markdown(f"**Zona {zi+1}**")
                    if not group:
                        st.info("Zona vacía.")
                        continue
                    
                    df = pd.DataFrame(group, columns=["Pareja"])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    st.markdown("---")
                    st.markdown("**:dart: Partidos y resultados**")
                    zone_name = f"Z{zi+1}"
                    zone_matches = [m for m in state["results"] if m["zone"]==zone_name]
                    
                    for i, match in enumerate(zone_matches):
                        st.caption(f"Partido {i+1}: {match['pair1']} vs {match['pair2']}")
                        sets_col = st.columns(3)
                        
                        num_sets = 0
                        if state["config"]["format"] == "one_set":
                            num_sets = 1
                        elif state["config"]["format"] == "best_of_3":
                            num_sets = 3
                        elif state["config"]["format"] == "best_of_5":
                            num_sets = 5

                        # asegurar que hay suficientes sets en el dict
                        while len(match.get("sets", [])) < num_sets:
                            match.setdefault("sets", []).append({"s1":0,"s2":0})

                        for si, s in enumerate(match["sets"]):
                            if si >= num_sets: continue # Solo mostrar los necesarios
                            with sets_col[0]:
                                match["sets"][si]["s1"] = st.number_input(f"Set {si+1} {match['pair1']}", min_value=0, value=s.get("s1",0), key=f"s{si}_{i}_z{zi}_s1")
                            with sets_col[1]:
                                match["sets"][si]["s2"] = st.number_input(f"Set {si+1} {match['pair2']}", min_value=0, value=s.get("s2",0), key=f"s{si}_{i}_z{zi}_s2")
                        
                        with sets_col[2]:
                            st.markdown("<br>",unsafe_allow_html=True) # Espacio para alinear
                            if match_has_winner(match["sets"]):
                                stats = compute_sets_stats(match["sets"])
                                winner_name = match['pair1'] if stats["sets1"] > stats["sets2"] else match['pair2']
                                st.success(f"Ganador: {winner_name}")
                            else:
                                st.warning("Sin ganador")
                               
                        st.markdown("---")

            if st.button("💾 Guardar resultados de grupo", type="primary", use_container_width=True):
                save_tournament(tid, state)
                st.success("Resultados de grupo guardados.")
                st.rerun()

            st.markdown("<br><br>", unsafe_allow_html=True)
            st.subheader("Clasificación por zona")
            zone_tables = []
            for zi, group in enumerate(state["groups"], start=0):
                zone_name = f"Z{zi+1}"
                if not group: continue
                table = standings_from_results(zone_name, group, state["results"], state["config"])
                zone_tables.append(table)
            
            for table in zone_tables:
                st.markdown(f"**Zona {table.iloc[0]['Zona']}**")
                # Add 'Pos' to table before displaying
                table_disp = table.copy()
                table_disp.index = table_disp["Pos"]
                st.dataframe(table_disp.drop(columns=["Pos","Zona"]), use_container_width=True)
                st.markdown("---")
            
            qualified_pairs = qualified_from_tables(zone_tables, state["config"]["top_per_zone"])
            st.markdown(f"**:trophy: Clasificados a Playoffs ({state['config']['top_per_zone']} por zona):**")
            
            qualified_df = pd.DataFrame(qualified_pairs, columns=["Zona","Posición","Pareja"])
            st.dataframe(qualified_df, hide_index=True)

    with tabs[2]:
        st.subheader("Playoffs (Llave de Eliminación)")
        
        qualified_pairs = qualified_from_tables(zone_tables, state["config"]["top_per_zone"])
        n_qualified = len(qualified_pairs)
        n_slots = next_pow2(n_qualified)
        
        st.info(f"Se clasifican {n_qualified} parejas. El cuadro de eliminación tendrá {n_slots} slots.")
        
        if st.button("Generar / Regenerar Llave", type="primary"):
            state["ko"]["matches"] = build_initial_ko(qualified_pairs)
            save_tournament(tid, state, make_snapshot=True)
            st.rerun()

        if state["ko"].get("matches"):
            st.markdown("---")
            matches = state["ko"]["matches"]
            round_matches = {}
            for m in matches:
                round_matches.setdefault(m["round"], []).append(m)
            
            current_round_name = sorted(round_matches.keys(), key=lambda x: ({"FN":3,"SF":2,"QF":1}.get(x,0), x), reverse=True)[0]
            
            st.subheader(f":trophy: {current_round_name} de Final")
            
            cols = st.columns(2)
            for m in round_matches[current_round_name]:
                with cols[0]:
                    with st.container(border=True):
                        st.markdown(f"**{m['label']}**")
                        st.markdown(f"{m['a']} vs {m['b']}")
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            s1 = st.number_input(f"Sets {m['a']}", value=compute_sets_stats(m['sets'])['sets1'], min_value=0, key=f"{m['label']}_s1")
                        with c2:
                            s2 = st.number_input(f"Sets {m['b']}", value=compute_sets_stats(m['sets'])['sets2'], min_value=0, key=f"{m['label']}_s2")
                        
                        sets_for_match = []
                        if s1 > s2:
                            sets_for_match = [{"s1": 1, "s2": 0}] * s1 + [{"s1": 0, "s2": 1}] * s2
                        elif s2 > s1:
                            sets_for_match = [{"s1": 1, "s2": 0}] * s2 + [{"s1": 0, "s2": 1}] * s1
                        
                        m["sets"] = sets_for_match
                        
                        if m['b'] == 'BYE':
                            st.success(f"Ganador: {m['a']}")
                        elif s1 != s2:
                            winner_name = m['a'] if s1 > s2 else m['b']
                            st.success(f"Ganador: {winner_name}")
                        else:
                            st.warning("Aún sin ganador")

            st.button("💾 Guardar resultados de Playoffs", on_click=lambda: save_tournament(tid, state), type="secondary")
            
            winners_this_round = advance_pairs_from_round(round_matches[current_round_name])
            if winners_this_round:
                next_round_name = make_next_round_name(current_round_name)
                if next_round_name:
                    st.info(f"Todas las partidas de {current_round_name} están completas. Listo para generar {next_round_name}.")
                    if st.button(f"➡️ Generar {next_round_name}", type="primary", key="next_round_btn"):
                        slots = winners_this_round
                        next_round_pairs = next_round(slots)
                        next_round_matches = pairs_to_matches(next_round_pairs, next_round_name)
                        state["ko"]["matches"] += next_round_matches
                        save_tournament(tid, state, make_snapshot=True)
                        st.rerun()
                else:
                    st.markdown("---")
                    st.subheader("¡Campeón!")
                    st.markdown(f"**Pareja Campeona: {winners_this_round[0]}**")
                    st.balloons()
            
    with tabs[3]:
        st.subheader("Link de acceso público")
        st.info("Copia y comparte este link para que los participantes y público puedan ver los resultados en vivo (sin poder editarlos).")
        # Generar el link público
        current_host = urlparse(st.experimental_get_query_params().get("host", [""])[0])
        parsed_url = urlparse(st.experimental_get_query_params().get("base_url", [f"http://localhost:8501"])[0])
        
        # Usar la URL base de la app, sin query params, y agregar los nuestros
        base_url = urlunparse(parsed_url._replace(query="", fragment=""))
        
        public_url = f"{base_url}?mode=public&tid={tid}"
        st.code(public_url)
        st.link_button("Abrir link público", url=public_url)

def viewer_dashboard(user):
    user_text = f"Usuario: <b>{user['username']}</b> &nbsp;|&nbsp; Rol: <code>{user['role']}</code> &nbsp;&nbsp;<a href='#' onclick='window.location.reload()'>Cerrar sesión</a>"
    inject_global_layout(user_text)
    
    st.header("Modo solo lectura")
    
    admin = get_user(user["assigned_admin"])
    if not admin:
        st.info("No tienes un administrador asignado.")
        return
        
    st.info(f"Estás viendo los torneos de **{admin['username']}**.")

    my_tourns = load_index_for_admin(admin["username"])
    if not my_tourns:
        st.info("Tu administrador no tiene torneos creados.")
        return
    
    selected_tid = st.selectbox(
        "Selecciona un torneo para ver", 
        options=[t["tournament_id"] for t in my_tourns],
        format_func=lambda tid: f"{load_tournament(tid)['meta']['t_name']} ({load_tournament(tid)['meta']['date']})"
    )
    if selected_tid:
        viewer_tournament(selected_tid)

def viewer_tournament(tid, public=False):
    state = load_tournament(tid)
    if not state:
        st.error("Torneo no encontrado.")
        return
    
    tourn_name = state["meta"]["t_name"]
    
    if public:
        st.markdown(f"### {tourn_name} - Modo Público")
    else:
        st.markdown(f"### {tourn_name} - Solo lectura")
    
    st.caption(f"Lugar: {state['meta']['place']} | Fecha: {state['meta']['date']} | Género: {state['meta']['gender']}")
    
    tabs = st.tabs(["🗂️ Grupos", "🏆 Playoffs"])
    
    with tabs[0]:
        st.subheader("Fixture y Clasificación por Zona")
        if not state["groups"]:
            st.info("Fixture no generado aún.")
            return

        cols = st.columns(state["config"]["num_zones"])
        zone_tables = []
        for zi, group in enumerate(state["groups"], start=0):
            zone_name = f"Z{zi+1}"
            if not group:
                with cols[zi]:
                    st.info("Zona vacía.")
                continue

            with cols[zi]:
                st.markdown(f"**Zona {zone_name}**")
                df = pd.DataFrame(group, columns=["Pareja"])
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.markdown("**:dart: Partidos y resultados**")
                zone_matches = [m for m in state["results"] if m["zone"]==zone_name]
                for match in zone_matches:
                    st.caption(f"{match['pair1']} vs {match['pair2']}")
                    sets_stats = compute_sets_stats(match["sets"])
                    if match_has_winner(match["sets"]):
                        winner_name = match['pair1'] if sets_stats['sets1']>sets_stats['sets2'] else match['pair2']
                        st.markdown(f"**Ganador**: **{winner_name}**")
                    else:
                        st.markdown("Sin ganador aún.")
                    st.markdown(f"Sets: {sets_stats['sets1']} - {sets_stats['sets2']} | Games: {sets_stats['games1']} - {sets_stats['games2']}")
                    st.markdown("---")
            
            table = standings_from_results(zone_name, group, state["results"], state["config"])
            zone_tables.append(table)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("Clasificación")
        for table in zone_tables:
            st.markdown(f"**Zona {table.iloc[0]['Zona']}**")
            table_disp = table.copy()
            table_disp.index = table_disp["Pos"]
            st.dataframe(table_disp.drop(columns=["Pos","Zona"]), use_container_width=True)
            st.markdown("---")

    with tabs[1]:
        st.subheader("Playoffs (Llave de Eliminación)")
        if not state["ko"]["matches"]:
            st.info("Playoffs aún no generados.")
            return
        
        matches = state["ko"]["matches"]
        round_matches = {}
        for m in matches:
            round_matches.setdefault(m["round"], []).append(m)
            
        current_round_name = sorted(round_matches.keys(), key=lambda x: ({"FN":3,"SF":2,"QF":1}.get(x,0), x), reverse=True)[0]
        st.markdown(f"### {current_round_name} de Final")
        
        for m in round_matches[current_round_name]:
            with st.container(border=True):
                st.markdown(f"**{m['label']}**")
                st.markdown(f"**{m['a']}** vs **{m['b']}**")
                
                sets_stats = compute_sets_stats(m["sets"])
                
                winner = None
                if m['b']=='BYE':
                    winner = m['a']
                elif sets_stats['sets1'] != sets_stats['sets2']:
                    winner = m['a'] if sets_stats['sets1']>sets_stats['sets2'] else m['b']
                
                if winner:
                    st.success(f"🏆 Ganador: **{winner}**")
                    if m['label']=="FINAL":
                        st.markdown(f"### ¡FELICITACIONES CAMPEONES!")
                else:
                    st.warning("Sin ganador aún.")
                
                st.markdown(f"Sets: {sets_stats['sets1']} - {sets_stats['sets2']} | Games: {sets_stats['games1']} - {sets_stats['games2']}")

def main():
    init_session()

    if st.query_params.get("mode",[""])[0] == "super":
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
        st.caption("iAPPs Pádel — v3.3.23")
        return

    if not st.session_state.get("auth_user"):
        inject_global_layout("No autenticado")
        login_form()
        st.caption("iAPPs Pádel — v3.3.23")
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
            st.session_state.current_tid = st.selectbox(
                "Torneo", 
                [t["tournament_id"] for t in load_index_for_admin(admin["username"])],
                format_func=lambda tid: load_tournament(tid)["meta"]["t_name"],
                key="viewer_select_tourn"
            )
            if st.session_state.current_tid:
                viewer_tournament(st.session_state.current_tid)
        else:
            st.warning("No tienes un administrador asignado. Contacta al Super Admin.")

if __name__ == "__main__":
    main()
