# app.py — v3.3.24
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

st.set_page_config(page_title="Torneo de Pádel — v3.3.24", layout="wide")

# ====== Estilos / colores ======
PRIMARY_BLUE = "#0D47A1"
LIME_GREEN  = "#AEEA00"
DARK_BLUE   = "#082D63"
DARK_GREY   = "#2f3b52"
LIGHT_GREY  = "#f5f7fa"

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
    "format": "best_of_3",  # one_set | best_of_3 | best_of_5
    "seeded_mode": False
}

rng = lambda off, seed: random.Random(int(seed) + int(off))

def create_groups(pairs, num_groups, seeded_mode=False, seeded_pairs=None, seed=42):
    r = random.Random(int(seed))
    groups = [[] for _ in range(num_groups)]
    
    if seeded_mode and seeded_pairs:
        # Asignar los cabezas de serie, uno por zona, de forma aleatoria.
        shuffled_seeded = seeded_pairs[:]
        r.shuffle(shuffled_seeded)
        for i, p in enumerate(shuffled_seeded):
            groups[i % num_groups].append(p)
        
        # Distribuir el resto de las parejas.
        non_seeded_pairs = [p for p in pairs if p not in seeded_pairs]
        shuffled_non_seeded = non_seeded_pairs[:]
        r.shuffle(shuffled_non_seeded)
        
        group_idx = 0
        for p in shuffled_non_seeded:
            # Encontrar el siguiente grupo con espacio
            while len(groups[group_idx % num_groups]) >= (len(pairs) / num_groups):
                group_idx += 1
            groups[group_idx % num_groups].append(p)
            group_idx += 1
    else:
        shuffled = pairs[:]
        r.shuffle(shuffled)
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
      .dark-header th {{ background-color: {DARK_GREY} !important; color:#fff !important; }}
      .zebra tr:nth-child(even) td  {{ background-color: {LIGHT_GREY} !important; }}
      .zebra tr:nth-child(odd) td   {{ background-color: #ffffff !important; }}
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
        "seeded_pairs": [],  # Nuevo campo para almacenar las parejas cabeza de serie
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
                if st.button("Cambiar PIN", type="primary", key=f"cpin_{usr['username']}"):
                    if len(new_pin)!=6 or not new_pin.isdigit():
                        st.error("PIN inválido.")
                    else:
                        usr["pin_hash"] = sha(new_pin)
                        set_user(usr)
                        st.success("PIN actualizado.")
                        st.rerun()
            with c2:
                new_role = st.selectbox("Cambiar rol a", ["TOURNAMENT_ADMIN", "VIEWER"], key=f"nr_{usr['username']}", index=["TOURNAMENT_ADMIN","VIEWER"].index(usr["role"]))
                if st.button("Cambiar rol", key=f"cr_{usr['username']}"):
                    if new_role != usr["role"]:
                        usr["role"] = new_role
                        set_user(usr)
                        st.success("Rol actualizado.")
                        st.rerun()
            with c3:
                action = "Desactivar" if usr.get("active",True) else "Activar"
                if st.button(action, key=f"act_{usr['username']}"):
                    usr["active"] = not usr.get("active", True)
                    set_user(usr)
                    st.success(f"Usuario {usr['username']} {'desactivado' if not usr['active'] else 'activado'}.")
                    st.rerun()
            with c4:
                if st.button("Eliminar usuario", key=f"del_{usr['username']}"):
                    if usr["username"] == "ADMIN":
                        st.error("No se puede eliminar el usuario 'ADMIN'.")
                    else:
                        users_list = load_users()
                        users_list = [u for u in users_list if u['username'] != usr['username']]
                        save_users(users_list)
                        st.success(f"Usuario {usr['username']} eliminado.")
                        st.rerun()

# ====== Admin (Torneos) ======
def load_index_for_admin(admin_username: str) -> List[Dict[str, Any]]:
    return [t for t in load_index() if t.get("admin_username")==admin_username]

def admin_dashboard(admin_user: Dict[str, Any]):
    user_text = f"Usuario: <b>{admin_user['username']}</b> &nbsp;|&nbsp; Rol: <code>{admin_user['role']}</code> &nbsp;&nbsp;<a href='#' onclick='window.location.reload()'>Cerrar sesión</a>"
    inject_global_layout(user_text)

    # st.header(f"Dashboard - {admin_user['username']}") # Removed double header
    st.markdown("### Mis Torneos")

    # Lista de torneos del admin
    index = load_index_for_admin(admin_user["username"])
    if not index:
        st.info("Aún no tienes torneos.")
    
    col_sel, col_new = st.columns([1,1])
    with col_sel:
        # Selector de torneo activo
        current_tid = st.selectbox("Seleccionar Torneo", [""] + [t["tournament_id"] for t in index], format_func=lambda tid: load_tournament(tid)["meta"]["t_name"] if tid else "— Nuevo Torneo —")
        if current_tid != st.session_state.current_tid:
            st.session_state.current_tid = current_tid
            st.session_state.last_hash = "" # reset hash to force load
            st.rerun()

    with col_new:
        if st.session_state.current_tid == "" and st.session_state.get("new_tourn_name"):
            st.session_state["new_tourn_name"] = ""
        with st.form("new_tourn_form"):
            t_name = st.text_input("Nombre del nuevo torneo", key="new_tourn_name").strip()
            t_id_suf = st.text_input("Identificador URL (opcional)", help="Si no lo pones, se generará uno aleatorio.").strip()
            date_col, place_col = st.columns(2)
            with date_col:
                date_str = st.date_input("Fecha", value=date.today(), key="new_tourn_date").isoformat()
            with place_col:
                place = st.text_input("Lugar (opcional)", key="new_tourn_place").strip()
            gender = st.selectbox("Género", ["mixto","masculino","femenino"], key="new_tourn_gender")
            if st.form_submit_button("Crear nuevo torneo", type="primary"):
                if not t_name:
                    st.error("El nombre es requerido.")
                else:
                    tid = t_id_suf or str(uuid.uuid4())
                    if any(t["tournament_id"] == tid for t in index):
                        st.error("Ya existe un torneo con ese identificador. Elige otro.")
                    else:
                        index.append({
                            "tournament_id": tid,
                            "t_name": t_name,
                            "admin_username": admin_user["username"],
                            "created_at": now_iso()
                        })
                        save_index(index)
                        new_tourn_state = tournament_state_template(admin_user["username"], {"tournament_id":tid, "t_name":t_name, "place":place, "date":date_str, "gender":gender})
                        save_tournament(tid, new_tourn_state, make_snapshot=False)
                        st.session_state.current_tid = tid
                        st.session_state.last_hash = "" # force reload
                        st.success(f"Torneo '{t_name}' creado.")
                        st.rerun()

    # Si hay torneo seleccionado, mostrar el editor
    if st.session_state.current_tid:
        tourn_tid = st.session_state.current_tid
        tourn_state = load_tournament(tourn_tid)
        st.session_state.last_hash = compute_state_hash(tourn_state)

        t_name = tourn_state["meta"]["t_name"]
        st.subheader(f"🛠️ {t_name}")

        with st.columns(3)[0]:
            if st.button("Eliminar torneo", key=f"del_tourn_{tourn_tid}"):
                if st.warning(f"¿Estás seguro de eliminar el torneo '{t_name}'? Esta acción es irreversible."):
                    try:
                        tourn_path(tourn_tid).unlink(missing_ok=True)
                        for f in snap_dir_for(tourn_tid).glob("snapshot_*.json"): f.unlink()
                        tourn_state = None
                        st.session_state.current_tid = ""
                        st.session_state.last_hash = ""
                        st.success("Torneo eliminado.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al eliminar: {e}")

        # TABS
        tab_config, tab_pairs, tab_zones, tab_playoffs, tab_standings = st.tabs(["📋 Configuración", "👫 Parejas", "📍 Zonas", "🏆 Playoffs", "📊 Posiciones"])

        with tab_config:
            st.markdown("### Configuración")
            current_config = tourn_state["config"]
            
            with st.form("tourn_config_form"):
                c_name, c_url = st.columns([0.7, 0.3])
                with c_name:
                    tourn_state["meta"]["t_name"] = st.text_input("Nombre del torneo", value=tourn_state["meta"]["t_name"], key="c_name")
                with c_url:
                    public_url = f"{urlunparse(urlparse(st.experimental_get_query_params.get('__query_params__'))[0:3] + (f'mode=public&tid={tourn_tid}', '', ''))}"
                    st.text_input("Link público", value=public_url, key="public_link", disabled=True)
                    st.button("📋 Icono de copiar", on_click=lambda: st.components.v1.html(f"""
                        <script>
                            navigator.clipboard.writeText("{public_url}").then(() => {{
                                alert("Link copiado!");
                            }});
                        </script>
                    """, height=0, width=0))
                
                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    tourn_state["meta"]["place"] = st.text_input("Lugar", value=tourn_state["meta"]["place"], key="c_place")
                with c2:
                    tourn_state["meta"]["date"] = st.text_input("Fecha", value=tourn_state["meta"]["date"], key="c_date")
                with c3:
                    tourn_state["meta"]["gender"] = st.selectbox("Género", ["mixto","masculino","femenino"], index=["mixto","masculino","femenino"].index(tourn_state["meta"]["gender"]), key="c_gender")
                
                st.markdown("---")
                
                c_pairs, c_zones, c_top, c_seed = st.columns(4)
                with c_pairs:
                    tourn_state["config"]["num_pairs"] = st.number_input("Nº total de parejas", min_value=2, value=current_config.get("num_pairs", DEFAULT_CONFIG["num_pairs"]))
                with c_zones:
                    tourn_state["config"]["num_zones"] = st.number_input("Nº de zonas", min_value=1, value=current_config.get("num_zones", DEFAULT_CONFIG["num_zones"]))
                with c_top:
                    tourn_state["config"]["top_per_zone"] = st.number_input("Pasan a Playoff", min_value=1, value=current_config.get("top_per_zone", DEFAULT_CONFIG["top_per_zone"]))
                with c_seed:
                    tourn_state["config"]["seed"] = st.number_input("Seed (semilla de sorteo)", value=current_config.get("seed", DEFAULT_CONFIG["seed"]))
                
                c_seeded_mode, c_format = st.columns(2)
                with c_seeded_mode:
                    tourn_state["config"]["seeded_mode"] = st.checkbox("Usar cabezas de serie", value=current_config.get("seeded_mode", DEFAULT_CONFIG["seeded_mode"]), key="seeded_mode_checkbox")
                with c_format:
                    tourn_state["config"]["format"] = st.selectbox("Formato de partidos", ["one_set","best_of_3","best_of_5"], index=["one_set","best_of_3","best_of_5"].index(current_config.get("format",DEFAULT_CONFIG["format"])))
                
                c_pts_w, c_pts_l = st.columns(2)
                with c_pts_w:
                    tourn_state["config"]["points_win"] = st.number_input("Puntos por victoria", value=current_config.get("points_win", DEFAULT_CONFIG["points_win"]))
                with c_pts_l:
                    tourn_state["config"]["points_loss"] = st.number_input("Puntos por derrota", value=current_config.get("points_loss", DEFAULT_CONFIG["points_loss"]))
                
                submitted = st.form_submit_button("Guardar configuración", type="primary")
                if submitted:
                    st.success("Configuración guardada.")
                    save_tournament(tourn_tid, tourn_state)
                    st.rerun()

        with tab_pairs:
            st.markdown("### Administración de Parejas")
            st.info("Ingresa el nombre de los jugadores para formar las parejas. Se les asignará un número automáticamente.")
            
            form_col, list_col = st.columns(2)
            
            with form_col:
                with st.form("add_pair"):
                    if st.session_state.get("j1_input"): st.session_state["j1_input"] = ""
                    if st.session_state.get("j2_input"): st.session_state["j2_input"] = ""
                    j1 = st.text_input("Jugador 1", key="j1_input").strip()
                    j2 = st.text_input("Jugador 2", key="j2_input").strip()
                    submitted = st.form_submit_button("Agregar pareja", type="primary")
                    
                    if submitted:
                        if not j1 or not j2:
                            st.warning("Debes ingresar el nombre de ambos jugadores.")
                        else:
                            pair_number = next_available_number(tourn_state["pairs"], tourn_state["config"]["num_pairs"])
                            if pair_number is not None:
                                new_pair_label = format_pair_label(pair_number, j1, j2)
                                tourn_state["pairs"].append(new_pair_label)
                                tourn_state["pairs"] = sorted(tourn_state["pairs"])
                                save_tournament(tourn_tid, tourn_state)
                                st.success(f"Pareja '{j1} / {j2}' agregada.")
                                st.rerun()
                            else:
                                st.error("Ya has alcanzado el número máximo de parejas configurado.")

            with list_col:
                if tourn_state["config"]["seeded_mode"]:
                    num_zones = tourn_state["config"]["num_zones"]
                    st.markdown(f"**Seleccionar hasta {num_zones} cabezas de serie:**")
                    seeded_pairs = st.multiselect(
                        "Selecciona las parejas cabezas de serie",
                        options=tourn_state["pairs"],
                        default=tourn_state["seeded_pairs"]
                    )
                    if len(seeded_pairs) > num_zones:
                        st.warning(f"No puedes seleccionar más de {num_zones} cabezas de serie.")
                    else:
                        tourn_state["seeded_pairs"] = seeded_pairs
                        if st.button("Guardar cabezas de serie", type="secondary"):
                            save_tournament(tourn_tid, tourn_state)
                            st.success("Cabezas de serie guardados.")
                            st.rerun()

                st.markdown("### Lista de Parejas")
                if not tourn_state["pairs"]:
                    st.info("No hay parejas ingresadas.")
                else:
                    for pair in tourn_state["pairs"]:
                        p_col, d_col = st.columns([0.9, 0.1])
                        with p_col:
                            is_seeded = pair in tourn_state.get("seeded_pairs", [])
                            pair_display = f"**{pair}** "
                            if is_seeded:
                                pair_display += " ⭐"
                            st.markdown(pair_display)
                        with d_col:
                            if st.button("🗑️", key=f"del_pair_{pair}"):
                                pair_number = parse_pair_number(pair)
                                tourn_state["pairs"] = remove_pair_by_number(tourn_state["pairs"], pair_number)
                                # Eliminar de la lista de cabezas de serie si estaba
                                if pair in tourn_state.get("seeded_pairs", []):
                                    tourn_state["seeded_pairs"].remove(pair)
                                save_tournament(tourn_tid, tourn_state)
                                st.success(f"Pareja '{pair}' eliminada.")
                                st.rerun()

        with tab_zones:
            st.markdown("### Sorteo y Fixture de Zonas")
            col_sort, col_regen = st.columns(2)
            with col_sort:
                if st.button("Sortear parejas", type="primary"):
                    if len(tourn_state["pairs"]) < tourn_state["config"]["num_pairs"]:
                        st.warning("El número de parejas ingresadas es menor que el configurado. ¿Deseas continuar?")
                        if st.button("Sí, continuar"):
                            tourn_state["config"]["num_pairs"] = len(tourn_state["pairs"])
                    
                    if len(tourn_state["pairs"]) % tourn_state["config"]["num_zones"] != 0:
                        st.warning("El número de parejas no es divisible por el número de zonas. Las zonas tendrán un número desigual de parejas.")

                    if tourn_state["config"]["seeded_mode"] and len(tourn_state["seeded_pairs"]) != tourn_state["config"]["num_zones"]:
                        st.warning(f"Debes seleccionar exactamente {tourn_state['config']['num_zones']} parejas cabezas de serie.")
                    else:
                        tourn_state["groups"] = create_groups(tourn_state["pairs"], tourn_state["config"]["num_zones"], tourn_state["config"]["seeded_mode"], tourn_state.get("seeded_pairs"), tourn_state["config"]["seed"])
                        tourn_state["results"] = build_fixtures(tourn_state["groups"])
                        tourn_state["ko"]["matches"] = []
                        st.success("Sorteo realizado. Grupos y fixture generados.")
                        save_tournament(tourn_tid, tourn_state)
                        st.rerun()
            
            with col_regen:
                if st.button("Regenerar Playoff", type="secondary"):
                    if tourn_state.get("groups"):
                        q = qualified_from_tables(
                            [standings_from_results(f"Z{i+1}", g, tourn_state["results"], tourn_state["config"]) for i,g in enumerate(tourn_state["groups"])],
                            tourn_state["config"]["top_per_zone"]
                        )
                        tourn_state["ko"]["matches"] = build_initial_ko(q)
                        st.success("Playoff regenerado.")
                        save_tournament(tourn_tid, tourn_state)
                        st.rerun()
            
            if tourn_state.get("groups"):
                st.subheader("Grupos")
                cols = st.columns(tourn_state["config"]["num_zones"])
                for i, group in enumerate(tourn_state["groups"]):
                    with cols[i]:
                        st.markdown(f"**Zona {i+1}**")
                        for p in group:
                            is_seeded = p in tourn_state.get("seeded_pairs", [])
                            st.markdown(f"- {p} {'(CS)' if is_seeded else ''}")
                
                st.subheader("Fixture de Zonas")
                for m in tourn_state["results"]:
                    with st.container(border=True):
                        st.markdown(f"**{m['zone']}** - {m['pair1']} vs {m['pair2']}")
                        fmt = tourn_state["config"]["format"]
                        is_complete = match_has_winner(m.get("sets", []))
                        
                        cols = st.columns([1,1,0.5])
                        sets = m.get("sets",[])
                        if st.button("Registrar resultado", key=f"res_{m['pair1']}_{m['pair2']}_{m['zone']}"):
                            if is_complete:
                                st.session_state[f"sets_{m['pair1']}_{m['pair2']}"] = sets
                            st.session_state[f"show_res_{m['pair1']}_{m['pair2']}"] = True
                            st.rerun()
                            
                        if st.session_state.get(f"show_res_{m['pair1']}_{m['pair2']}"):
                            with st.form(f"f_{m['pair1']}_{m['pair2']}"):
                                st.write(f"Sets para **{m['pair1']}** vs **{m['pair2']}**")
                                existing_sets = st.session_state.get(f"sets_{m['pair1']}_{m['pair2']}", [{"s1":0,"s2":0}] * (1 if fmt=="one_set" else 3))
                                new_sets = []
                                for i in range(len(existing_sets)):
                                    c1,c2 = st.columns(2)
                                    with c1:
                                        s1 = st.number_input(f"Set {i+1} - {m['pair1']}", min_value=0, value=existing_sets[i]["s1"], key=f"s{i}_1_{m['pair1']}_{m['pair2']}")
                                    with c2:
                                        s2 = st.number_input(f"Set {i+1} - {m['pair2']}", min_value=0, value=existing_sets[i]["s2"], key=f"s{i}_2_{m['pair1']}_{m['pair2']}")
                                    new_sets.append({"s1":s1,"s2":s2})

                                s_submitted = st.form_submit_button("Guardar resultado")
                                if s_submitted:
                                    m["sets"] = new_sets
                                    ok, err = validate_sets(fmt, m["sets"])
                                    if ok:
                                        save_tournament(tourn_tid, tourn_state)
                                        st.success("Resultado guardado.")
                                        st.session_state[f"show_res_{m['pair1']}_{m['pair2']}"] = False
                                        st.rerun()
                                    else:
                                        st.error(err)

        with tab_standings:
            st.markdown("### Posiciones por Zona")
            if tourn_state.get("groups"):
                zone_tables = [standings_from_results(f"Z{i+1}", g, tourn_state["results"], tourn_state["config"]) for i,g in enumerate(tourn_state["groups"])]
                cols = st.columns(tourn_state["config"]["num_zones"])
                for i, t in enumerate(zone_tables):
                    with cols[i]:
                        st.markdown(f"**Posiciones - Zona {i+1}**")
                        if not t.empty:
                            is_complete = zone_complete(f"Z{i+1}", tourn_state["results"], tourn_state["config"]["format"])
                            # Add qualified checkmark
                            if is_complete:
                                for j in range(len(t)):
                                    if t.iloc[j]["Pos"] <= tourn_state["config"]["top_per_zone"]:
                                        t.at[j, "pair"] = f"{t.iloc[j]['pair']} ✅"
                            
                            st.dataframe(t.style.set_table_styles([
                                {'selector': 'th', 'props': [('background-color', DARK_GREY), ('color', 'white')]},
                                {'selector': 'tr:nth-child(even)', 'props': [('background-color', LIGHT_GREY)]},
                                {'selector': 'tr:nth-child(odd)', 'props': [('background-color', 'white')]}
                            ]).hide(axis="index"), use_container_width=True)


def viewer_tournament(tid: str, public: bool=False):
    tourn_state = load_tournament(tid)
    if not tourn_state:
        st.warning("Torneo no encontrado.")
        return

    st.markdown("---")
    st.markdown(f"## {tourn_state['meta']['t_name']}")
    
    tab_config, tab_pairs, tab_zones, tab_playoffs, tab_standings = st.tabs(["📋 Configuración", "👫 Parejas", "📍 Zonas", "🏆 Playoffs", "📊 Posiciones"])

    with tab_config:
        st.markdown("### Configuración del Torneo")
        st.json(tourn_state["config"])
        st.json(tourn_state["meta"])

    with tab_pairs:
        st.markdown("### Lista de Parejas")
        for pair in tourn_state["pairs"]:
            is_seeded = pair in tourn_state.get("seeded_pairs", [])
            st.markdown(f"- {pair} {'(CS)' if is_seeded else ''}")
    
    with tab_zones:
        st.markdown("### Sorteo y Fixture de Zonas")
        if tourn_state.get("groups"):
            st.subheader("Grupos")
            cols = st.columns(tourn_state["config"]["num_zones"])
            for i, group in enumerate(tourn_state["groups"]):
                with cols[i]:
                    st.markdown(f"**Zona {i+1}**")
                    for p in group:
                        is_seeded = p in tourn_state.get("seeded_pairs", [])
                        st.markdown(f"- {p} {'(CS)' if is_seeded else ''}")
            
            st.subheader("Fixture de Zonas")
            for m in tourn_state["results"]:
                with st.container(border=True):
                    st.markdown(f"**{m['zone']}** - {m['pair1']} vs {m['pair2']}")
                    sets = m.get("sets", [])
                    if sets:
                        stats = compute_sets_stats(sets)
                        st.markdown(f"Resultado: **{stats['sets1']} - {stats['sets2']}** sets")
                        for i,s in enumerate(sets):
                            st.markdown(f"Set {i+1}: {s['s1']}-{s['s2']}")
        else:
            st.info("Aún no se ha realizado el sorteo.")
            
    with tab_playoffs:
        st.markdown("### Llaves de Playoff")
        if tourn_state["ko"]["matches"]:
            for m in tourn_state["ko"]["matches"]:
                st.markdown(f"### {m['label']} - {m['round']}")
                st.markdown(f"**{m['a']}** vs **{m['b']}**")
                sets = m.get("sets", [])
                if sets:
                    stats = compute_sets_stats(sets)
                    st.markdown(f"Resultado: **{stats['sets1']} - {stats['sets2']}** sets")
                    for i,s in enumerate(sets):
                        st.markdown(f"Set {i+1}: {s['s1']}-{s['s2']}")
        else:
            st.info("Aún no se ha generado la llave de playoff.")

    with tab_standings:
        st.markdown("### Posiciones por Zona")
        if tourn_state.get("groups"):
            zone_tables = [standings_from_results(f"Z{i+1}", g, tourn_state["results"], tourn_state["config"]) for i,g in enumerate(tourn_state["groups"])]
            cols = st.columns(tourn_state["config"]["num_zones"])
            for i, t in enumerate(zone_tables):
                with cols[i]:
                    st.markdown(f"**Posiciones - Zona {i+1}**")
                    if not t.empty:
                        is_complete = zone_complete(f"Z{i+1}", tourn_state["results"], tourn_state["config"]["format"])
                        # Add qualified checkmark
                        if is_complete:
                            for j in range(len(t)):
                                if t.iloc[j]["Pos"] <= tourn_state["config"]["top_per_zone"]:
                                    t.at[j, "pair"] = f"{t.iloc[j]['pair']} ✅"
                        
                        st.dataframe(t.style.set_table_styles([
                            {'selector': 'th', 'props': [('background-color', DARK_GREY), ('color', 'white')]},
                            {'selector': 'tr:nth-child(even)', 'props': [('background-color', LIGHT_GREY)]},
                            {'selector': 'tr:nth-child(odd)', 'props': [('background-color', 'white')]}
                        ]).hide(axis="index"), use_container_width=True)
                        
def main():
    init_session()

    if st.query_params.get("mode", [""])[0] == "super":
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
        st.caption("iAPPs Pádel — v3.3.24")
        return

    if not st.session_state.get("auth_user"):
        inject_global_layout("No autenticado")
        login_form()
        st.caption("iAPPs Pádel — v3.3.24")
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
            st.warning("No tienes torneos asignados para ver.")

if __name__ == "__main__":
    main()
