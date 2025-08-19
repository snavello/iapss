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

def brand_text_logo() -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="220" viewBox="0 0 660 200" role="img" aria-label="iAPPs PADEL TOURNAMENT">
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
    "use_seed_pairs": False,
    "seed_pairs": []
}

rng = lambda off, seed: random.Random(int(seed) + int(off))

def create_groups(pairs, num_groups, seed=42, seed_pairs_list=None):
    r = rng(seed, "groups")
    shuffled = pairs[:]
    if seed_pairs_list:
        seeded = [p for p in shuffled if parse_pair_number(p) in seed_pairs_list]
        unseeded = [p for p in shuffled if parse_pair_number(p) not in seed_pairs_list]
        r.shuffle(unseeded)
        shuffled = seeded + unseeded
    
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

def get_public_link(tid: str) -> str:
    url = st.get_app_host()
    return f"{url}?mode=public&tid={tid}"

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
      .compact-table td, .compact-table th {{ padding: 4px 8px; }}
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
            st.write(f"**{usr['username']}** — rol `{usr['role']}` — activo `{usr.get('active',True)}`")
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
    for f in (snap_dir_for(tid)).glob("*.json"):
        try:
            f.unlink()
        except Exception:
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
        return

    st.subheader("Abrir / eliminar torneo")
    names = [f"{t['date']} — {t['t_name']} ({t['gender']}) — {t['place']} — ID:{t['tournament_id']}" for t in my]
    selected = st.selectbox("Selecciona un torneo", names, index=0)
    sel = my[names.index(selected)]
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("Abrir torneo"):
            st.session_state.current_tid = sel["tournament_id"]
            st.rerun()
    with c2:
        if st.button("Eliminar torneo", type="secondary"):
            try:
                delete_tournament(user["username"], sel["tournament_id"])
                st.success("Torneo eliminado.")
                if st.session_state.get("current_tid")==sel["tournament_id"]:
                    st.session_state.current_tid=None
                st.rerun()
            except Exception as e:
                st.error(f"Error al eliminar el torneo: {e}")
    with c3:
        tid = sel["tournament_id"]
        public_link = get_public_link(tid)
        st.caption("Link público (solo lectura):")
        st.markdown(f"""
            <div class="copy-btn-container">
                <input type="text" value="{public_link}" id="publicLinkInput" readonly style="width:100%; border:none; background:none;"/>
                <button onclick="navigator.clipboard.writeText('{public_link}')" title="Copiar link" style="background:none; border:none;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clipboard" viewBox="0 0 16 16">
                        <path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/>
                        <path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/>
                    </svg>
                </button>
            </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.get("current_tid"):
        tournament_manager(user, st.session_state["current_tid"])

# ====== Gestor del Torneo ======
def tournament_manager(user: Dict[str, Any], tid: str):
    state = load_tournament(tid)
    if not state:
        st.error("No se encontró el torneo.")
        return

    tab_cfg, tab_pairs, tab_results, tab_tables, tab_ko, tab_persist = st.tabs(
        ["⚙️ Configuración", "👥 Parejas", "📝 Resultados", "📊 Tablas", "🗂️ Playoffs", "💾 Persistencia"]
    )
    cfg = state.get("config", DEFAULT_CONFIG.copy())
    
    # --- CONFIGURACIÓN ---
    with tab_cfg:
        st.subheader("Parámetros deportivos")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            cfg["t_name"] = st.text_input("Nombre del torneo", value=cfg.get("t_name"), key="t_name_cfg")
            cfg["num_pairs"] = st.number_input(
                "N° máximo de parejas", min_value=2, max_value=64,
                value=cfg.get("num_pairs"), key="n_pairs"
            )
        with c2:
            cfg["num_zones"] = st.number_input(
                "N° de zonas/grupos", min_value=1, max_value=8,
                value=cfg.get("num_zones"), key="n_zones"
            )
            cfg["top_per_zone"] = st.number_input(
                "Clasifican por zona", min_value=1, max_value=4,
                value=cfg.get("top_per_zone"), key="top_per_zone"
            )
        with c3:
            cfg["points_win"] = st.number_input(
                "Puntos por ganar", min_value=1, value=cfg.get("points_win"), key="p_win"
            )
            cfg["points_loss"] = st.number_input(
                "Puntos por perder", min_value=0, value=cfg.get("points_loss"), key="p_loss"
            )
        with c4:
            cfg["seed"] = st.number_input(
                "Semilla (orden de grupos)", value=cfg.get("seed"), key="seed_groups"
            )
            cfg["format"] = st.selectbox(
                "Formato de partidos", ["one_set", "best_of_3", "best_of_5"],
                index=["one_set", "best_of_3", "best_of_5"].index(cfg.get("format")),
                key="match_format"
            )
        
        cfg["use_seed_pairs"] = st.checkbox("Usar cabezas de serie (1 por zona)", value=cfg.get("use_seed_pairs"))
        if cfg["use_seed_pairs"]:
            all_pairs = sorted(state["pairs"], key=parse_pair_number)
            cfg["seed_pairs"] = st.multiselect(
                f"Selecciona {cfg['num_zones']} cabezas de serie",
                options=all_pairs,
                default=[p for p in cfg.get("seed_pairs",[]) if p in all_pairs],
                max_selections=cfg["num_zones"]
            )


        if st.button("Guardar configuración", type="primary"):
            state["config"] = cfg
            save_tournament(tid, state)
            st.success("Configuración guardada.")
            st.rerun()

        st.caption(f"Torneo: **{state['meta']['t_name']}** ({state['meta']['tournament_id']})")
        st.caption(f"Lugar: {state['meta']['place']} | Fecha: {state['meta']['date']} | Género: {state['meta']['gender']}")

    # --- PAREJAS ---
    with tab_pairs:
        st.subheader("Parejas inscritas")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("##### Agregar pareja")
            with st.form("add_pair_form"):
                n_avail = next_available_number(state["pairs"], cfg["num_pairs"])
                pair_num = st.number_input(
                    "N° de pareja", min_value=1, max_value=cfg["num_pairs"],
                    value=n_avail or 1, key="pair_num"
                )
                j1 = st.text_input("Jugador 1", key="j1")
                j2 = st.text_input("Jugador 2", key="j2")
                add_btn = st.form_submit_button("Agregar pareja", type="primary")

            if add_btn:
                if not j1 or not j2:
                    st.error("Por favor, ingresa los nombres de ambos jugadores.")
                elif len(state["pairs"]) >= cfg["num_pairs"]:
                    st.warning("Ya se alcanzó el número máximo de parejas para el torneo.")
                else:
                    pair_label = format_pair_label(pair_num, j1, j2)
                    if pair_label in state["pairs"]:
                        st.error("Este número de pareja ya está en uso.")
                    else:
                        state["pairs"].append(pair_label)
                        state["pairs"].sort()
                        save_tournament(tid, state, make_snapshot=False)
                        st.success(f"Pareja {pair_num} agregada.")
                        st.rerun()
        
        with c2:
            st.markdown("##### Parejas actuales")
            if not state["pairs"]:
                st.info("No hay parejas inscritas.")
            else:
                for i, p in enumerate(state["pairs"]):
                    c, r = st.columns([0.8, 0.2])
                    with c:
                        st.markdown(p)
                    with r:
                        pn = parse_pair_number(p)
                        if pn and st.button("🗑️", key=f"del_{pn}", help="Eliminar"):
                            state["pairs"] = remove_pair_by_number(state["pairs"], pn)
                            save_tournament(tid, state)
                            st.success(f"Pareja {pn} eliminada.")
                            st.rerun()
        
        st.divider()
        if st.button("Generar grupos/fixture", type="primary", use_container_width=True):
            if len(state["pairs"]) < 2:
                st.warning("Deben haber al menos 2 parejas para generar un fixture.")
            elif cfg["use_seed_pairs"] and len(cfg.get("seed_pairs",[])) != cfg["num_zones"]:
                st.warning(f"Debes seleccionar exactamente {cfg['num_zones']} cabezas de serie.")
            else:
                groups = create_groups(state["pairs"], cfg["num_zones"], cfg["seed"], cfg.get("seed_pairs"))
                state["groups"] = groups
                state["results"] = build_fixtures(groups)
                state["ko"]["matches"] = []
                save_tournament(tid, state)
                st.success("Grupos y fixture generados.")
                st.balloons()
                st.rerun()

    # --- RESULTADOS ---
    with tab_results:
        st.subheader("Resultados de fase de grupos")
        if not state.get("groups"):
            st.warning("Primero debes generar el fixture en la pestaña 'Parejas'.")
            return
        
        current_zone = st.selectbox("Selecciona una zona", [f"Z{i+1}" for i in range(cfg["num_zones"])])
        
        matches = [m for m in state["results"] if m["zone"] == current_zone]
        
        for i, m in enumerate(matches):
            st.markdown(f"**Partido {i+1}: {m['pair1']} vs {m['pair2']}**")
            with st.form(f"match_{i}"):
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.markdown(f"**{m['pair1']}**")
                    score1= st.number_input("Sets", min_value=0, value=m['sets'][0]['s1'] if m['sets'] else 0, key=f"s1_{i}")
                    golden1 = st.number_input("Puntos extra", min_value=0, value=m['golden1'], key=f"g1_{i}")
                with c2:
                    st.markdown(f"**{m['pair2']}**")
                    score2= st.number_input("Sets", min_value=0, value=m['sets'][0]['s2'] if m['sets'] else 0, key=f"s2_{i}")
                    golden2 = st.number_input("Puntos extra", min_value=0, value=m['golden2'], key=f"g2_{i}")
                
                submitted = st.form_submit_button("Guardar resultado")
                if submitted:
                    sets_data = [{"s1": score1, "s2": score2}]
                    state["results"][state["results"].index(m)]["sets"] = sets_data
                    state["results"][state["results"].index(m)]["golden1"] = golden1
                    state["results"][state["results"].index(m)]["golden2"] = golden2
                    save_tournament(tid, state)
                    st.success("Resultado guardado.")
                    st.rerun()
    
    # --- TABLAS ---
    with tab_tables:
        st.subheader("Tablas de posiciones por zona")
        if not state.get("groups"):
            st.warning("Primero debes generar el fixture.")
            return

        tables = []
        for i, group in enumerate(state["groups"]):
            zone = f"Z{i+1}"
            df = standings_from_results(zone, group, state["results"], cfg)
            tables.append(df)
            st.markdown(f"##### Zona {zone}")
            if not df.empty:
                cols = st.columns([1,1,1,1,1,1,1,1])
                for col, name in zip(cols, ["Pos","Zona","Pareja","PJ","PG","PP","GF","GC"]):
                    col.markdown(f"**{name}**")
                
                for _,row in df.iterrows():
                    cols = st.columns([1,1,1,1,1,1,1,1])
                    with cols[0]:
                        st.markdown(f"{int(row['Pos'])}")
                    with cols[1]:
                        st.markdown(row['Zona'])
                    with cols[2]:
                        st.markdown(row['pair'])
                    with cols[3]:
                        st.markdown(f"{int(row['PJ'])}")
                    with cols[4]:
                        st.markdown(f"{int(row['PG'])}")
                    with cols[5]:
                        st.markdown(f"{int(row['PP'])}")
                    with cols[6]:
                        st.markdown(f"{int(row['GF'])}")
                    with cols[7]:
                        st.markdown(f"{int(row['GC'])}")
            
            st.markdown("---")

    # --- PLAYOFFS ---
    with tab_ko:
        st.subheader("Fase de Playoffs")
        
        qualified_count = cfg["num_zones"] * cfg["top_per_zone"]
        st.info(f"Se clasificarán {qualified_count} parejas para los playoffs.")
        
        qualified_pairs = qualified_from_tables(tables, cfg["top_per_zone"])
        
        if not all(zone_complete(f"Z{i+1}", state["results"], cfg["format"]) for i in range(cfg["num_zones"])):
            st.warning("Completa todos los partidos de la fase de grupos para generar los playoffs.")
            return
        
        if st.button("Generar Playoffs", type="primary", use_container_width=True):
            state["ko"]["matches"] = build_initial_ko(qualified_pairs)
            state["ko"]["champions"] = None
            save_tournament(tid, state)
            st.success("Playoffs generados. ¡A jugar!")
            st.rerun()

        if not state["ko"]["matches"]:
            st.info("No hay playoffs generados.")
            return

        ko_matches = state["ko"]["matches"]
        
        round_names = sorted(list(set([m['round'] for m in ko_matches])), key=lambda x: ["QF","SF","FN"].index(x))

        for round_name in round_names:
            st.markdown(f"### {round_name}")
            round_matches = [m for m in ko_matches if m["round"]==round_name]
            
            for i, m in enumerate(round_matches):
                st.markdown(f"**Partido: {m['a']} vs {m['b']}**")
                with st.form(f"ko_{round_name}_{i}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**{m['a']}**")
                        s1 = st.number_input("Sets", min_value=0, value=compute_sets_stats(m['sets'])['sets1'], key=f"ko_s1_{round_name}_{i}")
                    with col2:
                        st.markdown(f"**{m['b']}**")
                        s2 = st.number_input("Sets", min_value=0, value=compute_sets_stats(m['sets'])['sets2'], key=f"ko_s2_{round_name}_{i}")

                    if st.form_submit_button("Guardar resultado"):
                        m['sets'] = [{'s1':s1, 's2':s2}]
                        save_tournament(tid, state)
                        st.success("Resultado guardado.")
                        st.rerun()

            winners = advance_pairs_from_round(round_matches)
            if winners:
                next_r = make_next_round_name(round_name)
                if next_r:
                    st.info(f"Todos los partidos de {round_name} están completos. Haz clic abajo para generar la siguiente ronda ({next_r}).")
                    if st.button(f"Generar {next_r}", key=f"next_{next_r}", use_container_width=True):
                        next_round_pairs = next_round(winners)
                        new_matches = pairs_to_matches(next_round_pairs, next_r)
                        state["ko"]["matches"].extend(new_matches)
                        save_tournament(tid, state)
                        st.success("Ronda generada.")
                        st.rerun()
                else: # Final
                    st.balloons()
                    st.success("¡Torneo finalizado!")
                    champion = winners[0] if winners else None
                    if champion:
                        st.markdown(f"#### ¡El campeón es: **{champion}**!")
                        state["ko"]["champions"] = champion
                        save_tournament(tid, state)
                        st.markdown(f"""
                        <style>
                            .st-emotion-cache-12m318y {{
                                background-color: #fff9c4 !important;
                                border: 1px solid #ffeb3b !important;
                            }}
                            .st-emotion-cache-12m318y p {{
                                font-size: 1.2rem;
                                font-weight: 700;
                            }}
                        </style>
                        """, unsafe_allow_html=True)
            else:
                st.warning(f"Faltan resultados en la ronda de {round_name} para continuar.")

    # --- PERSISTENCIA ---
    with tab_persist:
        st.subheader("Exportar / Importar")
        
        st.markdown("##### Exportar JSON del torneo")
        json_str = json.dumps(state, ensure_ascii=False, indent=2)
        st.download_button(
            label="Descargar JSON",
            data=json_str.encode("utf-8"),
            file_name=f"torneo_{tid}.json",
            mime="application/json"
        )

        st.markdown("##### Importar JSON (restaurar torneo)")
        uploaded_file = st.file_uploader("Sube un archivo .json", type="json")
        if uploaded_file:
            try:
                data = json.loads(uploaded_file.read().decode("utf-8"))
                new_tid = data.get("meta", {}).get("tournament_id")
                if not new_tid:
                    st.error("El archivo JSON no tiene un ID de torneo válido.")
                else:
                    save_tournament(new_tid, data)
                    st.session_state.current_tid = new_tid
                    st.success("Torneo restaurado con éxito.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error al procesar el archivo JSON: {e}")

        # PDF Export (optional)
        if REPORTLAB_OK:
            st.subheader("Exportar a PDF")
            if st.button("Generar PDF de Grupos y Fixture", use_container_width=True):
                if state.get("groups"):
                    pdf_bytes = generate_pdf_groups(state)
                    if pdf_bytes:
                        st.session_state["pdf_fixture_bytes"] = pdf_bytes
                        st.session_state["pdf_generated_at"] = now_iso()
                        st.success("PDF de Grupos generado.")
            
            if st.session_state.get("pdf_fixture_bytes"):
                st.download_button(
                    label="Descargar PDF de Grupos",
                    data=st.session_state.get("pdf_fixture_bytes"),
                    file_name=f"fixture_grupos_{tid}.pdf",
                    mime="application/pdf"
                )

            if st.button("Generar PDF de Playoffs", use_container_width=True):
                if state["ko"]["matches"]:
                    pdf_bytes = generate_pdf_ko(state)
                    if pdf_bytes:
                        st.session_state["pdf_playoffs_bytes"] = pdf_bytes
                        st.session_state["pdf_generated_at"] = now_iso()
                        st.success("PDF de Playoffs generado.")

            if st.session_state.get("pdf_playoffs_bytes"):
                st.download_button(
                    label="Descargar PDF de Playoffs",
                    data=st.session_state.get("pdf_playoffs_bytes"),
                    file_name=f"playoffs_{tid}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("La librería `reportlab` no está instalada. La exportación a PDF no está disponible.")

# PDF Functions (must be outside the main function)
def generate_pdf_groups(state: Dict[str,Any]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    flowables = []

    title_text = f"Torneo de Pádel: {state['meta'].get('t_name','')}"
    title = Paragraph(f"<font size='16'>{title_text}</font>", styles['h1'])
    flowables.append(title)
    flowables.append(Spacer(1, 0.5*cm))

    # Groups and Standings
    if state.get('groups'):
        for i, group in enumerate(state['groups']):
            zone = f"Z{i+1}"
            flowables.append(Paragraph(f"<font size='14'>Tabla de posiciones - Zona {zone}</font>", styles['h2']))
            df = standings_from_results(zone, group, state["results"], state["config"])
            if not df.empty:
                data = [list(df.columns)] + df.values.tolist()
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND', (0,1), (-1,-1), colors.white),
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ('BOX', (0,0), (-1,-1), 1, colors.black)
                ]))
                flowables.append(table)
            flowables.append(Spacer(1, 1*cm))

    # Fixture
    if state.get('results'):
        flowables.append(Paragraph("<font size='14'>Fixture de la Fase de Grupos</font>", styles['h2']))
        data = [["Zona", "Pareja 1", "Pareja 2", "Sets"]]
        for match in state['results']:
            sets_str = f"{compute_sets_stats(match['sets'])['sets1']}-{compute_sets_stats(match['sets'])['sets2']}" if match['sets'] else "N/A"
            data.append([match['zone'], match['pair1'], match['pair2'], sets_str])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        flowables.append(table)

    doc.build(flowables)
    buffer.seek(0)
    return buffer.getvalue()

def generate_pdf_ko(state: Dict[str, Any]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    flowables = []

    title_text = f"Playoffs - {state['meta'].get('t_name','')}"
    title = Paragraph(f"<font size='16'>{title_text}</font>", styles['h1'])
    flowables.append(title)
    flowables.append(Spacer(1, 0.5*cm))
    
    # KO bracket
    if state["ko"].get("matches"):
        matches = state["ko"]["matches"]
        round_names = sorted(list(set([m['round'] for m in matches])), key=lambda x: ["QF","SF","FN"].index(x))
        
        for round_name in round_names:
            flowables.append(Paragraph(f"<font size='14'>{round_name}</font>", styles['h2']))
            round_matches = [m for m in matches if m['round']==round_name]
            
            for m in round_matches:
                stats = compute_sets_stats(m['sets'])
                winner_text = ""
                if stats["sets1"] > stats["sets2"]:
                    winner_text = f"➡️ **Ganador: {m['a']}**"
                elif stats["sets2"] > stats["sets1"]:
                    winner_text = f"➡️ **Ganador: {m['b']}**"
                
                sets_str = f"({stats['sets1']}-{stats['sets2']})"
                
                flowables.append(Paragraph(f"<b>{m['a']}</b> vs <b>{m['b']}</b> &nbsp; Resultado: {sets_str}", styles['Normal']))
                flowables.append(Spacer(1, 0.2*cm))

    doc.build(flowables)
    buffer.seek(0)
    return buffer.getvalue()

def viewer_tournament(tid: str, public: bool=False):
    state = load_tournament(tid)
    if not state:
        st.error("Torneo no encontrado.")
        return
    
    # Simplified layout for viewers
    st.header(state['meta'].get('t_name',''))
    st.markdown(f"**Lugar:** {state['meta'].get('place','')}")
    st.markdown(f"**Fecha:** {state['meta'].get('date','')}")
    st.markdown("---")

    tabs_v = st.tabs(["👥 Parejas", "📊 Tablas", "🗂️ Playoffs"])

    # Pairs
    with tabs_v[0]:
        st.subheader("Parejas inscritas")
        if not state.get("pairs"):
            st.info("No hay parejas inscritas.")
        else:
            st.write(pd.DataFrame(state["pairs"], columns=["Pareja"]))

    # Tables
    with tabs_v[1]:
        st.subheader("Tablas de posiciones por zona")
        if not state.get("groups"):
            st.warning("La fase de grupos aún no ha comenzado.")
            return

        tables = []
        for i, group in enumerate(state["groups"]):
            zone = f"Z{i+1}"
            df = standings_from_results(zone, group, state["results"], state["config"])
            tables.append(df)
            st.markdown(f"##### Zona {zone}")
            if not df.empty:
                st.dataframe(df)
            
            st.markdown("---")

    # Playoffs
    with tabs_v[2]:
        st.subheader("Fase de Playoffs")
        if not state["ko"]["matches"]:
            st.info("Los playoffs aún no han sido generados.")
            return
        
        ko_matches = state["ko"]["matches"]
        round_names = sorted(list(set([m['round'] for m in ko_matches])), key=lambda x: ["QF","SF","FN"].index(x))

        for round_name in round_names:
            st.markdown(f"### {round_name}")
            round_matches = [m for m in ko_matches if m["round"]==round_name]
            
            for m in round_matches:
                stats = compute_sets_stats(m['sets'])
                winner_text = ""
                if stats["sets1"] > stats["sets2"]:
                    winner_text = f"➡️ **Ganador: {m['a']}**"
                elif stats["sets2"] > stats["sets1"]:
                    winner_text = f"➡️ **Ganador: {m['b']}**"
                
                sets_str = f"({stats['sets1']}-{stats['sets2']})"
                
                st.markdown(f"**{m['a']}** vs **{m['b']}** {sets_str} {winner_text}")
                st.markdown("---")
        
        if state["ko"].get("champions"):
            st.balloons()
            st.markdown(f"## 🏆 ¡El campeón del torneo es **{state['ko']['champions']}**! 🏆")


def main():
    if "mode" in st.query_params:
        params = st.query_params
        mode = params.get("mode", [""])[0]
        _tid = params.get("tid", [""])[0]
        
        if mode=="super":
            if st.session_state.get("auth_user") and st.session_state.auth_user["role"] == "SUPER_ADMIN":
                super_admin_panel()
            else:
                st.warning("Acceso denegado. Solo Super Admin.")
                login_form()
            return

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
                format_func=lambda tid: load_tournament(tid)["meta"]["t_name"]
            )
            if st.session_state.get("current_tid"):
                viewer_tournament(st.session_state["current_tid"])

if __name__ == '__main__':
    init_session()
    main()
