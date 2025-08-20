# padel_tournament_pro_multiuser_v3_3.py — v3.3.35
# Cambios/funciones claves:
# - Logo seguro (descarga URL RAW → data URI base64), con fallback SVG. Encabezado NO fijo.
# - SUPER_ADMIN: gestión de usuarios (alta/baja/roles/PIN), logo global de la app.
# - TOURNAMENT_ADMIN: torneos (config/parejas/resultados/tablas/playoffs/persistencia).
# - VIEWER: solo lectura (modo público o asociado a un admin).
# - Config: checkbox “Usar sistema de cabezas de serie” (seeds); marcar X seeds (= nº de zonas).
# - Sorteo de zonas con 1 seed por zona y reparto mínimo por zona = clasificados (top_per_zone).
# - Resultados: sets según formato + puntos de oro. Ganador resaltado.
# - Tablas: desempates PTS→DG→GP→sorteo; encabezado oscuro, zebra, ✓ en 1° y 2°.
# - Playoffs: genera ronda inicial acorde a Nº de clasificados (FN/SF/QF) y avanza hasta campeón.
# - Link público con botón “📋 Copiar”.
# - Persistencia: autosave + descarga/restore JSON (nombre con id + nombre + fecha + timestamp).
# - Sintaxis revisada; sin escrituras a session_state de widgets ya instanciados en el mismo tick.

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

st.set_page_config(page_title="iAPPs Pádel — v3.3.35", layout="wide")

# CSS básico para tablas HTML (zebra + header oscuro)
st.markdown("""
<style>
table.zebra tbody tr:nth-child(odd) { background: #f9fafb; }
table.dark-header thead th { background: #2f3b52; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ====== Persistencia (archivos) ======
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
    # Puedes cambiar esta URL RAW si deseas precargar otro logo.
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

# ====== Logo seguro (Data URI) + fallback SVG ======
PRIMARY_BLUE = "#0D47A1"
LIME_GREEN  = "#AEEA00"
DARK_BLUE   = "#082D63"

def _brand_svg(width_px: int = 200) -> str:
    # Fallback si no se puede descargar el logo remoto
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

@st.cache_data(show_spinner=False)
def _fetch_image_data_uri(url: str, timeout: float = 6.0) -> str:
    """Descarga una imagen remota y la convierte a data URI. Devuelve '' si falla."""
    try:
        if not url:
            return ""
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        content = resp.content
        # Detectar MIME simple por cabecera
        ct = (resp.headers.get("Content-Type") or "").lower()
        if "svg" in ct:
            mime = "image/svg+xml"
        elif "jpeg" in ct or "jpg" in ct:
            mime = "image/jpeg"
        elif "webp" in ct:
            mime = "image/webp"
        else:
            mime = "image/png"
        b64 = base64.b64encode(content).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""

def render_header_bar(user_name: str = "", role: str = "", logo_url: str = ""):
    """
    Encabezado simple NO fijo (no tapa el contenido y scrollea con la página).
    Logo pequeño, responsive, cargado como data URI (si falla, SVG fallback).
    """
    st.markdown(
        """
        <style>
        .iapps-header-row {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 6px 0 4px 0;
            border-bottom: 1px solid #e5e7eb;
        }
        .iapps-header-logo {
            max-height: 54px;
            width: auto;
            height: auto;
            max-width: 22vw;
            object-fit: contain;
        }
        .iapps-user { font-weight:600; }
        .iapps-role { color:#6b7280; margin-left:6px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="iapps-header-row">', unsafe_allow_html=True)
    c_logo, c_spacer, c_user = st.columns([2, 6, 3])

    with c_logo:
        data_uri = _fetch_image_data_uri(logo_url) if logo_url else ""
        if data_uri:
            st.markdown(f'<img class="iapps-header-logo" src="{data_uri}" alt="iAPPs">', unsafe_allow_html=True)
        else:
            st.markdown(_brand_svg(180), unsafe_allow_html=True)

    with c_spacer:
        st.markdown("")  # espacio

    with c_user:
        if user_name:
            st.markdown(
                f'<span class="iapps-user">{user_name}</span><span class="iapps-role">({role})</span>',
                unsafe_allow_html=True
            )
            # Si implementas logout, coloca el botón aquí.

    st.markdown('</div>', unsafe_allow_html=True)

def copy_to_clipboard_html(text: str, label: str = "Copiar"):
    """
    Botón HTML que copia `text` al portapapeles.
    """
    safe = (text or "").replace("\\", "\\\\").replace("'", "\\'")
    return f"""
    <button
      style="padding:4px 8px;border:1px solid #ddd;border-radius:6px;background:#f7f7f7;cursor:pointer"
      onclick="navigator.clipboard.writeText('{safe}');this.textContent='¡Copiado!';setTimeout(()=>this.textContent='📋 {label}',1200);"
      title="Copiar al portapapeles"
    >📋 {label}</button>
    """

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

# ====== Reglas del torneo ======
DEFAULT_CONFIG = {
    "t_name": "Open Pádel",
    "num_pairs": 16,
    "num_zones": 4,
    "top_per_zone": 2,
    "points_win": 2,
    "points_loss": 0,
    "seed": 42,
    "format": "best_of_3",  # one_set | best_of_3 | best_of_5
    "use_seeds": False
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

def pairs_seed_v1(winners: List[Tuple[str,int,str]], runners: List[Tuple[str,int,str]]) -> List[Tuple[str,str]]:
    # empareja 1ro vs 2do de otras zonas (rotación simple)
    if not winners or not runners:
        return []
    m = min(len(winners), len(runners))
    winners = winners[:m]; runners = runners[:m]
    rr = runners[1:] + runners[:1] if len(runners) > 1 else runners
    return list(zip([w for (_,_,w) in winners], [r for (_,_,r) in rr]))

def build_initial_ko(qualified: List[Tuple[str,int,str]]) -> List[Dict[str,Any]]:
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
            pairs = pairs_seed_v1(winners, runners)
            if not pairs or len(pairs) != 2:
                names = [q[2] for q in qualified]
                pairs = [(names[0], names[3]), (names[1], names[2])]
            return [{"round":"SF","label":f"SF{i+1}","a":a,"b":b,"sets":[],"goldenA":0,"goldenB":0} for i,(a,b) in enumerate(pairs)]
        if N == 8:
            pairs = pairs_seed_v1(winners, runners)
            if len(pairs) != 4:
                names = [q[2] for q in qualified]
                pairs = [(names[i], names[-(i+1)]) for i in range(4)]
            return [{"round":"QF","label":f"QF{i+1}","a":a,"b":b,"sets":[],"goldenA":0,"goldenB":0} for i,(a,b) in enumerate(pairs)]

    # N no potencia de 2 → BYEs
    names = [q[2] for q in sorted(qualified, key=lambda x:(x[1], x[0]))]
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
        lab_list = labels.get(round_name, [])
        lab = lab_list[i-1] if i-1 < len(lab_list) else f"{round_name}{i}"
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
        "seeded_pairs": []
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

# ====== SUPER ADMIN PANEL ======
def super_admin_panel():
    user = st.session_state["auth_user"]
    app_cfg = load_app_config()
    render_header_bar(
        user_name=user.get("username",""),
        role=user.get("role",""),
        logo_url=app_cfg.get("app_logo_url","")
    )

    st.header("👑 Panel de SUPER ADMIN")

    # Apariencia
    with st.expander("🎨 Apariencia (Logo global de la app)", expanded=True):
        url = st.text_input("URL pública del logotipo (RAW de GitHub recomendado)", value=app_cfg.get("app_logo_url","")).strip()
        if st.button("Guardar logo global", type="primary"):
            app_cfg["app_logo_url"] = url
            save_app_config(app_cfg)
            st.success("Logo global guardado. (Refrescá la página si no lo ves de inmediato)")

    st.divider()
    st.subheader("👥 Gestión de usuarios")

    users = load_users()
    # Crear usuario
    with st.form("create_user_form", clear_on_submit=True):
        c1,c2,c3,c4 = st.columns([3,2,2,3])
        with c1:
            new_u = st.text_input("Username nuevo").strip()
        with c2:
            new_role = st.selectbox("Rol", ["TOURNAMENT_ADMIN","VIEWER"])
        with c3:
            new_pin = st.text_input("PIN inicial (6 dígitos)", max_chars=6)
        assigned_admin = None
        with c4:
            if new_role == "VIEWER":
                admins=[x["username"] for x in users if x["role"]=="TOURNAMENT_ADMIN" and x.get("active",True)]
                assigned_admin = st.selectbox("Asignar a admin", admins) if admins else None
        subm = st.form_submit_button("Crear usuario", type="primary")
        if subm:
            if not new_u:
                st.error("Username requerido.")
            elif get_user(new_u):
                st.error("Ya existe un usuario con ese nombre.")
            elif len(new_pin)!=6 or not new_pin.isdigit():
                st.error("PIN inválido.")
            else:
                set_user({"username":new_u,"pin_hash":sha(new_pin),"role":new_role,
                          "assigned_admin":assigned_admin,"created_at":now_iso(),"active":True})
                st.success(f"Usuario {new_u} creado.")

    st.markdown("### Lista y edición")
    for usr in users:
        with st.container(border=True):
            st.write(f"**{usr['username']}** — rol `{usr['role']}` — activo `{usr.get('active',True)}`")
            c1,c2,c3,c4,c5 = st.columns([2,2,2,3,2])

            # Cambiar rol
            with c1:
                new_role = st.selectbox(
                    f"Rol de {usr['username']}",
                    ["SUPER_ADMIN","TOURNAMENT_ADMIN","VIEWER"],
                    index=["SUPER_ADMIN","TOURNAMENT_ADMIN","VIEWER"].index(usr["role"]),
                    key=f"role_{usr['username']}",
                    disabled=(usr["username"]=="ADMIN")
                )
            # Asignar admin a viewer
            with c2:
                if new_role=="VIEWER" or usr["role"]=="VIEWER":
                    admins=[x["username"] for x in users if x["role"]=="TOURNAMENT_ADMIN" and x.get("active",True)]
                    default_idx = len(admins)
                    if usr.get("assigned_admin") in admins:
                        default_idx = admins.index(usr.get("assigned_admin"))
                    new_assigned = st.selectbox(
                        f"Admin asignado ({usr['username']})",
                        admins + [None],
                        index=default_idx,
                        key=f"ass_{usr['username']}"
                    )
                else:
                    new_assigned = None
                    st.caption("—")

            # Activo/inactivo
            with c3:
                active_toggle = st.checkbox("Activo", value=usr.get("active",True), key=f"act_{usr['username']}")

            # PIN
            with c4:
                if usr["username"]=="ADMIN":
                    st.caption("PIN de ADMIN fijo en 199601 (no editable).")
                    pin_value = None
                else:
                    pin_value = st.text_input(f"Nuevo PIN ({usr['username']}) (opcional)", max_chars=6, key=f"pin_{usr['username']}")

            # Guardar cambios
            with c5:
                if st.button(f"💾 Guardar {usr['username']}", key=f"save_{usr['username']}"):
                    if usr["username"]=="ADMIN":
                        usr["role"] = "SUPER_ADMIN"
                        usr["pin_hash"] = sha("199601")  # asegurar
                    else:
                        usr["role"] = new_role
                        if pin_value:
                            if len(pin_value)==6 and pin_value.isdigit():
                                usr["pin_hash"] = sha(pin_value)
                            else:
                                st.error("PIN inválido (debe ser 6 dígitos).")
                                st.stop()
                        usr["assigned_admin"] = new_assigned if usr["role"]=="VIEWER" else None
                    usr["active"] = bool(active_toggle)
                    set_user(usr)
                    st.success("Cambios guardados.")

    st.caption("Iapps Padel Tournament · iAPPs Pádel — v3.3.35")

# ====== ADMIN (torneos) ======
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

def create_groups_unseeded(pairs: List[str], num_groups: int, top_per_zone: int, seed: int) -> List[List[str]]:
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

def create_groups_seeded(pairs: List[str], seeded_labels: List[str], num_groups: int, top_per_zone: int, seed: int) -> List[List[str]]:
    r = random.Random(int(seed))
    seeded = [p for p in pairs if p in seeded_labels]
    non_seeded = [p for p in pairs if p not in seeded_labels]
    r.shuffle(non_seeded)
    groups = [[] for _ in range(num_groups)]
    # 1) 1 seed por grupo
    for i, s in enumerate(seeded[:num_groups]):
        groups[i].append(s)
    # 2) mínimo por zona = top_per_zone
    min_per_zone = max(1, int(top_per_zone))
    total = len(pairs)
    desired_min_total = num_groups * min_per_zone
    gi = 0
    while non_seeded and sum(len(g) for g in groups) < min(total, desired_min_total):
        if len(groups[gi]) < min_per_zone:
            groups[gi].append(non_seeded.pop())
        gi = (gi + 1) % num_groups
    # 3) resto
    gi = 0
    while non_seeded:
        groups[gi].append(non_seeded.pop())
        gi = (gi + 1) % num_groups
    return groups

def admin_dashboard(admin_user: Dict[str, Any]):
    app_cfg = load_app_config()
    render_header_bar(
        user_name=admin_user.get("username",""),
        role=admin_user.get("role",""),
        logo_url=app_cfg.get("app_logo_url","")
    )

    st.header(f"Torneos de {admin_user['username']}")

    with st.expander("➕ Crear torneo nuevo", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        with c1: t_name = st.text_input("Nombre del torneo", value="Open Pádel")
        with c2: place = st.text_input("Lugar / Club", value="Mi Club")
        with c3: tdate = st.date_input("Fecha", value=date.today()).isoformat()
        with c4: gender = st.selectbox("Género", ["masculino","femenino","mixto"], index=2)
        if st.button("Crear torneo", type="primary"):
            tid = create_tournament(admin_user["username"], t_name, place, tdate, gender)
            st.session_state.current_tid = tid
            st.success(f"Torneo creado: {t_name} ({tid})")
            st.rerun()

    my = load_index_for_admin(admin_user["username"])
    if not my:
        st.info("Aún no tienes torneos.")
        return

    st.subheader("Abrir / eliminar torneo")
    names = [f"{t['date']} — {t['t_name']} ({t['gender']}) — {t['place']} — ID:{t['tournament_id']}" for t in my]
    selected = st.selectbox("Selecciona un torneo", names, index=0)
    sel = my[names.index(selected)]
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("Abrir torneo"): st.session_state.current_tid = sel["tournament_id"]; st.rerun()
    with c2:
        if st.button("Eliminar torneo", type="secondary"):
            delete_tournament(admin_user["username"], sel["tournament_id"])
            st.success("Torneo eliminado.")
            if st.session_state.get("current_tid")==sel["tournament_id"]:
                st.session_state.current_tid=None
            st.rerun()
    with c3:
        tid = sel["tournament_id"]
        st.caption("Link público (solo lectura):")
        # URL relativa (sirve dentro de la app). Si quieres full URL, reemplaza por la de tu app.
        public_url = f"?mode=public&tid={tid}"
        # public_url = f"https://iappspadel.streamlit.app/?mode=public&tid={tid}"
        st.code(public_url, language=None)
        st.markdown(copy_to_clipboard_html(public_url), unsafe_allow_html=True)

    if st.session_state.get("current_tid"):
        tournament_manager(admin_user, st.session_state["current_tid"])

# ====== Gestor del Torneo (Admin) ======
def tournament_manager(user: Dict[str, Any], tid: str):
    state = load_tournament(tid)
    if not state:
        st.error("No se encontró el torneo.")
        return

    tab_cfg, tab_pairs, tab_results, tab_tables, tab_ko, tab_persist = st.tabs(
        ["⚙️ Configuración", "👥 Parejas", "📝 Resultados", "📊 Tablas", "🏁 Playoffs", "💾 Persistencia"]
    )

    cfg = state.get("config", DEFAULT_CONFIG.copy())

    # --- CONFIG ---
    with tab_cfg:
        st.subheader("Datos del torneo")
        m = state.get("meta", {})
        st.write({"ID":m.get("tournament_id"),"Nombre":m.get("t_name"),"Lugar":m.get("place"),"Fecha":m.get("date"),"Género":m.get("gender"),"Admin":m.get("admin_username")})
        st.subheader("Configuración deportiva")
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

        # Semillas ON/OFF
        cfg["use_seeds"] = st.checkbox(
            "Usar sistema de cabezas de serie",
            value=bool(cfg.get("use_seeds", False)),
            help="Si está activo, podrás marcar X parejas como cabezas de serie (X = número de zonas)."
        )

        cA,cB = st.columns(2)
        with cA:
            if st.button("💾 Guardar configuración", type="primary"):
                state["config"] = {k:int(v) if isinstance(v,(int,float)) and k not in ["t_name","format","use_seeds"] else v for k,v in cfg.items()}
                save_tournament(tid, state); st.success("Configuración guardada.")
        with cB:
            if st.button("🎲 Sortear zonas (crear/rehacer fixture)"):
                pairs = state.get("pairs", [])
                if len(pairs) < cfg["num_zones"]:
                    st.error("Debe haber al menos tantas parejas como zonas.")
                else:
                    if cfg.get("use_seeds", False):
                        seeded = state.get("seeded_pairs", [])
                        if len(seeded) != int(cfg["num_zones"]):
                            st.error(f"Debes marcar exactamente {int(cfg['num_zones'])} cabezas de serie antes de sortear.")
                            st.stop()
                        groups = create_groups_seeded(pairs, seeded, int(cfg["num_zones"]), int(cfg["top_per_zone"]), int(cfg["seed"]))
                    else:
                        groups = create_groups_unseeded(pairs, int(cfg["num_zones"]), int(cfg["top_per_zone"]), int(cfg["seed"]))
                    state["groups"] = groups
                    state["results"] = build_fixtures(groups)
                    state["ko"] = {"matches": []}
                    save_tournament(tid, state); st.success("Zonas + fixture generados.")

    # --- PAREJAS ---
    with tab_pairs:
        st.subheader("Parejas")
        pairs = state.get("pairs", [])
        max_pairs = int(state.get("config", {}).get("num_pairs", 16))

        colL, colR = st.columns([1, 1])

        # ---- Izquierda: alta manual
        with colL:
            st.markdown("**Alta manual — una pareja por vez**")
            next_n = next_available_number(pairs, max_pairs)
            c1,c2,c3 = st.columns([1,3,3])
            with c1:
                st.text_input("N°", value=(str(next_n) if next_n else "—"), disabled=True, key=f"num_auto_{tid}")
            with c2:
                p1 = st.text_input("Jugador 1", key=f"p1_{tid}")
            with c3:
                p2 = st.text_input("Jugador 2", key=f"p2_{tid}")

            disabled_btn = (next_n is None)
            if st.button("Agregar", key=f"add_pair_{tid}", type="primary", disabled=disabled_btn):
                p1c, p2c = (p1 or "").strip(), (p2 or "").strip()
                if not p1c or not p2c:
                    st.error("Completá ambos nombres.")
                else:
                    label = format_pair_label(next_n, p1c, p2c)
                    pairs.append(label)
                    state["pairs"] = pairs
                    save_tournament(tid, state)
                    st.success(f"Agregada: {label}")
                    st.rerun()

            if next_n is None:
                st.warning(f"Se alcanzó el máximo de parejas configurado ({max_pairs}). Borrá alguna o aumenta el máximo en Configuración.")

        # ---- Derecha: listado con 🗑️
        with colR:
            st.markdown("**Listado**")
            if not pairs:
                st.info("Aún no hay parejas cargadas.")
            else:
                for p in pairs:
                    n = parse_pair_number(p) or "-"
                    c1, c2, c3 = st.columns([1, 8, 1])
                    with c1:
                        st.write(f"**{n:>02}**" if isinstance(n, int) else "**—**")
                    with c2:
                        st.write(p)
                    with c3:
                        if st.button("🗑️", key=f"del_{tid}_{p}", help="Eliminar pareja"):
                            state["pairs"] = [x for x in pairs if x != p]
                            save_tournament(tid, state)
                            st.rerun()

        st.divider()
        st.subheader("Cabezas de serie")
        use_seeds = bool(state.get("config", {}).get("use_seeds", False))
        num_groups = int(state.get("config", {}).get("num_zones", 4))
        if use_seeds:
            seeded = state.get("seeded_pairs", [])
            choices = pairs[:]
            current = [p for p in seeded if p in choices]
            st.caption(f"Selecciona exactamente {num_groups} parejas como cabezas de serie (1 por zona).")
            selected = st.multiselect(
                f"Cabezas de serie ({len(current)}/{num_groups})",
                options=choices,
                default=current,
                max_selections=num_groups
            )
            if st.button("💾 Guardar cabezas de serie"):
                if len(selected) != num_groups:
                    st.error(f"Debes seleccionar exactamente {num_groups}.")
                else:
                    state["seeded_pairs"] = selected
                    save_tournament(tid, state)
                    st.success("Cabezas de serie guardadas.")
        else:
            st.info("El torneo no usa cabezas de serie (activá la opción en Configuración).")

    # --- RESULTADOS ---
    with tab_results:
        st.subheader("Resultados — fase de grupos (sets + puntos de oro)")
        if not state.get("groups"):
            st.info("Primero crea/sortea zonas en Configuración.")
        else:
            fmt = state["config"].get("format","best_of_3")
            zones = sorted({m["zone"] for m in state["results"]})
            z_filter = st.selectbox("Filtrar por zona", ["(todas)"] + zones)
            pnames = sorted(set([m["pair1"] for m in state["results"]] + [m["pair2"] for m in state["results"]]))
            p_filter = st.selectbox("Filtrar por pareja", ["(todas)"] + pnames)

            listing = state["results"]
            if z_filter != "(todas)":
                listing = [m for m in listing if m["zone"]==z_filter]
            if p_filter != "(todas)":
                listing = [m for m in listing if m["pair1"]==p_filter or m["pair2"]==p_filter]

            for m in listing:
                idx = state["results"].index(m)
                with st.container(border=True):
                    title = f"**{m['zone']}** — {m['pair1']} vs {m['pair2']}"
                    stats_now = compute_sets_stats(m.get("sets", [])) if m.get("sets") else {"sets1":0,"sets2":0}
                    if m.get("sets") and match_has_winner(m["sets"]):
                        winner = m['pair1'] if stats_now["sets1"]>stats_now["sets2"] else m['pair2']
                        title += f"  <span style='display:inline-block;padding:2px 6px;border-radius:6px;background:#e8f5e9;color:#1b5e20;font-weight:600;margin-left:8px;'>🏆 {winner}</span>"
                    else:
                        title += "  <span style='color:#999'>(A definir)</span>"
                    st.markdown(title, unsafe_allow_html=True)

                    cur_sets = m.get("sets", [])
                    n_min, n_max = (1,1) if fmt=="one_set" else ((2,3) if fmt=="best_of_3" else (3,5))
                    n_sets = st.number_input(
                        "Sets jugados", min_value=n_min, max_value=n_max,
                        value=min(max(len(cur_sets), n_min), n_max),
                        key=f"ns_{tid}_{idx}"
                    )
                    new_sets = []
                    for si in range(n_sets):
                        cA,cB = st.columns(2)
                        with cA:
                            s1 = st.number_input(
                                f"Set {si+1} — games {m['pair1']}", 0, 20,
                                int(cur_sets[si]["s1"]) if si<len(cur_sets) and "s1" in cur_sets[si] else 0,
                                key=f"s1_{tid}_{idx}_{si}"
                            )
                        with cB:
                            s2 = st.number_input(
                                f"Set {si+1} — games {m['pair2']}", 0, 20,
                                int(cur_sets[si]["s2"]) if si<len(cur_sets) and "s2" in cur_sets[si] else 0,
                                key=f"s2_{tid}_{idx}_{si}"
                            )
                        new_sets.append({"s1":int(s1),"s2":int(s2)})
                    ok, msg = validate_sets(fmt, new_sets)
                    if not ok:
                        st.error(msg)
                    gC,gD = st.columns(2)
                    with gC:
                        g1 = st.number_input(f"Puntos de oro {m['pair1']}", 0, 200, int(m.get("golden1",0)), key=f"g1_{tid}_{idx}")
                    with gD:
                        g2 = st.number_input(f"Puntos de oro {m['pair2']}", 0, 200, int(m.get("golden2",0)), key=f"g2_{tid}_{idx}")

                    if st.button("Guardar partido", key=f"sv_{tid}_{idx}"):
                        stats = compute_sets_stats(new_sets)
                        if stats["sets1"] == stats["sets2"]:
                            st.error("Debe haber un ganador (no se permiten empates). Ajustá los sets.")
                        else:
                            state["results"][idx]["sets"] = new_sets
                            state["results"][idx]["golden1"] = int(g1)
                            state["results"][idx]["golden2"] = int(g2)
                            save_tournament(tid, state)
                            winner = m['pair1'] if stats["sets1"]>stats["sets2"] else m['pair2']
                            st.success(f"Partido guardado. 🏆 Ganó {winner}")
                            st.rerun()

    # --- TABLAS ---
    with tab_tables:
        st.subheader("Tablas por zona y clasificados")
        if not state.get("groups") or not state.get("results"):
            st.info("Aún no hay fixture o resultados.")
        else:
            cfg = state["config"]
            fmt = cfg.get("format","best_of_3")
            zone_tables = []
            all_complete = True
            for zi, group in enumerate(state["groups"], start=1):
                zone_name = f"Z{zi}"
                complete = zone_complete(zone_name, state["results"], fmt)
                status = "✅ Completa" if complete else "⏳ A definir"
                if not complete:
                    all_complete = False
                st.markdown(f"#### Tabla {zone_name} — {status}")

                table = standings_from_results(zone_name, group, state["results"], cfg)
                zone_tables.append(table)
                if table.empty:
                    st.info("Sin datos para mostrar todavía.")
                else:
                    # Agregar ✓ visual y ordenar columnas
                    show = table.rename(columns={"pair":"Pareja"})
                    show["✓"] = show["Pos"].apply(lambda x: "✅" if x in (1,2) else "")
                    show = show[["Zona","Pos","✓","Pareja","PJ","PG","PP","GF","GC","DG","GP","PTS"]]
                    st.markdown(show.to_html(index=False, classes=["zebra","dark-header"]), unsafe_allow_html=True)

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

    # --- PLAYOFFS ---
    with tab_ko:
        st.subheader("Playoffs (por sets + puntos de oro)")
        if not state.get("groups") or not state.get("results"):
            st.info("Necesitas tener zonas y resultados para definir clasificados.")
        else:
            cfg = state["config"]
            fmt = cfg.get("format","best_of_3")
            all_complete = all(zone_complete(f"Z{zi}", state["results"], fmt) for zi in range(1, len(state["groups"])+1))
            if not all_complete:
                st.info("⏳ A definir — Completa la fase de grupos para habilitar los playoffs.")
            else:
                zone_tables = []
                for zi, group in enumerate(state["groups"], start=1):
                    zone_name = f"Z{zi}"
                    table = standings_from_results(zone_name, group, state["results"], cfg)
                    zone_tables.append(table)
                qualified = qualified_from_tables(zone_tables, cfg["top_per_zone"])

                c1,c2 = st.columns(2)
                with c1:
                    if st.button("🔄 Regenerar Playoffs (desde clasificados)"):
                        state["ko"]["matches"] = build_initial_ko(qualified)
                        save_tournament(tid, state)
                        st.success("Playoffs regenerados.")
                        st.rerun()
                with c2:
                    st.caption("Usa esto si cambiaste resultados de zonas y querés rehacer la llave.")

                if not state["ko"]["matches"]:
                    state["ko"]["matches"] = build_initial_ko(qualified)
                    save_tournament(tid, state)

                round_order = ["QF","SF","FN"]
                final_champion = None

                for rname in round_order:
                    ms = [m for m in state["ko"]["matches"] if m.get("round")==rname]
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
                                title += f"  <span style='display:inline-block;padding:2px 6px;border-radius:6px;background:#e8f5e9;color:#1b5e20;font-weight:600;margin-left:8px;'>🏆 {winner}</span>"
                            else:
                                title += "  <span style='color:#999'>(A definir)</span>"
                            st.markdown(title, unsafe_allow_html=True)

                            cur_sets = m.get("sets", [])
                            n_min, n_max = (1,1) if fmt=="one_set" else ((2,3) if fmt=="best_of_3" else (3,5))
                            n_sets = st.number_input(
                                "Sets jugados", min_value=n_min, max_value=n_max,
                                value=min(max(len(cur_sets), n_min), n_max),
                                key=f"ko_ns_{tid}_{rname}_{idx}"
                            )
                            new_sets = []
                            for si in range(n_sets):
                                cA,cB = st.columns(2)
                                with cA:
                                    s1 = st.number_input(
                                        f"Set {si+1} — games {m['a']}", 0, 20,
                                        int(cur_sets[si]["s1"]) if si<len(cur_sets) and "s1" in cur_sets[si] else 0,
                                        key=f"ko_s1_{tid}_{rname}_{idx}_{si}"
                                    )
                                with cB:
                                    s2 = st.number_input(
                                        f"Set {si+1} — games {m['b']}", 0, 20,
                                        int(cur_sets[si]["s2"]) if si<len(cur_sets) and "s2" in cur_sets[si] else 0,
                                        key=f"ko_s2_{tid}_{rname}_{idx}_{si}"
                                    )
                                new_sets.append({"s1":int(s1),"s2":int(s2)})
                            ok, msg = validate_sets(fmt, new_sets)
                            if not ok:
                                st.error(msg)
                            gC,gD = st.columns(2)
                            with gC:
                                g1 = st.number_input(f"Puntos de oro {m['a']}", 0, 200, int(m.get("goldenA",0)), key=f"ko_g1_{tid}_{rname}_{idx}")
                            with gD:
                                g2 = st.number_input(f"Puntos de oro {m['b']}", 0, 200, int(m.get("goldenB",0)), key=f"ko_g2_{tid}_{rname}_{idx}")

                            if st.button("Guardar partido KO", key=f"ko_sv_{tid}_{rname}_{idx}"):
                                stats = compute_sets_stats(new_sets)
                                if stats["sets1"] == stats["sets2"]:
                                    rr = random.Random(int(state["config"].get("seed",42)) + idx)
                                    winner_rand = rr.choice([m['a'], m['b']])
                                    st.warning(f"Empate en sets → sorteo automático: **{winner_rand}**")
                                m["sets"] = new_sets
                                m["goldenA"] = int(g1)
                                m["goldenB"] = int(g2)
                                save_tournament(tid, state)
                                st.success("KO guardado.")
                                st.rerun()

                            # ganador (si lo hay)
                            if m.get("sets") and match_has_winner(m["sets"]):
                                statsF = compute_sets_stats(m["sets"])
                                winnerF = m['a'] if statsF["sets1"]>statsF["sets2"] else m['b']
                                advancing.append(winnerF)

                    # Avance de ronda
                    if rname == "FN":
                        if advancing and len(advancing)==1:
                            final_champion = advancing[0]
                            st.markdown(
                                f"<div style='padding:14px 18px;border-radius:10px;background:#fff9c4;border:1px solid #ffeb3b;font-size:1.1rem;font-weight:700;color:#795548;margin:8px 0;'>🏆 CAMPEÓN: {final_champion}</div>",
                                unsafe_allow_html=True
                            )
                            st.balloons()
                        continue

                    if advancing and len(advancing)>=1:
                        next_rname = make_next_round_name(rname)
                        if next_rname:
                            pairs = next_round(advancing)
                            # Elimina ronda siguiente existente y la vuelve a generar
                            state["ko"]["matches"] = [mm for mm in state["ko"]["matches"] if mm.get("round") not in (next_rname,)]
                            state["ko"]["matches"].extend(pairs_to_matches(pairs, next_rname))
                            save_tournament(tid, state)
                            st.info(f"Ronda {next_rname} preparada. Completa todos para llegar a la FINAL.")
                            st.rerun()
                    else:
                        st.info("⏳ A definir — Falta completar partidos de esta fase para avanzar.")

    # --- PERSISTENCIA ---
    with tab_persist:
        st.subheader("Persistencia (autosave + snapshots)")
        def sanitize_filename(s: str) -> str:
            return "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in s).strip("_")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.session_state.autosave = st.checkbox("Autosave", value=st.session_state.autosave)
        with c2:
            if st.button("💾 Guardar ahora"):
                save_tournament(tid, state)
                st.success("Guardado")
        with c3:
            meta = state.get("meta", {})
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{meta.get('tournament_id','')}_{sanitize_filename(meta.get('t_name',''))}_{meta.get('date','')}_{ts}.json"
            st.download_button(
                "⬇️ Descargar estado (JSON)",
                data=json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=fname,
                mime="application/json",
                key="dl_state_json"
            )
        with c4:
            up = st.file_uploader("⬆️ Cargar estado", type=["json"], key=f"up_{tid}")
            if up is not None:
                st.warning("⚠️ Restauración completa: guardaré el archivo subido como estado del torneo.")
                if st.button("Confirmar restauración", key=f"confirm_restore_{tid}", type="primary"):
                    try:
                        new_state = json.load(up)
                        save_tournament(tid, new_state)
                        st.success("Cargado y guardado.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al cargar: {e}")

    # Autosave simple
    current_hash = compute_state_hash(state)
    if st.session_state.autosave and current_hash != st.session_state.last_hash:
        save_tournament(tid, state)
        st.session_state.last_hash = current_hash

# ====== Viewer ======
def viewer_dashboard(user: Dict[str, Any]):
    app_cfg = load_app_config()
    render_header_bar(
        user_name=user.get("username",""),
        role=user.get("role",""),
        logo_url=app_cfg.get("app_logo_url","")
    )
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
    app_cfg = load_app_config()
    render_header_bar(
        user_name="Público" if public else "",
        role="VIEW",
        logo_url=app_cfg.get("app_logo_url","")
    )
    state = load_tournament(tid)
    if not state:
        st.error("No se encontró el torneo.")
        return
    st.subheader(f"{state['meta'].get('t_name')} — {state['meta'].get('place')} — {state['meta'].get('date')} — {state['meta'].get('gender')}")
    tab_over, tab_tables, tab_ko = st.tabs(["👀 General","📊 Tablas","🏁 Playoffs"])
    with tab_over:
        st.write("Parejas")
        dfp = pd.DataFrame({"Parejas": state.get("pairs", [])})
        st.table(dfp)
        if state.get("groups"):
            st.write("Zonas")
            for zi, group in enumerate(state["groups"], start=1):
                st.write(f"**Z{zi}**")
                st.table(pd.DataFrame({"Parejas": group}))
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
                    show = table.rename(columns={"pair":"Pareja"})
                    show["✓"] = show["Pos"].apply(lambda x: "✅" if x in (1,2) else "")
                    show = show[["Zona","Pos","✓","Pareja","PJ","PG","PP","GF","GC","DG","GP","PTS"]]
                    st.markdown(show.to_html(index=False, classes=["zebra","dark-header"]), unsafe_allow_html=True)
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

    if not st.session_state.get("auth_user"):
        app_cfg = load_app_config()
        render_header_bar(user_name="", role="", logo_url=app_cfg.get("app_logo_url",""))
        # Login
        st.markdown("### Ingreso — Usuario + PIN (6 dígitos)")
        with st.form("login", clear_on_submit=True):
            username = st.text_input("Usuario").strip()
            pin = st.text_input("PIN (6 dígitos)", type="password").strip()
            submitted = st.form_submit_button("Ingresar", type="primary")
        if submitted:
            user = get_user(username)
            if not user or not user.get("active", True):
                st.error("Usuario inexistente o inactivo.")
            elif len(pin)!=6 or not pin.isdigit():
                st.error("PIN inválido.")
            elif sha(pin) != user["pin_hash"]:
                st.error("PIN incorrecto.")
            else:
                st.session_state.auth_user = user
                st.success(f"Bienvenido {user['username']} ({user['role']})")
                st.rerun()
        st.caption("Iapps Padel Tournament · iAPPs Pádel — v3.3.35")
        return

    user = st.session_state["auth_user"]

    # Ruteo por rol
    if user["role"] == "SUPER_ADMIN":
        if mode=="public" and _tid:
            viewer_tournament(_tid, public=True)
        else:
            super_admin_panel()
        st.caption("Iapps Padel Tournament · iAPPs Pádel — v3.3.35")
        return

    if user["role"] == "TOURNAMENT_ADMIN":
        admin_dashboard(user)
        st.caption("Iapps Padel Tournament · iAPPs Pádel — v3.3.35")
        return

    if user["role"] == "VIEWER":
        if mode=="public" and _tid:
            viewer_tournament(_tid, public=True)
        else:
            viewer_dashboard(user)
        st.caption("Iapps Padel Tournament · iAPPs Pádel — v3.3.35")
        return

    st.error("Rol desconocido.")
    st.caption("Iapps Padel Tournament · iAPPs Pádel — v3.3.35")

if __name__ == "__main__":
    main()