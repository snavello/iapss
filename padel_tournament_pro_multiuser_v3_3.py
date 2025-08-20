
# padel_tournament_pro_multiuser_v3_4_0.py
# VersiÃ³n 3.4.0 â€” Cambios:
# 1) Enlace pÃºblico completo configurable por torneo (en Admin del torneo):
#    - Campo "URL base pÃºblica" y exposiciÃ³n del link completo con botÃ³n Abrir + copy integrado.
# 2) Sistema de "cabezas de serie":
#    - Toggle en ConfiguraciÃ³n: "Usar cabezas de serie" (default: NO).
#    - Marcado de parejas como "cabeza de serie" (sÃ­/no) â€” exactamente 1 por zona requerido para sortear.
#    - Sorteo: una semilla por zona al azar; el resto, distribuciÃ³n balanceada.
#    - Importar CSV con columna opcional "cabeza de serie" (sÃ­/no/true/false/1/0).
#    - Distintivo visual â­ en listados (parejas y zonas/fixture).
# 3) Regla de mÃ­nimo para sortear zonas:
#    - Se exige total_parejas >= num_zonas * top_per_zone.
# 4) Mantenimiento de todas las funcionalidades previas (resultados, PDF, autosave, roles).
#
# NOTA: Requiere streamlit >= 1.30 para el Ã­cono de "copiar" en bloques de cÃ³digo (st.code).
#
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

# ====== Dependencias opcionales para PDF ======
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

st.set_page_config(page_title="Torneo de PÃ¡del â€” Multiusuario v3.4.0", layout="wide")

# ====== Constantes de estilo/colores para el logo SVG ======
PRIMARY_BLUE = "#0D47A1"
LIME_GREEN  = "#AEEA00"
DARK_BLUE   = "#082D63"

# ====== Carpetas y persistencia ======
DATA_DIR = Path("data")
APP_CONFIG_PATH = DATA_DIR / "app_config.json"   # configuraciÃ³n global (logo_url)
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

# ====== Config App (global) ======
DEFAULT_APP_CONFIG = {
    "app_logo_url": ""  # URL pÃºblica del logo de la aplicaciÃ³n (opcional). Si vacÃ­o, se usa el SVG embebido.
}

def load_app_config() -> Dict[str, Any]:
    if not APP_CONFIG_PATH.exists():
        APP_CONFIG_PATH.write_text(json.dumps(DEFAULT_APP_CONFIG, indent=2), encoding="utf-8")
        return DEFAULT_APP_CONFIG.copy()
    try:
        return json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_APP_CONFIG.copy()

def save_app_config(cfg: Dict[str, Any]):
    APP_CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

# ====== Usuarios ======
DEFAULT_SUPER = {"username": "ADMIN", "pin_hash": sha("199601"), "role": "SUPER_ADMIN",
                 "assigned_admin": None, "created_at": now_iso(), "active": True}

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

# ====== Torneos ======
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
            try: old.unlink()
            except Exception: pass

# ====== Reglas del torneo (sin puntos por empate) ======
DEFAULT_CONFIG = {
    "t_name": "Open PÃ¡del",
    "num_pairs": 16,
    "num_zones": 4,
    "top_per_zone": 2,
    "points_win": 2,
    "points_loss": 0,
    "seed": 42,
    "format": "best_of_3",  # one_set | best_of_3 | best_of_5
    "use_seeds": False      # NUEVO
}

rng = lambda off, seed: random.Random(int(seed) + int(off))

def create_groups_simple(pairs, num_groups, seed=42):
    """Sorteo simple sin semillas: round-robin de lista mezclada."""
    r = random.Random(int(seed))
    shuffled = pairs[:]
    r.shuffle(shuffled)
    groups = [[] for _ in range(num_groups)]
    for i, p in enumerate(shuffled):
        groups[i % num_groups].append(p)
    return groups

def create_groups_with_seeds(all_pairs: List[str], seed_numbers: List[int], num_groups: int, seed_value: int) -> List[List[str]]:
    """Asigna 1 cabeza de serie por zona al azar y distribuye el resto balanceado (diferencia â‰¤ 1)."""
    # Mapear numero -> label
    def parse_pair_number(label: str) -> Optional[int]:
        try:
            left = label.split("â€”", 1)[0].strip()
            return int(left)
        except Exception:
            return None

    pairs_by_num = {parse_pair_number(lbl): lbl for lbl in all_pairs}
    seeds_labels = []
    for n in seed_numbers:
        if n in pairs_by_num:
            seeds_labels.append(pairs_by_num[n])

    # Embarajar semillas y asignar 1 por zona
    r = random.Random(int(seed_value) + 1000)
    r.shuffle(seeds_labels)
    groups = [[] for _ in range(num_groups)]
    for i in range(num_groups):
        groups[i].append(seeds_labels[i])

    # Resto de parejas (no semillas)
    seeds_set = set(seeds_labels)
    rest = [p for p in all_pairs if p not in seeds_set]
    r2 = random.Random(int(seed_value) + 2000)
    r2.shuffle(rest)

    # DistribuciÃ³n balanceada: siempre a la zona con menor tamaÃ±o
    for p in rest:
        sizes = [len(g) for g in groups]
        idx = sizes.index(min(sizes))
        groups[idx].append(p)

    return groups

def rr_schedule(group):
    return list(combinations(group, 2))

def build_fixtures(groups):
    rows = []
    if not groups: return []
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
        if n != 1: return False, "Formato a 1 set: debe haber exactamente 1 set."
    elif fmt == "best_of_3":
        if n < 2 or n > 3: return False, "Formato al mejor de 3: debe haber 2 o 3 sets."
    elif fmt == "best_of_5":
        if n < 3 or n > 5: return False, "Formato al mejor de 5: debe haber entre 3 y 5 sets."
    return True, ""

def compute_sets_stats(sets: List[Dict[str,int]]) -> Dict[str,int]:
    g1=g2=s1=s2=0
    for s in sets:
        a = int(s.get("s1",0)); b = int(s.get("s2",0))
        g1 += a; g2 += b
        if a>b: s1 += 1
        elif b>a: s2 += 1
    return {"games1": g1, "games2": g2, "sets1": s1, "sets2": s2}

def standings_from_results(zone_name, group_pairs, results_list, cfg):
    rows = [{"pair": p, "PJ": 0, "PG": 0, "PP": 0, "GF": 0, "GC": 0, "GP": 0, "PTS": 0} for p in group_pairs]
    table = pd.DataFrame(rows).set_index("pair")
    fmt = cfg.get("format","best_of_3")
    for m in results_list:
        if m["zone"] != zone_name: continue
        sets = m.get("sets", [])
        ok, _ = validate_sets(fmt, sets)
        if not ok: continue
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
        # Sin empates en fase de grupos
        if s1>s2:
            table.at[p1, "PG"] += 1; table.at[p2, "PP"] += 1
            table.at[p1, "PTS"] += cfg["points_win"]
            table.at[p2, "PTS"] += cfg["points_loss"]
        elif s2>s1:
            table.at[p2, "PG"] += 1; table.at[p1, "PP"] += 1
            table.at[p2, "PTS"] += cfg["points_win"]
            table.at[p1, "PTS"] += cfg["points_loss"]
    table["DG"] = table["GF"] - table["GC"]
    r = rng(0, cfg["seed"]) ; randmap = {p: r.random() for p in table.index}
    table["RND"] = table.index.map(randmap.get)
    table = table.sort_values(by=["PTS","DG","GP","RND"], ascending=[False,False,False,False]).reset_index()
    table.insert(0, "Zona", zone_name)
    table.insert(1, "Pos", range(1, len(table)+1))
    return table.drop(columns=["RND"])

def qualified_from_tables(zone_tables, k):
    qualified = []
    for table in zone_tables:
        if table.empty: continue
        z = table.iloc[0]["Zona"]
        q = table.head(int(k))
        for _, row in q.iterrows():
            qualified.append((z, int(row["Pos"]), row["pair"]))
    return qualified

def cross_bracket(qualified):
    winners = [(z, pos, p) for (z,pos,p) in qualified if pos==1]
    runners = [(z, pos, p) for (z,pos,p) in qualified if pos==2]
    if len(winners)==0 or len(runners)==0: return []
    runners_rot = runners[1:] + runners[:1] if len(runners)>1 else runners
    pairs = []
    for w, r in zip(winners, runners_rot):
        pairs.append((f"{w[0]}1", w[2], f"{r[0]}2", r[2]))
    return pairs

def next_round(slots: List[str]):
    out=[]; i=0
    while i < len(slots):
        if i+1 < len(slots): out.append((slots[i], slots[i+1])); i+=2
        else: out.append((slots[i], None)); i+=1
    return out

# ====== Branding (logo arriba-izquierda fijo) ======
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

def render_brand_top_left():
    app_cfg = load_app_config()
    url = (app_cfg or {}).get("app_logo_url", "").strip()
    if url:
        html = f"""
        <style>
          .brand-fixed {{
            position: fixed; top: 8px; left: 12px; z-index: 10000;
            pointer-events: none; /* no bloquea clics en UI */
          }}
        </style>
        <div class="brand-fixed"><img src="{url}" width="220"/></div>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        svg = brand_svg(220)
        html = f"""
        <style>
          .brand-fixed {{
            position: fixed; top: 8px; left: 12px; z-index: 10000;
            pointer-events: none;
          }}
        </style>
        <div class="brand-fixed">{svg}</div>
        """
        st.markdown(html, unsafe_allow_html=True)

# ====== SesiÃ³n ======
def init_session():
    st.session_state.setdefault("auth_user", None)
    st.session_state.setdefault("current_tid", None)
    st.session_state.setdefault("autosave", True)
    st.session_state.setdefault("last_hash", "")
    st.session_state.setdefault("pdf_fixture_bytes", None)
    st.session_state.setdefault("pdf_playoffs_bytes", None)
    st.session_state.setdefault("pdf_generated_at", None)

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
            "public_base_url": ""  # NUEVO: por torneo
        },
        "config": cfg,
        "pairs": [],
        "seeds": [],      # NUEVO: lista de nÃºmeros de pareja marcados como cabeza de serie
        "groups": None,
        "results": [],
        "ko": {"matches": []},
    }

# ====== Utilidades parejas ======
def parse_pair_number(label: str) -> Optional[int]:
    # Formato esperado: "NN â€” Nombre1 / Nombre2"
    try:
        left = label.split("â€”", 1)[0].strip()
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
    return None  # lleno

def format_pair_label(n: int, j1: str, j2: str) -> str:
    return f"{n:02d} â€” {j1.strip()} / {j2.strip()}"

def remove_pair_by_number(pairs: List[str], n: int) -> List[str]:
    out = []
    for p in pairs:
        pn = parse_pair_number(p)
        if pn != n:
            out.append(p)
    return out

def yes_no_to_bool(v) -> Optional[bool]:
    if v is None: return None
    s = str(v).strip().lower()
    if s in ("si","sÃ­","s","yes","true","1"): return True
    if s in ("no","n","false","0"): return False
    return None

def build_public_url(meta: Dict[str, Any], tid: str) -> str:
    base = (meta or {}).get("public_base_url", "") or ""
    base = base.strip()
    if not base:
        return f"?mode=public&tid={tid}"
    glue = "&" if "?" in base else "?"
    return f"{base}{glue}mode=public&tid={tid}"

def label_with_seed(label: str, seeds: List[int]) -> str:
    n = parse_pair_number(label)
    if n in (seeds or []):
        return f"â­ {label}"
    return label

# ====== Login ======
def login_form():
    st.markdown("### Ingreso â€” Usuario + PIN (6 dÃ­gitos)")
    with st.form("login"):
        username = st.text_input("Usuario").strip()
        pin = st.text_input("PIN (6 dÃ­gitos)", type="password").strip()
        submitted = st.form_submit_button("Ingresar", type="primary")
    if submitted:
        user = get_user(username)
        if not user or not user.get("active", True):
            st.error("Usuario inexistente o inactivo.") ; return
        if len(pin)!=6 or not pin.isdigit():
            st.error("PIN invÃ¡lido.") ; return
        if sha(pin) != user["pin_hash"]:
            st.error("PIN incorrecto.") ; return
        st.session_state.auth_user = user
        st.success(f"Bienvenido {user['username']} ({user['role']})")
        st.rerun()

# ====== Panel del SUPER_ADMIN ======
def super_admin_panel():
    st.header("Panel de ADMIN (Super Admin)")
    users = load_users()

    with st.expander("ðŸŽ¨ Apariencia (Logo global de la app)", expanded=True):
        app_cfg = load_app_config()
        url = st.text_input("URL pÃºblica del logotipo de la aplicaciÃ³n (no por torneo)", value=app_cfg.get("app_logo_url","")).strip()
        if st.button("Guardar logo global", type="primary"):
            app_cfg["app_logo_url"] = url
            save_app_config(app_cfg)
            st.success("Logo global guardado. (RefrescÃ¡ la pÃ¡gina si no lo ves de inmediato)")
        st.caption("Sugerencia: usa el vÃ­nculo RAW de GitHub u otro hosting pÃºblico.")

    with st.expander("âž• Crear usuario"):
        c1,c2,c3 = st.columns(3)
        with c1: u = st.text_input("Username nuevo").strip()
        with c2: role = st.selectbox("Rol", ["TOURNAMENT_ADMIN","VIEWER"])
        with c3: pin = st.text_input("PIN inicial (6)", max_chars=6).strip()
        assigned_admin=None
        if role=="VIEWER":
            admins=[x["username"] for x in users if x["role"]=="TOURNAMENT_ADMIN" and x.get("active",True)]
            assigned_admin = st.selectbox("Asignar a admin", admins) if admins else None
        if st.button("Crear usuario", type="primary"):
            if not u: st.error("Username requerido.")
            elif get_user(u): st.error("Ya existe.")
            elif len(pin)!=6 or not pin.isdigit(): st.error("PIN invÃ¡lido.")
            else:
                set_user({"username":u,"pin_hash":sha(pin),"role":role,"assigned_admin":assigned_admin,"created_at":now_iso(),"active":True})
                st.success(f"Usuario {u} creado.")

    st.subheader("Usuarios")
    for usr in users:
        with st.container(border=True):
            st.write(f"**{usr['username']}** â€” rol `{usr['role']}` â€” activo `{usr.get('active',True)}`")
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                if st.button(f"Reset PIN: {usr['username']}", key=f"rst_{usr['username']}"):
                    new_pin = st.text_input(f"Nuevo PIN para {usr['username']}", key=f"np_{usr['username']}", max_chars=6)
                    if new_pin and new_pin.isdigit() and len(new_pin)==6:
                        usr["pin_hash"] = sha(new_pin); set_user(usr); st.success("PIN actualizado.")
            with c2:
                if usr["role"]=="VIEWER":
                    admins=[x["username"] for x in users if x["role"]=="TOURNAMENT_ADMIN" and x.get("active",True)]
                    new_admin = st.selectbox(f"Admin de {usr['username']}", admins+[None], key=f"adm_{usr['username']}")
                    if st.button(f"Guardar admin {usr['username']}", key=f"sadm_{usr['username']}"):
                        usr["assigned_admin"]=new_admin; set_user(usr); st.success("Asignado.")
            with c3:
                active_toggle = st.checkbox("Activo", value=usr.get("active",True), key=f"act_{usr['username']}")
                if st.button(f"Guardar activo {usr['username']}", key=f"sact_{usr['username']}"):
                    usr["active"]=active_toggle; set_user(usr); st.success("Estado guardado.")
            with c4:
                if usr["username"]!="ADMIN" and st.button(f"Inactivar {usr['username']}", key=f"del_{usr['username']}"):
                    usr["active"] = False; set_user(usr); st.success("Inactivado.")

# ====== Ãrea del ADMIN de Torneo ======
def load_index_for_admin(admin_username: str) -> List[Dict[str, Any]]:
    idx = load_index()
    my = [t for t in idx if t.get("admin_username")==admin_username]
    def keyf(t):
        try: return datetime.fromisoformat(t.get("date"))
        except Exception: return datetime.min
    return sorted(my, key=keyf, reverse=True)

def create_tournament(admin_username: str, t_name: str, place: str, tdate: str, gender: str) -> str:
    tid = str(uuid.uuid4())[:8]
    meta = {"tournament_id":tid,"t_name":t_name,"place":place,"date":tdate,"gender":gender}
    state = tournament_state_template(admin_username, meta)
    save_tournament(tid, state)
    idx = load_index(); idx.append({"tournament_id":tid,"t_name":t_name,"place":place,"date":tdate,"gender":gender,"admin_username":admin_username,"created_at":now_iso()})
    save_index(idx)
    return tid

def delete_tournament(admin_username: str, tid: str):
    idx = load_index(); idx = [t for t in idx if not (t["tournament_id"]==tid and t["admin_username"]==admin_username)]
    save_index(idx)
    p = tourn_path(tid)
    if p.exists(): p.unlink()
    for f in (snap_dir_for(tid)).glob("*.json"):
        try: f.unlink()
        except Exception: pass

def admin_dashboard(user: Dict[str, Any]):
    st.header(f"Torneos de {user['username']}")
    with st.expander("âž• Crear torneo nuevo", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        with c1: t_name = st.text_input("Nombre del torneo", value="Open PÃ¡del")
        with c2: place = st.text_input("Lugar / Club", value="Mi Club")
        with c3: tdate = st.date_input("Fecha", value=date.today()).isoformat()
        with c4: gender = st.selectbox("GÃ©nero", ["masculino","femenino","mixto"], index=2)
        if st.button("Crear torneo", type="primary"):
            tid = create_tournament(user["username"], t_name, place, tdate, gender)
            st.session_state.current_tid = tid
            st.success(f"Torneo creado: {t_name} ({tid})")

    my = load_index_for_admin(user["username"])
    if not my: st.info("AÃºn no tienes torneos."); return

    st.subheader("Abrir / eliminar torneo")
    names = [f"{t['date']} â€” {t['t_name']} ({t['gender']}) â€” {t['place']} â€” ID:{t['tournament_id']}" for t in my]
    selected = st.selectbox("Selecciona un torneo", names, index=0)
    sel = my[names.index(selected)]
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("Abrir torneo"): st.session_state.current_tid = sel["tournament_id"]
    with c2:
        if st.button("Eliminar torneo", type="secondary"):
            delete_tournament(user["username"], sel["tournament_id"])
            st.success("Torneo eliminado.")
            if st.session_state.get("current_tid")==sel["tournament_id"]:
                st.session_state.current_tid=None
    with c3:
        tid = sel["tournament_id"]
        # Cargar meta para construir link pÃºblico completo
        state_tmp = load_tournament(tid) or {}
        meta_tmp = state_tmp.get("meta", {})
        full_link = build_public_url(meta_tmp, tid)
        st.caption("Link pÃºblico (solo lectura):")
        st.code(full_link)
        st.link_button("Abrir enlace pÃºblico", url=full_link)

    if st.session_state.get("current_tid"): tournament_manager(user, st.session_state["current_tid"])

# ====== Gestor del Torneo (Admin) ======
def tournament_manager(user: Dict[str, Any], tid: str):
    state = load_tournament(tid)
    if not state: st.error("No se encontrÃ³ el torneo."); return

    # Backfill de claves nuevas
    state.setdefault("seeds", [])
    state.setdefault("config", DEFAULT_CONFIG.copy())
    state["config"].setdefault("use_seeds", False)
    state.setdefault("meta", {})
    state["meta"].setdefault("public_base_url", "")

    render_brand_top_left()  # usa logo global

    tab_cfg, tab_pairs, tab_results, tab_tables, tab_ko, tab_persist = st.tabs(
        ["âš™ï¸ ConfiguraciÃ³n", "ðŸ‘¥ Parejas", "ðŸ“ Resultados", "ðŸ“Š Tablas", "ðŸ—‚ï¸ Playoffs", "ðŸ’¾ Persistencia"]
    )

    cfg = state.get("config", DEFAULT_CONFIG.copy())

    # --- CONFIG ---
    with tab_cfg:
        st.subheader("Datos del torneo")
        m = state.get("meta", {})
        st.write({"ID":m.get("tournament_id"),"Nombre":m.get("t_name"),"Lugar":m.get("place"),"Fecha":m.get("date"),"GÃ©nero":m.get("gender"),"Admin":m.get("admin_username")})

        st.subheader("ConfiguraciÃ³n deportiva")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            cfg["t_name"] = st.text_input("Nombre para mostrar", value=cfg.get("t_name","Open PÃ¡del"))
            cfg["num_pairs"] = st.number_input("Cantidad mÃ¡xima de parejas", 2, 256, int(cfg.get("num_pairs",16)), step=1)
        with c2:
            cfg["num_zones"] = st.number_input("Cantidad de zonas", 2, 32, int(cfg.get("num_zones",4)), step=1)
            cfg["top_per_zone"] = st.number_input("Clasifican por zona (Top N)", 1, 8, int(cfg.get("top_per_zone",2)), step=1)
        with c3:
            cfg["points_win"] = st.number_input("Puntos por victoria", 1, 10, int(cfg.get("points_win",2)), step=1)
            cfg["points_loss"] = st.number_input("Puntos por derrota", 0, 5, int(cfg.get("points_loss",0)), step=1)
        with c4:
            cfg["seed"] = st.number_input("Semilla (sorteo zonas)", 1, 999999, int(cfg.get("seed",42)), step=1)
        fmt = st.selectbox("Formato de partido", ["one_set","best_of_3","best_of_5"], index=["one_set","best_of_3","best_of_5"].index(cfg.get("format","best_of_3")))
        cfg["format"] = fmt

        st.markdown("â€”")
        cUse, cSave = st.columns([2,1])
        with cUse:
            cfg["use_seeds"] = st.checkbox("Usar cabezas de serie", value=bool(cfg.get("use_seeds", False)))
        with cSave:
            if st.button("ðŸ’¾ Guardar configuraciÃ³n", type="primary"):
                state["config"] = {k:int(v) if isinstance(v,(int,float)) and k not in ["t_name","format","use_seeds"] else v for k,v in cfg.items()}
                state["config"]["use_seeds"] = bool(cfg.get("use_seeds", False))
                save_tournament(tid, state); st.success("ConfiguraciÃ³n guardada.")

        st.divider()
        st.subheader("Enlace pÃºblico")
        base_url = st.text_input("URL base pÃºblica (ej: https://mis-torneos.com/app )", value=state["meta"].get("public_base_url","")).strip()
        if st.button("Guardar URL base pÃºblica"):
            state["meta"]["public_base_url"] = base_url
            save_tournament(tid, state)
            st.success("URL base pÃºblica guardada.")
        full_link = build_public_url(state["meta"], tid)
        st.caption("Enlace pÃºblico completo:")
        st.code(full_link)
        st.link_button("Abrir enlace pÃºblico", url=full_link)

        st.divider()
        cB,cC = st.columns(2)
        with cB:
            if st.button("ðŸŽ² Sortear zonas (crear/rehacer fixture)"):
                pairs = state.get("pairs", [])
                num_z = int(cfg.get("num_zones", 4))
                top_z = int(cfg.get("top_per_zone", 2))

                if len(pairs) < num_z:
                    st.error("Debe haber al menos tantas parejas como zonas.")
                elif len(pairs) < (num_z * top_z):
                    st.error(f"Regla de mÃ­nimo: se requieren al menos {num_z * top_z} parejas para sortear (num_zonas Ã— top_per_zone).")
                else:
                    use_seeds = bool(cfg.get("use_seeds", False))
                    if use_seeds:
                        seeds = state.get("seeds", []) or []
                        # Validaciones de semillas
                        if len(seeds) != num_z:
                            st.error(f"Debe haber exactamente {num_z} 'cabezas de serie' (1 por zona). Actualmente: {len(seeds)}.")
                        else:
                            # Verificar que las semillas existan
                            pair_nums = set([parse_pair_number(p) for p in pairs])
                            if not all(s in pair_nums for s in seeds):
                                st.error("Una o mÃ¡s 'cabezas de serie' no corresponden a parejas cargadas.")
                            else:
                                groups = create_groups_with_seeds(pairs, seeds, num_z, seed_value=int(cfg["seed"]))
                                state["groups"] = groups
                                state["results"] = build_fixtures(groups)
                                save_tournament(tid, state); st.success("Zonas + fixture generados (con cabezas de serie).")
                    else:
                        groups = create_groups_simple(pairs, int(cfg["num_zones"]), seed=int(cfg["seed"]))
                        state["groups"] = groups
                        state["results"] = build_fixtures(groups)
                        save_tournament(tid, state); st.success("Zonas + fixture generados.")

        with cC:
            if REPORTLAB_OK and st.button("ðŸ§¾ Generar PDFs"):
                with st.spinner("Generando PDFs..."):
                    buf1 = export_fixture_pdf(state)
                    buf2 = export_playoffs_pdf(state)
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
                st.download_button("â¬‡ï¸ Fixture (PDF)", data=st.session_state.pdf_fixture_bytes,
                                   file_name=f"fixture_{state['meta']['tournament_id']}.pdf", mime="application/pdf",
                                   key="dl_fixture_pdf")
            if st.session_state.pdf_playoffs_bytes:
                st.download_button("â¬‡ï¸ Playoffs (PDF)", data=st.session_state.pdf_playoffs_bytes,
                                   file_name=f"playoffs_{state['meta']['tournament_id']}.pdf", mime="application/pdf",
                                   key="dl_playoffs_pdf")
            if st.button("ðŸ§¹ Limpiar PDFs generados"):
                st.session_state.pdf_fixture_bytes = None
                st.session_state.pdf_playoffs_bytes = None
                st.session_state.pdf_generated_at = None
                st.success("Limpio.")

    # --- PAREJAS (manual + CSV; sin ediciÃ³n rÃ¡pida) ---
    with tab_pairs:
        st.subheader("Parejas")
        pairs = state.get("pairs", [])
        max_pairs = int(state.get("config", {}).get("num_pairs", 16))

        # Alta manual individual (con control de mÃ¡ximo y numeraciÃ³n 1..N)
        st.markdown("**Alta manual â€” una pareja por vez**")
        next_n = next_available_number(pairs, max_pairs)
        c1,c2,c3,c4 = st.columns([1,3,3,2])
        with c1:
            st.text_input("NÂ° pareja", value=(str(next_n) if next_n else "â€”"), disabled=True, key=f"num_auto_{tid}")
        with c2:
            p1 = st.text_input("Jugador 1", key=f"p1_{tid}")
        with c3:
            p2 = st.text_input("Jugador 2", key=f"p2_{tid}")
        with c4:
            disabled_btn = (next_n is None)
            if st.button("âž• Agregar pareja", key=f"add_pair_{tid}", type="primary", disabled=disabled_btn):
                p1c, p2c = (p1 or "").strip(), (p2 or "").strip()
                if not p1c or not p2c:
                    st.error("CompletÃ¡ ambos nombres.")
                else:
                    label = format_pair_label(next_n, p1c, p2c)
                    pairs.append(label)
                    state["pairs"] = pairs
                    save_tournament(tid, state)
                    st.success(f"Agregada: {label}")
                    st.experimental_rerun()
        if next_n is None:
            st.warning(f"Se alcanzÃ³ el mÃ¡ximo de parejas configurado ({max_pairs}). PodÃ©s borrar alguna o aumentar el mÃ¡ximo en ConfiguraciÃ³n.")

        st.divider()

        # Importar CSV con control de mÃ¡ximo
        st.markdown("**Importar CSV (opcional)**")
        st.caption("Formato: columnas `numero, jugador1, jugador2, cabeza de serie (opcional)`.\nEjemplo: `1, Juan Perez, Luis Diaz, sÃ­`.")
        up = st.file_uploader("Seleccionar CSV", type=["csv"], key=f"csv_{tid}")
        if up is not None:
            try:
                df = pd.read_csv(up, header=0)
            except Exception:
                up.seek(0)
                df = pd.read_csv(up, header=None, names=["numero","jugador1","jugador2","cabeza de serie"])
            cols = [c.strip().lower() for c in df.columns.tolist()]
            df.columns = cols
            required = {"numero","jugador1","jugador2"}
            if not required.issubset(set(df.columns)):
                st.error("El CSV debe contener columnas: numero, jugador1, jugador2 (y opcional: cabeza de serie)")
            else:
                # Convertir a lista formateada y respetar el mÃ¡ximo
                parsed = []
                new_seeds = []
                for _, row in df.iterrows():
                    try:
                        num = int(row["numero"])
                    except Exception:
                        continue
                    j1 = str(row["jugador1"]).strip()
                    j2 = str(row["jugador2"]).strip()
                    if j1 and j2 and num >= 1:
                        parsed.append((num, j1, j2))
                        # Semilla CSV
                        seed_col = None
                        for k in ("cabeza de serie","cabeza_de_serie","seed"):
                            if k in df.columns: seed_col = k; break
                        if seed_col is not None:
                            val = yes_no_to_bool(row.get(seed_col))
                            if val is True:
                                new_seeds.append(num)
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
                        st.warning("El CSV no contenÃ­a filas vÃ¡lidas dentro del rango permitido.")
                    else:
                        state["pairs"] = new_list
                        # Actualizar semillas (si hay columna). No forzamos el mÃ¡ximo acÃ¡; se valida al sortear.
                        if new_seeds:
                            # Filtrar a nÃºmeros realmente presentes en new_list
                            present_nums = set(parse_pair_number(lbl) for lbl in new_list)
                            state["seeds"] = [n for n in new_seeds if n in present_nums]
                        save_tournament(tid, state)
                        st.success(f"Importadas {len(new_list)} parejas (mÃ¡ximo {max_pairs}). Semillas detectadas: {len(state.get('seeds', []))}.")
                        st.experimental_rerun()
                else:
                    st.warning("No se encontraron filas vÃ¡lidas en el CSV.")

        st.divider()

        # Tabla de parejas + marcado de semillas (si aplica) + botÃ³n borrar por fila
        seeds = state.get("seeds", []) or []
        show_pairs = [label_with_seed(p, seeds) for p in pairs]
        if pairs:
            st.markdown("### Listado de parejas")
            df_pairs = pd.DataFrame({"Pareja": show_pairs})
            st.table(df_pairs)

            # Marcado de semillas si estÃ¡ habilitado
            if bool(state["config"].get("use_seeds", False)):
                st.markdown("**Cabezas de serie**")
                max_seeds = int(state["config"].get("num_zones", 4))
                current_nums = [parse_pair_number(p) for p in pairs if parse_pair_number(p) is not None]
                # Multi-select de nÃºmeros de pareja
                sel = st.multiselect(f"SeleccionÃ¡ hasta {max_seeds} 'cabezas de serie' (1 por zona)",
                                     options=current_nums,
                                     default=[s for s in seeds if s in current_nums])
                if len(sel) > max_seeds:
                    st.error(f"Seleccionaste {len(sel)} semillas. El mÃ¡ximo es {max_seeds}.")
                if st.button("ðŸ’¾ Guardar cabezas de serie"):
                    if len(sel) > max_seeds:
                        st.error(f"No se guardÃ³. MÃ¡ximo permitido: {max_seeds}.")
                    else:
                        state["seeds"] = [int(x) for x in sel]
                        save_tournament(tid, state)
                        st.success(f"Guardado. Semillas: {len(state['seeds'])}/{max_seeds}.")

            # Botones de borrado
            st.markdown("**Borrar pareja:**")
            cols = st.columns(4)
            per_row = 4
            for i, label in enumerate(pairs):
                n = parse_pair_number(label) or (i+1)
                col = cols[i % per_row]
                with col:
                    if st.button(f"ðŸ—‘ï¸ NÂº {n}", key=f"del_{tid}_{n}"):
                        state["pairs"] = remove_pair_by_number(pairs, n)
                        # Quitar semilla si existÃ­a
                        if n in state.get("seeds", []):
                            state["seeds"] = [x for x in state["seeds"] if x != n]
                        save_tournament(tid, state)
                        st.success(f"Eliminada pareja NÂº {n}.")
                        st.experimental_rerun()
        else:
            st.info("AÃºn no hay parejas cargadas.")

        # Zonas actuales (mostrar â­)
        if state.get("groups"):
            st.divider(); st.markdown("### Zonas")
            seeds_set = set(state.get("seeds", []) or [])
            for zi, group in enumerate(state["groups"], start=1):
                disp = [label_with_seed(p, seeds_set) for p in group]
                st.write(f"**Z{zi}**"); st.table(pd.DataFrame({"Parejas": disp}))

    # --- RESULTADOS ---
    with tab_results:
        st.subheader("Resultados â€” fase de grupos (sets + puntos de oro)")
        if not state.get("groups"):
            st.info("Primero crea/sortea zonas (en ConfiguraciÃ³n).")
        else:
            fmt = state["config"].get("format","best_of_3")
            zones = sorted({m["zone"] for m in state["results"]})
            z_filter = st.selectbox("Filtrar por zona", ["(todas)"] + zones)
            pnames = sorted(set([m["pair1"] for m in state["results"]] + [m["pair2"] for m in state["results"]]))
            p_filter = st.selectbox("Filtrar por pareja", ["(todas)"] + pnames)
            listing = state["results"]
            if z_filter != "(todas)": listing = [m for m in listing if m["zone"]==z_filter]
            if p_filter != "(todas)": listing = [m for m in listing if m["pair1"]==p_filter or m["pair2"]==p_filter]

            for m in listing:
                idx = state["results"].index(m)
                with st.container(border=True):
                    st.write(f"**{m['zone']}** â€” {m['pair1']} vs {m['pair2']}")
                    cur_sets = m.get("sets", [])
                    n_min, n_max = (1,1) if fmt=="one_set" else ((2,3) if fmt=="best_of_3" else (3,5))
                    n_sets = st.number_input("Sets jugados", min_value=n_min, max_value=n_max,
                                             value=min(max(len(cur_sets), n_min), n_max),
                                             key=f"ns_{tid}_{idx}")
                    new_sets = []
                    for si in range(n_sets):
                        cA,cB = st.columns(2)
                        with cA:
                            s1 = st.number_input(f"Set {si+1} â€” games {m['pair1']}", 0, 20,
                                                 int(cur_sets[si]["s1"]) if si<len(cur_sets) and "s1" in cur_sets[si] else 0,
                                                 key=f"s1_{tid}_{idx}_{si}")
                        with cB:
                            s2 = st.number_input(f"Set {si+1} â€” games {m['pair2']}", 0, 20,
                                                 int(cur_sets[si]["s2"]) if si<len(cur_sets) and "s2" in cur_sets[si] else 0,
                                                 key=f"s2_{tid}_{idx}_{si}")
                        new_sets.append({"s1":int(s1),"s2":int(s2)})
                    ok, msg = validate_sets(fmt, new_sets)
                    if not ok: st.error(msg)
                    gC,gD = st.columns(2)
                    with gC:
                        g1 = st.number_input(f"Puntos de oro {m['pair1']}", 0, 200, int(m.get("golden1",0)), key=f"g1_{tid}_{idx}")
                    with gD:
                        g2 = st.number_input(f"Puntos de oro {m['pair2']}", 0, 200, int(m.get("golden2",0)), key=f"g2_{tid}_{idx}")
                    if st.button("Guardar partido", key=f"sv_{tid}_{idx}"):
                        stats = compute_sets_stats(new_sets)
                        if stats["sets1"] == stats["sets2"]:
                            st.error("Debe haber un ganador (no se permiten empates en fase de grupos). AjustÃ¡ los sets.")
                        else:
                            state["results"][idx]["sets"] = new_sets
                            state["results"][idx]["golden1"] = int(g1)
                            state["results"][idx]["golden2"] = int(g2)
                            save_tournament(tid, state); st.success("Partido guardado.")

    # --- TABLAS ---
    with tab_tables:
        st.subheader("Tablas por zona y clasificados")
        if not state.get("groups") or not state.get("results"):
            st.info("AÃºn no hay fixture o resultados.")
        else:
            cfg = state["config"]
            zone_tables = []
            for zi, group in enumerate(state["groups"], start=1):
                zone_name = f"Z{zi}"
                table = standings_from_results(zone_name, group, state["results"], cfg)
                zone_tables.append(table)
                with st.expander(f"Tabla {zone_name}", expanded=True):
                    st.dataframe(table, use_container_width=True)
            qualified = qualified_from_tables(zone_tables, cfg["top_per_zone"])
            st.markdown("### Clasificados a Playoffs")
            if not qualified: st.info("Sin clasificados aÃºn.")
            else: st.table(pd.DataFrame([{"Zona":z,"Pos":pos,"Pareja":p} for (z,pos,p) in qualified]))

    # --- PLAYOFFS ---
    with tab_ko:
        st.subheader("Playoffs (por sets + puntos de oro)")
        if not state.get("groups") or not state.get("results"):
            st.info("Necesitas tener zonas y resultados para definir clasificados.")
        else:
            cfg = state["config"]; fmt = cfg.get("format","best_of_3")
            zone_tables = []
            for zi, group in enumerate(state["groups"], start=1):
                zone_name = f"Z{zi}"
                table = standings_from_results(zone_name, group, state["results"], cfg)
                zone_tables.append(table)
            qualified = qualified_from_tables(zone_tables, cfg["top_per_zone"])
            if len(qualified) < 2:
                st.info("Se requieren al menos dos zonas con clasificados.")
            else:
                initial = cross_bracket(qualified)
                st.markdown("#### Cruces iniciales")
                if not state["ko"]["matches"]:
                    for i,(tagA,a,tagB,b) in enumerate(initial, start=1):
                        state["ko"]["matches"].append({"round": "QF","label": f"QF{i}","a": a,"b": b,"sets": [],"goldenA": 0,"goldenB": 0})
                    save_tournament(tid, state)

                round_names = ["QF","SF","FN"]
                for rname in round_names:
                    ms = [m for m in state["ko"]["matches"] if m["round"]==rname]
                    if not ms: continue
                    st.markdown(f"### {rname}")
                    advancing = []
                    for idx, m in enumerate(ms, start=1):
                        with st.container(border=True):
                            st.write(f"**{m['label']}** â€” {m['a']} vs {m['b']}")
                            cur_sets = m.get("sets", [])
                            n_min, n_max = (1,1) if fmt=="one_set" else ((2,3) if fmt=="best_of_3" else (3,5))
                            n_sets = st.number_input("Sets jugados", min_value=n_min, max_value=n_max,
                                                     value=min(max(len(cur_sets), n_min), n_max),
                                                     key=f"ko_ns_{tid}_{rname}_{idx}")
                            new_sets = []
                            for si in range(n_sets):
                                cA,cB = st.columns(2)
                                with cA:
                                    s1 = st.number_input(f"Set {si+1} â€” games {m['a']}", 0, 20,
                                                         int(cur_sets[si]["s1"]) if si<len(cur_sets) and "s1" in cur_sets[si] else 0,
                                                         key=f"ko_s1_{tid}_{rname}_{idx}_{si}")
                                with cB:
                                    s2 = st.number_input(f"Set {si+1} â€” games {m['b']}", 0, 20,
                                                         int(cur_sets[si]["s2"]) if si<len(cur_sets) and "s2" in cur_sets[si] else 0,
                                                         key=f"ko_s2_{tid}_{rname}_{idx}_{si}")
                                new_sets.append({"s1":int(s1),"s2":int(s2)})
                            ok, msg = validate_sets(fmt, new_sets)
                            if not ok: st.error(msg)
                            gC,gD = st.columns(2)
                            with gC:
                                g1 = st.number_input(f"Puntos de oro {m['a']}", 0, 200, int(m.get("goldenA",0)), key=f"ko_g1_{tid}_{rname}_{idx}")
                            with gD:
                                g2 = st.number_input(f"Puntos de oro {m['b']}", 0, 200, int(m.get("goldenB",0)), key=f"ko_g2_{tid}_{rname}_{idx}")
                            if st.button("Guardar partido KO", key=f"ko_sv_{tid}_{rname}_{idx}"):
                                # En KO, si empatan en sets, sorteo automÃ¡tico
                                stats = compute_sets_stats(new_sets)
                                if stats["sets1"] == stats["sets2"]:
                                    rr = rng(9000+idx, cfg["seed"]); winner = rr.choice([m['a'], m['b']])
                                    st.warning(f"Empate en sets â†’ sorteo automÃ¡tico: **{winner}**")
                                m["sets"] = new_sets; m["goldenA"] = int(g1); m["goldenB"] = int(g2)
                                save_tournament(tid, state); st.success("KO guardado.")
                            stats = compute_sets_stats(new_sets)
                            if stats["sets1"] == stats["sets2"]:
                                rr = rng(9000+idx, cfg["seed"]); winner = rr.choice([m['a'], m['b']])
                                st.caption(f"Empate en sets â†’ sorteo: **{winner}**")
                            else:
                                winner = m['a'] if stats["sets1"]>stats["sets2"] else m['b']
                                st.caption(f"Ganador: **{winner}**")
                            advancing.append(winner)
                    if rname != "FN" and advancing:
                        pairs = next_round(advancing)
                        next_r = "SF" if rname=="QF" else "FN"
                        existing = [m for m in state["ko"]["matches"] if m["round"]==next_r]
                        labels = ["SF1","SF2"] if next_r=="SF" else ["FINAL"]
                        if not existing:
                            for j,(a,b) in enumerate(pairs, start=1):
                                state["ko"]["matches"].append({"round": next_r, "label": labels[min(j-1,len(labels)-1)], "a": a, "b": (b or "BYE"), "sets": [], "goldenA": 0, "goldenB": 0})
                            save_tournament(tid, state)

    # --- PERSISTENCIA ---
    with tab_persist:
        st.subheader("Persistencia (autosave + snapshots)")
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.session_state.autosave = st.checkbox("Autosave", value=st.session_state.autosave)
        with c2:
            if st.button("ðŸ’¾ Guardar ahora"): save_tournament(tid, state); st.success("Guardado")
        with c3:
            st.download_button("â¬‡ï¸ Descargar estado (JSON)",
                               data=json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8"),
                               file_name=f"{tid}.json", mime="application/json")
        with c4:
            up = st.file_uploader("â¬†ï¸ Cargar estado", type=["json"], key=f"up_{tid}")
            if up is not None:
                try:
                    new_state = json.load(up); save_tournament(tid, new_state); st.success("Cargado y guardado.")
                except Exception as e:
                    st.error(f"Error al cargar: {e}")

    # Autosave
    current_hash = compute_state_hash(state)
    if st.session_state.autosave and current_hash != st.session_state.last_hash:
        save_tournament(tid, state)
        st.toast("ðŸ’¾ Autosaved", icon="ðŸ’¾")
        st.session_state.last_hash = current_hash
    elif not st.session_state.autosave:
        st.session_state.last_hash = current_hash

# ====== Viewer (consulta) ======
def viewer_dashboard(user: Dict[str, Any]):
    render_brand_top_left()
    st.header(f"Vista de consulta â€” {user['username']}")
    if not user.get("assigned_admin"): st.warning("No asignado a un admin."); return
    my = load_index_for_admin(user["assigned_admin"])
    if not my: st.info("El admin asignado no tiene torneos."); return
    names = [f"{t['date']} â€” {t['t_name']} ({t['gender']}) â€” {t['place']} â€” ID:{t['tournament_id']}" for t in my]
    selected = st.selectbox("Selecciona un torneo para ver", names, index=0)
    sel = my[names.index(selected)]
    viewer_tournament(sel["tournament_id"])

def viewer_tournament(tid: str, public: bool=False):
    render_brand_top_left()
    state = load_tournament(tid)
    if not state:
        st.error("No se encontrÃ³ el torneo."); return
    # Backfill seeds/meta nuevos
    state.setdefault("seeds", [])
    state.setdefault("meta", {})
    state["meta"].setdefault("public_base_url", "")
    st.subheader(f"{state['meta'].get('t_name')} â€” {state['meta'].get('place')} â€” {state['meta'].get('date')} â€” {state['meta'].get('gender')}")
    tab_over, tab_tables, tab_ko = st.tabs(["ðŸ‘€ General","ðŸ“Š Tablas","ðŸ Playoffs"])
    with tab_over:
        st.write("Parejas")
        seeds = state.get("seeds", []) or []
        show_pairs = [label_with_seed(p, seeds) for p in state.get("pairs", [])]
        st.table(pd.DataFrame({"Parejas": show_pairs}))
        if state.get("groups"):
            st.write("Zonas")
            seeds_set = set(seeds)
            for zi, group in enumerate(state["groups"], start=1):
                disp = [label_with_seed(p, seeds_set) for p in group]
                st.write(f"**Z{zi}**"); st.table(pd.DataFrame({"Parejas": disp}))
    with tab_tables:
        if not state.get("groups") or not state.get("results"):
            st.info("Sin fixture/resultados aÃºn.")
        else:
            cfg = state["config"]
            for zi, group in enumerate(state["groups"], start=1):
                zone_name = f"Z{zi}"
                table = standings_from_results(zone_name, group, state["results"], cfg)
                with st.expander(f"Tabla {zone_name}", expanded=True): st.dataframe(table, use_container_width=True)
    with tab_ko:
        ko = state.get("ko", {"matches": []})
        if not ko.get("matches"):
            st.info("AÃºn no hay partidos de playoffs.")
        else:
            df = pd.DataFrame([
                {"Ronda": m.get("round",""), "Clave": m.get("label",""), "A": m.get("a",""), "B": m.get("b","")}
                for m in ko["matches"]
            ])
            st.dataframe(df, use_container_width=True)
    if public: st.info("Modo pÃºblico (solo lectura)")

# ====== ExportaciÃ³n de PDF ======
def export_fixture_pdf(state: Dict[str,Any]) -> Optional[BytesIO]:
    if not REPORTLAB_OK: return None
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    elems = []
    title = f"Fixture â€” {state['meta'].get('t_name')} â€” {state['meta'].get('place')} â€” {state['meta'].get('date')}"
    elems.append(Paragraph(title, styles['Title']))
    elems.append(Spacer(1, 12))
    if not state.get("groups"):
        elems.append(Paragraph("Sin zonas generadas.", styles['Normal']))
    else:
        for zi, group in enumerate(state["groups"], start=1):
            elems.append(Paragraph(f"Zona Z{zi}", styles['Heading2']))
            data = [["Parejas"]] + [[p] for p in group]
            t = Table(data, colWidths=[16*cm])
            t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
            elems.append(t); elems.append(Spacer(1,8))
        rows = [["Zona","Pareja 1","Pareja 2"]]
        for m in state["results"]:
            rows.append([m["zone"], m["pair1"], m["pair2"]])
        elems.append(Paragraph("Partidos (fase de grupos)", styles['Heading2']))
        t2 = Table(rows, colWidths=[2*cm, 7*cm, 7*cm])
        t2.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
        elems.append(t2)
    doc.build(elems)
    buf.seek(0)
    return buf

def export_playoffs_pdf(state: Dict[str,Any]) -> Optional[BytesIO]:
    if not REPORTLAB_OK: return None
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.0*cm, rightMargin=1.0*cm, topMargin=1.0*cm, bottomMargin=1.0*cm)
    styles = getSampleStyleSheet()
    elems = []
    elems.append(Paragraph(f"Playoffs â€” {state['meta'].get('t_name')}", styles['Title']))
    elems.append(Spacer(1, 10))
    ko = state.get("ko", {"matches": []})
    rounds = ["QF","SF","FN"]
    for r in rounds:
        ms = [m for m in ko.get("matches", []) if m.get("round")==r]
        if not ms: continue
        elems.append(Paragraph(r, styles['Heading2']))
        rows = [["Clave","A","B","Sets A-B","Ptos Oro A-B"]]
        for m in ms:
            stats = compute_sets_stats(m.get("sets", []))
            sets_str = f"{stats['sets1']}-{stats['sets2']}"
            gp_str = f"{m.get('goldenA',0)}-{m.get('goldenB',0)}"
            rows.append([m.get("label",""), m.get("a",""), m.get("b",""), sets_str, gp_str])
        t = Table(rows, colWidths=[3*cm, 5*cm, 5*cm, 3*cm, 3*cm])
        t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
        elems.append(t); elems.append(Spacer(1,8))
    doc.build(elems)
    buf.seek(0)
    return buf

# ====== Entrada de la App ======
def init_app():
    # Query params
    if hasattr(st, "query_params"): params = st.query_params
    else: params = st.experimental_get_query_params()

    init_session()

    mode = params.get("mode", [""])
    mode = mode[0] if isinstance(mode, list) else mode
    _tid = params.get("tid", [""])
    _tid = _tid[0] if isinstance(_tid, list) else _tid

    # Logo global fijo
    render_brand_top_left()

    if mode=="public" and _tid:
        viewer_tournament(_tid, public=True)
        st.caption("iAPPs PÃ¡del â€” v3.4.0")
        return

    # Login o paneles
    if not st.session_state.get("auth_user"):
        login_form()
        st.caption("iAPPs PÃ¡del â€” v3.4.0")
        return

    user = st.session_state["auth_user"]

    # barra superior (sin caption de versiÃ³n aquÃ­)
    top = st.columns([4,3,3,1])
    with top[0]: st.markdown(f"**Usuario:** {user['username']} Â· Rol: `{user['role']}`")
    with top[1]: st.link_button("Abrir Super Admin", url="?mode=super")
    with top[2]: st.button("Cerrar sesiÃ³n", on_click=lambda: st.session_state.update({"auth_user":None,"current_tid":None}))
    st.divider()

    if user["role"]=="SUPER_ADMIN" and (mode=="super"):
        super_admin_panel()
    elif user["role"]=="SUPER_ADMIN":
        super_admin_panel()
    elif user["role"]=="TOURNAMENT_ADMIN":
        admin_dashboard(user)
    elif user["role"]=="VIEWER":
        viewer_dashboard(user)
    else:
        st.error("Rol desconocido.")

    # Footer de versiÃ³n (muy pequeÃ±o y abajo)
    st.caption("iAPPs PÃ¡del â€” v3.4.0")

init_app()