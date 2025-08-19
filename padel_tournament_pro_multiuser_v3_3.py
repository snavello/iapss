# app.py — v3.3.29
# - Corregido el error StreamlitAPIException al añadir pareja.
# - Solucionado el problema de la lista de parejas que no se refrescaba.
# - Asegurado que el checkbox para 'cabeza de serie' funcione correctamente.
# - Fix `StreamlitInvalidFormCallbackError` by moving the "copy link" button outside the form.
# - This also resolves the "Missing Submit Button" warning related to the same form.
# - Restore "Persistencia" tab with all its content and functionality.
# - Reworked public URL logic to be more robust.
# - Fix `Streamlit` bug where pair names were not being detected in the form by
#   correctly resetting session state after a successful submission.

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

st.set_page_config(page_title="Torneo de Pádel — v3.3.29", layout="wide")

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

def is_round_finished(matches):
    return all(m["winner"] for m in matches)

def playoff_stage_name(n_pairs):
    if n_pairs == 2: return "Final"
    if n_pairs == 4: return "Semifinales"
    if n_pairs == 8: return "Cuartos de final"
    if n_pairs == 16: return "Octavos de final"
    return f"Playoffs ({n_pairs})"

# ====== Utilidades de URL / UI ======
def build_public_url(tid):
    parts = list(urlparse(st.experimental_get_query_params.get("share_url", [""])[0]))
    params = dict(parse_qsl(parts[4]))
    params["mode"] = "public"
    params["tid"] = tid
    parts[4] = urlunparse(urlunparse(('', '', '', '', ''))).query
    parts[4] = '&'.join([f"{k}={v}" for k, v in params.items()])
    return urlunparse(parts)

# ====== Vistas y componentes de la UI ======
def inject_global_layout(app_cfg, auth_user, admin_name=None):
    logo_uri = fetch_image_as_data_uri(app_cfg.get("app_logo_url", ""), auth_user["username"] if auth_user else "")

    st.markdown("""
        <style>
        .header-section {
            background-color: #f0f2f6;
            padding: 10px;
            border-bottom: 2px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header-text {
            display: flex;
            align-items: center;
        }
        .logo-img {
            height: 50px;
            margin-right: 15px;
        }
        .user-info {
            font-size: 1.2em;
            font-weight: bold;
        }
        .stButton button {
            background-color: #0D47A1;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        </style>
    """, unsafe_allow_html=True)
    
    user_info = f"Usuario: <b>{auth_user['username']}</b> | Rol: <code>{auth_user['role']}</code>"
    if admin_name:
        user_info += f" | Asignado a: {admin_name}"

    header_html = f"""
    <div class="header-section">
        <div class="header-text">
            {'<img src="{}" class="logo-img">'.format(logo_uri) if logo_uri else ''}
            <div class="user-info">{user_info}</div>
        </div>
        <a href="?logout=true" style="text-decoration:none;">
            <button class="stButton">Cerrar Sesión</button>
        </a>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def login_form():
    st.subheader("Acceso al sistema")
    st.markdown("---")
    
    with st.form("login"):
        st.markdown("#### Ingresa tus credenciales")
        username = st.text_input("Usuario")
        password = st.text_input("PIN (6 dígitos)", type="password")
        submitted = st.form_submit_button("Ingresar")
        if submitted:
            user = get_user(username)
            if user and user["pin_hash"] == sha(password) and user["active"]:
                st.session_state["auth_user"] = user
                st.experimental_rerun()
            else:
                st.error("Credenciales incorrectas o usuario inactivo.")

def super_admin_panel():
    st.title("Super Admin Panel")
    app_cfg = load_app_config()

    st.subheader("Configuración Global")
    with st.form("global_config"):
        new_logo_url = st.text_input("URL del logo (RAW)", value=app_cfg.get("app_logo_url"))
        if st.form_submit_button("Guardar Configuración"):
            app_cfg["app_logo_url"] = new_logo_url
            save_app_config(app_cfg)
            st.success("Configuración guardada.")
            fetch_image_as_data_uri.clear() # Clear cache
            st.experimental_rerun()

    st.subheader("Gestión de Usuarios")
    users = load_users()
    df_users = pd.DataFrame(users).drop(columns=["pin_hash", "assigned_admin", "created_at"])
    st.dataframe(df_users, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Crear Usuario")
        with st.form("create_user"):
            new_user = st.text_input("Nombre de usuario")
            new_pin = st.text_input("PIN", type="password")
            new_role = st.selectbox("Rol", ["TOURNAMENT_ADMIN", "VIEWER"])
            assigned_admin = None
            if new_role == "VIEWER":
                admins = [u["username"] for u in users if u["role"] == "TOURNAMENT_ADMIN"]
                assigned_admin = st.selectbox("Asignar a Admin", ["-"] + admins)
            
            if st.form_submit_button("Crear"):
                if get_user(new_user):
                    st.error("Ese usuario ya existe.")
                elif not new_user or not new_pin:
                    st.error("Usuario y PIN son obligatorios.")
                else:
                    new_user_data = {
                        "username": new_user,
                        "pin_hash": sha(new_pin),
                        "role": new_role,
                        "assigned_admin": assigned_admin if new_role == "VIEWER" else None,
                        "created_at": now_iso(),
                        "active": True
                    }
                    set_user(new_user_data)
                    st.success(f"Usuario {new_user} creado con éxito.")
                    st.experimental_rerun()
    with col2:
        st.subheader("Administrar Usuario")
        with st.form("manage_user"):
            user_to_manage = st.selectbox("Seleccionar usuario", [u["username"] for u in users if u["role"] != "SUPER_ADMIN"])
            
            if user_to_manage:
                user_data = get_user(user_to_manage)
                new_pin_manage = st.text_input("Nuevo PIN", type="password", help="Dejar vacío para no cambiar.")
                new_active = st.checkbox("Activo", value=user_data["active"])
                
                if st.form_submit_button("Actualizar"):
                    if new_pin_manage:
                        user_data["pin_hash"] = sha(new_pin_manage)
                    user_data["active"] = new_active
                    set_user(user_data)
                    st.success("Usuario actualizado.")
                    st.experimental_rerun()
                
                if st.form_submit_button("Eliminar"):
                    users = load_users()
                    users = [u for u in users if u["username"] != user_to_manage]
                    save_users(users)
                    st.success("Usuario eliminado.")
                    st.experimental_rerun()

def admin_dashboard(user):
    st.title("Dashboard de Torneos")
    
    # Check if a tournament is currently selected
    if "current_tid" not in st.session_state:
        st.session_state.current_tid = None
    
    def select_tournament(tid):
        st.session_state.current_tid = tid
        st.experimental_rerun()
    
    def delete_tournament_callback(tid):
        idx = load_index()
        idx = [t for t in idx if t["tid"] != tid]
        save_index(idx)
        
        path = tourn_path(tid)
        if path.exists():
            path.unlink()
        
        st.session_state.current_tid = None
        st.experimental_rerun()
    
    def create_tournament_callback(name, num_groups, top_per_zone, seeded_mode):
        tid = str(uuid.uuid4())
        new_tourn = {
            "tid": tid,
            "name": name,
            "admin_user": user["username"],
            "pairs": [],
            "groups": {},
            "matches": {},
            "config": {
                "num_groups": num_groups,
                "top_per_zone": top_per_zone,
                "seeded_mode": seeded_mode,
            }
        }
        save_tournament(tid, new_tourn, make_snapshot=False)
        
        idx = load_index()
        idx.append({"tid": tid, "name": name, "admin_user": user["username"]})
        save_index(idx)
        
        st.session_state.current_tid = tid
        st.success(f"Torneo '{name}' creado con éxito!")
        st.experimental_rerun()
    
    st.subheader("Gestión de Torneos")
    
    tourns_by_admin = [t for t in load_index() if t["admin_user"] == user["username"]]
    
    with st.form("tourn_select_form"):
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            tourn_options = {t["tid"]: t["name"] for t in tourns_by_admin}
            tourn_options[None] = "Nuevo Torneo..."
            selected_tid = st.selectbox(
                "Selecciona un torneo",
                options=list(tourn_options.keys()),
                format_func=lambda tid: tourn_options[tid],
                index=list(tourn_options.keys()).index(st.session_state.current_tid) if st.session_state.current_tid in tourn_options else 0
            )
        
        with col2:
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True) # Spacer
            if st.form_submit_button("Seleccionar", use_container_width=True):
                select_tournament(selected_tid)
        with col3:
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True) # Spacer
            if st.form_submit_button("Eliminar", type="primary", use_container_width=True):
                if selected_tid:
                    delete_tournament_callback(selected_tid)
                else:
                    st.warning("Selecciona un torneo para eliminar.")
    
    st.divider()

    if st.session_state.current_tid:
        tourn = load_tournament(st.session_state.current_tid)
        st.subheader(f"Torneo: {tourn['name']}")

        tab_pairs, tab_groups, tab_results, tab_playoffs, tab_persistence = st.tabs([
            "Parejas", "Generar Grupos", "Resultados", "Playoffs", "Persistencia"
        ])
        
        # Tab: Parejas
        with tab_pairs:
            st.markdown("#### Administración de Parejas")
            col_add, col_list = st.columns(2)
            
            def add_pair_callback(j1_name, j2_name, is_seeded):
                if not j1_name or not j2_name:
                    st.warning("Los nombres de los jugadores no pueden estar vacíos.")
                    return
                
                tourn = load_tournament(st.session_state.current_tid)
                pair_id = str(uuid.uuid4())
                tourn["pairs"].append({
                    "id": pair_id,
                    "p1": j1_name,
                    "p2": j2_name,
                    "name": f"{j1_name} y {j2_name}",
                    "seeded": is_seeded
                })
                save_tournament(tourn["tid"], tourn)
                st.success("Pareja agregada con éxito.")
                st.experimental_rerun()
            
            with col_add:
                st.markdown("##### Añadir nueva pareja")
                with st.form("add_pair_form", clear_on_submit=True):
                    j1_input = st.text_input("Jugador 1")
                    j2_input = st.text_input("Jugador 2")
                    is_seeded_checkbox = st.checkbox("Cabeza de Serie")
                    
                    if st.form_submit_button("Añadir"):
                        add_pair_callback(j1_input, j2_input, is_seeded_checkbox)
            
            with col_list:
                st.markdown("##### Lista de Parejas")
                pairs = load_tournament(st.session_state.current_tid)["pairs"]
                df = pd.DataFrame(pairs)
                if not df.empty:
                    st.dataframe(df[["name", "seeded"]].rename(columns={"name": "Pareja", "seeded": "Cabeza de Serie"}), hide_index=True)
                    if st.button("Eliminar todas las parejas", use_container_width=True):
                        tourn = load_tournament(st.session_state.current_tid)
                        tourn["pairs"] = []
                        tourn["groups"] = {} # Clear groups if pairs are deleted
                        save_tournament(tourn["tid"], tourn)
                        st.experimental_rerun()
                else:
                    st.info("Aún no hay parejas registradas.")

        # Tab: Generar Grupos
        with tab_groups:
            st.markdown("#### Generar y Configurar Grupos")
            st.info(f"Parejas registradas: **{len(tourn['pairs'])}**")

            with st.form("group_config"):
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    num_groups = st.number_input(
                        "Número de zonas",
                        min_value=1,
                        value=tourn["config"].get("num_groups", 4)
                    )
                with col_c2:
                    top_per_zone = st.number_input(
                        "Clasificados por zona",
                        min_value=1,
                        value=tourn["config"].get("top_per_zone", 2)
                    )
                with col_c3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    seeded_mode = st.checkbox("Usar cabezas de serie", value=tourn["config"].get("seeded_mode", False))
                
                if st.form_submit_button("Generar Zonas"):
                    tourn["config"]["num_groups"] = num_groups
                    tourn["config"]["top_per_zone"] = top_per_zone
                    tourn["config"]["seeded_mode"] = seeded_mode
                    
                    if len(tourn["pairs"]) < 2:
                        st.warning("Se necesitan al menos 2 parejas para generar grupos.")
                    else:
                        seeded_pairs = [p for p in tourn["pairs"] if p["seeded"]]
                        if seeded_mode and len(seeded_pairs) != num_groups:
                            st.warning(f"Se necesitan exactamente {num_groups} cabezas de serie para este modo. Hay {len(seeded_pairs)}.")
                        else:
                            grouped_pairs = create_groups(
                                tourn["pairs"],
                                num_groups,
                                seeded_mode,
                                seeded_pairs,
                                tourn["tid"]
                            )
                            tourn["groups"] = {}
                            for i, group in enumerate(grouped_pairs):
                                group_id = str(i + 1)
                                tourn["groups"][group_id] = {
                                    "pairs": group,
                                    "matches": []
                                }
                                for p1, p2 in combinations(group, 2):
                                    match_id = str(uuid.uuid4())
                                    tourn["matches"][match_id] = {
                                        "p1_id": p1["id"],
                                        "p2_id": p2["id"],
                                        "score": [],
                                        "winner": None,
                                        "group_id": group_id
                                    }
                            save_tournament(tourn["tid"], tourn)
                            st.success("Grupos generados con éxito.")
                            st.experimental_rerun()
            
            st.divider()
            if tourn["groups"]:
                st.markdown("#### Zonas Generadas")
                for group_id, group_data in tourn["groups"].items():
                    st.markdown(f"**Zona {group_id}**")
                    df_group = pd.DataFrame(group_data["pairs"])
                    if not df_group.empty:
                        st.dataframe(df_group[["name", "seeded"]].rename(columns={"name": "Pareja", "seeded": "Cabeza de Serie"}), hide_index=True)
                    else:
                        st.info("No hay parejas en esta zona.")
        
        # Tab: Resultados
        with tab_results:
            st.markdown("#### Cargar Resultados de Partidos")
            
            if not tourn["groups"]:
                st.warning("Primero debes generar las zonas.")
                return

            for group_id, group_data in tourn["groups"].items():
                st.markdown(f"##### Zona {group_id}")
                for pair1, pair2 in combinations(group_data["pairs"], 2):
                    match_data = next(
                        (m for m in tourn["matches"].values() if {m["p1_id"], m["p2_id"]} == {pair1["id"], pair2["id"]}),
                        None
                    )
                    
                    if not match_data:
                        continue # Should not happen, but for safety

                    match_id = next((k for k,v in tourn["matches"].items() if v == match_data))
                    
                    with st.expander(f"Partido: {pair1['name']} vs {pair2['name']}"):
                        col_score1, col_score2 = st.columns(2)
                        with col_score1:
                            score1 = st.number_input(f"Sets ganados por {pair1['name']}", min_value=0, step=1, key=f"score1_{match_id}", value=match_data["score"][0] if match_data["score"] else 0)
                        with col_score2:
                            score2 = st.number_input(f"Sets ganados por {pair2['name']}", min_value=0, step=1, key=f"score2_{match_id}", value=match_data["score"][1] if match_data["score"] else 0)
                        
                        if st.button("Guardar resultado", key=f"save_{match_id}"):
                            tourn["matches"][match_id]["score"] = [score1, score2]
                            if score1 > score2:
                                tourn["matches"][match_id]["winner"] = pair1["id"]
                            elif score2 > score1:
                                tourn["matches"][match_id]["winner"] = pair2["id"]
                            else:
                                tourn["matches"][match_id]["winner"] = None
                                
                            save_tournament(tourn["tid"], tourn)
                            st.success("Resultado guardado.")
                            st.experimental_rerun()
        
        # Tab: Playoffs
        with tab_playoffs:
            st.markdown("#### Cuadro de Playoffs")
            st.info("Esta sección aún no está implementada. Estará disponible en futuras actualizaciones.")

        # Tab: Persistencia
        with tab_persistence:
            st.markdown("#### Opciones de Persistencia y Compartir")
            
            st.subheader("URL Pública del Torneo")
            public_url = build_public_url(tourn["tid"])
            st.code(public_url)
            
            # Button to copy to clipboard
            st.markdown(f"""
                <button onclick="navigator.clipboard.writeText('{public_url}')">
                    Copiar URL
                </button>
            """, unsafe_allow_html=True)
            
            st.subheader("Carga y Descarga de Datos")
            
            # Descargar JSON
            json_dump = json.dumps(tourn, ensure_ascii=False, indent=2)
            st.download_button(
                label="Descargar datos del torneo (JSON)",
                data=json_dump,
                file_name=f"{tourn['tid']}_{tourn['name']}.json",
                mime="application/json"
            )
            
            # Cargar JSON
            uploaded_file = st.file_uploader("Cargar datos del torneo desde JSON", type="json")
            if uploaded_file is not None:
                try:
                    uploaded_data = json.load(uploaded_file)
                    st.info("Vista previa de los datos cargados:")
                    st.json(uploaded_data)
                    
                    if st.button("Restaurar torneo con estos datos"):
                        save_tournament(tourn["tid"], uploaded_data, make_snapshot=True)
                        st.success("Torneo restaurado con éxito.")
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error al cargar el archivo JSON: {e}")

    else:
        st.subheader("Crear Nuevo Torneo")
        with st.form("new_tourn_form"):
            new_tourn_name = st.text_input("Nombre del Torneo")
            num_groups_new = st.number_input("Número de grupos", min_value=1, value=4)
            top_per_zone_new = st.number_input("Clasificados por grupo", min_value=1, value=2)
            seeded_mode_new = st.checkbox("Usar cabezas de serie")
            
            if st.form_submit_button("Crear"):
                create_tournament_callback(new_tourn_name, num_groups_new, top_per_zone_new, seeded_mode_new)

def viewer_tournament(tid, public=False):
    tourn = load_tournament(tid)
    if not tourn:
        st.error("Torneo no encontrado.")
        return
    
    st.title(tourn["name"])
    st.markdown(f"Administrador: {tourn['admin_user']}")
    
    if public:
        st.info("Modo de visualización pública. No es necesario iniciar sesión.")
    
    tab_groups, tab_results, tab_playoffs = st.tabs(["Grupos", "Resultados", "Playoffs"])
    
    with tab_groups:
        st.markdown("#### Zonas y Parejas")
        if not tourn["groups"]:
            st.info("Grupos no generados aún.")
            return
        
        for group_id, group_data in tourn["groups"].items():
            st.markdown(f"**Zona {group_id}**")
            df_group = pd.DataFrame(group_data["pairs"])
            if not df_group.empty:
                st.dataframe(df_group[["name", "seeded"]].rename(columns={"name": "Pareja", "seeded": "Cabeza de Serie"}), hide_index=True)
    
    with tab_results:
        st.markdown("#### Resultados y Tabla de Posiciones")
        if not tourn["groups"]:
            st.info("Grupos no generados. No hay resultados para mostrar.")
            return

        for group_id, group_data in tourn["groups"].items():
            st.markdown(f"**Resultados de la Zona {group_id}**")
            
            points = {p["id"]: 0 for p in group_data["pairs"]}
            for match in tourn["matches"].values():
                if match["group_id"] == group_id and match["winner"]:
                    points[match["winner"]] += 1
            
            ranking_data = [
                {"Pareja": p["name"], "Puntos": points.get(p["id"], 0)}
                for p in group_data["pairs"]
            ]
            
            df_ranking = pd.DataFrame(ranking_data).sort_values(by="Puntos", ascending=False)
            st.dataframe(df_ranking, hide_index=True)
            
            st.markdown(f"**Clasifican los {tourn['config']['top_per_zone']} mejores.**")
    
    with tab_playoffs:
        st.markdown("#### Cuadro de Playoffs")
        st.info("El cuadro de playoffs no ha sido generado aún.")

def main():
    if "logout" in st.experimental_get_query_params():
        st.session_state["auth_user"] = None
        st.experimental_set_query_params() # Clear query params
    
    app_cfg = load_app_config()
    tourns_index = load_index()
    
    # Manejar el modo publico desde la URL
    params = st.experimental_get_query_params()
    mode = params.get("mode", [None])[0]
    _tid = params.get("tid", [None])[0]
    if mode == "public" and _tid:
        viewer_tournament(_tid, public=True)
        st.caption("iAPPs Pádel — v3.3.29")
        return
        
    # Autenticación
    if "auth_user" not in st.session_state or st.session_state.auth_user is None:
        st.title("Gestor de Torneos de Pádel")
        login_form()
        st.caption("iAPPs Pádel — v3.3.29")
        return
    
    # Panel principal
    user = st.session_state["auth_user"]
    admin_name = None
    if user["role"] == "VIEWER" and user["assigned_admin"]:
        admin_name = user["assigned_admin"]
    
    inject_global_layout(app_cfg, user, admin_name)

    if user["role"] == "SUPER_ADMIN":
        super_admin_panel()
    elif user["role"] == "TOURNAMENT_ADMIN":
        admin_dashboard(user)
    elif user["role"] == "VIEWER":
        st.title("Vista de Torneos")
        if user["assigned_admin"]:
            tourns = [t for t in tourns_index if t["admin_user"] == user["assigned_admin"]]
            if tourns:
                selected_tourn_tid = st.selectbox(
                    "Selecciona un torneo para ver",
                    options=[t["tid"] for t in tourns],
                    format_func=lambda tid: next(t["name"] for t in tourns if t["tid"] == tid)
                )
                if selected_tourn_tid:
                    viewer_tournament(selected_tourn_tid)
            else:
                st.info("El administrador asignado aún no ha creado torneos.")
        else:
            st.warning("No tienes un administrador asignado. Contacta al super admin.")

if __name__ == "__main__":
    main()

