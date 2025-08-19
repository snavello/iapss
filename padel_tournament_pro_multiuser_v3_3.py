# app.py — v3.3.27
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

st.set_page_config(page_title="Torneo de Pádel — v3.3.27", layout="wide")

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

def get_tournament_path(tournament_id: str) -> Path:
    return TOURN_DIR / f"tourn_{tournament_id}.json"

def load_tournament(tournament_id: str) -> Optional[Dict[str, Any]]:
    path = get_tournament_path(tournament_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None

def save_tournament(tourn: Dict[str, Any]):
    path = get_tournament_path(tourn["tournament_id"])
    path.write_text(json.dumps(tourn, ensure_ascii=False, indent=2), encoding="utf-8")

def delete_tournament(tournament_id: str):
    idx = load_index()
    idx = [t for t in idx if t["tournament_id"] != tournament_id]
    save_index(idx)

    tourn_path = get_tournament_path(tournament_id)
    if tourn_path.exists():
        tourn_path.unlink()

    snap_path = SNAP_ROOT / f"snapshots_{tournament_id}"
    if snap_path.exists():
        for f in snap_path.glob("*.json"):
            f.unlink()
        snap_path.rmdir()

# ====== Gestión de Torneos por Administrador ======
def load_index_for_admin(admin_username: str) -> List[Dict[str, Any]]:
    all_tourns = load_index()
    return [t for t in all_tourns if t["admin"] == admin_username]

# ====== Funciones de Torneo ======
def create_new_tournament(user: Dict[str, Any], tournament_name: str, start_date: date) -> Dict[str, Any]:
    new_id = str(uuid.uuid4())
    tourn_data = {
        "tournament_id": new_id,
        "name": tournament_name,
        "admin": user["username"],
        "created_at": now_iso(),
        "status": "SETUP",
        "start_date": start_date.isoformat(),
        "config": {
            "seeded_mode": False,
            "seeded_pairs": [],
            "groups_stage": True
        },
        "pairs": [],
        "groups": {},
        "qualifiers": [],
        "playoffs": {},
    }
    idx = load_index()
    idx.append({"tournament_id": new_id, "name": tournament_name, "admin": user["username"]})
    save_index(idx)
    save_tournament(tourn_data)
    st.session_state.current_tid = new_id
    st.success(f"Torneo '{tournament_name}' creado exitosamente!")
    return tourn_data

def add_pair(tourn: Dict[str, Any], player1: str, player2: str):
    tourn["pairs"].append({"player1": player1.strip().title(), "player2": player2.strip().title()})
    save_tournament(tourn)

def delete_pair(tourn: Dict[str, Any], pair_to_delete: Dict[str, str]):
    tourn["pairs"] = [p for p in tourn["pairs"] if p != pair_to_delete]
    save_tournament(tourn)

def generate_groups(tourn: Dict[str, Any], num_groups: int, num_teams_per_group: int, seeded_mode: bool, seeded_pairs: List[Dict[str, str]]):
    all_pairs = list(tourn["pairs"])
    if len(all_pairs) < num_groups * num_teams_per_group:
        st.error(f"Se necesitan al menos {num_groups * num_teams_per_group} parejas para generar {num_groups} grupos con {num_teams_per_group} parejas cada uno.")
        return

    if seeded_mode:
        seeded_pairs_names = [f"{p['player1']} / {p['player2']}" for p in seeded_pairs]
        non_seeded_pairs = [p for p in all_pairs if f"{p['player1']} / {p['player2']}" not in seeded_pairs_names]
        if len(seeded_pairs) < num_groups:
            st.error(f"Se necesitan al menos {num_groups} parejas cabeza de serie para el número de grupos seleccionado.")
            return
        random.shuffle(non_seeded_pairs)
        groups_list = [[] for _ in range(num_groups)]
        for i in range(num_groups):
            groups_list[i].append(seeded_pairs[i])
        remaining_pairs = non_seeded_pairs
        while any(len(g) < num_teams_per_group for g in groups_list):
            for i in range(len(groups_list)):
                if len(groups_list[i]) < num_teams_per_group and remaining_pairs:
                    groups_list[i].append(remaining_pairs.pop(0))
    else:
        random.shuffle(all_pairs)
        groups_list = [all_pairs[i:i + num_teams_per_group] for i in range(0, len(all_pairs), num_teams_per_group)]

    groups = {}
    for i, group_pairs in enumerate(groups_list):
        group_name = chr(ord('A') + i)
        groups[group_name] = {
            "pairs": group_pairs,
            "matches": {},
            "table": []
        }
        group_pairs_names = [f"{p['player1']} / {p['player2']}" for p in group_pairs]
        match_combinations = list(combinations(group_pairs_names, 2))
        random.shuffle(match_combinations)
        for i, (p1, p2) in enumerate(match_combinations):
            groups[group_name]["matches"][f"match_{i+1}"] = {
                "pair1": p1,
                "pair2": p2,
                "score_pair1": None,
                "score_pair2": None,
                "completed": False
            }
    tourn["groups"] = groups
    tourn["status"] = "GROUP_STAGE"
    tourn["config"]["seeded_mode"] = seeded_mode
    tourn["config"]["seeded_pairs"] = seeded_pairs
    save_tournament(tourn)
    st.success("Grupos y partidos generados exitosamente!")

def generate_playoffs(tourn: Dict[str, Any], num_qualifiers: int):
    qualifiers = tourn.get("qualifiers", [])
    if len(qualifiers) != num_qualifiers:
        st.warning(f"Error: Se esperaban {num_qualifiers} clasificados, pero se encontraron {len(qualifiers)}. Por favor, clasifique a los jugadores en la etapa de grupos primero.")
        tourn["playoffs"] = {}
        save_tournament(tourn)
        return

    tourn["playoffs"] = {}

    if num_qualifiers == 2:
        tourn["playoffs"]["FINAL"] = {
            "match_1": {
                "pair1": qualifiers[0],
                "pair2": qualifiers[1],
                "score_pair1": None,
                "score_pair2": None,
                "completed": False
            }
        }
    elif num_qualifiers == 4:
        tourn["playoffs"]["SF"] = {
            "match_1": {
                "pair1": qualifiers[0], "pair2": qualifiers[3],
                "score_pair1": None, "score_pair2": None, "completed": False
            },
            "match_2": {
                "pair1": qualifiers[1], "pair2": qualifiers[2],
                "score_pair1": None, "score_pair2": None, "completed": False
            }
        }
        tourn["playoffs"]["FINAL"] = {
            "match_1": {
                "pair1": "Ganador SF 1", "pair2": "Ganador SF 2",
                "score_pair1": None, "score_pair2": None, "completed": False
            }
        }
    elif num_qualifiers == 8:
        tourn["playoffs"]["QF"] = {
            "match_1": {
                "pair1": qualifiers[0], "pair2": qualifiers[7],
                "score_pair1": None, "score_pair2": None, "completed": False
            },
            "match_2": {
                "pair1": qualifiers[1], "pair2": qualifiers[6],
                "score_pair1": None, "score_pair2": None, "completed": False
            },
            "match_3": {
                "pair1": qualifiers[2], "pair2": qualifiers[5],
                "score_pair1": None, "score_pair2": None, "completed": False
            },
            "match_4": {
                "pair1": qualifiers[3], "pair2": qualifiers[4],
                "score_pair1": None, "score_pair2": None, "completed": False
            }
        }
        tourn["playoffs"]["SF"] = {
            "match_1": {
                "pair1": "Ganador QF 1", "pair2": "Ganador QF 2",
                "score_pair1": None, "score_pair2": None, "completed": False
            },
            "match_2": {
                "pair1": "Ganador QF 3", "pair2": "Ganador QF 4",
                "score_pair1": None, "score_pair2": None, "completed": False
            }
        }
        tourn["playoffs"]["FINAL"] = {
            "match_1": {
                "pair1": "Ganador SF 1", "pair2": "Ganador SF 2",
                "score_pair1": None, "score_pair2": None, "completed": False
            }
        }

    tourn["status"] = "PLAYOFFS"
    tourn["champion"] = None
    save_tournament(tourn)
    st.success("Cuadro de playoffs generado exitosamente!")

def update_playoffs(tourn: Dict[str, Any]):
    playoffs = tourn.get("playoffs", {})
    if not playoffs:
        return

    # Helper para actualizar nombres de ganadores en la siguiente ronda
    def update_winner(stage, match_key, winner_name):
        stage_map = {"QF": "SF", "SF": "FINAL"}
        if stage in stage_map:
            next_stage_key = stage_map[stage]
            next_stage = playoffs.get(next_stage_key)
            if next_stage:
                if match_key == "match_1" and stage == "QF":
                    next_stage["match_1"]["pair1"] = winner_name
                elif match_key == "match_2" and stage == "QF":
                    next_stage["match_1"]["pair2"] = winner_name
                elif match_key == "match_3" and stage == "QF":
                    next_stage["match_2"]["pair1"] = winner_name
                elif match_key == "match_4" and stage == "QF":
                    next_stage["match_2"]["pair2"] = winner_name
                elif match_key == "match_1" and stage == "SF":
                    next_stage["match_1"]["pair1"] = winner_name
                elif match_key == "match_2" and stage == "SF":
                    next_stage["match_1"]["pair2"] = winner_name
    
    stages = ["QF", "SF", "FINAL"]
    for stage in stages:
        if stage in playoffs:
            for match_key, match_data in playoffs[stage].items():
                if match_data["completed"] and match_data.get("score_pair1") is not None and match_data.get("score_pair2") is not None:
                    winner = match_data["pair1"] if match_data["score_pair1"] > match_data["score_pair2"] else match_data["pair2"]
                    match_data["winner"] = winner
                    if stage == "FINAL":
                        tourn["champion"] = winner
                    else:
                        update_winner(stage, match_key, winner)
    
    save_tournament(tourn)

# ====== Generación de PDF (opcional) ======
def generate_pdf(tourn: Dict[str, Any]):
    if not REPORTLAB_OK:
        st.error("ReportLab no está instalado. No se puede generar el PDF. Usa `pip install reportlab` para habilitar esta función.")
        return

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=cm, leftMargin=cm, topMargin=cm, bottomMargin=cm)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>Torneo de Pádel:</b> {tourn['name']}", styles['Title']))
    elements.append(Spacer(1, 0.5 * cm))

    # Grupos
    if tourn.get("groups"):
        elements.append(Paragraph("<b>Etapa de Grupos</b>", styles['Heading2']))
        for group_name, group_data in tourn["groups"].items():
            elements.append(Paragraph(f"<b>Grupo {group_name}</b>", styles['h3']))
            table_data = [["Pareja 1", "Pareja 2", "Resultado"]]
            for match_id, match in group_data["matches"].items():
                score = f"{match['score_pair1']} - {match['score_pair2']}" if match["completed"] else "Pendiente"
                table_data.append([match["pair1"], match["pair2"], score])

            table_style = TableStyle([
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.white)
            ])
            table = Table(table_data)
            table.setStyle(table_style)
            elements.append(table)
            elements.append(Spacer(1, 0.5 * cm))

    # Playoffs
    if tourn.get("playoffs"):
        elements.append(Paragraph("<b>Playoffs</b>", styles['Heading2']))
        for stage, stage_data in tourn["playoffs"].items():
            elements.append(Paragraph(f"<b>{stage}</b>", styles['h3']))
            table_data = [["Partido", "Pareja 1", "Pareja 2", "Resultado"]]
            for match_id, match in stage_data.items():
                score = f"{match['score_pair1']} - {match['score_pair2']}" if match["completed"] else "Pendiente"
                table_data.append([match_id, match["pair1"], match["pair2"], score])

            table_style = TableStyle([
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.white)
            ])
            table = Table(table_data)
            table.setStyle(table_style)
            elements.append(table)
            elements.append(Spacer(1, 0.5 * cm))

    doc.build(elements)
    buffer.seek(0)
    st.download_button(
        label="Descargar PDF",
        data=buffer,
        file_name=f"torneo_{tourn['name'].replace(' ', '_')}.pdf",
        mime="application/pdf",
    )

# ====== Vistas de la app ======

def inject_global_layout(title: str):
    """Injects a global layout with app title, logo, and a horizontal rule."""
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {LIGHT_GREY};
            }}
            .main .block-container {{
                padding-top: 1rem;
                padding-bottom: 1rem;
            }}
            .stAlert > div {{
                border-radius: 0.5rem;
            }}
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
                font-size:1rem;
                font-weight: bold;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: {DARK_BLUE};
            }}
            .stButton>button {{
                border: 2px solid {DARK_BLUE};
                background-color: {PRIMARY_BLUE};
                color: white;
                border-radius: 0.5rem;
                padding: 0.5rem 1rem;
                font-weight: bold;
            }}
            .stButton>button:hover {{
                background-color: {LIME_GREEN};
                color: {DARK_BLUE};
            }}
            .stTextInput>div>div>input {{
                border-radius: 0.5rem;
            }}
            .stDateInput>div>div>input {{
                border-radius: 0.5rem;
            }}
            .stMarkdown h3 {{
                color: {DARK_GREY};
            }}
            .stDataFrame {{
                border-radius: 0.5rem;
            }}
            .stTable {{
                border-radius: 0.5rem;
            }}
            .stSelectbox>div>div {{
                border-radius: 0.5rem;
            }}
            .stForm {{
                background-color: white;
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid #ddd;
            }}
            .stAlert {{
                border-left: 5px solid {PRIMARY_BLUE};
            }}
            .st-emotion-cache-1jmveo5 {{
                border-radius: 0.5rem;
            }}
            .st-emotion-cache-1jmveo5 > div > div {{
                border-radius: 0.5rem;
            }}
            .st-emotion-cache-1jmveo5 > div > div > div {{
                border-radius: 0.5rem;
            }}
            .st-emotion-cache-1jmveo5 > div > div > div > div {{
                border-radius: 0.5rem;
            }}

            .main-header {{
                display: flex;
                align-items: center;
                gap: 1rem;
                margin-bottom: 0.5rem;
            }}
            .header-text {{
                font-size: 2rem;
                font-weight: bold;
                color: {DARK_BLUE};
            }}
            .app-logo {{
                height: 50px;
                width: auto;
            }}
            .public-link-container {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            .public-link-container .st-emotion-cache-1jmveo5 {{
                flex-grow: 1;
            }}
            .public-link-container .stButton > button {{
                width: 50px;
                padding: 0.5rem;
                display: flex;
                justify-content: center;
            }}
            .clipboard-icon {{
                font-size: 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    app_config = load_app_config()
    st.markdown(f'<div class="main-header"><img src="{fetch_image_as_data_uri(app_config["app_logo_url"])}" class="app-logo" /><span class="header-text">{title}</span></div>', unsafe_allow_html=True)
    st.divider()

def login_form():
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Usuario")
        pin = st.text_input("PIN", type="password")
        submitted = st.form_submit_button("Ingresar")

        if submitted:
            user = get_user(username)
            if user and sha(pin) == user["pin_hash"]:
                if user["active"]:
                    st.session_state["auth_user"] = user
                    st.success("¡Ingreso exitoso!")
                    st.experimental_rerun()
                else:
                    st.error("Tu cuenta ha sido desactivada. Contacta al Super Admin.")
            else:
                st.error("Usuario o PIN incorrectos.")

def super_admin_panel():
    inject_global_layout("Panel de Super Admin")
    st.subheader("Gestión de Usuarios")

    users = load_users()
    users_df = pd.DataFrame(users)
    users_df = users_df.drop(columns=["pin_hash"])
    st.dataframe(users_df, use_container_width=True)

    with st.form("add_user_form"):
        st.subheader("Agregar/Editar Usuario")
        username = st.text_input("Usuario", key="add_username").strip()
        pin = st.text_input("PIN", type="password", key="add_pin").strip()
        role = st.selectbox("Rol", ["SUPER_ADMIN", "TOURNAMENT_ADMIN", "VIEWER"], key="add_role")
        active = st.checkbox("Activo", value=True, key="add_active")
        assigned_admin = None
        if role == "VIEWER":
            admin_users = [u["username"] for u in users if u["role"] == "TOURNAMENT_ADMIN"]
            assigned_admin = st.selectbox("Administrador Asignado", admin_users, key="add_assigned_admin")

        submitted = st.form_submit_button("Guardar Usuario")
        if submitted:
            if not username or not pin:
                st.error("Usuario y PIN no pueden estar vacíos.")
            else:
                new_user = {
                    "username": username,
                    "pin_hash": sha(pin),
                    "role": role,
                    "assigned_admin": assigned_admin,
                    "created_at": now_iso(),
                    "active": active
                }
                set_user(new_user)
                st.success(f"Usuario '{username}' guardado.")
                st.experimental_rerun()

    st.subheader("Configuración de la App")
    app_config = load_app_config()
    with st.form("app_config_form"):
        st.subheader("Logo de la App")
        new_logo_url = st.text_input("URL del logo (RAW URL)", value=app_config.get("app_logo_url", ""))
        submitted = st.form_submit_button("Actualizar logo")
        if submitted:
            app_config["app_logo_url"] = new_logo_url
            save_app_config(app_config)
            st.cache_data.clear()
            st.success("Logo actualizado. Recargando la página...")
            st.experimental_rerun()


def viewer_tournament(tid: str, public: bool = False):
    tourn = load_tournament(tid)
    if not tourn:
        st.error("Torneo no encontrado.")
        return

    inject_global_layout(f"Torneo: {tourn['name']}")

    if public:
        st.warning("Estás viendo el modo público de este torneo.")
    
    st.markdown(f"**Creado por:** {tourn['admin']} &nbsp;&nbsp;**Estado:** `{tourn['status']}`")
    st.markdown(f"**Fecha de inicio:** {datetime.fromisoformat(tourn['start_date']).strftime('%d/%m/%Y')}")

    if tourn["status"] == "SETUP":
        st.info("El torneo aún no ha comenzado.")
        st.subheader("Parejas inscritas")
        pairs_df = pd.DataFrame(tourn["pairs"])
        st.dataframe(pairs_df, hide_index=True)
    elif tourn["status"] == "GROUP_STAGE":
        st.subheader("Etapa de Grupos")
        for group_name, group_data in tourn["groups"].items():
            st.markdown(f"### Grupo {group_name}")
            matches_df = pd.DataFrame([
                {
                    "Partido": f"Match {i+1}",
                    "Pareja 1": match["pair1"],
                    "Pareja 2": match["pair2"],
                    "Resultado": f"{match['score_pair1']} - {match['score_pair2']}" if match["completed"] else "Pendiente"
                } for i, (match_id, match) in enumerate(group_data["matches"].items())
            ])
            st.table(matches_df.set_index("Partido"))

            st.markdown("#### Tabla de Posiciones")
            # Ordenar la tabla: Puntos, PG, DG
            sorted_table = sorted(group_data["table"], key=lambda x: (x.get("Puntos", 0), x.get("PG", 0), x.get("DG", 0)), reverse=True)
            table_df = pd.DataFrame(sorted_table)
            st.table(table_df.set_index("Pareja"))

    elif tourn["status"] == "PLAYOFFS":
        st.subheader("Playoffs")
        champion = tourn.get("champion")
        for stage, stage_data in tourn["playoffs"].items():
            st.markdown(f"### Ronda: {stage}")
            for match_id, match in stage_data.items():
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.info(match["pair1"])
                with col2:
                    st.text(f"vs")
                with col3:
                    st.info(match["pair2"])
                
                if match["completed"]:
                    winner = match.get("winner")
                    winner_text = f"¡Ganador: {winner}!" if winner else "Partido completo"
                    st.success(f"Resultado: {match['score_pair1']} - {match['score_pair2']} | {winner_text}")
                    if stage == "FINAL" and winner == champion:
                        st.balloons()
                else:
                    st.warning("Partido pendiente")
                st.markdown("---")
        
        if champion:
            st.success(f"¡El campeón del torneo es: {champion}!")
            st.balloons()

    st.subheader("Parejas participantes")
    pairs_df = pd.DataFrame(tourn["pairs"])
    st.dataframe(pairs_df, hide_index=True)
    
    if REPORTLAB_OK:
        generate_pdf(tourn)

def admin_dashboard(user: Dict[str, Any]):
    inject_global_layout("Administración de Torneos")

    st.markdown(f"**Usuario:** {user['username']} · Rol: `{user['role']}`")
    
    # Manejar creación o selección de torneo
    tournaments = load_index_for_admin(user["username"])
    tourn_options = {t["name"]: t["tournament_id"] for t in tournaments}
    
    selected_tourn_name = st.selectbox("Seleccionar Torneo", ["Crear nuevo..."] + list(tourn_options.keys()))
    
    if selected_tourn_name == "Crear nuevo...":
        with st.form("create_tournament"):
            st.subheader("Crear Nuevo Torneo")
            tourn_name = st.text_input("Nombre del Torneo")
            start_date = st.date_input("Fecha de Inicio", datetime.now().date())
            if st.form_submit_button("Crear Torneo"):
                if tourn_name:
                    create_new_tournament(user, tourn_name, start_date)
                    st.experimental_rerun()
                else:
                    st.error("El nombre del torneo no puede estar vacío.")
        st.stop()

    tid = tourn_options[selected_tourn_name]
    st.session_state.current_tid = tid
    tourn = load_tournament(tid)
    tourn_state = st.session_state.get("tourn_state", {})

    st.subheader(f"Torneo: {tourn['name']}")
    
    public_url_prefix = urlunparse(urlparse(st.experimental_get_query_params()["__script_name__"][0])._replace(query=""))
    public_url = f"{public_url_prefix}?mode=public&tid={tid}"
    
    with st.expander("Compartir y Opciones"):
        st.markdown(f"**Estado:** `{tourn['status']}` &nbsp;&nbsp;**Fecha de inicio:** {datetime.fromisoformat(tourn['start_date']).strftime('%d/%m/%Y')}")
        st.markdown(f"**Creado por:** {tourn['admin']}")
        
        st.markdown("### Enlace Público del Torneo")
        st.text_input("Compartir este enlace", value=public_url, key="public_link")
        st.button("Copiar al portapapeles", key="copy_button", on_click=lambda: st.runtime.legacy_caching.write_clipboard(public_url))
        
        if REPORTLAB_OK:
            generate_pdf(tourn)

        if st.button("Eliminar Torneo", help="Elimina el torneo y todos sus datos."):
            delete_tournament(tid)
            if "current_tid" in st.session_state:
                del st.session_state["current_tid"]
            st.success("Torneo eliminado.")
            st.experimental_rerun()

    tab_pairs, tab_groups, tab_playoffs, tab_persistency = st.tabs(["Parejas", "Grupos", "Playoffs", "Persistencia"])

    with tab_pairs:
        st.subheader("Administrar Parejas")
        col_form, col_list = st.columns([1, 1])
        with col_form:
            with st.form("add_pair_form"):
                st.subheader("Agregar Pareja")
                p1_name = st.text_input("Nombre Jugador 1")
                p2_name = st.text_input("Nombre Jugador 2")
                seeded_pair_selection = st.checkbox("Marcar como cabeza de serie")
                
                add_pair_button = st.form_submit_button("Añadir Pareja")

                if add_pair_button:
                    if p1_name and p2_name:
                        new_pair = {"player1": p1_name.strip().title(), "player2": p2_name.strip().title()}
                        add_pair(tourn, new_pair["player1"], new_pair["player2"])
                        if seeded_pair_selection:
                            if "seeded_pairs" not in tourn["config"]:
                                tourn["config"]["seeded_pairs"] = []
                            tourn["config"]["seeded_pairs"].append(new_pair)
                            save_tournament(tourn)
                        st.success(f"Pareja {p1_name} / {p2_name} agregada.")
                        st.experimental_rerun()
                    else:
                        st.error("Por favor, ingresa los nombres de ambos jugadores.")

        with col_list:
            st.subheader("Lista de Parejas")
            if tourn["pairs"]:
                pairs_df = pd.DataFrame(tourn["pairs"])
                pairs_df["Pareja"] = pairs_df["player1"] + " / " + pairs_df["player2"]
                
                # Checkbox para cabezas de serie
                st.markdown("### Opciones de Agrupación")
                seeded_mode = st.checkbox("Usar cabezas de serie", value=tourn["config"].get("seeded_mode", False))
                
                if seeded_mode:
                    all_pairs_list = pairs_df["Pareja"].tolist()
                    # AQUI ESTÁ LA CORRECCIÓN CRÍTICA
                    default_seeded_pairs = [f"{p['player1']} / {p['player2']}" for p in tourn["config"].get("seeded_pairs", [])]
                    seeded_pairs_names = st.multiselect(
                        "Seleccionar parejas cabeza de serie",
                        options=all_pairs_list,
                        default=default_seeded_pairs
                    )
                    selected_seeded_pairs = [
                        {"player1": name.split(" / ")[0], "player2": name.split(" / ")[1]}
                        for name in seeded_pairs_names
                    ]
                    
                    if selected_seeded_pairs != tourn["config"].get("seeded_pairs", []):
                        tourn["config"]["seeded_pairs"] = selected_seeded_pairs
                        save_tournament(tourn)
                        st.info("Parejas cabeza de serie actualizadas.")
                
                if st.button("Eliminar parejas seleccionadas"):
                    # Esta parte del código necesita ser ajustada si se implementa una UI para seleccionar y borrar
                    pass
                
                st.markdown("### Parejas Inscritas")
                st.dataframe(pairs_df[["Pareja"]], use_container_width=True, hide_index=True)
                
                col_del, col_space = st.columns([1, 4])
                with col_del:
                    pair_to_delete_name = st.selectbox("Seleccionar pareja para eliminar", [""] + pairs_df["Pareja"].tolist())
                
                if st.button("Eliminar pareja"):
                    if pair_to_delete_name:
                        p1, p2 = pair_to_delete_name.split(" / ")
                        pair_to_delete = {"player1": p1, "player2": p2}
                        delete_pair(tourn, pair_to_delete)
                        st.success(f"Pareja {pair_to_delete_name} eliminada.")
                        st.experimental_rerun()


    with tab_groups:
        st.subheader("Generar Grupos y Partidos")
        
        num_pairs = len(tourn["pairs"])
        if num_pairs == 0:
            st.warning("No hay parejas para generar grupos.")
            st.stop()
            
        with st.form("generate_groups_form"):
            num_groups = st.number_input("Número de grupos", min_value=1, value=1, step=1, help="Debe haber al menos 2 parejas por grupo")
            
            pairs_per_group = 2
            if num_pairs > num_groups:
                pairs_per_group = max(2, num_pairs // num_groups)

            num_teams_per_group = st.number_input("Parejas por grupo", min_value=2, value=pairs_per_group, step=1)
            
            seeded_mode = tourn["config"].get("seeded_mode", False)
            seeded_pairs_count = len(tourn["config"].get("seeded_pairs", []))
            
            if seeded_mode and seeded_pairs_count < num_groups:
                st.warning(f"Tienes activado el modo de cabezas de serie, pero necesitas al menos {num_groups} parejas marcadas como cabezas de serie para generar {num_groups} grupos. Actualmente tienes {seeded_pairs_count}.")
                
            if st.form_submit_button("Generar Grupos"):
                if num_pairs < num_groups * num_teams_per_group:
                    st.error(f"Se necesitan al menos {num_groups * num_teams_per_group} parejas en total.")
                else:
                    generate_groups(tourn, num_groups, num_teams_per_group, seeded_mode, tourn["config"].get("seeded_pairs", []))
                    st.experimental_rerun()
        
        if tourn["status"] == "GROUP_STAGE":
            st.subheader("Resultados de Grupos")
            groups = tourn["groups"]
            for group_name, group_data in groups.items():
                st.markdown(f"### Grupo {group_name}")
                
                matches_df = pd.DataFrame([
                    {
                        "Partido": f"Match {i+1}",
                        "Pareja 1": match["pair1"],
                        "Pareja 2": match["pair2"],
                        "Resultado": f"{match['score_pair1']} - {match['score_pair2']}" if match["completed"] else "Pendiente"
                    } for i, (match_id, match) in enumerate(group_data["matches"].items())
                ])
                st.table(matches_df.set_index("Partido"))
                
                with st.form(f"update_group_{group_name}"):
                    st.subheader("Ingresar Resultados de Partidos")
                    match_to_update = st.selectbox(
                        "Seleccionar Partido",
                        list(group_data["matches"].keys()),
                        format_func=lambda x: f"{group_data['matches'][x]['pair1']} vs {group_data['matches'][x]['pair2']}"
                    )
                    
                    if match_to_update:
                        current_match = group_data["matches"][match_to_update]
                        col_score1, col_score2 = st.columns(2)
                        with col_score1:
                            score1 = st.number_input(f"Puntos {current_match['pair1']}", min_value=0, value=current_match['score_pair1'] if current_match['score_pair1'] is not None else 0, key=f"score1_{group_name}_{match_to_update}")
                        with col_score2:
                            score2 = st.number_input(f"Puntos {current_match['pair2']}", min_value=0, value=current_match['score_pair2'] if current_match['score_pair2'] is not None else 0, key=f"score2_{group_name}_{match_to_update}")
                        
                        update_button = st.form_submit_button("Actualizar Resultado")
                        
                        if update_button:
                            if score1 == score2:
                                st.error("El resultado no puede ser un empate.")
                            else:
                                groups[group_name]["matches"][match_to_update]["score_pair1"] = score1
                                groups[group_name]["matches"][match_to_update]["score_pair2"] = score2
                                groups[group_name]["matches"][match_to_update]["completed"] = True
                                save_tournament(tourn)
                                st.success("Resultado actualizado.")
                                st.experimental_rerun()
            
            if st.button("Calcular Tabla de Posiciones y Clasificados"):
                calculate_group_tables(tourn)
                st.experimental_rerun()

            if tourn.get("qualifiers"):
                st.markdown("### Clasificados a Playoffs")
                st.dataframe(pd.DataFrame({"Pareja": tourn["qualifiers"]}))
            
    with tab_playoffs:
        st.subheader("Generar y Administrar Playoffs")
        
        num_qualifiers = len(tourn.get("qualifiers", []))
        if num_qualifiers not in [2, 4, 8]:
            st.warning("Debes tener 2, 4 u 8 clasificados para generar los playoffs.")
        
        if st.button("Generar Cuadro de Playoffs"):
            generate_playoffs(tourn, num_qualifiers)
            st.experimental_rerun()
            
        if st.button("Regenerar Cuadro de Playoffs"):
            generate_playoffs(tourn, num_qualifiers)
            st.experimental_rerun()

        if tourn["status"] == "PLAYOFFS":
            st.subheader("Resultados de Playoffs")
            for stage, stage_data in tourn["playoffs"].items():
                st.markdown(f"### Ronda: {stage}")
                for match_id, match in stage_data.items():
                    with st.form(f"update_{stage}_{match_id}"):
                        st.markdown(f"**Partido: {match['pair1']} vs {match['pair2']}**")
                        if not match["completed"]:
                            col_score1, col_score2 = st.columns(2)
                            with col_score1:
                                score1 = st.number_input(f"Puntos {match['pair1']}", min_value=0, value=0, key=f"{stage}_{match_id}_score1")
                            with col_score2:
                                score2 = st.number_input(f"Puntos {match['pair2']}", min_value=0, value=0, key=f"{stage}_{match_id}_score2")
                            
                            update_button = st.form_submit_button("Actualizar")
                            
                            if update_button:
                                if score1 == score2:
                                    st.error("El resultado no puede ser un empate.")
                                else:
                                    tourn["playoffs"][stage][match_id]["score_pair1"] = score1
                                    tourn["playoffs"][stage][match_id]["score_pair2"] = score2
                                    tourn["playoffs"][stage][match_id]["completed"] = True
                                    update_playoffs(tourn)
                                    st.success("Resultado actualizado.")
                                    st.experimental_rerun()
                        else:
                            winner = match["pair1"] if match["score_pair1"] > match["score_pair2"] else match["pair2"]
                            st.info(f"Resultado: {match['score_pair1']} - {match['score_pair2']}")
                            st.success(f"Ganador: {winner}")
                            st.markdown("---")
            
            champion = tourn.get("champion")
            if champion:
                st.success(f"¡El campeón del torneo es: {champion}!")
                st.balloons()


    with tab_persistency:
        st.subheader("Copia de Seguridad y Restauración")
        st.info("Aquí puedes guardar y restaurar manualmente el estado del torneo. Los cambios automáticos también se guardan, pero esta es tu red de seguridad.")

        if st.button("Guardar estado actual"):
            save_snapshot(tourn)
            st.success("Estado actual guardado.")
        
        st.markdown("### Restaurar desde una copia")
        snapshots = get_snapshots(tourn["tournament_id"])
        
        if not snapshots:
            st.info("No hay copias de seguridad disponibles.")
        else:
            snap_options = {s["timestamp"]: s["path"] for s in snapshots}
            selected_snap_ts = st.selectbox(
                "Seleccionar copia de seguridad",
                list(snap_options.keys()),
                format_func=lambda ts: f"Guardado el {ts}"
            )
            
            if selected_snap_ts and st.button("Restaurar"):
                snap_path = snap_options[selected_snap_ts]
                restored_tourn = load_snapshot(snap_path)
                if restored_tourn:
                    save_tournament(restored_tourn)
                    st.success("Estado del torneo restaurado.")
                    st.experimental_rerun()
                else:
                    st.error("Error al restaurar la copia de seguridad.")


# ====== Funciones de Persistencia (Snapshots) ======
def get_snapshot_path(tournament_id: str) -> Path:
    return SNAP_ROOT / f"snapshots_{tournament_id}"

def get_snapshots(tournament_id: str) -> List[Dict[str, Any]]:
    path = get_snapshot_path(tournament_id)
    if not path.exists():
        return []
    snapshots = []
    for f in path.glob("*.json"):
        try:
            timestamp = f.stem.replace(f"snap_{tournament_id}_", "")
            snapshots.append({"timestamp": timestamp, "path": str(f)})
        except:
            pass
    snapshots.sort(key=lambda x: x["timestamp"], reverse=True)
    return snapshots

def save_snapshot(tourn: Dict[str, Any]):
    path = get_snapshot_path(tourn["tournament_id"])
    path.mkdir(exist_ok=True)
    
    # Limpiar snapshots antiguos
    snapshots = sorted(path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in snapshots[KEEP_SNAPSHOTS-1:]:
        f.unlink()
        
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snap_path = path / f"snap_{tourn['tournament_id']}_{ts}.json"
    snap_path.write_text(json.dumps(tourn, ensure_ascii=False, indent=2), encoding="utf-8")

def load_snapshot(snap_path: str) -> Optional[Dict[str, Any]]:
    p = Path(snap_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

# ====== Lógica de cálculo de tabla de grupo ======
def calculate_group_tables(tourn: Dict[str, Any]):
    if tourn["status"] != "GROUP_STAGE":
        st.warning("El torneo debe estar en la etapa de grupos para calcular las tablas.")
        return

    qualifiers = []
    for group_name, group_data in tourn["groups"].items():
        table = {}
        for pair in group_data["pairs"]:
            pair_name = f"{pair['player1']} / {pair['player2']}"
            table[pair_name] = {"Puntos": 0, "PG": 0, "PP": 0, "DG": 0, "Pareja": pair_name}

        for match_id, match in group_data["matches"].items():
            if match["completed"]:
                p1, p2 = match["pair1"], match["pair2"]
                s1, s2 = match["score_pair1"], match["score_pair2"]
                
                if s1 > s2: # Ganador p1
                    table[p1]["PG"] += 1
                    table[p2]["PP"] += 1
                    table[p1]["Puntos"] += 2
                    table[p2]["Puntos"] += 1
                else: # Ganador p2
                    table[p1]["PP"] += 1
                    table[p2]["PG"] += 1
                    table[p2]["Puntos"] += 2
                    table[p1]["Puntos"] += 1
                
                table[p1]["DG"] += (s1 - s2)
                table[p2]["DG"] += (s2 - s1)

        # Ordenar por Puntos, PG, DG (descendente)
        sorted_pairs = sorted(table.values(), key=lambda x: (x["Puntos"], x["PG"], x["DG"]), reverse=True)
        tourn["groups"][group_name]["table"] = sorted_pairs
        
        # Clasificar 2 por grupo
        qualifiers.extend([p["Pareja"] for p in sorted_pairs[:2]])

    tourn["qualifiers"] = qualifiers
    save_tournament(tourn)
    st.success("Tablas de posiciones y clasificados calculados.")

# ====== Funciones de usuario/login (modificadas para la web) ======
def check_auth_status():
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None

# ====== Main ======
def main():
    check_auth_status()
    
    # Check for public viewer mode
    params = st.query_params
    mode = params.get("mode", [None])[0]
    _tid = params.get("tid", [None])[0]

    if mode == "public" and _tid:
        viewer_tournament(_tid, public=True)
        st.caption("iAPPs Pádel — v3.3.27")
        return

    if not st.session_state.get("auth_user"):
        inject_global_layout("No autenticado")
        login_form()
        st.caption("iAPPs Pádel — v3.3.27")
        return

    user = st.session_state["auth_user"]
    
    # Redirección si un Super Admin intenta entrar sin el modo 'super'
    if user["role"] == "SUPER_ADMIN" and params.get("mode", [None])[0] != "super":
        st.warning("Acceso denegado. Solo Super Admin.")
        st.session_state["auth_user"] = None
        st.experimental_rerun()
        return
    
    if user["role"] == "SUPER_ADMIN":
        super_admin_panel()
    elif user["role"] == "TOURNAMENT_ADMIN":
        admin_dashboard(user)
    elif user["role"] == "VIEWER":
        st.info("Modo solo lectura. Puedes ver los torneos de tu administrador asignado.")
        admin = get_user(user["assigned_admin"])
        if admin:
            st.session_state.current_tid = st.selectbox("Torneo", [t["tournament_id"] for t in load_index_for_admin(admin["username"])], format_func=lambda tid: load_tournament(tid)["name"])
            if st.session_state.current_tid:
                viewer_tournament(st.session_state.current_tid)
        else:
            st.warning("No tienes un administrador asignado o la cuenta del administrador no existe.")
            st.caption("iAPPs Pádel — v3.3.27")


if __name__ == "__main__":
    main()

