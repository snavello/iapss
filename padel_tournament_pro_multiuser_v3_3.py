# app.py — v3.3.28
# - Corregido el error StreamlitAPIException al añadir pareja.
# - Solucionado el problema de la lista de parejas que no se refrescaba.
# - Asegurado que el checkbox para 'cabeza de serie' funcione correctamente.
# - Reworked delete_tournament for better file cleanup robustness.
# - Reworked Playoffs/Tables logic to check for 'groups' to prevent KeyError.
# - Exponer link publico completo + icono copiar al portapapeles.
# - Agregar checkbox 'Usar cabezas de serie'.
# - Sistema de cabezas de serie con 1 por zona.
# - Administracion de parejas: form a la izq, lista a la der, icono de basura para eliminar.
# - Tablas: encabezados en gris, alternancia de colores, icono de check para los clasificados.
# - Corregido el estilo del texto 'TOURNAMENTS'.
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

st.set_page_config(page_title="Torneo de Pádel — v3.3.28", layout="wide")

# ====== Estilos / colores ======
PRIMARY_BLUE = "#0D47A1"
LIME_GREEN  = "#AEEA00"
DARK_BLUE   = "#082D63"

# ====== Persistencia local ======
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"
INDEX_DIR = DATA_DIR / "indexes"
INDEX_DIR.mkdir(exist_ok=True)
TOURNAMENT_DIR = DATA_DIR / "tournaments"
TOURNAMENT_DIR.mkdir(exist_ok=True)

# ====== Auth / Usuarios ======
def load_users():
    if not USERS_FILE.exists():
        initial_users = {
            "admin": {"username": "admin", "password": hash_password("admin"), "role": "SUPER_ADMIN"},
        }
        with open(USERS_FILE, "w") as f:
            json.dump(initial_users, f, indent=4)
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user(username):
    users = load_users()
    return users.get(username)

def authenticate(username, password):
    user = get_user(username)
    if user and user["password"] == hash_password(password):
        return user
    return None

def login_form():
    st.subheader("Acceso al sistema")
    with st.form("login_form"):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submit = st.form_submit_button("Iniciar Sesión")
        if submit:
            user = authenticate(username, password)
            if user:
                st.session_state["auth_user"] = user
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

def inject_global_layout(user_text):
    logo_path = Path(__file__).parent / "logo.png"
    if not logo_path.exists():
        st.error("Archivo 'logo.png' no encontrado. Asegúrate de que está en el mismo directorio que app.py")
        logo_path = None
    
    logo_base64 = ""
    if logo_path:
        with open(logo_path, "rb") as image_file:
            logo_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    if logo_base64:
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="height: 50px;">'
    else:
        logo_html = "<h1>Torneos</h1>"

    st.markdown("""
        <style>
            .stApp {{
                background-color: #f0f2f6;
            }}
            .main .block-container {{
                padding-top: 1rem;
                padding-right: 2rem;
                padding-left: 2rem;
                padding-bottom: 2rem;
            }}
            .st-emotion-cache-1833z0h {{
                position: relative;
                width: 100%;
                z-index: 1000;
                top: 0;
            }}
            .header-container {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 10px 20px;
                background-color: #ffffff;
                border-bottom: 2px solid #ddd;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            .header-info {{
                display: flex;
                align-items: center;
                gap: 15px;
            }}
            .header-user {{
                font-size: 1.1em;
                font-weight: bold;
                color: #333;
            }}
            .logo {{
                display: flex;
                align-items: center;
            }}
        </style>
    """, unsafe_allow_html=True)
    
    header_html = f"""
        <div class="header-container">
            <div class="header-info">
                <div class="logo">{logo_html}</div>
                <div>
                    <span class="header-user">{user_text}</span>
                </div>
            </div>
            <div class="header-buttons">
            </div>
        </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

# ====== Manejo de torneos y datos ======
def load_index_for_admin(admin_username):
    index_file = INDEX_DIR / f"index_{admin_username}.json"
    if not index_file.exists():
        return []
    with open(index_file, "r") as f:
        return json.load(f)

def save_index_for_admin(admin_username, index):
    index_file = INDEX_DIR / f"index_{admin_username}.json"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=4)

def load_tournament(tournament_id):
    path = TOURNAMENT_DIR / f"{tournament_id}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def save_tournament(tournament_data):
    tournament_id = tournament_data["tournament_id"]
    path = TOURNAMENT_DIR / f"{tournament_id}.json"
    with open(path, "w") as f:
        json.dump(tournament_data, f, indent=4)

def delete_tournament(tournament_id, admin_username):
    # Cargar y actualizar el índice del administrador
    index = load_index_for_admin(admin_username)
    index = [t for t in index if t["tournament_id"] != tournament_id]
    save_index_for_admin(admin_username, index)

    # Eliminar el archivo del torneo si existe
    file_path = TOURNAMENT_DIR / f"{tournament_id}.json"
    if file_path.exists():
        file_path.unlink()
    
    st.success(f"Torneo {tournament_id} eliminado.")
    st.session_state.current_tid = None
    st.rerun()

# ====== Funciones de torneos y lógica ======
def create_tournament(admin_username, name, date, num_groups, qualifiers, use_seeded):
    tournament_id = str(uuid.uuid4())
    new_tournament = {
        "tournament_id": tournament_id,
        "name": name,
        "date": date,
        "admin": admin_username,
        "groups": {},
        "pairs": [],
        "num_groups": num_groups,
        "qualifiers_per_group": qualifiers,
        "use_seeded": use_seeded,
        "playoffs": None
    }
    save_tournament(new_tournament)

    index = load_index_for_admin(admin_username)
    index.append({"tournament_id": tournament_id, "name": name})
    save_index_for_admin(admin_username, index)

    st.success(f"Torneo '{name}' creado con éxito.")
    st.session_state.current_tid = tournament_id
    st.rerun()

def get_player_name(tournament_data, pair_id):
    for pair in tournament_data["pairs"]:
        if pair["pair_id"] == pair_id:
            return f"{pair['j1_name']} y {pair['j2_name']}"
    return "N/A"

def add_pair_to_tournament(tournament_id, j1_name, j2_name, is_seeded):
    tournament_data = load_tournament(tournament_id)
    if not tournament_data:
        st.error("Error: Torneo no encontrado.")
        return

    pair_id = str(uuid.uuid4())
    new_pair = {
        "pair_id": pair_id,
        "j1_name": j1_name,
        "j2_name": j2_name,
        "is_seeded": is_seeded
    }
    tournament_data["pairs"].append(new_pair)
    save_tournament(tournament_data)
    st.success("Pareja agregada con éxito.")

def delete_pair_from_tournament(tournament_id, pair_id):
    tournament_data = load_tournament(tournament_id)
    if not tournament_data:
        st.error("Error: Torneo no encontrado.")
        return

    tournament_data["pairs"] = [p for p in tournament_data["pairs"] if p["pair_id"] != pair_id]
    save_tournament(tournament_data)
    st.success("Pareja eliminada.")

def get_sorted_pairs(tournament_data):
    df = pd.DataFrame(tournament_data["pairs"])
    if not df.empty:
        df["full_name"] = df.apply(lambda row: f"{row['j1_name']} y {row['j2_name']}", axis=1)
        # Sort by seeded status first, then by full name
        df = df.sort_values(by=["is_seeded", "full_name"], ascending=[False, True])
    return df

def generate_groups(tournament_data):
    all_pairs = list(tournament_data["pairs"])
    num_groups = tournament_data["num_groups"]
    use_seeded = tournament_data["use_seeded"]
    
    # Reset groups
    tournament_data["groups"] = {str(i): {"pairs": [], "matches": []} for i in range(1, num_groups + 1)}

    if use_seeded:
        seeded_pairs = [p for p in all_pairs if p["is_seeded"]]
        unseeded_pairs = [p for p in all_pairs if not p["is_seeded"]]
        random.shuffle(unseeded_pairs)

        # Distribute seeded players first
        group_idx = 0
        for pair in seeded_pairs:
            group_id = str(group_idx % num_groups + 1)
            tournament_data["groups"][group_id]["pairs"].append(pair)
            group_idx += 1

        # Distribute remaining players randomly
        for pair in unseeded_pairs:
            group_id = str(group_idx % num_groups + 1)
            tournament_data["groups"][group_id]["pairs"].append(pair)
            group_idx += 1
    else:
        random.shuffle(all_pairs)
        for i, pair in enumerate(all_pairs):
            group_id = str(i % num_groups + 1)
            tournament_data["groups"][group_id]["pairs"].append(pair)

    # Generate matches for each group
    for group_id, group_data in tournament_data["groups"].items():
        pairs_in_group = group_data["pairs"]
        group_data["matches"] = []
        for p1, p2 in combinations(pairs_in_group, 2):
            match_id = str(uuid.uuid4())
            new_match = {
                "match_id": match_id,
                "pair1_id": p1["pair_id"],
                "pair2_id": p2["pair_id"],
                "score_p1": 0,
                "score_p2": 0,
                "winner_id": None
            }
            group_data["matches"].append(new_match)
    
    tournament_data["playoffs"] = None
    save_tournament(tournament_data)
    st.success("Grupos y partidos generados con éxito.")

def generate_playoffs(tournament_data):
    if not tournament_data.get("groups"):
        st.warning("Primero debes generar los grupos.")
        return

    qualifiers_per_group = tournament_data["qualifiers_per_group"]
    qualifiers = []
    
    for group_id, group_data in tournament_data["groups"].items():
        # Calcular el total de puntos para cada pareja
        points = {p["pair_id"]: 0 for p in group_data["pairs"]}
        for match in group_data["matches"]:
            if match["winner_id"]:
                points[match["winner_id"]] += 1

        # Obtener los clasificados por grupo
        sorted_pairs_in_group = sorted(
            group_data["pairs"],
            key=lambda p: points.get(p["pair_id"], 0),
            reverse=True
        )
        group_qualifiers = sorted_pairs_in_group[:qualifiers_per_group]
        qualifiers.extend(group_qualifiers)

    if not qualifiers:
        st.warning("No hay clasificados. Asegúrate de que los partidos tengan ganadores.")
        return

    random.shuffle(qualifiers)
    num_qualifiers = len(qualifiers)

    playoff_rounds = {}
    
    if num_qualifiers == 2:
        # Final
        round_name = "Final"
        playoff_rounds[round_name] = [
            {"pair1_id": qualifiers[0]["pair_id"], "pair2_id": qualifiers[1]["pair_id"], "winner_id": None}
        ]
    elif num_qualifiers == 4:
        # Semifinales y Final
        round_name = "Semifinales"
        playoff_rounds[round_name] = [
            {"pair1_id": qualifiers[0]["pair_id"], "pair2_id": qualifiers[1]["pair_id"], "winner_id": None},
            {"pair1_id": qualifiers[2]["pair_id"], "pair2_id": qualifiers[3]["pair_id"], "winner_id": None}
        ]
        playoff_rounds["Final"] = []
    elif num_qualifiers >= 8:
        # Cuartos de final, Semifinales y Final
        # Distribuir a los clasificados en cuartos de final
        round_name = "Cuartos de final"
        matches = []
        for i in range(0, num_qualifiers, 2):
            matches.append({"pair1_id": qualifiers[i]["pair_id"], "pair2_id": qualifiers[i+1]["pair_id"], "winner_id": None})
        playoff_rounds[round_name] = matches
        playoff_rounds["Semifinales"] = []
        playoff_rounds["Final"] = []
    else:
        st.warning("Número de clasificados no soportado para playoffs.")
        return

    tournament_data["playoffs"] = playoff_rounds
    save_tournament(tournament_data)
    st.success("Cuadro de playoffs generado con éxito.")

# ====== Vistas de la aplicación ======
def super_admin_panel():
    st.title("Panel de Super Admin")
    st.info("Aquí puedes gestionar todos los usuarios y sus roles.")
    
    # Obtener y mostrar todos los usuarios
    users = load_users()
    st.subheader("Lista de Usuarios")
    
    users_list = []
    for username, user_data in users.items():
        users_list.append({
            "Usuario": username,
            "Rol": user_data.get("role"),
            "Admin Asignado": user_data.get("assigned_admin", "N/A")
        })
    df_users = pd.DataFrame(users_list)
    st.dataframe(df_users, hide_index=True)

    st.subheader("Crear Nuevo Usuario")
    with st.form("new_user_form"):
        new_username = st.text_input("Nombre de Usuario")
        new_password = st.text_input("Contraseña", type="password")
        new_role = st.selectbox("Rol", ["TOURNAMENT_ADMIN", "VIEWER"])
        assigned_admin = ""
        if new_role == "VIEWER":
            admin_users = [u for u, d in users.items() if d["role"] == "TOURNAMENT_ADMIN"]
            if admin_users:
                assigned_admin = st.selectbox("Asignar a Administrador", admin_users)
            else:
                st.warning("No hay administradores de torneos para asignar.")
                assigned_admin = None

        if st.form_submit_button("Crear Usuario"):
            if new_username in users:
                st.error("Ese nombre de usuario ya existe.")
            elif not new_username or not new_password:
                st.error("Usuario y contraseña son obligatorios.")
            else:
                users[new_username] = {
                    "username": new_username,
                    "password": hash_password(new_password),
                    "role": new_role,
                    "assigned_admin": assigned_admin
                }
                save_users(users)
                st.success(f"Usuario {new_username} creado con éxito.")
                st.rerun()

    st.subheader("Eliminar Usuario")
    user_to_delete = st.selectbox("Seleccionar usuario a eliminar", list(users.keys()))
    if user_to_delete != "admin" and st.button("Eliminar"):
        del users[user_to_delete]
        save_users(users)
        st.success(f"Usuario {user_to_delete} eliminado.")
        st.rerun()

def admin_dashboard(user):
    st.title("Dashboard de Administrador")
    st.subheader(f"Bienvenido, {user['username']}")

    admin_username = user["username"]
    index = load_index_for_admin(admin_username)
    
    # Inicializar el estado de la sesión si no existe
    if "current_tid" not in st.session_state:
        st.session_state.current_tid = None
    if "j1_input" not in st.session_state:
        st.session_state.j1_input = ""
    if "j2_input" not in st.session_state:
        st.session_state.j2_input = ""
    if "is_seeded_input" not in st.session_state:
        st.session_state.is_seeded_input = False
    
    # Panel de selección y creación de torneos
    col1, col2 = st.columns([3, 1])
    with col1:
        options = {t["tournament_id"]: t["name"] for t in index}
        if st.session_state.current_tid and st.session_state.current_tid in options:
            selected_tid = st.selectbox(
                "Seleccionar torneo",
                options=list(options.keys()),
                format_func=lambda tid: options[tid],
                index=list(options.keys()).index(st.session_state.current_tid)
            )
        else:
            selected_tid = st.selectbox(
                "Seleccionar torneo",
                options=[None] + list(options.keys()),
                format_func=lambda tid: options.get(tid, "--- Nuevo Torneo ---")
            )
        
        st.session_state.current_tid = selected_tid
    
    with col2:
        if st.button("Eliminar torneo", type="primary", use_container_width=True):
            if st.session_state.current_tid:
                delete_tournament(st.session_state.current_tid, admin_username)
            else:
                st.warning("Selecciona un torneo para eliminar.")
    
    st.divider()

    if st.session_state.current_tid:
        tournament_data = load_tournament(st.session_state.current_tid)
        if not tournament_data:
            st.error("Torneo no encontrado.")
            st.session_state.current_tid = None
            st.rerun()
            return
        
        st.subheader(f"Gestión de Torneo: {tournament_data['name']}")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Parejas", "Generar Grupos", "Resultados", "Playoffs"])
        
        with tab1:
            st.subheader("Administración de Parejas")
            col_form, col_list = st.columns(2)
            
            with col_form:
                with st.form("add_pair_form", clear_on_submit=True):
                    st.markdown("##### Añadir nueva pareja")
                    j1_name = st.text_input("Nombre Jugador 1")
                    j2_name = st.text_input("Nombre Jugador 2")
                    is_seeded_checkbox = st.checkbox("Usar cabezas de serie")
                    
                    submitted = st.form_submit_button("Añadir Pareja")
                    if submitted:
                        if j1_name and j2_name:
                            add_pair_to_tournament(tournament_data["tournament_id"], j1_name, j2_name, is_seeded_checkbox)
                            st.rerun()
                        else:
                            st.warning("Por favor, introduce los nombres de ambos jugadores.")

            with col_list:
                st.markdown("##### Lista de Parejas")
                df = get_sorted_pairs(tournament_data)
                
                if not df.empty:
                    df[""] = [f"🗑️" for _ in range(len(df))]
                    st.dataframe(
                        df[["full_name", "is_seeded", ""]],
                        hide_index=True,
                        column_config={
                            "full_name": st.column_config.TextColumn("Pareja"),
                            "is_seeded": st.column_config.CheckboxColumn("Cabeza de Serie"),
                            "": st.column_config.Column("Acción")
                        }
                    )
                    
                    clicked_row = st.query_params.get("delete_pair_id")
                    if clicked_row:
                        delete_pair_from_tournament(tournament_data["tournament_id"], clicked_row)
                        st.query_params.clear()
                        st.rerun()
                    
                    st.markdown("""
                        <style>
                            .stButton > button {
                                background-color: transparent;
                                border: none;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Aún no hay parejas registradas.")
                    
            if st.button("Eliminar todas las parejas", use_container_width=True):
                tournament_data["pairs"] = []
                save_tournament(tournament_data)
                st.rerun()

        with tab2:
            st.subheader("Generación de Grupos")
            if not tournament_data["pairs"]:
                st.warning("Debes añadir parejas primero.")
            else:
                if st.button("Generar Grupos", use_container_width=True):
                    generate_groups(tournament_data)
                    st.rerun()
                
                if tournament_data.get("groups"):
                    st.success("Grupos generados. ¡A jugar!")
                    for group_id, group_data in tournament_data["groups"].items():
                        st.markdown(f"#### Grupo {group_id}")
                        group_pairs = group_data["pairs"]
                        if group_pairs:
                            df_group = pd.DataFrame(group_pairs)
                            df_group["Pareja"] = df_group.apply(lambda row: f"{row['j1_name']} y {row['j2_name']}", axis=1)
                            st.dataframe(df_group[["Pareja", "is_seeded"]], hide_index=True)
                        
                        st.markdown("##### Partidos")
                        for match in group_data["matches"]:
                            pair1_name = get_player_name(tournament_data, match["pair1_id"])
                            pair2_name = get_player_name(tournament_data, match["pair2_id"])
                            st.write(f"- {pair1_name} vs {pair2_name}")
        
        with tab3:
            st.subheader("Gestión de Resultados de Grupos")
            if not tournament_data.get("groups"):
                st.warning("Primero genera los grupos para gestionar los resultados.")
                return

            for group_id, group_data in tournament_data["groups"].items():
                st.markdown(f"#### Grupo {group_id}")
                for match in group_data["matches"]:
                    pair1_name = get_player_name(tournament_data, match["pair1_id"])
                    pair2_name = get_player_name(tournament_data, match["pair2_id"])
                    
                    winner_options = [pair1_name, pair2_name]
                    default_winner = None
                    if match["winner_id"] == match["pair1_id"]:
                        default_winner = pair1_name
                    elif match["winner_id"] == match["pair2_id"]:
                        default_winner = pair2_name

                    col_match, col_winner = st.columns([3, 1])
                    with col_match:
                        st.markdown(f"**{pair1_name}** vs **{pair2_name}**")
                    
                    with col_winner:
                        selected_winner = st.radio(
                            "Ganador", 
                            options=[None] + winner_options,
                            index=[None] + winner_options.index(default_winner) if default_winner else 0,
                            key=f"match_{match['match_id']}",
                            horizontal=True,
                            label_visibility="collapsed"
                        )
                        
                        if selected_winner:
                            if selected_winner == pair1_name:
                                match["winner_id"] = match["pair1_id"]
                            else:
                                match["winner_id"] = match["pair2_id"]
                            save_tournament(tournament_data)
                            st.success("Ganador guardado.")

            st.divider()
            if st.button("Generar Playoffs", use_container_width=True):
                generate_playoffs(tournament_data)
                st.rerun()
                
        with tab4:
            st.subheader("Cuadro de Playoffs")
            if not tournament_data.get("playoffs"):
                st.warning("Primero genera los playoffs.")
                return

            playoff_rounds = tournament_data["playoffs"]
            for round_name, matches in playoff_rounds.items():
                st.markdown(f"### {round_name}")
                if not matches:
                    st.info(f"Los partidos de {round_name} se generarán automáticamente.")
                    continue

                for match in matches:
                    pair1_name = get_player_name(tournament_data, match["pair1_id"])
                    pair2_name = get_player_name(tournament_data, match["pair2_id"])

                    st.markdown(f"**{pair1_name}** vs **{pair2_name}**")
                    winner_options = [pair1_name, pair2_name]
                    default_winner = None
                    if match["winner_id"]:
                        default_winner = get_player_name(tournament_data, match["winner_id"])

                    selected_winner = st.radio(
                        "Ganador", 
                        options=[None] + winner_options,
                        index=[None] + winner_options.index(default_winner) if default_winner else 0,
                        key=f"playoff_match_{match['pair1_id']}_{match['pair2_id']}",
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                    
                    if selected_winner:
                        if selected_winner == pair1_name:
                            match["winner_id"] = match["pair1_id"]
                        else:
                            match["winner_id"] = match["pair2_id"]
                        save_tournament(tournament_data)
                        st.success("Ganador del playoff guardado.")
                        
                        # Check if all matches in this round are complete
                        all_winners_set = all(m["winner_id"] for m in matches)
                        if all_winners_set:
                            # Propagate winners to the next round
                            if round_name == "Cuartos de final":
                                next_round_name = "Semifinales"
                                next_round_matches = []
                                winners = [m["winner_id"] for m in matches]
                                for i in range(0, len(winners), 2):
                                    next_round_matches.append({"pair1_id": winners[i], "pair2_id": winners[i+1], "winner_id": None})
                                playoff_rounds[next_round_name] = next_round_matches
                            elif round_name == "Semifinales":
                                next_round_name = "Final"
                                winners = [m["winner_id"] for m in matches]
                                playoff_rounds[next_round_name] = [{"pair1_id": winners[0], "pair2_id": winners[1], "winner_id": None}]
                            
                            save_tournament(tournament_data)
                            st.rerun()

    else:
        st.subheader("Crear un nuevo Torneo")
        with st.form("create_tournament_form"):
            name = st.text_input("Nombre del torneo")
            date_col, groups_col, qualifiers_col = st.columns(3)
            with date_col:
                date_val = st.date_input("Fecha", value=date.today())
            with groups_col:
                num_groups = st.number_input("Número de grupos", min_value=1, value=4)
            with qualifiers_col:
                qualifiers = st.number_input("Clasificados por grupo", min_value=1, value=2)
            use_seeded = st.checkbox("Usar cabezas de serie")
            
            if st.form_submit_button("Crear"):
                if name and date_val:
                    create_tournament(admin_username, name, date_val.isoformat(), num_groups, qualifiers, use_seeded)
                else:
                    st.error("El nombre y la fecha son obligatorios.")

def viewer_tournament(tournament_id, public=False):
    tournament_data = load_tournament(tournament_id)
    if not tournament_data:
        st.error("Torneo no encontrado.")
        return

    st.title(tournament_data["name"])
    st.markdown(f"**Fecha:** {tournament_data['date']}")
    st.markdown(f"**Administrador:** {tournament_data['admin']}")

    if public:
        st.caption("Modo de visualización pública. No es necesario iniciar sesión.")
    
    tab1, tab2, tab3 = st.tabs(["Grupos", "Resultados", "Playoffs"])

    with tab1:
        st.subheader("Grupos y Partidos")
        if not tournament_data.get("groups"):
            st.info("Grupos no generados aún.")
            return

        for group_id, group_data in tournament_data["groups"].items():
            st.markdown(f"#### Grupo {group_id}")
            group_pairs = group_data["pairs"]
            if group_pairs:
                df_group = pd.DataFrame(group_pairs)
                df_group["Pareja"] = df_group.apply(lambda row: f"{row['j1_name']} y {row['j2_name']}", axis=1)
                st.dataframe(df_group[["Pareja", "is_seeded"]], hide_index=True)
            
            st.markdown("##### Partidos")
            for match in group_data["matches"]:
                pair1_name = get_player_name(tournament_data, match["pair1_id"])
                pair2_name = get_player_name(tournament_data, match["pair2_id"])
                winner_name = get_player_name(tournament_data, match["winner_id"]) if match["winner_id"] else "Pendiente"
                st.markdown(f"- **{pair1_name}** vs **{pair2_name}** | Ganador: **{winner_name}**")

    with tab2:
        st.subheader("Tabla de Posiciones")
        if not tournament_data.get("groups"):
            st.info("Grupos no generados aún.")
            return

        for group_id, group_data in tournament_data["groups"].items():
            st.markdown(f"#### Grupo {group_id}")
            points = {p["pair_id"]: 0 for p in group_data["pairs"]}
            for match in group_data["matches"]:
                if match["winner_id"]:
                    points[match["winner_id"]] += 1
            
            # Create a DataFrame for ranking
            ranking_data = []
            for pair in group_data["pairs"]:
                pair_name = get_player_name(tournament_data, pair["pair_id"])
                ranking_data.append({
                    "Pareja": pair_name,
                    "Puntos": points.get(pair["pair_id"], 0)
                })
            
            df_ranking = pd.DataFrame(ranking_data).sort_values(by="Puntos", ascending=False).reset_index(drop=True)
            df_ranking.index = df_ranking.index + 1
            st.dataframe(df_ranking)
            
            num_qualifiers = tournament_data["qualifiers_per_group"]
            st.info(f"Los primeros {num_qualifiers} clasificados pasarán a los playoffs.")

    with tab3:
        st.subheader("Cuadro de Playoffs")
        if not tournament_data.get("playoffs"):
            st.info("El cuadro de playoffs no ha sido generado aún.")
            return

        playoff_rounds = tournament_data["playoffs"]
        for round_name, matches in playoff_rounds.items():
            st.markdown(f"### {round_name}")
            if not matches:
                st.info("Esperando resultados de la ronda anterior.")
                continue

            for match in matches:
                pair1_name = get_player_name(tournament_data, match["pair1_id"])
                pair2_name = get_player_name(tournament_data, match["pair2_id"])
                winner_name = get_player_name(tournament_data, match["winner_id"]) if match["winner_id"] else "Pendiente"
                
                if round_name == "Final" and match["winner_id"]:
                    st.markdown(f"🏆 ¡**{winner_name}** es el campeón! 🏆")
                
                st.markdown(f"- **{pair1_name}** vs **{pair2_name}** | Ganador: **{winner_name}**")

def main():
    if not USERS_FILE.exists():
        load_users()

    params = st.query_params
    mode = params.get("mode", [None])[0]
    _tid = params.get("tid", [None])[0]

    if mode == "public" and _tid:
        viewer_tournament(_tid, public=True)
        st.caption("iAPPs Pádel — v3.3.28")
        return
        
    if not st.session_state.get("auth_user"):
        inject_global_layout("No autenticado")
        login_form()
        st.caption("iAPPs Pádel — v3.3.28")
        return

    user = st.session_state["auth_user"]
    
    user_text = f"Usuario: <b>{user['username']}</b> &nbsp;|&nbsp; Rol: <code>{user['role']}</code> &nbsp;&nbsp;<a href='#' onclick='window.location.reload()'>Cerrar sesión</a>"
    inject_global_layout(user_text)

    if user["role"] == "SUPER_ADMIN":
        super_admin_panel()
    elif user["role"] == "TOURNAMENT_ADMIN":
        admin_dashboard(user)
    elif user["role"] == "VIEWER":
        st.info("Modo solo lectura. Puedes ver los torneos de tu administrador asignado.")
        admin = get_user(user["assigned_admin"])
        if admin:
            st.session_state.current_tid = st.selectbox(
                "Torneo", 
                [t["tournament_id"] for t in load_index_for_admin(admin["username"])],
                format_func=lambda tid: load_tournament(tid)["name"]
            )
            viewer_tournament(st.session_state.current_tid)
        else:
            st.warning("No tienes un administrador de torneo asignado.")

if __name__ == "__main__":
    main()
