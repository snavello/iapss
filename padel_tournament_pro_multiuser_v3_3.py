# app.py — v3.3.28
# - Fix `KeyError: 'admin'` by checking for the key's existence before accessing it.
# - This ensures robustness when handling older or corrupted tournament index files.

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
DARK_GREY   = "#262730"
LIGHT_GREY  = "#8D8D8D"

def inject_global_layout(text=""):
    """
    Inyecta el layout global con el logo y el texto de estado.
    """
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
            @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');
            
            html, body, [class*="st-"] {{
                font-family: 'Roboto', sans-serif;
            }}
            .st-emotion-cache-18ni7ap {{
                background-color: transparent !important;
            }}
            .st-emotion-cache-j7qwjs {{
                background-color: transparent !important;
                border-bottom: 2px solid {LIME_GREEN};
                padding-bottom: 5px;
            }}
            .st-emotion-cache-1j0r504 {{
                font-size: 1.5rem;
                font-weight: bold;
                color: {PRIMARY_BLUE};
                text-align: center;
                margin-top: -10px;
                margin-bottom: 10px;
            }}
            .st-emotion-cache-q8sso8 {{
                color: {PRIMARY_BLUE};
            }}
            .st-emotion-cache-p2wz25, .st-emotion-cache-17l1q3e {{
                font-weight: bold;
                color: {PRIMARY_BLUE} !important;
            }}
            .st-emotion-cache-1ae431c, .st-emotion-cache-7ym5gk {{
                border: 1px solid {PRIMARY_BLUE};
                border-radius: 15px;
                padding: 10px;
                background-color: rgba(13, 71, 161, 0.05);
            }}
            .st-emotion-cache-z59b8q {{
                font-size: 1rem;
            }}
            .st-emotion-cache-1ghy2tt {{
                color: {DARK_BLUE} !important;
            }}
            .st-emotion-cache-ocqbe5 {{
                border-color: {PRIMARY_BLUE};
            }}
            .st-emotion-cache-1h61j49 {{
                background-color: {PRIMARY_BLUE};
                color: white;
            }}
            .st-emotion-cache-j4a070, .st-emotion-cache-17l1q3e, .st-emotion-cache-qg0a5a {{
                color: {LIME_GREEN} !important;
                font-weight: bold;
            }}
            
            /* Custom styles for tables */
            .st-emotion-cache-13ln4jf p {{
                margin-bottom: 0.1rem;
            }}
            .st-emotion-cache-13ln4jf {{
                background-color: #f0f2f6; /* Light gray background for tables */
                padding: 10px;
                border-radius: 10px;
            }}

            /* Match scores */
            .match-score {{
                font-size: 1.1em;
                font-weight: bold;
                color: {PRIMARY_BLUE};
            }}
            
            .st-emotion-cache-1avf059 {{
                color: {DARK_BLUE};
                font-size: 1.5rem;
                font-weight: bold;
            }}
            .st-emotion-cache-1g81u1p {{
                font-size: 1rem;
                color: {DARK_GREY};
            }}
            .st-emotion-cache-1r6y1l0 {{
                border-color: {LIME_GREEN} !important;
            }}
            
            /* New styles for tournament table */
            .tournament-table th {{
                background-color: {DARK_GREY};
                color: white;
                font-weight: bold;
                text-align: center;
            }}
            .tournament-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .tournament-table tr:hover {{
                background-color: #f1f1f1;
            }}
            .tournament-table td {{
                vertical-align: middle;
            }}

            .champion-row td {{
                background-color: #ffeb3b !important; /* Amber for champion highlight */
                font-weight: bold;
            }}
            
            .st-emotion-cache-1ghy2tt p {{
                font-size: 1.25rem;
                font-weight: bold;
                color: {DARK_BLUE};
            }}
        </style>
        <div style="
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            padding-bottom: 10px; 
            border-bottom: 2px solid {LIME_GREEN};
        ">
            <h1 style="color: {PRIMARY_BLUE}; margin: 0; font-size: 2em;">
                <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNzUiIGhlaWdodD0iMTAwIiB2aWV3Qm94PSIwIDAgMTc1IDEwMCI+CiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgcj0iNTAiIGZpbGw9IiNBMUVBMDAiLz4KICA8Y2lyY2xlIGN4PSIxMjUiIGN5PSI1MCIgcj0iNTAiIGZpbGw9IiMwRDQ3QTEiLz4KPC9zdmc+" alt="iAPPs Padel Logo" style="height: 50px; margin-right: 15px;"/>
                iAPPs Pádel
            </h1>
            <div style="font-size: 0.9em; color: {DARK_BLUE}; font-weight: bold;">
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ====== Persistencia local ======
DATA_DIR = Path("data")
APP_USERS_FILE = DATA_DIR / "app_users.json"
TOURNAMENT_INDEX_FILE = DATA_DIR / "tournament_index.json"

# Helper functions for persistence
def save_data(data: Dict, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_data(file_path: Path) -> Dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_user(username: str) -> Optional[Dict]:
    users = load_data(APP_USERS_FILE)
    return users.get(username)

def get_tournament_data(tid: str) -> Optional[Dict]:
    return load_data(DATA_DIR / f"{tid}.json")

def save_tournament_data(tid: str, data: Dict):
    save_data(data, DATA_DIR / f"{tid}.json")

def load_index() -> List[Dict]:
    return load_data(TOURNAMENT_INDEX_FILE).get("tournaments", [])

def save_index(index: List[Dict]):
    save_data({"tournaments": index}, TOURNAMENT_INDEX_FILE)
    
def delete_tournament_data(tid: str):
    file_path = DATA_DIR / f"{tid}.json"
    if file_path.exists():
        file_path.unlink()

def load_index_for_admin(admin_username: str) -> List[Dict]:
    all_tourns = load_index()
    # FIX: Use a safe check to prevent KeyError
    return [t for t in all_tourns if "admin" in t and t["admin"] == admin_username]

# ====== Autenticación ======
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_initial_users():
    if not APP_USERS_FILE.exists():
        initial_users = {
            "admin": {"password": hash_password("admin"), "role": "SUPER_ADMIN"},
        }
        save_data(initial_users, APP_USERS_FILE)

def authenticate(username: str, password: str) -> Optional[Dict]:
    users = load_data(APP_USERS_FILE)
    if username in users and users[username]["password"] == hash_password(password):
        return {"username": username, "role": users[username]["role"], "assigned_admin": users[username].get("assigned_admin")}
    return None

def login_form():
    st.subheader("Acceso")
    with st.form("login_form"):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submit_button = st.form_submit_button("Ingresar")
        if submit_button:
            user = authenticate(username, password)
            if user:
                st.session_state["auth_user"] = user
                st.success(f"Bienvenido, {user['username']}!")
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

# ====== Generación de Torneo ======
def create_tournament_struct(name: str, admin_username: str, pairs: List[List[str]], use_seeding: bool) -> Dict:
    num_pairs = len(pairs)
    if num_pairs % 2 != 0:
        st.error("Número de parejas impar. Se necesita un número par.")
        return None

    if use_seeding and num_pairs % 4 != 0:
        st.warning("Para usar cabezas de serie, el número de parejas debe ser un múltiplo de 4. Se deshabilitará la opción.")
        use_seeding = False

    tourn_id = str(uuid.uuid4())
    tournament_data = {
        "tournament_id": tourn_id,
        "name": name,
        "admin": admin_username,
        "pairs": pairs,
        "teams": {},
        "groups": [],
        "playoffs": {},
        "use_seeding": use_seeding,
        "status": "initial"
    }

    teams = {}
    for i, pair in enumerate(pairs):
        team_id = f"t{i+1}"
        teams[team_id] = {"pair": pair, "group": None, "points": 0, "set_wins": 0, "set_losses": 0, "game_wins": 0, "game_losses": 0}
    tournament_data["teams"] = teams

    num_groups = num_pairs // 4
    groups = [{"teams": [], "matches": []} for _ in range(num_groups)]
    
    if use_seeding:
        # Sort pairs by seed
        sorted_pairs = sorted(pairs, key=lambda p: (int(p[2]) if len(p) > 2 and p[2].isdigit() else float('inf')))
        seeded_pairs = sorted_pairs[:num_groups]
        unseeded_pairs = sorted_pairs[num_groups:]
        random.shuffle(unseeded_pairs)

        # Distribute seeded pairs, one per group
        for i, pair in enumerate(seeded_pairs):
            team_id = [k for k, v in teams.items() if v['pair'] == pair][0]
            teams[team_id]['group'] = i + 1
            groups[i]['teams'].append(team_id)

        # Distribute remaining pairs
        group_idx = 0
        for pair in unseeded_pairs:
            team_id = [k for k, v in teams.items() if v['pair'] == pair][0]
            teams[team_id]['group'] = group_idx + 1
            groups[group_idx]['teams'].append(team_id)
            group_idx = (group_idx + 1) % num_groups

    else:
        # Old random distribution
        team_ids = list(teams.keys())
        random.shuffle(team_ids)
        for i, team_id in enumerate(team_ids):
            group_idx = i % num_groups
            teams[team_id]['group'] = group_idx + 1
            groups[group_idx]['teams'].append(team_id)

    # Generate matches for each group
    for group in groups:
        group_teams = group["teams"]
        group_matches = []
        for pair1, pair2 in combinations(group_teams, 2):
            group_matches.append({"team1": pair1, "team2": pair2, "score": []})
        group["matches"] = group_matches
    
    tournament_data["groups"] = groups
    tournament_data["status"] = "groups_created"

    return tournament_data

def generate_playoffs(tournament_data: Dict, num_qualifiers: int = 4):
    qualifiers = []
    
    # Calculate group standings
    group_standings = []
    for group_idx, group in enumerate(tournament_data["groups"]):
        df_group = pd.DataFrame([tournament_data["teams"][t] for t in group["teams"]])
        df_group['team_id'] = group["teams"]
        df_group.set_index('team_id', inplace=True)
        df_group.sort_values(by=["points", "set_wins", "game_wins"], ascending=False, inplace=True)
        group_standings.append(df_group)

    # Determine qualifiers
    if num_qualifiers == 2: # 2 to Final
        for df in group_standings:
            qualifiers.append(df.index[0])
    elif num_qualifiers == 4: # 4 to Semi-Final
        for df in group_standings:
            qualifiers.append(df.index[0])
        for df in group_standings:
            qualifiers.append(df.index[1])
    elif num_qualifiers == 8: # 8 to Quarter-Final
        for df in group_standings:
            qualifiers.append(df.index[0])
        for df in group_standings:
            qualifiers.append(df.index[1])
        for df in group_standings:
            qualifiers.append(df.index[2])
        for df in group_standings:
            qualifiers.append(df.index[3])
    else:
        st.error(f"Número de clasificados no soportado: {num_qualifiers}")
        return tournament_data

    # Ensure we have enough qualifiers
    if len(qualifiers) < num_qualifiers:
        st.warning("No hay suficientes clasificados para generar el cuadro de playoffs completo.")
        return tournament_data

    # Shuffle qualifiers for fair bracket
    random.shuffle(qualifiers)

    # Generate playoff structure
    playoffs = {"QF": [], "SF": [], "FN": []}
    
    if num_qualifiers == 8:
        # Quarter-Finals (QF)
        for i in range(4):
            match = {"team1": qualifiers[i*2], "team2": qualifiers[i*2 + 1], "score": [], "winner": None}
            playoffs["QF"].append(match)
        
        # Semi-Finals (SF)
        for i in range(2):
            match = {"team1": None, "team2": None, "score": [], "winner": None}
            playoffs["SF"].append(match)
            
        # Final (FN)
        playoffs["FN"].append({"team1": None, "team2": None, "score": [], "winner": None})

    elif num_qualifiers == 4:
        # Semi-Finals (SF)
        for i in range(2):
            match = {"team1": qualifiers[i*2], "team2": qualifiers[i*2 + 1], "score": [], "winner": None}
            playoffs["SF"].append(match)
            
        # Final (FN)
        playoffs["FN"].append({"team1": None, "team2": None, "score": [], "winner": None})

    elif num_qualifiers == 2:
        # Final (FN)
        playoffs["FN"].append({"team1": qualifiers[0], "team2": qualifiers[1], "score": [], "winner": None})
    
    tournament_data["playoffs"] = playoffs
    tournament_data["status"] = "playoffs_created"
    save_tournament_data(tournament_data["tournament_id"], tournament_data)
    
    return tournament_data

def calculate_group_points(score: List[int], current_score: List[int]) -> Tuple[int, int, int, int]:
    new_wins, new_losses, new_games_w, new_games_l = 0, 0, 0, 0
    if not score:
        return 0, 0, 0, 0

    if not current_score:
        current_score = [0, 0]

    set_diff = score[0] - score[1]
    
    if set_diff > 0:
        new_wins += 1
    elif set_diff < 0:
        new_losses += 1
    
    new_games_w += score[0]
    new_games_l += score[1]
    
    return new_wins, new_losses, new_games_w, new_games_l
    
def get_winner_from_score(score: List[int]) -> Optional[int]:
    """
    Determina el ganador de un partido de playoff basado en el puntaje.
    Retorna 1 para team1, 2 para team2, None si no hay ganador.
    """
    if len(score) < 2:
        return None
    
    # Simple best of 3 sets logic
    team1_sets = score[0]
    team2_sets = score[1]

    if team1_sets > team2_sets:
        return 1
    elif team2_sets > team1_sets:
        return 2
    
    return None

def get_playoff_winner(match: Dict) -> Optional[str]:
    score = match["score"]
    winner_idx = get_winner_from_score(score)
    if winner_idx == 1:
        return match["team1"]
    elif winner_idx == 2:
        return match["team2"]
    return None

def update_tournament_status(tournament_data: Dict):
    if not tournament_data.get("groups"):
        tournament_data["status"] = "initial"
        return
        
    all_group_matches_played = True
    for group in tournament_data["groups"]:
        for match in group["matches"]:
            if not match["score"]:
                all_group_matches_played = False
                break
        if not all_group_matches_played:
            break
            
    if all_group_matches_played:
        if not tournament_data.get("playoffs") or not tournament_data["playoffs"].get("FN"):
            # Playoffs not yet created, update to groups_completed
            tournament_data["status"] = "groups_completed"
            return
        
        playoffs = tournament_data["playoffs"]
        champions = []

        # Check for champion in Final
        final_match = playoffs["FN"][0]
        if final_match["winner"]:
            champions.append(final_match["winner"])
        
        if len(champions) > 0:
            tournament_data["status"] = "completed"
        elif any(match["score"] for match in playoffs["FN"]):
            tournament_data["status"] = "final_in_progress"
        elif playoffs.get("SF") and any(match["score"] for match in playoffs["SF"]):
            tournament_data["status"] = "semi_final_in_progress"
        elif playoffs.get("QF") and any(match["score"] for match in playoffs["QF"]):
            tournament_data["status"] = "quarter_final_in_progress"
        elif tournament_data["status"] == "groups_completed":
             # Already in this state, do nothing
             pass
        else:
            tournament_data["status"] = "playoffs_created"
            
    else:
        tournament_data["status"] = "groups_created"

def get_champion(tournament_data: Dict) -> Optional[List[str]]:
    playoffs = tournament_data.get("playoffs", {})
    if playoffs.get("FN") and playoffs["FN"][0].get("winner"):
        winner_id = playoffs["FN"][0]["winner"]
        return tournament_data["teams"][winner_id]["pair"]
    return None

def generate_pdf(tournament_data: Dict) -> Optional[BytesIO]:
    if not REPORTLAB_OK:
        st.warning("No se pudo importar reportlab. La función de PDF está deshabilitada.")
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"<b>Torneo de Pádel: {tournament_data['name']}</b>", styles['Title']))
    story.append(Spacer(1, 0.5 * cm))

    # Groups
    story.append(Paragraph("<b>Resultados de la Fase de Grupos</b>", styles['H2']))
    story.append(Spacer(1, 0.2 * cm))

    for i, group in enumerate(tournament_data["groups"]):
        story.append(Paragraph(f"<b>Grupo {i+1}</b>", styles['h3']))
        
        # Table of matches
        match_data = [["Equipo 1", "Equipo 2", "Resultado"]]
        for match in group["matches"]:
            team1_name = ' & '.join(tournament_data['teams'][match['team1']]['pair'])
            team2_name = ' & '.join(tournament_data['teams'][match['team2']]['pair'])
            score_text = f"{match['score'][0]}-{match['score'][1]}" if match['score'] else "Pendiente"
            match_data.append([team1_name, team2_name, score_text])
        
        table = Table(match_data, colWidths=[6*cm, 6*cm, 4*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.5 * cm))

    # Playoffs
    if tournament_data.get("playoffs"):
        story.append(Paragraph("<b>Cuadro de Playoffs</b>", styles['H2']))
        story.append(Spacer(1, 0.2 * cm))
        for stage, matches in tournament_data["playoffs"].items():
            story.append(Paragraph(f"<b>{stage}</b>", styles['h3']))
            
            match_data = [["Equipo 1", "Equipo 2", "Resultado", "Ganador"]]
            for match in matches:
                team1_name = ' & '.join(tournament_data['teams'][match['team1']]['pair']) if match.get("team1") else "TBD"
                team2_name = ' & '.join(tournament_data['teams'][match['team2']]['pair']) if match.get("team2") else "TBD"
                score_text = f"{match['score'][0]}-{match['score'][1]}" if match['score'] else "Pendiente"
                winner_name = ' & '.join(tournament_data['teams'][match['winner']]['pair']) if match.get("winner") else "Pendiente"
                match_data.append([team1_name, team2_name, score_text, winner_name])
            
            table = Table(match_data, colWidths=[4*cm, 4*cm, 3*cm, 5*cm])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(table)
            story.append(Spacer(1, 0.5 * cm))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ====== Vistas ======
def super_admin_panel():
    st.title("Panel de Super Administrador")
    tab1, tab2 = st.tabs(["Administrar Usuarios", "Administrar Torneos"])

    with tab1:
        st.subheader("Crear y Eliminar Usuarios")
        with st.form("user_form"):
            new_username = st.text_input("Usuario")
            new_password = st.text_input("Contraseña", type="password")
            new_role = st.selectbox("Rol", ["TOURNAMENT_ADMIN", "VIEWER", "SUPER_ADMIN"])
            assigned_admin = None
            if new_role == "VIEWER":
                all_admins = [u for u, data in load_data(APP_USERS_FILE).items() if data.get("role") == "TOURNAMENT_ADMIN"]
                assigned_admin = st.selectbox("Asignar a Admin", all_admins)
            
            col1, col2 = st.columns(2)
            with col1:
                create_button = st.form_submit_button("Crear Usuario")
            with col2:
                delete_button = st.form_submit_button("Eliminar Usuario")
            
            if create_button:
                if new_username and new_password:
                    users = load_data(APP_USERS_FILE)
                    if new_username in users:
                        st.error("El usuario ya existe.")
                    else:
                        user_data = {"password": hash_password(new_password), "role": new_role}
                        if assigned_admin:
                            user_data["assigned_admin"] = assigned_admin
                        users[new_username] = user_data
                        save_data(users, APP_USERS_FILE)
                        st.success(f"Usuario {new_username} creado con rol {new_role}.")
                        st.rerun()
                else:
                    st.error("Por favor, ingrese un usuario y contraseña.")
            
            if delete_button:
                if new_username:
                    users = load_data(APP_USERS_FILE)
                    if new_username in users:
                        del users[new_username]
                        save_data(users, APP_USERS_FILE)
                        st.success(f"Usuario {new_username} eliminado.")
                        st.rerun()
                    else:
                        st.error("El usuario no existe.")

        st.subheader("Lista de Usuarios")
        users = load_data(APP_USERS_FILE)
        df_users = pd.DataFrame([{"Usuario": k, "Rol": v['role'], "Asignado a Admin": v.get("assigned_admin", "N/A")} for k, v in users.items()])
        st.dataframe(df_users, hide_index=True)

    with tab2:
        st.subheader("Eliminar Torneos")
        all_tournaments = load_index()
        tourn_ids = [t["tournament_id"] for t in all_tournaments]
        tourn_names = {t["tournament_id"]: t["name"] for t in all_tournaments}
        
        tid_to_delete = st.selectbox("Seleccionar torneo para eliminar", tourn_ids, format_func=lambda tid: tourn_names.get(tid, tid))
        if st.button("Eliminar Torneo Seleccionado"):
            if tid_to_delete:
                tourn_file_path = DATA_DIR / f"{tid_to_delete}.json"
                if tourn_file_path.exists():
                    tourn_file_path.unlink()
                
                new_index = [t for t in all_tournaments if t["tournament_id"] != tid_to_delete]
                save_index(new_index)
                st.success(f"Torneo {tourn_names.get(tid_to_delete, tid_to_delete)} eliminado.")
                st.rerun()
            else:
                st.warning("Seleccione un torneo para eliminar.")


def admin_dashboard(user: Dict):
    st.title("Panel de Administrador de Torneos")
    
    tournaments = load_index_for_admin(user["username"])
    if not tournaments:
        st.info("No tienes torneos. Crea uno para empezar.")
    
    tab_titles = ["Crear Torneo"] + [t["name"] for t in tournaments]
    tabs = st.tabs(tab_titles)
    
    # "Crear Torneo" tab
    with tabs[0]:
        st.subheader("Nuevo Torneo")
        with st.form("new_tournament_form"):
            tournament_name = st.text_input("Nombre del Torneo")
            pairs_text = st.text_area("Lista de Parejas (una por línea, separados por '&')")
            use_seeding = st.checkbox("Usar cabezas de serie (ej: Jugador A & Jugador B | 1)")
            
            submit_button = st.form_submit_button("Crear Torneo")
            
            if submit_button:
                if not tournament_name:
                    st.error("El nombre del torneo es obligatorio.")
                elif not pairs_text:
                    st.error("Debe ingresar las parejas.")
                else:
                    pairs_list = [p.strip().split('&') for p in pairs_text.split('\n') if p.strip()]
                    
                    # Normalize pairs and remove empty spaces
                    normalized_pairs = []
                    for pair in pairs_list:
                        # Normalize the pair name, removing whitespace
                        normalized_pair = [p.strip() for p in pair if p.strip()]
                        # Check for seeding info
                        seeding = None
                        if len(normalized_pair) > 2 and normalized_pair[-1].isdigit():
                            seeding = int(normalized_pair.pop())
                        
                        if len(normalized_pair) != 2:
                            st.error(f"Formato de pareja incorrecto: '{' & '.join(pair)}'. Cada pareja debe tener exactamente dos jugadores.")
                            return
                        
                        final_pair = normalized_pair
                        if seeding is not None:
                            final_pair.append(str(seeding))
                            
                        normalized_pairs.append(final_pair)

                    if len(normalized_pairs) % 2 != 0:
                        st.error("El número de parejas debe ser par.")
                        return

                    tournament_data = create_tournament_struct(tournament_name, user["username"], normalized_pairs, use_seeding)
                    if tournament_data:
                        save_tournament_data(tournament_data["tournament_id"], tournament_data)
                        
                        # Update index
                        index = load_index()
                        index.append({"tournament_id": tournament_data["tournament_id"], "name": tournament_name, "admin": user["username"]})
                        save_index(index)
                        
                        st.success(f"Torneo '{tournament_name}' creado exitosamente!")
                        st.session_state.current_tid = tournament_data["tournament_id"]
                        st.rerun()

    # Tournament tabs
    for i, tourn_info in enumerate(tournaments):
        with tabs[i + 1]:
            tournament_id = tourn_info["tournament_id"]
            viewer_tournament(tournament_id, public=False)


def viewer_tournament(tid: str, public: bool):
    tournament_data = get_tournament_data(tid)
    if not tournament_data:
        st.error("Torneo no encontrado.")
        return

    st.title(tournament_data["name"])

    st.subheader(f"Estado: {tournament_data['status'].replace('_', ' ').capitalize()}")

    if not public:
        # Show public link and persistence options only for admins
        public_url = get_public_url(tid)
        st.markdown(
            f"""
            <div style="
                border: 1px solid {LIGHT_GREY};
                padding: 10px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 20px;
            ">
                <span style="font-weight: bold;">Link Público:</span>
                <span style="font-family: monospace; flex-grow: 1; margin: 0 10px; overflow-wrap: break-word;">{public_url}</span>
                <button
                    onclick="navigator.clipboard.writeText('{public_url}');"
                    style="
                        background-color: transparent;
                        border: none;
                        color: {PRIMARY_BLUE};
                        cursor: pointer;
                        font-size: 1.5rem;
                    "
                    title="Copiar al portapapeles"
                >
                    <i class="fa-regular fa-clipboard"></i>
                </button>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("Persistencia y Gestión del Torneo"):
            st.subheader("Importar/Exportar Datos")
            col1, col2 = st.columns(2)
            
            with col1:
                json_data = json.dumps(tournament_data, indent=4)
                st.download_button(
                    label="Descargar datos del torneo (.json)",
                    data=json_data,
                    file_name=f"{tournament_data['name'].replace(' ', '_')}.json",
                    mime="application/json"
                )
            
            with col2:
                uploaded_file = st.file_uploader("Cargar datos del torneo (.json)", type="json")
                if uploaded_file:
                    try:
                        new_data = json.load(uploaded_file)
                        st.session_state.current_tid = new_data["tournament_id"]
                        save_tournament_data(new_data["tournament_id"], new_data)
                        st.success("Datos del torneo cargados exitosamente. Recargando...")
                        st.rerun()
                    except (json.JSONDecodeError, KeyError) as e:
                        st.error(f"Error al cargar el archivo: {e}")

            if REPORTLAB_OK:
                pdf_buffer = generate_pdf(tournament_data)
                if pdf_buffer:
                    st.download_button(
                        label="Descargar PDF",
                        data=pdf_buffer,
                        file_name=f"{tournament_data['name'].replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
            
            st.subheader("Eliminar Torneo")
            if st.button("Eliminar este Torneo", type="primary"):
                index = load_index()
                new_index = [t for t in index if t["tournament_id"] != tid]
                save_index(new_index)
                delete_tournament_data(tid)
                st.success(f"Torneo '{tournament_data['name']}' eliminado.")
                st.session_state.pop("current_tid", None)
                st.rerun()

    # --- Pairs management ---
    if not public and tournament_data.get("status", "initial") == "initial":
        st.subheader("Administrar Parejas")
        col1, col2 = st.columns([1, 1])
        with col1:
            with st.form("add_pair_form"):
                player1 = st.text_input("Jugador 1")
                player2 = st.text_input("Jugador 2")
                seed_number = st.number_input("Cabeza de Serie (opcional)", min_value=1, step=1, value=1)
                add_pair_button = st.form_submit_button("Añadir Pareja")

            if add_pair_button and player1 and player2:
                new_pair = [player1, player2]
                if seed_number > 1:
                    new_pair.append(str(seed_number))
                
                # Check for duplicates
                pair_exists = False
                for p in tournament_data["pairs"]:
                    existing_normalized = set([x.lower() for x in p[:2]])
                    new_normalized = set([x.lower() for x in new_pair[:2]])
                    if existing_normalized == new_normalized:
                        pair_exists = True
                        break
                
                if not pair_exists:
                    tournament_data["pairs"].append(new_pair)
                    save_tournament_data(tid, tournament_data)
                    st.success("Pareja añadida. Recargando...")
                    st.rerun()
                else:
                    st.warning("Esa pareja ya está registrada.")

        with col2:
            st.markdown("### Parejas Registradas")
            pair_list = []
            for i, p in enumerate(tournament_data["pairs"]):
                pair_name = " & ".join(p[:2])
                seed_info = f" ({p[2]})" if len(p) > 2 else ""
                
                delete_button_key = f"delete_pair_{i}"
                if st.button(f"🗑️ {pair_name}{seed_info}", key=delete_button_key):
                    del tournament_data["pairs"][i]
                    save_tournament_data(tid, tournament_data)
                    st.success("Pareja eliminada. Recargando...")
                    st.rerun()
                
    if not public and tournament_data.get("status") == "initial":
        st.subheader("Generar Grupos y Cuadro")
        num_pairs = len(tournament_data["pairs"])
        if num_pairs < 4:
            st.warning("Necesita al menos 4 parejas para generar grupos.")
        else:
            if num_pairs % 2 != 0:
                st.warning("El número de parejas debe ser par para continuar.")
            else:
                use_seeding = st.checkbox("Usar cabezas de serie", tournament_data.get("use_seeding", False))
                num_qualifiers = st.selectbox("Número de clasificados para Playoffs", options=[2, 4, 8], index=1)
                if st.button("Generar Grupos y Playoffs", type="primary"):
                    new_tourn_data = create_tournament_struct(
                        tournament_data["name"], 
                        tournament_data["admin"], 
                        tournament_data["pairs"],
                        use_seeding
                    )
                    if new_tourn_data:
                        # Regenerate playoffs with the correct number of qualifiers
                        generate_playoffs(new_tourn_data, num_qualifiers)
                        save_tournament_data(tid, new_tourn_data)
                        st.success("Grupos y Playoffs generados. ¡Listo para jugar!")
                        st.rerun()
    
    # --- Groups ---
    if tournament_data.get("groups"):
        st.subheader("Fase de Grupos")
        groups = tournament_data["groups"]
        cols_per_row = 3
        group_cols = st.columns(cols_per_row)
        
        for i, group in enumerate(groups):
            with group_cols[i % cols_per_row]:
                st.markdown(f"### Grupo {i+1}")
                st.markdown("---")
                
                # Standings Table
                group_teams_data = []
                for team_id in group["teams"]:
                    team = tournament_data["teams"][team_id]
                    pair_name = " & ".join(team["pair"][:2])
                    group_teams_data.append({
                        "Pareja": pair_name,
                        "Puntos": team["points"],
                        "Sets Ganados": team["set_wins"],
                        "Juegos Ganados": team["game_wins"]
                    })
                
                df_group = pd.DataFrame(group_teams_data)
                df_group.sort_values(by=["Puntos", "Sets Ganados", "Juegos Ganados"], ascending=False, inplace=True)
                
                st.markdown("#### Tabla de Posiciones")
                st.dataframe(
                    df_group,
                    hide_index=True,
                    use_container_width=True
                )
                
                # Display matches
                st.markdown("#### Partidos")
                for match_idx, match in enumerate(group["matches"]):
                    team1_name = " & ".join(tournament_data["teams"][match["team1"]]["pair"][:2])
                    team2_name = " & ".join(tournament_data["teams"][match["team2"]]["pair"][:2])
                    
                    st.markdown(f"**{team1_name} vs {team2_name}**")
                    
                    current_score = match.get("score", [])
                    
                    if not public:
                        with st.form(f"score_form_g{i}_m{match_idx}"):
                            col_score1, col_score2 = st.columns(2)
                            with col_score1:
                                score1 = st.number_input(f"Sets {team1_name}", min_value=0, value=current_score[0] if current_score else 0)
                            with col_score2:
                                score2 = st.number_input(f"Sets {team2_name}", min_value=0, value=current_score[1] if current_score else 0)
                            
                            submit_score = st.form_submit_button("Guardar Resultado")
                            if submit_score:
                                if score1 == score2:
                                    st.warning("Un partido no puede ser un empate.")
                                else:
                                    # Update points for old score (if exists)
                                    old_score = match["score"]
                                    if old_score:
                                        old_wins1, old_losses1, old_games_w1, old_games_l1 = calculate_group_points(old_score)
                                        tournament_data["teams"][match["team1"]]["points"] -= (1 if old_wins1 > 0 else 0)
                                        tournament_data["teams"][match["team2"]]["points"] -= (1 if old_losses1 > 0 else 0)
                                        tournament_data["teams"][match["team1"]]["set_wins"] -= old_wins1
                                        tournament_data["teams"][match["team1"]]["set_losses"] -= old_losses1
                                        tournament_data["teams"][match["team2"]]["set_wins"] -= old_losses1
                                        tournament_data["teams"][match["team2"]]["set_losses"] -= old_wins1
                                        tournament_data["teams"][match["team1"]]["game_wins"] -= old_games_w1
                                        tournament_data["teams"][match["team1"]]["game_losses"] -= old_games_l1
                                        tournament_data["teams"][match["team2"]]["game_wins"] -= old_games_l1
                                        tournament_data["teams"][match["team2"]]["game_losses"] -= old_games_w1

                                    # Update with new score
                                    match["score"] = [score1, score2]
                                    new_wins1, new_losses1, new_games_w1, new_games_l1 = calculate_group_points([score1, score2])

                                    tournament_data["teams"][match["team1"]]["points"] += (3 if new_wins1 > 0 else 0) + (1 if new_losses1 > 0 else 0)
                                    tournament_data["teams"][match["team2"]]["points"] += (3 if new_losses1 > 0 else 0) + (1 if new_wins1 > 0 else 0)
                                    
                                    tournament_data["teams"][match["team1"]]["set_wins"] += new_wins1
                                    tournament_data["teams"][match["team1"]]["set_losses"] += new_losses1
                                    tournament_data["teams"][match["team2"]]["set_wins"] += new_losses1
                                    tournament_data["teams"][match["team2"]]["set_losses"] += new_wins1
                                    
                                    tournament_data["teams"][match["team1"]]["game_wins"] += new_games_w1
                                    tournament_data["teams"][match["team1"]]["game_losses"] += new_games_l1
                                    tournament_data["teams"][match["team2"]]["game_wins"] += new_games_l1
                                    tournament_data["teams"][match["team2"]]["game_losses"] += new_games_w1
                                    
                                    update_tournament_status(tournament_data)
                                    save_tournament_data(tid, tournament_data)
                                    st.success("Resultado guardado. Recargando...")
                                    st.rerun()
                    else:
                        if current_score:
                            st.info(f"Resultado: **{current_score[0]} - {current_score[1]}**")
                        else:
                            st.warning("Resultado pendiente")

    # --- Playoffs ---
    if tournament_data.get("playoffs") and tournament_data.get("groups"):
        st.subheader("Fase de Playoffs")
        
        # Determine number of qualifiers
        num_qualifiers = len(tournament_data["playoffs"]["FN"]) * 2 if tournament_data["playoffs"].get("SF") else 2

        if not public and tournament_data["status"] == "groups_completed":
            if st.button("Generar cuadro de playoffs", type="primary"):
                generate_playoffs(tournament_data, num_qualifiers)
                st.success("Cuadro de playoffs generado.")
                st.rerun()

        # Display qualifiers table
        st.markdown("#### Clasificados a Playoffs")
        qualifiers = []
        for group_idx, group in enumerate(tournament_data["groups"]):
            df_group = pd.DataFrame([tournament_data["teams"][t] for t in group["teams"]])
            df_group['team_id'] = group["teams"]
            df_group.set_index('team_id', inplace=True)
            df_group.sort_values(by=["points", "set_wins", "game_wins"], ascending=False, inplace=True)
            
            # Use `head(num_qualifiers)` to get top qualifiers
            num_to_qualify = num_qualifiers // len(tournament_data["groups"])
            
            for rank, team_id in enumerate(df_group.index.tolist()):
                if rank < num_to_qualify:
                    qualifiers.append({
                        "Pareja": " & ".join(tournament_data["teams"][team_id]["pair"][:2]),
                        "Grupo": group_idx + 1,
                        "Ranking": rank + 1,
                        "Clasificado": "✅"
                    })
        
        df_qualifiers = pd.DataFrame(qualifiers)
        if not df_qualifiers.empty:
            df_qualifiers.sort_values(by=["Grupo", "Ranking"], ascending=True, inplace=True)
            st.dataframe(df_qualifiers, hide_index=True, use_container_width=True)

        playoffs = tournament_data["playoffs"]
        
        for stage, matches in playoffs.items():
            st.markdown(f"### {stage.replace('QF', 'Cuartos de Final').replace('SF', 'Semifinal').replace('FN', 'Final')}")
            
            match_cols = st.columns(len(matches))
            for match_idx, match in enumerate(matches):
                with match_cols[match_idx]:
                    team1_id = match.get("team1")
                    team2_id = match.get("team2")
                    
                    team1_name = " & ".join(tournament_data["teams"][team1_id]["pair"][:2]) if team1_id else "Por Definir"
                    team2_name = " & ".join(tournament_data["teams"][team2_id]["pair"][:2]) if team2_id else "Por Definir"

                    st.markdown(f"**{team1_name} vs {team2_name}**")
                    
                    current_score = match.get("score", [])
                    
                    if not public and team1_id and team2_id:
                        with st.form(f"playoff_form_{stage}_{match_idx}"):
                            col_p1, col_p2 = st.columns(2)
                            with col_p1:
                                score1 = st.number_input(f"Sets {team1_name}", min_value=0, value=current_score[0] if current_score else 0)
                            with col_p2:
                                score2 = st.number_input(f"Sets {team2_name}", min_value=0, value=current_score[1] if current_score else 0)
                            
                            submit_button = st.form_submit_button("Guardar Resultado")
                            if submit_button:
                                if score1 == score2:
                                    st.warning("Un partido de playoff no puede ser un empate.")
                                else:
                                    match["score"] = [score1, score2]
                                    winner_id = get_playoff_winner(match)
                                    if winner_id == match["team1"]:
                                        match["winner"] = match["team1"]
                                    elif winner_id == match["team2"]:
                                        match["winner"] = match["team2"]
                                    
                                    # Advance winner to next round
                                    if stage == "QF":
                                        next_match_idx = match_idx // 2
                                        if match_idx % 2 == 0:
                                            playoffs["SF"][next_match_idx]["team1"] = match["winner"]
                                        else:
                                            playoffs["SF"][next_match_idx]["team2"] = match["winner"]
                                    elif stage == "SF":
                                        next_match_idx = match_idx // 2
                                        if match_idx % 2 == 0:
                                            playoffs["FN"][next_match_idx]["team1"] = match["winner"]
                                        else:
                                            playoffs["FN"][next_match_idx]["team2"] = match["winner"]
                                    
                                    update_tournament_status(tournament_data)
                                    save_tournament_data(tid, tournament_data)
                                    st.success("Resultado guardado. Recargando...")
                                    st.rerun()
                    else:
                        if current_score:
                            st.info(f"Resultado: **{current_score[0]} - {current_score[1]}**")
                        else:
                            st.warning("Resultado pendiente")

        # Display champion
        champion = get_champion(tournament_data)
        if champion:
            st.markdown(f"## 🎉 ¡Felicidades a los campeones! 🎉")
            st.markdown(f"<h1 style='text-align: center; color: {LIME_GREEN};'>{' & '.join(champion)}</h1>", unsafe_allow_html=True)
            
            
def get_public_url(tid: str) -> str:
    current_url = urlparse(st.experimental_get_query_params.get('url', [''])[0])
    query_params = dict(parse_qsl(current_url.query))
    
    query_params.pop('mode', None)
    query_params.pop('tid', None)
    
    query_params['mode'] = 'public'
    query_params['tid'] = tid
    
    return urlunparse(current_url._replace(query=requests.compat.urlencode(query_params)))

# ====== Main App ======
def main():
    create_initial_users()
    
    params = st.query_params
    mode = params.get("mode", [""])[0]
    _tid = params.get("tid", [""])[0]
    
    if mode == "super":
        if st.session_state.get("auth_user") and st.session_state.auth_user["role"] == "SUPER_ADMIN":
            inject_global_layout("Panel Super Admin")
            super_admin_panel()
        else:
            inject_global_layout("Acceso Denegado")
            st.warning("Acceso denegado. Solo Super Admin.")
            login_form()
        return

    if mode == "public" and _tid:
        inject_global_layout(f"Torneo Público: {_tid}")
        viewer_tournament(_tid, public=True)
        st.caption("iAPPs Pádel — v3.3.28")
        return

    if not st.session_state.get("auth_user"):
        inject_global_layout("No autenticado")
        login_form()
        st.caption("iAPPs Pádel — v3.3.28")
        return

    user = st.session_state["auth_user"]
    
    user_text = f"Usuario: <b>{user['username']}</b> &nbsp;|&nbsp; Rol: <code>{user['role']}</code>"
    inject_global_layout(user_text)

    # Use columns to position logout button
    top = st.columns([4, 1])
    with top[1]:
        if st.button("Cerrar sesión"):
            st.session_state.update({"auth_user": None, "current_tid": None})
            st.rerun()

    if user["role"] == "SUPER_ADMIN":
        super_admin_panel()
    elif user["role"] == "TOURNAMENT_ADMIN":
        admin_dashboard(user)
    elif user["role"] == "VIEWER":
        st.info("Modo solo lectura. Puedes ver los torneos de tu administrador asignado.")
        admin = get_user(user["assigned_admin"])
        if admin:
            # Check if there are tournaments for the assigned admin
            tournaments = load_index_for_admin(admin["username"])
            if tournaments:
                st.session_state.current_tid = st.selectbox(
                    "Torneo", 
                    [t["tournament_id"] for t in tournaments], 
                    format_func=lambda tid: load_tournament(tid)["name"]
                )
                if st.session_state.get("current_tid"):
                    viewer_tournament(st.session_state.current_tid, public=True)
            else:
                st.warning("No hay torneos disponibles para tu administrador.")
        else:
            st.warning("Administrador asignado no encontrado.")

if __name__ == "__main__":
    main()
