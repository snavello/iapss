# padel_tournament_pro_v1_1_1.py  —  v1.1.1
# -----------------------------------------------------------------------------
# Persistencia en GitHub (repo privado):
#   users.json, app_config.json, tournaments/index.json,
#   tournaments/{tid}.json + tournaments/{tid}/snapshots/YYYYMMDD_HHMMSS.json
# Correcciones:
#   • Procesamiento de formularios DENTRO del with st.form(...) en:
#       - Alta manual de parejas
#       - Resultados de grupos
#       - Partidos de Playoffs (KO)
#   • st.rerun() inmediato tras guardado para refrescar tablas/estado
#   • Evitar escritura tardía en st.session_state (errores de Streamlit)
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import requests, base64, hashlib, json, time, uuid
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple
from itertools import combinations
import random

APP_VERSION = "v1.1.1"

# ===========================
# ====== ESTILOS / UI =======
# ===========================
st.set_page_config(page_title=f"iAPPs Pádel · {APP_VERSION}", layout="wide")

PRIMARY_BLUE = "#0D47A1"
LIME_GREEN  = "#AEEA00"
DARK_BLUE   = "#082D63"

st.markdown("""
<style>
.iapps-header-row{
  display:flex;align-items:center;gap:12px;padding:6px 0 4px 0;
  border-bottom:1px solid #e5e7eb;position:sticky;top:0;background:#fff;z-index:20;
}
.iapps-header-logo{max-height:54px;width:auto;height:auto;max-width:22vw;object-fit:contain;}
.iapps-user{font-weight:600;}
.iapps-role{color:#6b7280;margin-left:6px;}
table.zebra tbody tr:nth-child(odd){background:#f9fafb;}
table.dark-header thead th{background:#2f3b52;color:#fff;}
.thin .stNumberInput,.thin .stTextInput,.thin .stSelectbox,.thin .stSlider{max-width:160px!important;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# ====== PERSISTENCIA EN GITHUB (cable) =====
# ==========================================
GITHUB_API = "https://api.github.com"

def _gh_conf():
    cfg = st.secrets.get("github", {})
    for k in ["token", "owner", "repo", "branch"]:
        if k not in cfg or not cfg[k]:
            raise RuntimeError(f"Falta github.{k} en st.secrets")
    return cfg

def _headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

def _repo_base(owner: str, repo: str) -> str:
    return f"{GITHUB_API}/repos/{owner}/{repo}"

class GitHubStore:
    def __init__(self, owner: str, repo: str, branch: str, token: str):
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.token = token
        st.session_state.setdefault("_gh_sha_cache", {})

    @classmethod
    def from_secrets(cls) -> "GitHubStore":
        cfg = _gh_conf()
        return cls(cfg["owner"], cfg["repo"], cfg["branch"], cfg["token"])

    def _contents_url(self, path: str) -> str:
        return f"{_repo_base(self.owner, self.repo)}/contents/{path}"

    def get_file(self, path: str) -> Tuple[Optional[bytes], Optional[str]]:
        url = self._contents_url(path)
        params = {"ref": self.branch}
        r = requests.get(url, headers=_headers(self.token), params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            content_b64 = data.get("content", "")
            sha = data.get("sha", None)
            if data.get("encoding") == "base64" and content_b64:
                raw = base64.b64decode(content_b64)
            else:
                raw = content_b64.encode("utf-8")
            st.session_state["_gh_sha_cache"][path] = sha
            return raw, sha
        elif r.status_code == 404:
            return None, None
        else:
            raise RuntimeError(f"GET {path} falló: {r.status_code} {r.text}")

    def put_file(self, path: str, content: bytes, message: str, sha: Optional[str]=None) -> str:
        url = self._contents_url(path)
        payload = {
            "message": message,
            "content": base64.b64encode(content).decode("ascii"),
            "branch": self.branch
        }
        if sha:
            payload["sha"] = sha
        r = requests.put(url, headers=_headers(self.token), json=payload, timeout=20)
        if r.status_code in (200, 201):
            new_sha = r.json()["content"]["sha"]
            st.session_state["_gh_sha_cache"][path] = new_sha
            return new_sha
        else:
            raise RuntimeError(f"PUT {path} falló: {r.status_code} {r.text}")

    def get_json(self, path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        content, sha = self.get_file(path)
        if content is None:
            return None, None
        try:
            return json.loads(content.decode("utf-8")), sha
        except Exception as e:
            raise RuntimeError(f"JSON inválido en {path}: {e}")

    def put_json(self, path: str, obj: Dict[str, Any], message: str, sha: Optional[str]=None) -> str:
        raw = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        return self.put_file(path, raw, message, sha)

class DataRepo:
    USERS = "users.json"
    APPCFG = "app_config.json"
    INDEX  = "tournaments/index.json"
    TOURN  = "tournaments/{tid}.json"
    SNAP   = "tournaments/{tid}/snapshots/{ts}.json"

    def __init__(self, store: GitHubStore):
        self.store = store

    def load_users(self) -> List[Dict[str, Any]]:
        data, _sha = self.store.get_json(self.USERS)
        if data is None:
            admin_hash = hashlib.sha256("199601".encode("utf-8")).hexdigest()
            default = [{
                "username":"ADMIN", "pin_hash":admin_hash,
                "role":"SUPER_ADMIN", "assigned_admin":None,
                "created_at":datetime.now().isoformat(), "active": True
            }]
            self.store.put_json(self.USERS, default, "bootstrap users.json (ADMIN)")
            return default
        return data

    def save_users(self, users: List[Dict[str, Any]]):
        _, sha = self.store.get_json(self.USERS)
        try:
            self.store.put_json(self.USERS, users, "update users.json", sha=sha)
        except RuntimeError as e:
            raise RuntimeError("Conflicto al guardar usuarios. Recargá la página.") from e

    def load_app_config(self) -> Dict[str, Any]:
        data, _ = self.store.get_json(self.APPCFG)
        if data is None:
            default = {
                "app_logo_url": "https://raw.githubusercontent.com/snavello/iapss/main/1000138052.png",
                "app_base_url": "https://iappspadel.streamlit.app"
            }
            self.store.put_json(self.APPCFG, default, "bootstrap app_config.json")
            return default
        return data

    def save_app_config(self, cfg: Dict[str, Any]):
        _, sha = self.store.get_json(self.APPCFG)
        try:
            self.store.put_json(self.APPCFG, cfg, "update app_config.json", sha=sha)
        except RuntimeError as e:
            raise RuntimeError("Conflicto al guardar app_config. Recargá y reintentá.") from e

    def load_index(self) -> List[Dict[str, Any]]:
        data, _ = self.store.get_json(self.INDEX)
        if data is None:
            empty: List[Dict[str, Any]] = []
            self.store.put_json(self.INDEX, empty, "bootstrap tournaments/index.json")
            return empty
        return data

    def save_index(self, idx: List[Dict[str, Any]]):
        _, sha = self.store.get_json(self.INDEX)
        try:
            self.store.put_json(self.INDEX, idx, "update tournaments/index.json", sha=sha)
        except RuntimeError as e:
            raise RuntimeError("Conflicto al guardar índice de torneos. Recargá y reintentá.") from e

    def tourn_path(self, tid: str) -> str:
        return self.TOURN.format(tid=tid)

    def snap_path(self, tid: str, ts: str) -> str:
        return self.SNAP.format(tid=tid, ts=ts)

    def load_tournament(self, tid: str) -> Dict[str, Any]:
        path = self.tourn_path(tid)
        data, _ = self.store.get_json(path)
        return data or {}

    def save_tournament(self, tid: str, obj: Dict[str, Any], make_snapshot: bool = True):
        path = self.tourn_path(tid)
        _, sha = self.store.get_json(path)
        try:
            self.store.put_json(path, obj, f"update {path}", sha=sha)
        except RuntimeError as e:
            raise RuntimeError("Conflicto al guardar el torneo. Recargá la página y reintentá.") from e
        if make_snapshot:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                self.store.put_json(self.snap_path(tid, ts), obj, f"snapshot {tid} {ts}")
            except Exception:
                pass

def _data_repo():
    if "DATA_REPO" not in st.session_state:
        store = GitHubStore.from_secrets()
        st.session_state["DATA_REPO"] = DataRepo(store)
    return st.session_state["DATA_REPO"]

# ==============================
# ====== HELPERS / BRANDING ====
# ==============================
def now_iso(): return datetime.now().isoformat()
def sha(s:str)->str: return hashlib.sha256(s.encode("utf-8")).hexdigest()

@st.cache_data(show_spinner=False)
def _fetch_image_data_uri(url:str, timeout:float=6.0)->str:
    try:
        if not url: return ""
        r=requests.get(url,timeout=timeout); r.raise_for_status()
        content=r.content; ct=(r.headers.get("Content-Type") or "").lower()
        if "svg" in ct: mime="image/svg+xml"
        elif "jpeg" in ct or "jpg" in ct: mime="image/jpeg"
        elif "webp" in ct: mime="image/webp"
        else: mime="image/png"
        b64=base64.b64encode(content).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""

def _brand_svg(width_px:int=180)->str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" viewBox="0 0 660 200" role="img" aria-label="iAPPs PADEL TOURNAMENT">
  <defs><linearGradient id="g1" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="{PRIMARY_BLUE}" /><stop offset="100%" stop-color="{DARK_BLUE}" /></linearGradient></defs>
  <rect x="0" y="0" width="660" height="200" fill="transparent"/>
  <text x="8" y="65" font-family="Inter, Segoe UI, Roboto, Arial, sans-serif" font-weight="800" font-size="74" fill="url(#g1)" letter-spacing="2">iAPP</text>
  <text x="445" y="65" font-family="Inter, Segoe UI, Roboto, Arial, sans-serif" font-weight="900" font-size="72" fill="{LIME_GREEN}">s</text>
  <text x="8" y="125" font-family="Inter, Segoe UI, Roboto, Arial, sans-serif" font-weight="800" font-size="76" fill="{PRIMARY_BLUE}" letter-spacing="4">PADEL</text>
  <text x="8" y="182" font-family="Inter, Segoe UI, Roboto, Arial, sans-serif" font-weight="700" font-size="58" fill="{PRIMARY_BLUE}" letter-spacing="6">TOURNAMENT</text>
</svg>"""

def render_header_bar(user_name:str="", role:str="", logo_url:str=""):
    st.markdown('<div class="iapps-header-row">', unsafe_allow_html=True)
    c_logo, c_spacer, c_user = st.columns([2,6,3])
    with c_logo:
        data_uri=_fetch_image_data_uri(logo_url) if logo_url else ""
        if data_uri:
            st.markdown(f'<img class="iapps-header-logo" src="{data_uri}" alt="iAPPs">', unsafe_allow_html=True)
        else:
            st.markdown(_brand_svg(180), unsafe_allow_html=True)
    with c_spacer: st.markdown("")
    with c_user:
        if user_name:
            st.markdown(f'<span class="iapps-user">{user_name}</span><span class="iapps-role"> ({role})</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =======================================
# ====== MODELOS / LÓGICA DE TORNEO =====
# =======================================
DEFAULT_CONFIG = {
    "t_name":"Open Pádel",
    "num_pairs":16,
    "num_zones":4,
    "top_per_zone":2,
    "points_win":2,
    "points_loss":0,
    "seed":42,
    "format":"best_of_3",
    "use_seeds":False
}

def rr_schedule(group:List[str])->List[Tuple[str,str]]:
    return list(combinations(group,2))

def build_fixtures(groups:List[List[str]])->List[Dict[str,Any]]:
    rows=[]
    for zi,group in enumerate(groups, start=1):
        zone=f"Z{zi}"
        for a,b in rr_schedule(group):
            rows.append({"zone":zone,"pair1":a,"pair2":b,"sets":[],"golden1":0,"golden2":0})
    return rows

def validate_sets(fmt:str, sets:List[Dict[str,int]])->Tuple[bool,str]:
    n=len(sets)
    if fmt=="one_set":
        if n!=1: return False,"Formato a 1 set: exactamente 1 set."
    elif fmt=="best_of_3":
        if n<2 or n>3: return False,"Al mejor de 3: 2 o 3 sets."
    elif fmt=="best_of_5":
        if n<3 or n>5: return False,"Al mejor de 5: entre 3 y 5 sets."
    return True,""

def compute_sets_stats(sets:List[Dict[str,int]])->Dict[str,int]:
    g1=g2=s1=s2=0
    for s in sets:
        a=int(s.get("s1",0)); b=int(s.get("s2",0))
        g1+=a; g2+=b
        if a>b: s1+=1
        elif b>a: s2+=1
    return {"games1":g1,"games2":g2,"sets1":s1,"sets2":s2}

def match_has_winner(sets:List[Dict[str,int]])->bool:
    stt=compute_sets_stats(sets); return stt["sets1"]!=stt["sets2"]

def zone_complete(zone_name:str, results_list:List[Dict[str,Any]], fmt:str)->bool:
    ms=[m for m in results_list if m["zone"]==zone_name]
    if not ms: return False
    for m in ms:
        ok,_=validate_sets(fmt,m.get("sets",[]))
        if not ok or not match_has_winner(m.get("sets",[])): return False
    return True

def standings_from_results(zone_name, group_pairs, results_list, cfg, seeded_set:Optional[set]=None):
    rows=[{"pair":p,"PJ":0,"PG":0,"PP":0,"GF":0,"GC":0,"GP":0,"PTS":0} for p in group_pairs]
    table=pd.DataFrame(rows).set_index("pair")
    fmt=cfg.get("format","best_of_3")
    for m in results_list:
        if m["zone"]!=zone_name: continue
        sets=m.get("sets",[])
        ok,_=validate_sets(fmt,sets)
        if not ok or not match_has_winner(sets): continue
        stats=compute_sets_stats(sets)
        p1,p2=m["pair1"],m["pair2"]
        g1,g2=stats["games1"],stats["games2"]
        s1,s2=stats["sets1"],stats["sets2"]
        for p in (p1,p2):
            table.at[p,"PJ"]+=1
        table.at[p1,"GF"]+=g1; table.at[p1,"GC"]+=g2
        table.at[p2,"GF"]+=g2; table.at[p2,"GC"]+=g1
        table.at[p1,"GP"]+=int(m.get("golden1",0))
        table.at[p2,"GP"]+=int(m.get("golden2",0))
        if s1>s2:
            table.at[p1,"PG"]+=1; table.at[p2,"PP"]+=1
            table.at[p1,"PTS"]+=cfg["points_win"]; table.at[p2,"PTS"]+=cfg["points_loss"]
        else:
            table.at[p2,"PG"]+=1; table.at[p1,"PP"]+=1
            table.at[p2,"PTS"]+=cfg["points_win"]; table.at[p1,"PTS"]+=cfg["points_loss"]
    table["DG"]=table["GF"]-table["GC"]
    r=random.Random(int(cfg.get("seed",42)))
    table["RND"]=table.index.map(lambda p: r.random())
    table=table.sort_values(by=["PTS","DG","GP","RND"],ascending=[False,False,False,False]).reset_index()
    if seeded_set:
        table["Pareja"]=table["pair"].apply(lambda x: f"🔴 {x}" if x in seeded_set else f"{x}")
    else:
        table["Pareja"]=table["pair"]
    table.insert(0,"Zona",zone_name); table.insert(1,"Pos",range(1,len(table)+1))
    table=table.drop(columns=["RND","pair"])
    return table

def qualified_from_tables(zone_tables,k):
    qualified=[]
    for table in zone_tables:
        if table.empty: continue
        z=table.iloc[0]["Zona"]; q=table.head(int(k))
        for _,row in q.iterrows():
            qualified.append((z, int(row["Pos"]), row["Pareja"].replace("🔴 ","")))
    return qualified

def starting_round_name_for(n:int)->str:
    if n<=2: return "FN"
    if n<=4: return "SF"
    if n<=8: return "QF"
    if n<=16: return "R16"
    return "R32"

def _mid_for(round_name:str,label:str,a:str,b:str)->str:
    base=f"{round_name}::{label}::{a}::{b}"
    import hashlib as _h
    h=_h.md5(base.encode("utf-8")).hexdigest()[:10]
    return f"{round_name}_{h}"

def ensure_match_ids(matches:List[Dict[str,Any]]):
    for m in matches:
        if not m.get("mid"):
            rn=m.get("round","R"); lb=m.get("label","M"); a=m.get("a","A"); b=m.get("b","B")
            m["mid"]=_mid_for(rn,lb,a,b)

def round_labels_map(round_name:str, count:int)->List[str]:
    if round_name=="FN": return ["FINAL"]
    if round_name=="SF": return [f"SF{i+1}" for i in range(count)]
    if round_name=="QF": return [f"QF{i+1}" for i in range(count)]
    if round_name=="R16": return [f"R16-{i+1}" for i in range(count)]
    if round_name=="R32": return [f"R32-{i+1}" for i in range(count)]
    return [f"{round_name}{i+1}" for i in range(count)]

def m_best_of(fmt:str)->int:
    return 1 if fmt=="one_set" else (3 if fmt=="best_of_3" else 5)

def build_initial_ko(qualified:List[Tuple[str,int,str]], best_of_fmt:str="best_of_3")->List[Dict[str,Any]]:
    N=len(qualified)
    if N==0: return []
    start_round=starting_round_name_for(N)
    names=[q[2] for q in sorted(qualified, key=lambda x:(x[1],x[0]))]
    target = {"FN":2,"SF":4,"QF":8,"R16":16,"R32":32}[start_round]
    while len(names) < target: names.append("BYE")
    pairs=[]
    for i in range(target//2):
        a=names[i]; b=names[-(i+1)]
        pairs.append((a,b))
    labels=round_labels_map(start_round, len(pairs))
    out=[]
    for i,(a,b) in enumerate(pairs):
        lab=labels[i] if i < len(labels) else f"{start_round}{i+1}"
        m={"round":start_round,"label":lab,"a":a,"b":b,"sets":[],"goldenA":0,"goldenB":0,"best_of":m_best_of(best_of_fmt)}
        m["mid"]=_mid_for(start_round,lab,a,b); out.append(m)
    return out

def next_round(slots:List[str]):
    out=[]; i=0
    while i<len(slots):
        if i+1<len(slots): out.append((slots[i],slots[i+1])); i+=2
        else: out.append((slots[i],None)); i+=1
    return out

def make_next_round_name(current:str)->Optional[str]:
    order=["R32","R16","QF","SF","FN"]
    if current=="FN": return None
    try: i=order.index(current)
    except ValueError: return None
    return order[i+1]

# ===== Wrappers de persistencia =====
def _repo(): return _data_repo()
def load_users()->List[Dict[str,Any]]: return _repo().load_users()
def save_users(users:List[Dict[str,Any]]): _repo().save_users(users)
def load_app_config()->Dict[str,Any]: return _repo().load_app_config()
def save_app_config(cfg:Dict[str,Any]): _repo().save_app_config(cfg)
def load_index()->List[Dict[str,Any]]: return _repo().load_index()
def save_index(idx:List[Dict[str,Any]]): _repo().save_index(idx)
def load_tournament(tid:str)->Dict[str,Any]: return _repo().load_tournament(tid)
def save_tournament(tid:str, obj:Dict[str,Any], make_snapshot:bool=True): _repo().save_tournament(tid, obj, make_snapshot=make_snapshot)

# ===== Autenticación y header =====
def render_header(user=None):
    cfg=load_app_config()
    logo=cfg.get("app_logo_url","")
    if user:
        render_header_bar(user.get("username",""), user.get("role",""), logo)
    else:
        render_header_bar("", "", logo)

def get_user(username:str)->Optional[Dict[str,Any]]:
    for u in load_users():
        if u["username"].lower()==username.lower():
            return u
    return None

def set_user(user:Dict[str,Any]):
    users=load_users()
    for i,u in enumerate(users):
        if u["username"].lower()==user["username"].lower():
            users[i]=user; save_users(users); return
    users.append(user); save_users(users)

# ===== SUPER ADMIN =====
def super_admin_panel():
    user = st.session_state["auth_user"]
    render_header(user)
    st.header("👑 Panel de SUPER ADMIN")

    with st.expander("🎨 Apariencia (Logo global y dominio público)", expanded=True):
        app_cfg=load_app_config()
        url = st.text_input("URL pública del logotipo (RAW de GitHub recomendado)", value=app_cfg.get("app_logo_url","")).strip()
        base = st.text_input("Dominio base de la app (para link público)", value=app_cfg.get("app_base_url","")).strip()
        if st.button("Guardar apariencia", type="primary"):
            app_cfg["app_logo_url"] = url
            app_cfg["app_base_url"] = base or app_cfg.get("app_base_url","")
            save_app_config(app_cfg)
            st.success("Apariencia guardada.")

    st.divider()
    st.subheader("👥 Gestión de usuarios")

    users = load_users()
    with st.form("create_user_form", clear_on_submit=True):
        c1,c2,c3,c4 = st.columns([3,2,2,3])
        with c1: new_u = st.text_input("Username nuevo").strip()
        with c2: new_role = st.selectbox("Rol", ["TOURNAMENT_ADMIN","VIEWER"])
        with c3: new_pin = st.text_input("PIN inicial (6 dígitos)", max_chars=6)
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
    users = load_users()
    for usr in users:
        with st.container(border=True):
            st.write(f"**{usr['username']}** — rol `{usr['role']}` — activo `{usr.get('active',True)}`")
            c1,c2,c3,c4,c5 = st.columns([2,2,2,3,2])
            with c1:
                new_role = st.selectbox(
                    f"Rol de {usr['username']}",
                    ["SUPER_ADMIN","TOURNAMENT_ADMIN","VIEWER"],
                    index=["SUPER_ADMIN","TOURNAMENT_ADMIN","VIEWER"].index(usr["role"]),
                    key=f"role_{usr['username']}",
                    disabled=(usr["username"]=="ADMIN")
                )
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
            with c3:
                active_toggle = st.checkbox("Activo", value=usr.get("active",True), key=f"act_{usr['username']}")
            with c4:
                if usr["username"]=="ADMIN":
                    st.caption("PIN de ADMIN fijo en 199601 (no editable).")
                    pin_value = None
                else:
                    pin_value = st.text_input(f"Nuevo PIN ({usr['username']}) (opcional)", max_chars=6, key=f"pin_{usr['username']}")
            with c5:
                if st.button(f"💾 Guardar {usr['username']}", key=f"save_{usr['username']}"):
                    if usr["username"]=="ADMIN":
                        usr["role"] = "SUPER_ADMIN"
                        usr["pin_hash"] = sha("199601")
                    else:
                        usr["role"] = new_role
                        if pin_value:
                            if len(pin_value)==6 and pin_value.isdigit():
                                usr["pin_hash"] = sha(pin_value)
                            else:
                                st.error("PIN inválido (6 dígitos)."); st.stop()
                        usr["assigned_admin"] = new_assigned if usr["role"]=="VIEWER" else None
                    usr["active"] = bool(active_toggle)
                    set_user(usr)
                    st.success("Cambios guardados.")

    st.caption(f"Iapps Padel Tournament · iAPPs Pádel — {APP_VERSION}")

# ===== ADMIN (torneos) =====
def load_index_for_admin(admin_username:str)->List[Dict[str,Any]]:
    idx=load_index(); my=[t for t in idx if t.get("admin_username")==admin_username]
    def keyf(t):
        try: return datetime.fromisoformat(t.get("date"))
        except Exception: return datetime.min
    return sorted(my, key=keyf, reverse=True)

def create_tournament(admin_username:str, t_name:str, place:str, tdate:str, gender:str)->str:
    tid=str(uuid.uuid4())[:8]
    meta={"tournament_id":tid,"t_name":t_name,"place":place,"date":tdate,"gender":gender}
    state={
        "meta":{**meta,"admin_username":admin_username,"created_at":now_iso()},
        "config":{
            "t_name":t_name,"num_pairs":16,"num_zones":4,"top_per_zone":2,
            "points_win":2,"points_loss":0,"seed":42,"format":"best_of_3","use_seeds":False
        },
        "pairs":[],
        "groups":None,
        "results":[],
        "ko":{"matches":[]},
        "seeded_pairs":[]
    }
    save_tournament(tid,state)
    idx=load_index()
    idx.append({"tournament_id":tid,"t_name":t_name,"place":place,"date":tdate,"gender":gender,"admin_username":admin_username,"created_at":now_iso()})
    save_index(idx)
    return tid

def delete_tournament(admin_username:str, tid:str):
    idx=load_index()
    idx=[t for t in idx if not (t["tournament_id"]==tid and t["admin_username"]==admin_username)]
    save_index(idx)
    st.info("El archivo del torneo queda en el repo de datos (histórico). Si lo deseás, borralo manualmente allí.")

def parse_pair_number(label:str)->Optional[int]:
    try:
        left=label.split("—",1)[0].strip()
        return int(left)
    except Exception:
        return None

def next_available_number(pairs:List[str], max_pairs:int)->Optional[int]:
    used=set()
    for p in pairs:
        n=parse_pair_number(p)
        if n is not None: used.add(n)
    for n in range(1, max_pairs+1):
        if n not in used: return n
    return None

def format_pair_label(n:int, j1:str, j2:str)->str:
    return f"{n:02d} — {j1.strip()} / {j2.strip()}"
def next_round(slots:List[str]):
    out=[]; i=0
    while i<len(slots):
        if i+1<len(slots): out.append((slots[i],slots[i+1])); i+=2
        else: out.append((slots[i],None)); i+=1
    return out

def make_next_round_name(current:str)->Optional[str]:
    order=["R32","R16","QF","SF","FN"]
    if current=="FN": return None
    try: i=order.index(current)
    except ValueError: return None
    return order[i+1]

# =================================================
# ====== PERSISTENCIA (wrappers hacia GitHub) =====
# =================================================
def _repo(): return _data_repo()

def load_users()->List[Dict[str,Any]]:
    return _repo().load_users()

def save_users(users:List[Dict[str,Any]]):
    _repo().save_users(users)

def load_app_config()->Dict[str,Any]:
    return _repo().load_app_config()

def save_app_config(cfg:Dict[str,Any]):
    _repo().save_app_config(cfg)

def load_index()->List[Dict[str,Any]]:
    return _repo().load_index()

def save_index(idx:List[Dict[str,Any]]):
    _repo().save_index(idx)

def load_tournament(tid:str)->Dict[str,Any]:
    return _repo().load_tournament(tid)

def save_tournament(tid:str, obj:Dict[str,Any], make_snapshot:bool=True):
    _repo().save_tournament(tid, obj, make_snapshot=make_snapshot)

# ===================================
# ====== UTILIDAD / AUTENTICACIÓN ===
# ===================================
def render_header(user=None):
    cfg=load_app_config()
    logo=cfg.get("app_logo_url","")
    if user:
        render_header_bar(user.get("username",""), user.get("role",""), logo)
    else:
        render_header_bar("", "", logo)

def get_user(username:str)->Optional[Dict[str,Any]]:
    for u in load_users():
        if u["username"].lower()==username.lower():
            return u
    return None

def set_user(user:Dict[str,Any]):
    users=load_users()
    for i,u in enumerate(users):
        if u["username"].lower()==user["username"].lower():
            users[i]=user; save_users(users); return
    users.append(user); save_users(users)

# ============================
# ====== SUPER ADMIN UI ======
# ============================
def super_admin_panel():
    user = st.session_state["auth_user"]
    render_header(user)
    st.header("👑 Panel de SUPER ADMIN")

    # Apariencia
    with st.expander("🎨 Apariencia (Logo global y dominio público)", expanded=True):
        app_cfg=load_app_config()
        url = st.text_input("URL pública del logotipo (RAW de GitHub recomendado)", value=app_cfg.get("app_logo_url","")).strip()
        base = st.text_input("Dominio base de la app (para link público)", value=app_cfg.get("app_base_url","")).strip()
        if st.button("Guardar apariencia", type="primary"):
            app_cfg["app_logo_url"] = url
            app_cfg["app_base_url"] = base or app_cfg.get("app_base_url","")
            save_app_config(app_cfg)
            st.success("Apariencia guardada.")

    st.divider()
    st.subheader("👥 Gestión de usuarios")

    users = load_users()
    # Crear usuario (procesado dentro del form)
    with st.form("create_user_form", clear_on_submit=True):
        c1,c2,c3,c4 = st.columns([3,2,2,3])
        with c1: new_u = st.text_input("Username nuevo").strip()
        with c2: new_role = st.selectbox("Rol", ["TOURNAMENT_ADMIN","VIEWER"])
        with c3: new_pin = st.text_input("PIN inicial (6 dígitos)", max_chars=6)
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
                st.rerun()

    st.markdown("### Lista y edición")
    users = load_users()
    for usr in users:
        with st.container(border=True):
            st.write(f"**{usr['username']}** — rol `{usr['role']}` — activo `{usr.get('active',True)}`")
            c1,c2,c3,c4,c5 = st.columns([2,2,2,3,2])
            with c1:
                new_role = st.selectbox(
                    f"Rol de {usr['username']}",
                    ["SUPER_ADMIN","TOURNAMENT_ADMIN","VIEWER"],
                    index=["SUPER_ADMIN","TOURNAMENT_ADMIN","VIEWER"].index(usr["role"]),
                    key=f"role_{usr['username']}",
                    disabled=(usr["username"]=="ADMIN")
                )
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
            with c3:
                active_toggle = st.checkbox("Activo", value=usr.get("active",True), key=f"act_{usr['username']}")
            with c4:
                if usr["username"]=="ADMIN":
                    st.caption("PIN de ADMIN fijo en 199601 (no editable).")
                    pin_value = None
                else:
                    pin_value = st.text_input(f"Nuevo PIN ({usr['username']}) (opcional)", max_chars=6, key=f"pin_{usr['username']}")
            with c5:
                if st.button(f"💾 Guardar {usr['username']}", key=f"save_{usr['username']}"):
                    if usr["username"]=="ADMIN":
                        usr["role"] = "SUPER_ADMIN"
                        usr["pin_hash"] = sha("199601")
                    else:
                        usr["role"] = new_role
                        if pin_value:
                            if len(pin_value)==6 and pin_value.isdigit():
                                usr["pin_hash"] = sha(pin_value)
                            else:
                                st.error("PIN inválido (6 dígitos)."); st.stop()
                        usr["assigned_admin"] = new_assigned if usr["role"]=="VIEWER" else None
                    usr["active"] = bool(active_toggle)
                    set_user(usr)
                    st.success("Cambios guardados.")
                    st.rerun()

    st.caption(f"Iapps Padel Tournament · iAPPs Pádel — {APP_VERSION}")

# ===========================
# ====== ADMIN (torneos) ====
# ===========================
def load_index_for_admin(admin_username:str)->List[Dict[str,Any]]:
    idx=load_index(); my=[t for t in idx if t.get("admin_username")==admin_username]
    def keyf(t):
        try: return datetime.fromisoformat(t.get("date"))
        except Exception: return datetime.min
    return sorted(my, key=keyf, reverse=True)

def create_tournament(admin_username:str, t_name:str, place:str, tdate:str, gender:str)->str:
    tid=str(uuid.uuid4())[:8]
    meta={"tournament_id":tid,"t_name":t_name,"place":place,"date":tdate,"gender":gender}
    state={
        "meta":{**meta,"admin_username":admin_username,"created_at":now_iso()},
        "config":{
            "t_name":t_name,"num_pairs":16,"num_zones":4,"top_per_zone":2,
            "points_win":2,"points_loss":0,"seed":42,"format":"best_of_3","use_seeds":False
        },
        "pairs":[],
        "groups":None,
        "results":[],
        "ko":{"matches":[]},
        "seeded_pairs":[]
    }
    save_tournament(tid,state)
    idx=load_index()
    idx.append({"tournament_id":tid,"t_name":t_name,"place":place,"date":tdate,"gender":gender,"admin_username":admin_username,"created_at":now_iso()})
    save_index(idx)
    return tid

def delete_tournament(admin_username:str, tid:str):
    idx=load_index()
    idx=[t for t in idx if not (t["tournament_id"]==tid and t["admin_username"]==admin_username)]
    save_index(idx)
    st.info("El archivo del torneo queda en el repo de datos (histórico). Si lo deseás, borralo manualmente allí.")

def parse_pair_number(label:str)->Optional[int]:
    try:
        left=label.split("—",1)[0].strip()
        return int(left)
    except Exception:
        return None

def next_available_number(pairs:List[str], max_pairs:int)->Optional[int]:
    used=set()
    for p in pairs:
        n=parse_pair_number(p)
        if n is not None: used.add(n)
    for n in range(1, max_pairs+1):
        if n not in used: return n
    return None

def format_pair_label(n:int, j1:str, j2:str)->str:
    return f"{n:02d} — {j1.strip()} / {j2.strip()}"

def create_groups_unseeded(pairs:List[str], num_groups:int, top_per_zone:int, seed:int)->List[List[str]]:
    r=random.Random(int(seed)); pool=pairs[:]; r.shuffle(pool)
    groups=[[] for _ in range(num_groups)]
    min_per_zone=max(1,int(top_per_zone)); total=len(pool); desired_min_total=num_groups*min_per_zone
    gi=0
    while pool and sum(len(g) for g in groups)<min(total,desired_min_total):
        if len(groups[gi])<min_per_zone: groups[gi].append(pool.pop())
        gi=(gi+1)%num_groups
    gi=0
    while pool:
        groups[gi].append(pool.pop()); gi=(gi+1)%num_groups
    return groups

def create_groups_seeded(pairs:List[str], seeded_labels:List[str], num_groups:int, top_per_zone:int, seed:int)->List[List[str]]:
    r=random.Random(int(seed))
    seeded=[p for p in pairs if p in seeded_labels]
    non_seeded=[p for p in pairs if p not in seeded_labels]
    r.shuffle(non_seeded)
    groups=[[] for _ in range(num_groups)]
    for i,s in enumerate(seeded[:num_groups]): groups[i].append(s)
    min_per_zone=max(1,int(top_per_zone)); total=len(pairs); desired_min_total=num_groups*min_per_zone
    gi=0
    while non_seeded and sum(len(g) for g in groups)<min(total,desired_min_total):
        if len(groups[gi])<min_per_zone: groups[gi].append(non_seeded.pop())
        gi=(gi+1)%num_groups
    gi=0
    while non_seeded:
        groups[gi].append(non_seeded.pop()); gi=(gi+1)%num_groups
    return groups

def tournament_manager(user:Dict[str,Any], tid:str):
    state=load_tournament(tid)
    if not state:
        st.error("No se encontró el torneo."); return

    cfg=state.get("config",DEFAULT_CONFIG.copy())

    tab_cfg, tab_pairs, tab_results, tab_tables, tab_ko = st.tabs(["⚙️ Configuración","👥 Parejas","📝 Resultados","📊 Tablas","🏁 Playoffs"])

    # -------- CONFIGURACIÓN --------
    with tab_cfg:
        st.subheader("Configuración deportiva")
        c1,c2,c3,c4=st.columns(4)
        with c1:
            cfg["t_name"]=st.text_input("Nombre del torneo", value=cfg.get("t_name","Open Pádel"))
            cfg["num_pairs"]=st.number_input("Máximo de parejas",2,256,int(cfg.get("num_pairs",16)),step=1)
        with c2:
            cfg["num_zones"]=st.number_input("Cantidad de zonas",2,32,int(cfg.get("num_zones",4)),step=1)
            cfg["top_per_zone"]=st.number_input("Clasifican por zona",1,8,int(cfg.get("top_per_zone",2)),step=1)
        with c3:
            cfg["points_win"]=st.number_input("Puntos por victoria",1,10,int(cfg.get("points_win",2)),step=1)
            cfg["points_loss"]=st.number_input("Puntos por derrota",0,5,int(cfg.get("points_loss",0)),step=1)
        with c4:
            cfg["seed"]=st.number_input("Semilla (sorteo zonas)",1,999999,int(cfg.get("seed",42)),step=1)
        fmt=st.selectbox("Formato de partido",["one_set","best_of_3","best_of_5"], index=["one_set","best_of_3","best_of_5"].index(cfg.get("format","best_of_3")))
        cfg["format"]=fmt
        cfg["use_seeds"]=st.checkbox("Usar sistema de cabezas de serie", value=bool(cfg.get("use_seeds",False)))

        colA,colB=st.columns(2)
        with colA:
            if st.button("💾 Guardar configuración", type="primary"):
                state["config"]=cfg
                save_tournament(tid,state)
                st.success("Configuración guardada.")
        with colB:
            if st.button("🎲 Sortear zonas (crear/rehacer fixture)"):
                pairs=state.get("pairs",[])
                if len(pairs)<cfg["num_zones"]:
                    st.error("Debe haber al menos tantas parejas como zonas.")
                else:
                    if cfg.get("use_seeds",False):
                        seeded=state.get("seeded_pairs",[])
                        if len(seeded)!=int(cfg["num_zones"]):
                            st.error(f"Seleccioná exactamente {int(cfg['num_zones'])} cabezas de serie."); st.stop()
                        groups=create_groups_seeded(pairs, seeded, int(cfg["num_zones"]), int(cfg["top_per_zone"]), int(cfg["seed"]))
                    else:
                        groups=create_groups_unseeded(pairs, int(cfg["num_zones"]), int(cfg["top_per_zone"]), int(cfg["seed"]))
                    state["groups"]=groups; state["results"]=build_fixtures(groups); state["ko"]={"matches":[]}
                    save_tournament(tid,state)
                    st.success("Zonas + fixture generados.")

        st.divider()
        # Persistencia: backup/restore del estado del torneo
        st.subheader("💾 Backup/Restore del torneo (JSON)")
        meta=state.get("meta",{}); ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        def sanitize_filename(s:str)->str:
            return "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in s).strip("_")
        fname=f"{meta.get('tournament_id','')}_{sanitize_filename(cfg.get('t_name',''))}_{meta.get('date','')}_{ts}.json"
        payload=json.dumps(state,ensure_ascii=False,indent=2).encode("utf-8")
        st.download_button("⬇️ Descargar estado (JSON)", data=payload, file_name=fname, mime="application/json", key=f"dl_state_json_{tid}_{ts}")
        up=st.file_uploader("⬆️ Cargar estado", type=["json"], key=f"up_{tid}")
        if up is not None:
            st.warning("⚠️ Restauración completa: reemplaza el estado actual por el archivo subido.")
            if st.button("Confirmar restauración", key=f"confirm_restore_{tid}", type="primary"):
                try:
                    new_state=json.load(up)
                    save_tournament(tid,new_state)
                    st.success("Cargado y guardado."); st.rerun()
                except Exception as e:
                    st.error(f"Error al cargar: {e}")

        st.divider()
        # Link público
        app_cfg=load_app_config()
        public_url=f"{app_cfg.get('app_base_url','https://iappspadel.streamlit.app')}/?mode=public&tid={tid}"
        st.caption("Link público (solo lectura) — copia manual:")
        st.text_input("URL pública", value=public_url, disabled=False, label_visibility="collapsed", key=f"pub_{tid}")

    # -------- PAREJAS --------
    with tab_pairs:
        st.subheader("Parejas")
        pairs=state.get("pairs",[]); max_pairs=int(state.get("config",{}).get("num_pairs",16))
        colL,colR=st.columns([1,1])
        with colL:
            st.markdown("**Alta manual — una pareja por vez**")
            next_n=next_available_number(pairs,max_pairs)
            # PROCESAR DENTRO DEL FORM (evita pérdida de valores)
            with st.form(f"add_pair_form_{tid}", clear_on_submit=True):
                c1,c2,c3=st.columns([1,3,3])
                with c1: st.text_input("N°", value=(str(next_n) if next_n else "—"), disabled=True)
                with c2: p1=st.text_input("Jugador 1")
                with c3: p2=st.text_input("Jugador 2")
                subm=st.form_submit_button("Agregar", type="primary", disabled=(next_n is None))
                if subm:
                    p1c,p2c=(p1 or "").strip(),(p2 or "").strip()
                    if not p1c or not p2c: st.error("Completá ambos nombres.")
                    else:
                        label=format_pair_label(next_n,p1c,p2c); pairs=state.get("pairs",[]); pairs.append(label); state["pairs"]=pairs
                        save_tournament(tid,state)
                        st.success(f"Agregada: {label}")
                        st.rerun()
            if next_n is None: st.warning(f"Se alcanzó el máximo de parejas ({max_pairs}).")
        with colR:
            st.markdown("**Listado**")
            if not pairs: st.info("Aún no hay parejas cargadas.")
            else:
                for p in pairs:
                    n=parse_pair_number(p) or "-"
                    c1,c2,c3=st.columns([1,8,1])
                    with c1: st.write(f"**{n:>02}**" if isinstance(n,int) else "**—**")
                    with c2: st.write(p)
                    with c3:
                        if st.button("🗑️", key=f"del_{tid}_{p}", help="Eliminar pareja"):
                            state["pairs"]=[x for x in pairs if x!=p]; save_tournament(tid,state); st.rerun()

        st.divider()
        st.subheader("Cabezas de serie")
        use_seeds=bool(state.get("config",{}).get("use_seeds",False))
        num_groups=int(state.get("config",{}).get("num_zones",4))
        if use_seeds:
            seeded=state.get("seeded_pairs",[]); choices=pairs[:]; current=[p for p in seeded if p in choices]
            st.caption(f"Selecciona exactamente {num_groups} parejas como cabezas de serie (1 por zona).")
            selected=st.multiselect(f"Cabezas de serie ({len(current)}/{num_groups})", options=choices, default=current, max_selections=num_groups)
            if st.button("💾 Guardar cabezas de serie", key=f"save_seeds_{tid}"):
                if len(selected)!=num_groups: st.error(f"Debes seleccionar exactamente {num_groups}.")
                else:
                    state["seeded_pairs"]=selected; save_tournament(tid,state); st.success("Cabezas de serie guardadas.")
        else:
            st.info("El torneo no usa cabezas de serie (activá la opción en Configuración).")

        st.divider()
        st.subheader("Sorteo de zonas (acceso rápido)")
        if st.button("🎲 Sortear zonas", key=f"sorteo_pairs_{tid}"):
            cfg=state["config"]; pairs=state.get("pairs",[])
            if len(pairs)<cfg["num_zones"]:
                st.error("Debe haber al menos tantas parejas como zonas.")
            else:
                if cfg.get("use_seeds",False):
                    seeded=state.get("seeded_pairs",[])
                    if len(seeded)!=int(cfg["num_zones"]):
                        st.error(f"Debes marcar {int(cfg['num_zones'])} cabezas de serie antes del sorteo."); st.stop()
                    groups=create_groups_seeded(pairs, seeded, int(cfg["num_zones"]), int(cfg["top_per_zone"]), int(cfg["seed"]))
                else:
                    groups=create_groups_unseeded(pairs, int(cfg["num_zones"]), int(cfg["top_per_zone"]), int(cfg["seed"]))
                state["groups"]=groups; state["results"]=build_fixtures(groups); state["ko"]={"matches":[]}
                save_tournament(tid,state)
                st.success("Zonas + fixture generados."); st.rerun()

    # -------- RESULTADOS (GRUPOS) --------
    with tab_results:
        st.subheader("Resultados — fase de grupos (sets + puntos de oro)")
        if not state.get("groups"):
            st.info("Primero crea/sortea zonas en Configuración o Parejas.")
        else:
            st.markdown("<div class='thin'>", unsafe_allow_html=True)
            fmt=state["config"].get("format","best_of_3")
            zones=sorted({m["zone"] for m in state["results"]})
            z_filter=st.selectbox("Filtrar por zona",["(todas)"]+zones)
            pnames=sorted(set([m["pair1"] for m in state["results"]]+[m["pair2"] for m in state["results"]]))
            p_filter=st.selectbox("Filtrar por pareja",["(todas)"]+pnames)
            listing=state["results"]
            if z_filter!="(todas)": listing=[m for m in listing if m["zone"]==z_filter]
            if p_filter!="(todas)": listing=[m for m in listing if m["pair1"]==p_filter or m["pair2"]==p_filter]

            for m in listing:
                with st.container(border=True):
                    stats_now=compute_sets_stats(m.get("sets",[])) if m.get("sets") else {"sets1":0,"sets2":0}
                    title=f"**{m['zone']}** — {m['pair1']} vs {m['pair2']}"
                    if m.get("sets") and match_has_winner(m["sets"]):
                        winner=m['pair1'] if stats_now["sets1"]>stats_now["sets2"] else m['pair2']
                        title+=f"  <span style='display:inline-block;padding:2px 6px;border-radius:6px;background:#e8f5e9;color:#1b5e20;font-weight:600;margin-left:8px;'>🏆 {winner}</span>"
                    else:
                        title+="  <span style='color:#999'>(A definir)</span>"
                    st.markdown(title, unsafe_allow_html=True)

                    cur_sets=m.get("sets",[])
                    n_min,n_max=(1,1) if fmt=="one_set" else ((2,3) if fmt=="best_of_3" else (3,5))
                    form_key=f"grp_form__{tid}__{m['zone']}__{m['pair1']}__{m['pair2']}"
                    # PARCHE: procesar dentro del form y guardar ahí
                    with st.form(form_key, clear_on_submit=False):
                        cN,_,_=st.columns([1,1,1])
                        with cN:
                            n_sets=st.number_input("Sets jugados", min_value=n_min, max_value=n_max,
                                value=min(max(len(cur_sets),n_min),n_max), key=f"ns_{form_key}")
                        new_sets=[]
                        for si in range(n_sets):
                            cA,cB,_=st.columns([1,1,1])
                            with cA:
                                s1=st.number_input(f"Set {si+1} — {m['pair1']}",0,20,
                                    int(cur_sets[si]["s1"]) if si<len(cur_sets) and "s1" in cur_sets[si] else 0,
                                    key=f"s1_{form_key}_{si}")
                            with cB:
                                s2=st.number_input(f"Set {si+1} — {m['pair2']}",0,20,
                                    int(cur_sets[si]["s2"]) if si<len(cur_sets) and "s2" in cur_sets[si] else 0,
                                    key=f"s2_{form_key}_{si}")
                            new_sets.append({"s1":int(s1),"s2":int(s2)})
                        ok,msg=validate_sets(fmt,new_sets)
                        if not ok: st.error(msg)
                        gC,gD,_=st.columns([1,1,1])
                        with gC: g1=st.number_input(f"Puntos de oro {m['pair1']}",0,200,int(m.get("golden1",0)),key=f"g1_{form_key}")
                        with gD: g2=st.number_input(f"Puntos de oro {m['pair2']}",0,200,int(m.get("golden2",0)),key=f"g2_{form_key}")
                        submitted=st.form_submit_button("Guardar este partido")
                        if submitted:
                            stats=compute_sets_stats(new_sets)
                            if stats["sets1"]==stats["sets2"]:
                                st.error("Debe haber un ganador (no se permiten empates).")
                            else:
                                m["sets"]=new_sets; m["golden1"]=int(g1); m["golden2"]=int(g2)
                                save_tournament(tid,state)
                                st.success("✔ Partido de grupos guardado")
                                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    # -------- TABLAS --------
    with tab_tables:
        st.subheader("Tablas de posiciones por zona")
        if not state.get("groups") or not state.get("results"):
            st.info("Generá primero las zonas y el fixture.")
        else:
            cfg_here=state["config"]; fmt_here=cfg_here.get("format","best_of_3")
            seeded_set=set(state.get("seeded_pairs",[])) if cfg_here.get("use_seeds",False) else set()
            for zi,group in enumerate(state["groups"], start=1):
                zone_name=f"Z{zi}"
                status="✅ Completa" if zone_complete(zone_name,state["results"],fmt_here) else "⏳ A definir"
                st.markdown(f"#### Tabla {zone_name} — {status}")
                table=standings_from_results(zone_name,group,state["results"],cfg_here,seeded_set=seeded_set)
                if table.empty: st.info("Sin datos para mostrar todavía.")
                else:
                    show=table[["Zona","Pos","Pareja","PJ","PG","PP","GF","GC","DG","GP","PTS"]]
                    st.markdown(show.to_html(index=False, classes=["zebra","dark-header"]), unsafe_allow_html=True)

    # -------- PLAYOFFS (KO) --------
    def ensure_ko_created_or_progress(state_obj:dict):
        fmt=state_obj["config"].get("format","best_of_3")
        all_complete=all(zone_complete(f"Z{zi}",state_obj["results"],fmt) for zi in range(1,len(state_obj["groups"])+1))
        if not all_complete: return False
        zone_tables=[]
        for zi,group in enumerate(state_obj["groups"], start=1):
            zone_name=f"Z{zi}"
            table=standings_from_results(zone_name,group,state_obj["results"],state_obj["config"])
            zone_tables.append(table)
        qualified=qualified_from_tables(zone_tables, state_obj["config"]["top_per_zone"])
        if not state_obj.get("ko"): state_obj["ko"]={"matches":[]}
        if not state_obj["ko"]["matches"]:
            init = build_initial_ko(qualified, best_of_fmt=state_obj["config"].get("format","best_of_3"))
            ensure_match_ids(init)
            state_obj["ko"]["matches"] = init
            return True
        return False

    def ko_widget_key(tid_:str, mid_:str, name:str)->str:
        return f"{name}__{tid_}__{mid_}"

    def save_ko_match_atomic(tid_: str, mid: str, new_sets: List[Dict[str,int]], g1: int, g2: int, max_retries:int=6) -> bool:
        for attempt in range(max_retries):
            fresh = load_tournament(tid_) or {}
            ko = fresh.setdefault("ko", {}).setdefault("matches", [])
            found=False
            for mm in ko:
                if mm.get("mid")==mid:
                    mm["sets"]=new_sets; mm["goldenA"]=int(g1); mm["goldenB"]=int(g2)
                    found=True; break
            if not found:
                ko.append({"mid":mid,"round":"?","label":"?","a":"?","b":"?","sets":new_sets,"goldenA":int(g1),"goldenB":int(g2)})
            try:
                save_tournament(tid_,fresh, make_snapshot=True)
            except RuntimeError:
                pass  # conflicto → reintenta
            verify = load_tournament(tid_) or {}
            vko = verify.get("ko", {}).get("matches", [])
            persisted=False
            for vm in vko:
                if vm.get("mid")==mid:
                    if len(vm.get("sets",[]))==len(new_sets) and int(vm.get("goldenA",0))==int(g1) and int(vm.get("goldenB",0))==int(g2):
                        ok_all=True
                        for i in range(len(new_sets)):
                            a=vm["sets"][i]; b=new_sets[i]
                            if int(a.get("s1",-1))!=int(b.get("s1",-2)) or int(a.get("s2",-1))!=int(b.get("s2",-2)): ok_all=False; break
                        if ok_all: persisted=True
                    break
            if persisted: return True
            time.sleep(0.15*(attempt+1))
        return False

    def render_playoff_match(tid_:str, match:dict, tourn_state:dict):
        mid=match.get("mid")
        if not mid:
            match["mid"]=_mid_for(match.get("round","R"),match.get("label","M"),match.get("a","A"),match.get("b","B"))
            mid=match["mid"]

        fmt_local=tourn_state["config"].get("format","best_of_3")
        best_of=match.get("best_of", m_best_of(fmt_local))
        n_min=best_of//2 + 1 if best_of>1 else 1
        n_max=best_of

        cur_sets=match.get("sets",[])
        title=f"**{match['label']}** — {match['a']} vs {match['b']}"
        if cur_sets and match_has_winner(cur_sets):
            stats_now=compute_sets_stats(cur_sets)
            winner=match['a'] if stats_now["sets1"]>stats_now["sets2"] else match['b']
            title+=(
                "  <span style='display:inline-block;padding:2px 6px;border-radius:6px;"
                "background:#e8f5e9;color:#1b5e20;font-weight:600;margin-left:8px;'>"
                f"🏆 {winner}</span>"
            )
        else:
            title+="  <span style='color:#999'>(A definir)</span>"
        st.markdown(title, unsafe_allow_html=True)

        form_key=f"ko_form__{tid_}__{mid}"
        # PARCHE: procesar y guardar dentro del form
        with st.form(form_key, clear_on_submit=False):
            cN,_,_=st.columns([1,1,1])
            with cN:
                n_sets=st.number_input("Sets jugados", min_value=n_min, max_value=n_max,
                    value=min(max(len(cur_sets),n_min),n_max), key=ko_widget_key(tid_,mid,"ko_ns"))

            new_sets=[]
            for si in range(n_sets):
                cA,cB,_=st.columns([1,1,1])
                with cA:
                    s1_val=int(cur_sets[si]["s1"]) if si<len(cur_sets) and "s1" in cur_sets[si] else 0
                    s1=st.number_input(f"Set {si+1} — {match['a']}",0,20,s1_val, key=ko_widget_key(tid_,mid,f"ko_s{si+1}_a"))
                with cB:
                    s2_val=int(cur_sets[si]["s2"]) if si<len(cur_sets) and "s2" in cur_sets[si] else 0
                    s2=st.number_input(f"Set {si+1} — {match['b']}",0,20,s2_val, key=ko_widget_key(tid_,mid,f"ko_s{si+1}_b"))
                new_sets.append({"s1":int(s1),"s2":int(s2)})

            ok,msg=validate_sets(fmt_local,new_sets)
            if not ok: st.error(msg)

            gC,gD,_=st.columns([1,1,1])
            with gC:
                g1=st.number_input(f"Puntos de oro {match['a']}",0,200,int(match.get("goldenA",0)), key=ko_widget_key(tid_,mid,"ko_g1"))
            with gD:
                g2=st.number_input(f"Puntos de oro {match['b']}",0,200,int(match.get("goldenB",0)), key=ko_widget_key(tid_,mid,"ko_g2"))

            submitted=st.form_submit_button("Guardar este partido KO")
            if submitted:
                stats=compute_sets_stats(new_sets)
                if stats["sets1"]==stats["sets2"]:
                    st.error("Debe haber un ganador en KO (no se permiten empates)."); return
                saving_key = f"saving_{mid}"
                if st.session_state.get(saving_key):
                    st.info("Guardando, por favor espera…"); st.stop()
                st.session_state[saving_key] = True
                with st.spinner("Guardando partido KO…"):
                    ok = save_ko_match_atomic(tid_, mid, new_sets, g1, g2, max_retries=6)
                st.session_state[saving_key] = False
                if not ok:
                    st.error("No se pudo confirmar el guardado tras varios intentos. Recargá e intentá nuevamente.")
                    st.stop()
                st.success("✔ Partido KO guardado")
                st.rerun()

    def render_playoff_round(tid_:str, round_name:str, matches:list, tourn_state:dict):
        for m in matches:
            with st.container(border=True):
                render_playoff_match(tid_, m, tourn_state)

    with tab_ko:
        st.subheader("Playoffs (por sets + puntos de oro)")
        if not state.get("groups") or not state.get("results"):
            st.info("Necesitas tener zonas y resultados para definir clasificados.")
        else:
            created = ensure_ko_created_or_progress(state)
            if created:
                save_tournament(tid,state)
                st.info("Ronda inicial de KO creada."); st.rerun()

            ensure_match_ids(state["ko"]["matches"])
            round_order=["R32","R16","QF","SF","FN"]
            for rname in round_order:
                ms=[m for m in state["ko"]["matches"] if m.get("round")==rname]
                if not ms: continue
                st.markdown(f"### {rname}")
                render_playoff_round(tid, rname, ms, state)

            # Progresión automática
            progressed=False
            for rname in ["R32","R16","QF","SF"]:
                ms=[m for m in state["ko"]["matches"] if m.get("round")==rname]
                if not ms: continue
                next_r=make_next_round_name(rname)
                if not next_r: continue
                if any(m.get("round")==next_r for m in state["ko"]["matches"]):
                    continue
                all_done=True; adv=[]
                for m in ms:
                    sets=m.get("sets",[])
                    if not sets or not match_has_winner(sets):
                        all_done=False; break
                    statsF=compute_sets_stats(sets)
                    adv.append(m['a'] if statsF["sets1"]>statsF["sets2"] else m['b'])
                if all_done and adv:
                    pairs=next_round(adv)
                    new_ms=[]
                    labels=round_labels_map(next_r, len(pairs))
                    for i,(a,b) in enumerate(pairs):
                        lab=labels[i] if i<len(labels) else f"{next_r}{i+1}"
                        m={"round":next_r,"label":lab,"a":a,"b":b or "BYE","sets":[],"goldenA":0,"goldenB":0,"best_of":m_best_of(state["config"].get("format","best_of_3"))}
                        m["mid"]=_mid_for(next_r,lab,a,b or "BYE"); new_ms.append(m)
                    state["ko"]["matches"].extend(new_ms)
                    progressed=True
            if progressed:
                save_tournament(tid,state)
                st.info("Ronda siguiente preparada."); st.rerun()

            # Campeón
            finals=[m for m in state["ko"]["matches"] if m.get("round")=="FN"]
            for fm in finals:
                sets=fm.get("sets",[])
                if sets and match_has_winner(sets):
                    statsF=compute_sets_stats(sets)
                    champion=fm['a'] if statsF["sets1"]>statsF["sets2"] else fm['b']
                    st.markdown(
                        f"<div style='padding:14px 18px;border-radius:10px;background:#fff9c4;border:1px solid #ffeb3b;font-size:1.1rem;font-weight:700;color:#795548;margin:8px 0;'>🏆 CAMPEÓN: {champion}</div>",
                        unsafe_allow_html=True
                    )
                    st.balloons()
                    break

# ==========================
# ====== VIEWER (RO) =======
# ==========================
def viewer_dashboard(user:Dict[str,Any]):
    render_header(user)
    st.header(f"Vista de consulta — {user['username']}")
    if not user.get("assigned_admin"):
        st.warning("No asignado a un admin."); return
    my=load_index_for_admin(user["assigned_admin"])
    if not my: st.info("El admin asignado no tiene torneos."); return
    names=[f"{t['date']} — {t['t_name']} ({t['gender']}) — {t['place']} — ID:{t['tournament_id']}" for t in my]
    selected=st.selectbox("Selecciona un torneo para ver", names, index=0)
    sel=my[names.index(selected)]
    viewer_tournament(sel["tournament_id"])

def viewer_tournament(tid:str, public:bool=False):
    cfg=load_app_config()
    render_header_bar("Público" if public else "", "VIEW", cfg.get("app_logo_url",""))
    state=load_tournament(tid)
    if not state: st.error("No se encontró el torneo."); return
    st.subheader(f"{state['meta'].get('t_name')} — {state['meta'].get('place')} — {state['meta'].get('date')} — {state['meta'].get('gender')}")
    tab_over,tab_tables,tab_ko=st.tabs(["👀 General","📊 Tablas","🏁 Playoffs"])
    with tab_over:
        st.write("Parejas"); dfp=pd.DataFrame({"Parejas":state.get("pairs",[])})
        st.table(dfp)
        if state.get("groups"):
            st.write("Zonas")
            for zi,group in enumerate(state["groups"], start=1):
                st.write(f"**Z{zi}**"); st.table(pd.DataFrame({"Parejas":group}))
    with tab_tables:
        if not state.get("groups") or not state.get("results"): st.info("Sin fixture/resultados aún.")
        else:
            cfg2=state["config"]; fmt=cfg2.get("format","best_of_3")
            seeded_set=set(state.get("seeded_pairs",[])) if cfg2.get("use_seeds",False) else set()
            for zi,group in enumerate(state["groups"], start=1):
                zone_name=f"Z{zi}"
                status="✅ Completa" if zone_complete(zone_name,state["results"],fmt) else "⏳ A definir"
                st.markdown(f"#### Tabla {zone_name} — {status}")
                table=standings_from_results(zone_name,group,state["results"],cfg2,seeded_set=seeded_set)
                if table.empty: st.info("Sin datos para mostrar todavía.")
                else:
                    show=table[["Zona","Pos","Pareja","PJ","PG","PP","GF","GC","DG","GP","PTS"]]
                    st.markdown(show.to_html(index=False, classes=["zebra","dark-header"]), unsafe_allow_html=True)
    with tab_ko:
        ko=state.get("ko",{"matches":[]})
        if not ko.get("matches"): st.info("Aún no hay partidos de playoffs.")
        else:
            rows=[]
            for m in ko["matches"]:
                stats=compute_sets_stats(m.get("sets",[])) if m.get("sets") else {"sets1":0,"sets2":0}
                res="A definir"
                if m.get("sets") and match_has_winner(m["sets"]): res=f"{stats['sets1']}-{stats['sets2']}"
                rows.append({"Ronda":m.get("round",""),"Clave":m.get("label",""),"A":m.get("a",""),"B":m.get("b",""),"Resultado":res})
            dfo=pd.DataFrame(rows)
            st.markdown(dfo.to_html(index=False, classes=["zebra","dark-header"]), unsafe_allow_html=True)
    if public: st.info("Modo público (solo lectura)")

# ================
# ===== MAIN =====
# ================
def init_session():
    st.session_state.setdefault("auth_user",None)

def main():
    # Query params
    try: params=st.query_params
    except Exception: params=st.experimental_get_query_params()
    mode=params.get("mode",[""]); mode=mode[0] if isinstance(mode,list) else mode
    _tid=params.get("tid",[""]); _tid=_tid[0] if isinstance(_tid,list) else _tid

    init_session()
    # Login
    if not st.session_state.get("auth_user"):
        render_header(None)
        st.markdown("### Ingreso — Usuario + PIN (6 dígitos)")
        with st.form("login", clear_on_submit=True):
            username=st.text_input("Usuario").strip()
            pin=st.text_input("PIN (6 dígitos)", type="password").strip()
            submitted=st.form_submit_button("Ingresar", type="primary")
        if submitted:
            user=get_user(username)
            if not user or not user.get("active",True): st.error("Usuario inexistente o inactivo.")
            elif len(pin)!=6 or not pin.isdigit(): st.error("PIN inválido.")
            elif sha(pin)!=user["pin_hash"]: st.error("PIN incorrecto.")
            else:
                st.session_state.auth_user=user; st.success(f"Bienvenido {user['username']} ({user['role']})"); st.rerun()
        st.caption(f"Iapps Padel Tournament · iAPPs Pádel — {APP_VERSION}"); return

    user=st.session_state["auth_user"]
    if user["role"]=="SUPER_ADMIN":
        if mode=="public" and _tid: viewer_tournament(_tid, public=True)
        else: super_admin_panel()
        st.caption(f"Iapps Padel Tournament · iAPPs Pádel — {APP_VERSION}"); return
    if user["role"]=="TOURNAMENT_ADMIN":
        render_header(user)
        st.header(f"Torneos de {user['username']}")
        with st.expander("➕ Crear torneo nuevo", expanded=True):
            c1,c2,c3,c4=st.columns(4)
            with c1: t_name=st.text_input("Nombre del torneo", value="Open Pádel")
            with c2: place=st.text_input("Lugar / Club", value="Mi Club")
            with c3: tdate=st.date_input("Fecha", value=date.today()).isoformat()
            with c4: gender=st.selectbox("Género", ["masculino","femenino","mixto"], index=2)
            if st.button("Crear torneo", type="primary"):
                tid=create_tournament(user["username"],t_name,place,tdate,gender)
                st.success(f"Torneo creado: {t_name} ({tid})"); st.rerun()

        my=load_index_for_admin(user["username"])
        if not my:
            st.info("Aún no tienes torneos."); st.caption(f"Iapps Padel Tournament · iAPPs Pádel — {APP_VERSION}"); return

        st.subheader("Abrir / eliminar torneo")
        names=[f"{t['date']} — {t['t_name']} ({t['gender']}) — {t['place']} — ID:{t['tournament_id']}" for t in my]
        selected=st.selectbox("Selecciona un torneo", names, index=0)
        sel=my[names.index(selected)]
        c1,c2,c3=st.columns(3)
        with c1:
            if st.button("Abrir torneo"): tournament_manager(user, sel["tournament_id"]); st.stop()
        with c2:
            if st.button("Eliminar torneo", type="secondary"):
                delete_tournament(user["username"], sel["tournament_id"])
                st.success("Torneo eliminado del índice."); st.rerun()
        with c3:
            app_cfg=load_app_config()
            tid=sel["tournament_id"]
            st.caption("Link público (solo lectura) — copia manual:")
            public_url=f"{app_cfg.get('app_base_url','https://iappspadel.streamlit.app')}/?mode=public&tid={tid}"
            st.text_input("URL pública", value=public_url, disabled=False, label_visibility="collapsed", key=f"pub_{tid}")
        st.caption(f"Iapps Padel Tournament · iAPPs Pádel — {APP_VERSION}"); return

    if user["role"]=="VIEWER":
        if mode=="public" and _tid: viewer_tournament(_tid, public=True)
        else: viewer_dashboard(user)
        st.caption(f"Iapps Padel Tournament · iAPPs Pádel — {APP_VERSION}"); return

    st.error("Rol desconocido.")
    st.caption(f"Iapps Padel Tournament · iAPPs Pádel — {APP_VERSION}")

if __name__=="__main__":
    main()