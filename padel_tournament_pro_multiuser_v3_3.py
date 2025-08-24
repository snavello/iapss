# padel_tournament_pro_v1_1_1.py
# Iapps Pádel Tournament — Persistencia en GitHub
# Requisitos: streamlit, requests, pandas
# Secrets necesarios en Streamlit:
# [github]
# token  = "ghp_xxx..."
# owner  = "TU_USUARIO_O_ORG"
# repo   = "iapps-data"
# branch = "main"

import os, json, time, random, uuid, base64
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

APP_VERSION = "v1.1.1"

# ============================
# ====== CSS / ESTILOS  ======
# ============================
BASE_CSS = """
<style>
:root{
  --zebra1:#fafafa; --zebra2:#f0f0f0; --dark:#37474f; --ok:#1b5e20; --lightok:#e8f5e9;
  --warn:#f57f17; --yellow:#fff9c4; --border:#e0e0e0;
}
.zebra tr:nth-child(even){ background:var(--zebra2)!important; }
.zebra tr:nth-child(odd){ background:var(--zebra1)!important; }
.dark-header thead tr th{ background:#424242!important; color:white!important; padding:8px!important; }
.copyable{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
.header-wrap{ display:flex; align-items:center; gap:16px; border-bottom:1px solid var(--border); padding:8px 0 6px 0; margin-bottom:6px; position:sticky; top:0; background:white; z-index:100; }
.header-wrap img{ max-height:48px; width:auto; }
.header-user{ margin-left:auto; font-size:0.9rem; color:#555; }
.separator{ border-bottom:1px solid var(--border); margin:6px 0 10px 0; }
.win-badge{ display:inline-block; padding:2px 6px; border-radius:6px; background: var(--lightok); color: var(--ok); font-weight:600; margin-left:8px; }
.champion{ padding:14px 18px; border-radius:10px; background: var(--yellow); border:1px solid #ffeb3b; font-size:1.1rem; font-weight:700; color:#795548; margin:8px 0; }
.thin{ margin-top:-10px; }
.small-note{ font-size:0.8rem; color:#777; }
</style>
"""
st.set_page_config(page_title="Iapps Pádel", layout="wide")
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ============================
# ====== UTILIDADES BASE =====
# ============================
def now_iso()->str:
    return datetime.now().isoformat(timespec="seconds")

def sha(pin:str)->str:
    # hash simple para PIN (no criptográfico; suficiente para este caso)
    import hashlib
    return hashlib.sha256(pin.encode("utf-8")).hexdigest()

def sanitize_filename(s:str)->str:
    return "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in s).strip("_")

def http_headers(token:str)->Dict[str,str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

def gh_file_url(owner:str, repo:str, path:str, branch:str)->str:
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"

def gh_put_file(owner:str, repo:str, path:str, branch:str, token:str, content_bytes:bytes, message:str, sha_old:Optional[str]=None)->Dict[str,Any]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("ascii"),
        "branch": branch
    }
    if sha_old:
        payload["sha"] = sha_old
    r = requests.put(url, headers=http_headers(token), json=payload, timeout=20)
    if r.status_code not in (200,201):
        raise RuntimeError(f"PUT {path} falló: {r.status_code} — {r.text[:200]}")
    return r.json()

def gh_get_file(owner:str, repo:str, path:str, branch:str, token:str)->Tuple[Optional[bytes], Optional[str]]:
    url = gh_file_url(owner, repo, path, branch)
    r = requests.get(url, headers=http_headers(token), timeout=20)
    if r.status_code == 200:
        data=r.json()
        b64=data.get("content","")
        sha_=data.get("sha")
        if b64:
            return base64.b64decode(b64), sha_
        return b"", sha_
    elif r.status_code == 404:
        return None, None
    else:
        raise RuntimeError(f"GET {path} falló: {r.status_code} — {r.text[:200]}")

# ================================
# ====== PERSISTENCIA GITHUB =====
# ================================
class GitHubDataRepo:
    def __init__(self):
        try:
            self.token  = st.secrets["github"]["token"]
            self.owner  = st.secrets["github"]["owner"]
            self.repo   = st.secrets["github"]["repo"]
            self.branch = st.secrets["github"].get("branch","main")
        except Exception as e:
            st.error("Faltan secrets de GitHub. Configura [github] en Secrets.")
            raise

    def _load_json(self, path:str)->Tuple[Optional[Any], Optional[str]]:
        raw, sha0 = gh_get_file(self.owner, self.repo, path, self.branch, self.token)
        if raw is None:
            return None, None
        try:
            return json.loads(raw.decode("utf-8")), sha0
        except Exception:
            return None, sha0

    def _save_json(self, path:str, obj:Any, msg:str):
        current, sha0 = self._load_json(path)
        content = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        gh_put_file(self.owner, self.repo, path, self.branch, self.token, content, msg, sha_old=sha0)

    # archivos base
    def load_users(self)->List[Dict[str,Any]]:
        users, _ = self._load_json("users.json")
        if users is None:
            # bootstrap
            users = [{
                "username":"ADMIN","pin_hash":sha("199601"),
                "role":"SUPER_ADMIN","created_at":now_iso(),"active":True
            }]
            self._save_json("users.json", users, "bootstrap users.json")
        return users

    def save_users(self, users:List[Dict[str,Any]]):
        self._save_json("users.json", users, "update users.json")

    def load_app_config(self)->Dict[str,Any]:
        cfg, _ = self._load_json("app_config.json")
        if cfg is None:
            cfg = {
                "app_logo_url": "https://raw.githubusercontent.com/snavello/iapss/refs/heads/main/1000138052.png",
                "app_base_url": "https://iappspadel.streamlit.app"
            }
            self._save_json("app_config.json", cfg, "bootstrap app_config.json")
        return cfg

    def save_app_config(self, cfg:Dict[str,Any]):
        self._save_json("app_config.json", cfg, "update app_config.json")

    def load_index(self)->List[Dict[str,Any]]:
        idx, _ = self._load_json("tournaments/index.json")
        if idx is None:
            idx = []
            self._save_json("tournaments/index.json", idx, "bootstrap tournaments/index.json")
        return idx

    def save_index(self, idx:List[Dict[str,Any]]):
        self._save_json("tournaments/index.json", idx, "update tournaments/index.json")

    def load_tournament(self, tid:str)->Dict[str,Any]:
        obj, _ = self._load_json(f"tournaments/{tid}.json")
        return obj or {}

    def save_tournament(self, tid:str, obj:Dict[str,Any], make_snapshot:bool=True):
        self._save_json(f"tournaments/{tid}.json", obj, f"update tournaments/{tid}.json")
        if make_snapshot:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = f"tournaments/{tid}/snapshots/{ts}.json"
            self._save_json(snap_path, obj, f"snapshot {tid} {ts}")

# Wrapper singleton
_repo_singleton = None
def _data_repo()->GitHubDataRepo:
    global _repo_singleton
    if _repo_singleton is None:
        _repo_singleton = GitHubDataRepo()
    return _repo_singleton

# Atajos
def load_users(): return _data_repo().load_users()
def save_users(u): _data_repo().save_users(u)
def load_app_config(): return _data_repo().load_app_config()
def save_app_config(c): _data_repo().save_app_config(c)
def load_index(): return _data_repo().load_index()
def save_index(i): _data_repo().save_index(i)
def load_tournament(tid): return _data_repo().load_tournament(tid)
def save_tournament(tid,obj,make_snapshot=True): _data_repo().save_tournament(tid,obj,make_snapshot)

# ============================
# ====== HEADER / BRAND  =====
# ============================
def render_header_bar(username:str, role:str, logo_url:str):
    cols = st.columns([1,5,3])
    with cols[0]:
        if logo_url:
            st.markdown(
                f'<div class="header-wrap"><img src="{logo_url}" alt="logo" /></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="header-wrap">Iapps Pádel</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
    with cols[2]:
        if username:
            st.markdown(f"<div class='header-user'>Usuario: <b>{username}</b> — Rol: <code>{role}</code></div>", unsafe_allow_html=True)
            if st.button("Cerrar sesión", key="logout_btn"):
                st.session_state["auth_user"]=None
                st.rerun()

def render_header(user=None):
    cfg = load_app_config()
    logo = cfg.get("app_logo_url","")
    if user:
        render_header_bar(user.get("username",""), user.get("role",""), logo)
    else:
        render_header_bar("", "", logo)

# ============================
# ====== LÓGICA TORNEOS  =====
# ============================
DEFAULT_CONFIG = {
    "t_name":"Open Pádel","num_pairs":16,"num_zones":4,"top_per_zone":2,
    "points_win":2,"points_loss":0,"seed":42,"format":"best_of_3","use_seeds":False
}

def create_tournament(admin_username:str, t_name:str, place:str, tdate:str, gender:str)->str:
    tid = str(uuid.uuid4())[:8]
    meta={"tournament_id":tid,"t_name":t_name,"place":place,"date":tdate,"gender":gender}
    state = {
        "meta":{**meta,"admin_username":admin_username,"created_at":now_iso()},
        "config":{**DEFAULT_CONFIG,"t_name":t_name},
        "pairs":[],
        "groups":None,
        "results":[],
        "ko":{"matches":[]},
        "seeded_pairs":[]
    }
    save_tournament(tid,state)
    idx = load_index()
    idx.append({**meta,"admin_username":admin_username,"created_at":now_iso()})
    save_index(idx)
    return tid

def delete_tournament(admin_username:str, tid:str):
    idx=load_index()
    idx=[t for t in idx if not (t["tournament_id"]==tid and t["admin_username"]==admin_username)]
    save_index(idx)
    st.info("Se eliminó del índice. Los archivos históricos quedan en el repo de datos.")

def load_index_for_admin(admin_username:str)->List[Dict[str,Any]]:
    idx=load_index()
    my=[t for t in idx if t.get("admin_username")==admin_username]
    def keyf(t):
        try: return datetime.fromisoformat(t.get("date"))
        except Exception: return datetime.min
    return sorted(my, key=keyf, reverse=True)

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

# ====== ZONAS / FIXTURE (grupos) ======
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
    non=[p for p in pairs if p not in seeded_labels]
    r.shuffle(non)
    groups=[[] for _ in range(num_groups)]
    for i,s in enumerate(seeded[:num_groups]): groups[i].append(s)
    min_per_zone=max(1,int(top_per_zone)); total=len(pairs); desired_min_total=num_groups*min_per_zone
    gi=0
    while non and sum(len(g) for g in groups)<min(total,desired_min_total):
        if len(groups[gi])<min_per_zone: groups[gi].append(non.pop())
        gi=(gi+1)%num_groups
    gi=0
    while non:
        groups[gi].append(non.pop()); gi=(gi+1)%num_groups
    return groups

def build_fixtures(groups:List[List[str]])->List[Dict[str,Any]]:
    fixtures=[]
    for zi, group in enumerate(groups, start=1):
        zone=f"Z{zi}"
        g=group[:]
        # round robin simple
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                fixtures.append({
                    "zone":zone, "pair1":g[i], "pair2":g[j],
                    "sets":[], "golden1":0, "golden2":0
                })
    return fixtures

# ====== Validación / Estadísticos de Sets ======
def m_best_of(fmt:str)->int:
    return 1 if fmt=="one_set" else (3 if fmt=="best_of_3" else 5)

def validate_sets(fmt:str, sets:List[Dict[str,int]])->Tuple[bool,str]:
    best_of=m_best_of(fmt)
    if not sets: return False,"Debes cargar al menos 1 set."
    if len(sets)>best_of: return False,f"Máximo {best_of} sets."
    s1=s2=0
    for s in sets:
        a=int(s.get("s1",0)); b=int(s.get("s2",0))
        if a==b: return False,"No puede haber sets empatados."
        if a>b: s1+=1
        else: s2+=1
        if s1>best_of//2 or s2>best_of//2:
            # ya hay ganador, puedes truncar si el usuario cargó demás
            pass
    if s1==s2: return False,"Debe haber ganador en sets."
    return True,"OK"

def compute_sets_stats(sets:List[Dict[str,int]])->Dict[str,int]:
    s1=s2=gf1=gf2=0
    for s in sets:
        a=int(s.get("s1",0)); b=int(s.get("s2",0))
        if a>b: s1+=1
        elif b>a: s2+=1
        gf1+=a; gf2+=b
    return {"sets1":s1,"sets2":s2,"gf1":gf1,"gf2":gf2}

def match_has_winner(sets:List[Dict[str,int]])->bool:
    if not sets: return False
    stt=compute_sets_stats(sets)
    return stt["sets1"]!=stt["sets2"]

# ====== Tablas por zona ======
def standings_from_results(zone:str, group:List[str], results:List[Dict[str,Any]], cfg:Dict[str,Any], seeded_set:Optional[set]=None)->pd.DataFrame:
    seeded_set = seeded_set or set()
    # init
    rows={p:{"Zona":zone,"Pareja":p,"PJ":0,"PG":0,"PP":0,"GF":0,"GC":0,"DG":0,"GP":0,"PTS":0,"Seed":(p in seeded_set)} for p in group}
    for m in results:
        if m["zone"]!=zone or not m.get("sets"): continue
        if not match_has_winner(m["sets"]): continue
        stt=compute_sets_stats(m["sets"])
        p1,p2=m["pair1"],m["pair2"]
        g1,g2=stt["sets1"],stt["sets2"]
        rows[p1]["PJ"]+=1; rows[p2]["PJ"]+=1
        rows[p1]["GF"]+=stt["gf1"]; rows[p1]["GC"]+=stt["gf2"]
        rows[p2]["GF"]+=stt["gf2"]; rows[p2]["GC"]+=stt["gf1"]
        rows[p1]["DG"]=rows[p1]["GF"]-rows[p1]["GC"]
        rows[p2]["DG"]=rows[p2]["GF"]-rows[p2]["GC"]
        rows[p1]["GP"]+= int(m.get("golden1",0))
        rows[p2]["GP"]+= int(m.get("golden2",0))
        if g1>g2:
            rows[p1]["PG"]+=1; rows[p2]["PP"]+=1
            rows[p1]["PTS"]+=int(cfg.get("points_win",2)); rows[p2]["PTS"]+=int(cfg.get("points_loss",0))
        else:
            rows[p2]["PG"]+=1; rows[p1]["PP"]+=1
            rows[p2]["PTS"]+=int(cfg.get("points_win",2)); rows[p1]["PTS"]+=int(cfg.get("points_loss",0))
    df=pd.DataFrame(list(rows.values()))
    if df.empty: return df
    df["Pos"]=range(1,len(df)+1)
    df=df.sort_values(by=["PTS","DG","GP"], ascending=[False,False,False])
    df["Pos"]=range(1,len(df)+1)
    # Marca visual para seed
    df["Pareja"] = df.apply(lambda r: f"🔴 {r['Pareja']}" if r.get("Seed") else r["Pareja"], axis=1)
    return df

def zone_complete(zone:str, results:List[Dict[str,Any]], fmt:str)->bool:
    # completa si todos los partidos de la zona tienen ganador
    for m in results:
        if m["zone"]!=zone: continue
        if not m.get("sets"): return False
        if not match_has_winner(m["sets"]): return False
    return True

def qualified_from_tables(zone_tables:List[pd.DataFrame], top:int)->List[str]:
    out=[]
    for df in zone_tables:
        if df is None or df.empty: continue
        k = min(top, len(df))
        out += list(df.sort_values(by=["PTS","DG","GP"], ascending=[False,False,False]).head(k)["Pareja"])
    # remover el posible prefijo de seed "🔴 "
    out = [p[2:].strip() if p.startswith("🔴") else p for p in out]
    return out

# ====== KO ======
def next_round(slots:List[str])->List[Tuple[str, Optional[str]]]:
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

def round_labels_map(r:str, n:int)->List[str]:
    if r=="R32": return [f"R32-{i+1}" for i in range(n)]
    if r=="R16": return [f"R16-{i+1}" for i in range(n)]
    if r=="QF":  return [f"QF-{i+1}"  for i in range(n)]
    if r=="SF":  return [f"SF-{i+1}"  for i in range(n)]
    if r=="FN":  return [f"Final"     for _ in range(n)]
    return [f"{r}-{i+1}" for i in range(n)]

def _mid_for(r,label,a,b): return f"{r}::{label}::{a}#{b}"

def ensure_match_ids(matches:List[Dict[str,Any]]):
    for m in matches:
        if not m.get("mid"):
            m["mid"] = _mid_for(m.get("round","R"), m.get("label",""), m.get("a",""), m.get("b",""))

def build_initial_ko(qualified:List[str], best_of_fmt:str)->List[Dict[str,Any]]:
    n = len(qualified)
    if n<=2:
        r="FN"; pairs=next_round(qualified)
    elif n<=4:
        r="SF"; pairs=next_round(qualified)
    elif n<=8:
        r="QF"; pairs=next_round(qualified)
    elif n<=16:
        r="R16"; pairs=next_round(qualified)
    else:
        r="R32"; pairs=next_round(qualified)
    labels = round_labels_map(r, len(pairs))
    best = m_best_of(best_of_fmt)
    out=[]
    for i,(a,b) in enumerate(pairs):
        lab = labels[i] if i<len(labels) else f"{r}-{i+1}"
        out.append({"round":r, "label":lab, "a":a, "b":b or "BYE", "sets":[], "goldenA":0, "goldenB":0, "best_of":best})
    return out

# Guardado atómico de KO con verificación
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

# ============================
# ====== SUPER ADMIN UI  =====
# ============================
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

def super_admin_panel():
    user = st.session_state["auth_user"]
    render_header(user)
    st.header("👑 Panel de SUPER ADMIN")

    with st.expander("🎨 Apariencia (Logo global y dominio público)", expanded=True):
        app_cfg=load_app_config()
        url = st.text_input("URL pública del logotipo (RAW de GitHub recomendado)", value=app_cfg.get("app_logo_url",""), key="sa_logo_url").strip()
        base = st.text_input("Dominio base de la app (para link público)", value=app_cfg.get("app_base_url",""), key="sa_base_url").strip()
        if st.button("Guardar apariencia", type="primary", key="sa_save_brand"):
            app_cfg["app_logo_url"] = url
            app_cfg["app_base_url"] = base or app_cfg.get("app_base_url","")
            save_app_config(app_cfg)
            st.success("Apariencia guardada.")

    st.divider()
    st.subheader("👥 Gestión de usuarios")

    users = load_users()
    with st.form("create_user_form", clear_on_submit=True):
        c1,c2,c3,c4 = st.columns([3,2,2,3])
        with c1: new_u = st.text_input("Username nuevo", key="new_u").strip()
        with c2: new_role = st.selectbox("Rol", ["TOURNAMENT_ADMIN","VIEWER"], key="new_role")
        with c3: new_pin = st.text_input("PIN inicial (6 dígitos)", max_chars=6, key="new_pin")
        assigned_admin = None
        with c4:
            if new_role == "VIEWER":
                admins=[x["username"] for x in users if x["role"]=="TOURNAMENT_ADMIN" and x.get("active",True)]
                assigned_admin = st.selectbox("Asignar a admin", admins, key="new_assigned") if admins else None
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

# ============================
# ====== ADMIN / TORNEOS  ====
# ============================
def tournament_manager(user:Dict[str,Any], tid:str):
    state=load_tournament(tid)
    if not state:
        st.error("No se encontró el torneo."); return

    cfg=state.get("config",DEFAULT_CONFIG.copy())
    tab_cfg, tab_pairs, tab_results, tab_tables, tab_ko = st.tabs(["⚙️ Configuración","👥 Parejas","📝 Resultados","📊 Tablas","🏁 Playoffs"])

    # -------- Configuración --------
    with tab_cfg:
        st.subheader("Configuración deportiva")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cfg["t_name"] = st.text_input("Nombre del torneo", value=cfg.get("t_name","Open Pádel"), key=f"cfg_t_name_{tid}")
            cfg["num_pairs"] = st.number_input("Máximo de parejas", 2, 256, int(cfg.get("num_pairs",16)), step=1, key=f"cfg_num_pairs_{tid}")
        with c2:
            cfg["num_zones"] = st.number_input("Cantidad de zonas", 2, 32, int(cfg.get("num_zones",4)), step=1, key=f"cfg_num_zones_{tid}")
            cfg["top_per_zone"] = st.number_input("Clasifican por zona", 1, 8, int(cfg.get("top_per_zone",2)), step=1, key=f"cfg_top_per_zone_{tid}")
        with c3:
            cfg["points_win"] = st.number_input("Puntos por victoria", 1, 10, int(cfg.get("points_win",2)), step=1, key=f"cfg_points_win_{tid}")
            cfg["points_loss"] = st.number_input("Puntos por derrota", 0, 5, int(cfg.get("points_loss",0)), step=1, key=f"cfg_points_loss_{tid}")
        with c4:
            cfg["seed"] = st.number_input("Semilla (sorteo zonas)", 1, 999999, int(cfg.get("seed",42)), step=1, key=f"cfg_seed_{tid}")

        fmt = st.selectbox("Formato de partido", ["one_set","best_of_3","best_of_5"],
                           index=["one_set","best_of_3","best_of_5"].index(cfg.get("format","best_of_3")),
                           key=f"cfg_format_{tid}")
        cfg["format"] = fmt
        cfg["use_seeds"] = st.checkbox("Usar sistema de cabezas de serie", value=bool(cfg.get("use_seeds",False)), key=f"cfg_use_seeds_{tid}")

        colA, colB = st.columns(2)
        with colA:
            if st.button("💾 Guardar configuración", type="primary", key=f"cfg_save_{tid}"):
                state["config"]=cfg
                save_tournament(tid,state)
                st.success("Configuración guardada.")
        with colB:
            if st.button("🎲 Sortear zonas (crear/rehacer fixture)", key=f"cfg_draw_{tid}"):
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
        st.subheader("💾 Backup/Restore del torneo (JSON)")
        meta=state.get("meta",{}); ts=datetime.now().strftime("%Y%m%d_%H%M%S")
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
        app_cfg=load_app_config()
        public_url=f"{app_cfg.get('app_base_url','https://iappspadel.streamlit.app')}/?mode=public&tid={tid}"
        st.caption("Link público (solo lectura) — copia manual:")
        st.text_input("URL pública", value=public_url, disabled=False, label_visibility="collapsed", key=f"pub_url_cfg_{tid}")

    # -------- Parejas --------
    with tab_pairs:
        st.subheader("Parejas")
        pairs=state.get("pairs",[]); max_pairs=int(state.get("config",{}).get("num_pairs",16))
        colL,colR=st.columns([1,1])
        with colL:
            st.markdown("**Alta manual — una pareja por vez**")
            next_n=next_available_number(pairs,max_pairs)
            with st.form(f"add_pair_form_{tid}", clear_on_submit=True):
                c1,c2,c3=st.columns([1,3,3])
                with c1: st.text_input("N°", value=(str(next_n) if next_n else "—"), disabled=True, key=f"pnum_{tid}")
                with c2: p1=st.text_input("Jugador 1", key=f"p1_{tid}")
                with c3: p2=st.text_input("Jugador 2", key=f"p2_{tid}")
                subm=st.form_submit_button("Agregar", type="primary", disabled=(next_n is None), key=f"padd_{tid}")
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
            selected=st.multiselect(f"Cabezas de serie ({len(current)}/{num_groups})", options=choices, default=current, max_selections=num_groups, key=f"seeds_{tid}")
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

    # -------- Resultados (grupos) --------
    with tab_results:
        st.subheader("Resultados — fase de grupos (sets + puntos de oro)")
        if not state.get("groups"):
            st.info("Primero crea/sortea zonas en Configuración o Parejas.")
        else:
            st.markdown("<div class='thin'>", unsafe_allow_html=True)
            fmt=state["config"].get("format","best_of_3")
            zones=sorted({m["zone"] for m in state["results"]})
            z_filter=st.selectbox("Filtrar por zona",["(todas)"]+zones, key=f"res_zone_{tid}")
            pnames=sorted(set([m["pair1"] for m in state["results"]]+[m["pair2"] for m in state["results"]]))
            p_filter=st.selectbox("Filtrar por pareja",["(todas)"]+pnames, key=f"res_pair_{tid}")
            listing=state["results"]
            if z_filter!="(todas)": listing=[m for m in listing if m["zone"]==z_filter]
            if p_filter!="(todas)": listing=[m for m in listing if m["pair1"]==p_filter or m["pair2"]==p_filter]

            for m in listing:
                with st.container(border=True):
                    stats_now=compute_sets_stats(m.get("sets",[])) if m.get("sets") else {"sets1":0,"sets2":0}
                    title=f"**{m['zone']}** — {m['pair1']} vs {m['pair2']}"
                    if m.get("sets") and match_has_winner(m["sets"]):
                        winner=m['pair1'] if stats_now["sets1"]>stats_now["sets2"] else m['pair2']
                        title+=f"  <span class='win-badge'>🏆 {winner}</span>"
                    else:
                        title+="  <span style='color:#999'>(A definir)</span>"
                    st.markdown(title, unsafe_allow_html=True)

                    cur_sets=m.get("sets",[])
                    n_min,n_max=(1,1) if fmt=="one_set" else ((2,3) if fmt=="best_of_3" else (3,5))
                    form_key=f"grp_form__{tid}__{m['zone']}__{m['pair1']}__{m['pair2']}"
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
                        submitted=st.form_submit_button("Guardar este partido", key=f"btn_{form_key}")
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

    # -------- Tablas --------
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

    # -------- Playoffs (KO) --------
    with tab_ko:
        st.subheader("Playoffs (por sets + puntos de oro)")
        if not state.get("groups") or not state.get("results"):
            st.info("Necesitas tener zonas y resultados para definir clasificados.")
        else:
            # crear ronda inicial si corresponde
            fmt=state["config"].get("format","best_of_3")
            all_complete=all(zone_complete(f"Z{zi}",state["results"],fmt) for zi in range(1,len(state["groups"])+1))
            if all_complete and not state["ko"]["matches"]:
                zone_tables=[]
                for zi,group in enumerate(state["groups"], start=1):
                    zone_name=f"Z{zi}"
                    table=standings_from_results(zone_name,group,state["results"],state["config"])
                    zone_tables.append(table)
                qualified=qualified_from_tables(zone_tables, state["config"]["top_per_zone"])
                init=build_initial_ko(qualified, best_of_fmt=state["config"].get("format","best_of_3"))
                ensure_match_ids(init)
                state["ko"]["matches"]=init
                save_tournament(tid,state)
                st.info("Ronda inicial de KO creada."); st.rerun()

            ensure_match_ids(state["ko"]["matches"])
            def ko_widget_key(tid_:str, mid_:str, name:str)->str:
                return f"{name}__{tid_}__{mid_}"

            def render_playoff_match(match:dict):
                mid=match.get("mid")
                title=f"**{match['label']}** — {match['a']} vs {match['b']}"
                cur_sets=match.get("sets",[])
                if cur_sets and match_has_winner(cur_sets):
                    stats_now=compute_sets_stats(cur_sets)
                    winner=match['a'] if stats_now["sets1"]>stats_now["sets2"] else match['b']
                    title+=f"  <span class='win-badge'>🏆 {winner}</span>"
                else:
                    title+="  <span style='color:#999'>(A definir)</span>"
                st.markdown(title, unsafe_allow_html=True)

                fmt_local=state["config"].get("format","best_of_3")
                best_of=m_best_of(fmt_local)
                n_min = best_of//2 + 1 if best_of>1 else 1
                n_max = best_of

                form_key=f"ko_form__{tid}__{mid}"
                with st.form(form_key, clear_on_submit=False):
                    cN,_,_=st.columns([1,1,1])
                    with cN:
                        n_sets=st.number_input("Sets jugados", min_value=n_min, max_value=n_max,
                            value=min(max(len(cur_sets),n_min),n_max), key=ko_widget_key(tid,mid,"ko_ns"))
                    new_sets=[]
                    for si in range(n_sets):
                        cA,cB,_=st.columns([1,1,1])
                        with cA:
                            s1_val=int(cur_sets[si]["s1"]) if si<len(cur_sets) and "s1" in cur_sets[si] else 0
                            s1=st.number_input(f"Set {si+1} — {match['a']}",0,20,s1_val, key=ko_widget_key(tid,mid,f"ko_s{si+1}_a"))
                        with cB:
                            s2_val=int(cur_sets[si]["s2"]) if si<len(cur_sets) and "s2" in cur_sets[si] else 0
                            s2=st.number_input(f"Set {si+1} — {match['b']}",0,20,s2_val, key=ko_widget_key(tid,mid,f"ko_s{si+1}_b"))
                        new_sets.append({"s1":int(s1),"s2":int(s2)})
                    ok,msg=validate_sets(fmt_local,new_sets)
                    if not ok: st.error(msg)
                    gC,gD,_=st.columns([1,1,1])
                    with gC:
                        g1=st.number_input(f"Puntos de oro {match['a']}",0,200,int(match.get("goldenA",0)), key=ko_widget_key(tid,mid,"ko_g1"))
                    with gD:
                        g2=st.number_input(f"Puntos de oro {match['b']}",0,200,int(match.get("goldenB",0)), key=ko_widget_key(tid,mid,"ko_g2"))
                    submitted=st.form_submit_button("Guardar este partido KO", key=ko_widget_key(tid,mid,"ko_save"))
                    if submitted:
                        stats=compute_sets_stats(new_sets)
                        if stats["sets1"]==stats["sets2"]:
                            st.error("Debe haber un ganador en KO (no se permiten empates)."); return
                        saving_key = f"saving_{mid}"
                        if st.session_state.get(saving_key):
                            st.info("Guardando, por favor espera…"); st.stop()
                        st.session_state[saving_key] = True
                        with st.spinner("Guardando partido KO…"):
                            ok = save_ko_match_atomic(tid, mid, new_sets, g1, g2, max_retries=6)
                        st.session_state[saving_key] = False
                        if not ok:
                            st.error("No se pudo confirmar el guardado tras varios intentos. Recargá e intentá nuevamente.")
                            st.stop()
                        st.success("✔ Partido KO guardado")
                        st.rerun()

            # Pintar rondas existentes
            round_order=["R32","R16","QF","SF","FN"]
            for rname in round_order:
                ms=[m for m in state["ko"]["matches"] if m.get("round")==rname]
                if not ms: continue
                st.markdown(f"### {rname}")
                for m in ms:
                    with st.container(border=True):
                        render_playoff_match(m)

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

            # Campeón (si hay final resuelta)
            finals=[m for m in state["ko"]["matches"] if m.get("round")=="FN"]
            for fm in finals:
                sets=fm.get("sets",[])
                if sets and match_has_winner(sets):
                    statsF=compute_sets_stats(sets)
                    champion=fm['a'] if statsF["sets1"]>statsF["sets2"] else fm['b']
                    st.markdown(f"<div class='champion'>🏆 CAMPEÓN: {champion}</div>", unsafe_allow_html=True)
                    st.balloons()
                    break

# ============================
# ====== VIEWER (solo RO) ====
# ============================
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

# ============================
# ====== DASHBOARD ADMIN  ====
# ============================
def admin_dashboard(user:Dict[str,Any]):
    render_header(user)
    st.header(f"Torneos de {user['username']}")
    with st.expander("➕ Crear torneo nuevo", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1: t_name = st.text_input("Nombre del torneo", value="Open Pádel", key="create_t_name")
        with c2: place = st.text_input("Lugar / Club", value="Mi Club", key="create_place")
        with c3: tdate = st.date_input("Fecha", value=date.today(), key="create_date").isoformat()
        with c4: gender = st.selectbox("Género", ["masculino","femenino","mixto"], index=2, key="create_gender")
        if st.button("Crear torneo", type="primary", key="create_btn"):
            tid=create_tournament(user["username"],t_name,place,tdate,gender)
            st.success(f"Torneo creado: {t_name} ({tid})"); st.rerun()

    my=load_index_for_admin(user["username"])
    if not my:
        st.info("Aún no tienes torneos."); st.caption(f"Iapps Padel Tournament · iAPPs Pádel — {APP_VERSION}"); return

    st.subheader("Abrir / eliminar torneo")
    names=[f"{t['date']} — {t['t_name']} ({t['gender']}) — {t['place']} — ID:{t['tournament_id']}" for t in my]
    selected=st.selectbox("Selecciona un torneo", names, index=0, key="adm_sel_t")
    sel=my[names.index(selected)]
    c1,c2,c3=st.columns(3)
    with c1:
        if st.button("Abrir torneo", key="open_t"):
            tournament_manager(user, sel["tournament_id"]); st.stop()
    with c2:
        if st.button("Eliminar torneo", type="secondary", key="del_t"):
            delete_tournament(user["username"], sel["tournament_id"])
            st.success("Torneo eliminado del índice."); st.rerun()
    with c3:
        app_cfg=load_app_config()
        tid=sel["tournament_id"]
        st.caption("Link público (solo lectura) — copia manual:")
        public_url=f"{app_cfg.get('app_base_url','https://iappspadel.streamlit.app')}/?mode=public&tid={tid}"
        st.text_input("URL pública", value=public_url, disabled=False, label_visibility="collapsed", key=f"pub_url_admin_{tid}")

    st.caption(f"Iapps Padel Tournament · iAPPs Pádel — {APP_VERSION}")

# ============================
# ============ MAIN ==========
# ============================
def init_session():
    st.session_state.setdefault("auth_user",None)

def main():
    try: params=st.query_params
    except Exception: params=st.experimental_get_query_params()
    mode=params.get("mode",[""]); mode=mode[0] if isinstance(mode,list) else mode
    _tid=params.get("tid",[""]); _tid=_tid[0] if isinstance(_tid,list) else _tid

    init_session()
    if not st.session_state.get("auth_user"):
        render_header(None)
        st.markdown("### Ingreso — Usuario + PIN (6 dígitos)")
        with st.form("login", clear_on_submit=True):
            username=st.text_input("Usuario", key="login_user").strip()
            pin=st.text_input("PIN (6 dígitos)", type="password", key="login_pin").strip()
            submitted=st.form_submit_button("Ingresar", type="primary", key="login_btn")
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
        if mode=="public" and _tid: viewer_tournament(_tid, public=True); st.caption(f"Iapps Padel Tournament · iAPPs Pádel — {APP_VERSION}"); return
        admin_dashboard(user); return
    if user["role"]=="VIEWER":
        if mode=="public" and _tid: viewer_tournament(_tid, public=True)
        else:
            # viewer interno: ver torneos del admin asignado
            render_header(user)
            st.header(f"Vista de consulta — {user['username']}")
            if not user.get("assigned_admin"):
                st.warning("No asignado a un admin."); return
            my=load_index_for_admin(user["assigned_admin"])
            if not my: st.info("El admin asignado no tiene torneos."); return
            names=[f"{t['date']} — {t['t_name']} ({t['gender']}) — {t['place']} — ID:{t['tournament_id']}" for t in my]
            selected=st.selectbox("Selecciona un torneo para ver", names, index=0, key="vw_sel")
            sel=my[names.index(selected)]
            viewer_tournament(sel["tournament_id"])
        st.caption(f"Iapps Padel Tournament · iAPPs Pádel — {APP_VERSION}"); return

    st.error("Rol desconocido.")
    st.caption(f"Iapps Padel Tournament · iAPPs Pádel — {APP_VERSION}")

if __name__=="__main__":
    main()