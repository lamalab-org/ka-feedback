"""
Streamlit App for Blind Rating of Kinetic Fitting Results (Round 1 vs Round 2)

Features:
1. Blind evaluation of 12 candidate runs (6 pairs of before/after human feedback).
2. Deterministic shuffling per user to prevent bias without losing position on refresh.
3. Completely hidden metadata/directory names to avoid info leaks.
4. Aggregated Admin View for Round 1 vs Round 2 head-to-head comparison.
"""

import streamlit as st
import regex as re
import glob
import json
import base64
import warnings
import hashlib
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
import h5py

# =============================================================================
# Configuration & Hardcoded Mapping
# =============================================================================

PAIRS =[
    {
        "type": "No Feedback (Claude Sonnet 4.5)",
        "r1_path": "round1/kinetic_fitting_no_fb_claudesonnet45_3",
        "r2_path": "round2/kinetic_fitting_no_fb_claudesonnet45_hf1_3",
    },
    {
        "type": "No Feedback (GPT-4o)",
        "r1_path": "round1/kinetic_fitting_no_fb_gpt4o_feedback_4",
        "r2_path": "round2/kinetic_fitting_no_fb_gpt4o_feedback_hf1_4",
    },
    {
        "type": "Text+Chemistry Feedback every round (Claude Sonnet 4.5)",
        "r1_path": "round1/kinetic_fitting_tasks_with_visionfb_and_chemistryopus_no_text_3",
        "r2_path": "round2/kinetic_fitting_tasks_with_visionfb_and_chemistryopus_no_text_hf1_3",
    },
    {
        "type": "Text Feedback (Claude Sonnet 4.5)",
        "r1_path": "round1/kinetic_fitting_with_fb_claudesonnet45_3",
        "r2_path": "round2/kinetic_fitting_with_fb_claudesonnet45_hf1_3",
    },
    {
        "type": "Text+Vision Feedback (Claude Sonnet 4.5)",
        "r1_path": "round1/kinetic_fitting_with_visionfb_claudesonnet45_5",
        "r2_path": "round2/kinetic_fitting_with_visionfb_claudesonnet45_hf1_5",
    },
    {
        "type": "Text+Vision+Chemistry Feedback (Claude Sonnet 4.5)",
        "r1_path": "round1/kinetic_fitting_with_visionfb_claudesonnet45_with_opus_fb_3",
        "r2_path": "round2/kinetic_fitting_with_visionfb_claudesonnet45_with_chem_fb_hf1_3",
    }
]

DATA_PATH = "data/experimental_data.h5"

# GitHub configuration
GITHUB_API = "https://api.github.com"
RATINGS_ROOT = "feedback_ratings"

# Performance settings
MAX_POINTS_PER_TRACE = 200
USE_WEBGL = True

# Color scheme for species
SPECIES_COLORS = {
    "O2": "#2ecc71", "H2O2": "#3498db", "Ru_Dimer": "#27ae60",
    "Ru2_dim": "#27ae60", "Inactive": "#9b59b6", "Ru_inactive": "#9b59b6",
    "Ru2_inactive": "#9b59b6", "RuII": "#e67e22", "RuIII": "#e74c3c",
    "RuII_ex": "#f1c40f", "S2O8": "#8e44ad", "SO4": "#d35400",
}

PARAM_COLORS =[
    "#440154", "#482878", "#3e4989", "#31688e", "#26838f",
    "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"
]

# =============================================================================
# Helper Utilities
# =============================================================================

def _slugify(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")

def downsample_data(x: np.ndarray, y: np.ndarray, max_points: int = MAX_POINTS_PER_TRACE) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    indices = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return x[indices], y[indices]

def get_shuffled_runs(username: str) -> List[dict]:
    """Deterministically shuffles the 12 candidate runs based on the username."""
    runs =[]
    for pair in PAIRS:
        runs.append({"path": pair["r1_path"], "round": "Round 1 (Before FB)", "type": pair["type"]})
        runs.append({"path": pair["r2_path"], "round": "Round 2 (After FB)", "type": pair["type"]})

    # Create a stable seed based on the username
    seed = int(hashlib.md5(username.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    rng.shuffle(runs)

    for i, run in enumerate(runs):
        run["candidate_id"] = f"Candidate {i+1}"

    return runs

# =============================================================================
# GitHub Storage (Adapted for JSON Dicts)
# =============================================================================

def _github_headers() -> dict:
    token = st.secrets.get("GITHUB_TOKEN", "")
    if not token:
        st.error("⚠️ `GITHUB_TOKEN` is not set in Streamlit secrets.")
        st.stop()
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

def _github_branch() -> str:
    return st.secrets.get("GITHUB_BRANCH", "main")

def _github_repo() -> str:
    return st.secrets.get("GITHUB_REPO", "")

def _github_contents_url(repo_path: str) -> str:
    repo = _github_repo()
    return f"{GITHUB_API}/repos/{repo}/contents/{repo_path}"

def _load_json_from_github(repo_path: str) -> Tuple[dict, Optional[str]]:
    url = _github_contents_url(repo_path)
    params = {"ref": _github_branch()}
    resp = requests.get(url, headers=_github_headers(), params=params)
    if resp.status_code == 200:
        payload = resp.json()
        content = base64.b64decode(payload["content"]).decode("utf-8")
        return json.loads(content), payload["sha"]
    return {}, None

def _save_json_to_github(repo_path: str, data: dict, sha: Optional[str] = None) -> bool:
    url = _github_contents_url(repo_path)
    content_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
    encoded = base64.b64encode(content_bytes).decode("utf-8")
    body = {
        "message": f"rating update: {repo_path} @ {datetime.now().isoformat()}",
        "content": encoded,
        "branch": _github_branch(),
    }
    if sha is not None:
        body["sha"] = sha
    resp = requests.put(url, headers=_github_headers(), json=body)
    if resp.status_code not in (200, 201):
        st.error(f"Failed to save ratings ({resp.status_code}): {resp.text}")
        return False
    return True

def load_user_ratings(username: str) -> Tuple[dict, Optional[str]]:
    path = f"{RATINGS_ROOT}/{_slugify(username)}.json"
    return _load_json_from_github(path)

def save_user_ratings(username: str, data: dict, sha: Optional[str]) -> bool:
    path = f"{RATINGS_ROOT}/{_slugify(username)}.json"
    return _save_json_to_github(path, data, sha)

# =============================================================================
# Math / ODE / Data (Identical caching as previous)
# =============================================================================

@dataclass
class Reaction:
    equation: str
    type: str
    reactants: List[str]
    products: List[str]
    stoichiometry: Dict[str, float]
    quantum_yield: Optional[Tuple[float, float]] = None
    fitted_k: Optional[float] = None
    fitted_quantum_yield: Optional[float] = None

    @classmethod
    def from_dict(cls, rxn_dict: dict) -> "Reaction":
        equation = rxn_dict["equation"].replace("<->", "->")
        left, right = equation.split("->")
        reactants =[s.strip() for s in left.split("+") if s.strip()]
        products =[s.strip() for s in right.split("+") if s.strip()]

        ignored = {"hv", "H2O", "OH", "products", "H", "H+", ""}
        stoich = {}

        def parse(species_str):
            if not species_str: return 1, ""
            parts = species_str.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                return int(parts[0]), parts[1]
            return 1, species_str

        for r in reactants:
            c, s = parse(r)
            if s and s not in ignored: stoich[s] = stoich.get(s, 0) - c
        for p in products:
            c, s = parse(p)
            if s and s not in ignored: stoich[s] = stoich.get(s, 0) + c

        return cls(
            equation=rxn_dict["equation"], type=rxn_dict["type"],
            reactants=reactants, products=products, stoichiometry=stoich,
            quantum_yield=rxn_dict.get("quantum_yield"),
            fitted_k=rxn_dict.get("fitted_k"),
            fitted_quantum_yield=rxn_dict.get("fitted_quantum_yield"),
        )

def create_ode_func(reactions: List[Reaction], species_idx: Dict[str, int]):
    PATHLENGTH, EPSILON_RU_II, EPSILON_RU_III, AVOGADRO_NUMBER = 2.25, 8500.0, 540.0, 6.022e23
    VOLUME_L = PATHLENGTH * 1e-3

    def ode_func(y: np.ndarray, t: float, params: dict, conditions: dict) -> np.ndarray:
        dydt = np.zeros_like(y)
        c_ru_ii_M = y[species_idx["RuII"]] * 1e-6 if "RuII" in species_idx else 0
        c_ru_iii_M = y[species_idx["RuIII"]] * 1e-6 if "RuIII" in species_idx else 0
        absorbance_tot = ((c_ru_ii_M * EPSILON_RU_II) + (c_ru_iii_M * EPSILON_RU_III)) * PATHLENGTH

        absorptance_factor = fraction_ru_ii = fraction_ru_iii = 0
        if absorbance_tot >= 1e-9:
            absorptance_factor = 1 - 10 ** (-absorbance_tot)
            fraction_ru_ii = (c_ru_ii_M * EPSILON_RU_II * PATHLENGTH) / absorbance_tot
            fraction_ru_iii = (c_ru_iii_M * EPSILON_RU_III * PATHLENGTH) / absorbance_tot

        if conditions.get("photon_flux"):
            incident_flux = (conditions["photon_flux"] / AVOGADRO_NUMBER / VOLUME_L) * 1e6
        else:
            irr = conditions.get("irradiance", 1000) * 1e-4 / 4.41e-19
            incident_flux = (irr / PATHLENGTH / AVOGADRO_NUMBER * 1000) * 1e6

        for i, rxn in enumerate(reactions):
            rate = 0.0
            if rxn.type == "light":
                flux = 0.0
                if "RuII" in rxn.reactants: flux = incident_flux * absorptance_factor * fraction_ru_ii
                elif "RuIII" in rxn.reactants: flux = incident_flux * absorptance_factor * fraction_ru_iii
                rate = flux * params.get(f"qy_{i}", rxn.quantum_yield[0] if rxn.quantum_yield else 0.1)
            else:
                rate = params.get(f"k_{i}", 1.0)
                for r in rxn.reactants:
                    if r in species_idx: rate *= y[species_idx[r]] ** (2 if rxn.equation.startswith(f"2 {r}") else 1)

            for s, coeff in rxn.stoichiometry.items():
                if s in species_idx: dydt[species_idx[s]] += coeff * rate

        return dydt
    return ode_func

@st.cache_data(ttl=3600, show_spinner=False)
def simulate_species_evolution_cached(
    network_json: str, params_json: str, conditions_json: str, t_min: float, t_max: float, n_points: int = 200
):
    network, params, conditions = json.loads(network_json), json.loads(params_json), json.loads(conditions_json)
    reactions =[Reaction.from_dict(r) for r in network.get("reactions", [])]
    
    all_species = set()
    for rxn in reactions: all_species.update(rxn.stoichiometry.keys())
    species_list = sorted({s for s in all_species if s and s.strip()})
    species_idx = {sp: i for i, sp in enumerate(species_list)}
    
    ode_func = create_ode_func(reactions, species_idx)
    time_points = np.linspace(t_min, t_max, n_points)
    
    y0 = np.zeros(len(species_list))
    if "RuII" in species_idx: y0[species_idx["RuII"]] = conditions.get("c_Ru", 10)
    if "S2O8" in species_idx: y0[species_idx["S2O8"]] = conditions.get("c_S2O8", 6000)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_solution = odeint(ode_func, y0, time_points, args=(params, conditions), rtol=1e-6, atol=1e-8)

    return y_solution.tolist(), species_list, species_idx, time_points.tolist()

def create_synthetic_experiment_data() -> dict:
    time = np.linspace(0, 600, 100)
    oxygen = np.maximum(50 * (1 - np.exp(-time / 100)) + np.random.normal(0, 1, len(time)), 0)
    return {"time": time.tolist(), "oxygen": oxygen.tolist(), "metadata": {"c_Ru": 50.0, "c_S2O8": 6000.0, "irradiance": 1000.0, "pH": 7.0}}

def create_synthetic_dataset() -> Dict[str, dict]:
    return {"Synthetic_1": create_synthetic_experiment_data()} # Minimal mock

# =============================================================================
# Plotting Functions (Copied identically)
# =============================================================================

def create_interactive_concentration_plot(network, params, exp_data, exp_name="Rep Exp"):
    time_exp, oxygen_exp = np.array(exp_data["time"]), np.array(exp_data["oxygen"])
    time_ds, oxygen_ds = downsample_data(time_exp, oxygen_exp)

    y_sol, s_list, s_idx, t_sim = simulate_species_evolution_cached(
        json.dumps(network, default=str), json.dumps(params, default=str),
        json.dumps(exp_data["metadata"], default=str), float(time_exp.min()), float(time_exp.max())
    )
    y_sol, t_sim = np.array(y_sol), np.array(t_sim)

    bulk = {"S2O8", "SO4"}
    cats = {"RuII", "RuIII", "Ru_Dimer", "Ru2_dim", "Inactive", "Ru_inactive", "Ru2_inactive", "RuII_ex"}
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("A: Products & Intermediates", "B: Catalyst Species"))
    ScatterClass = go.Scattergl if USE_WEBGL else go.Scatter

    fig.add_trace(ScatterClass(x=time_ds, y=oxygen_ds, mode="markers", name="[O₂] Exp", marker=dict(color="gray", size=5), legend="legend1"), row=1, col=1)

    for sp in[s for s in s_list if s not in bulk]:
        if sp in s_idx: fig.add_trace(go.Scatter(x=t_sim, y=y_sol[:, s_idx[sp]], mode="lines", name=f"[{sp}]", line=dict(color=SPECIES_COLORS.get(sp, "#666")), legend="legend1"), row=1, col=1)

    for sp in[s for s in s_list if s in cats]:
        if sp in s_idx: fig.add_trace(go.Scatter(x=t_sim, y=y_sol[:, s_idx[sp]], mode="lines", name=f"[{sp}]", line=dict(color=SPECIES_COLORS.get(sp, "#666")), legend="legend2"), row=1, col=2)

    fig.update_layout(title=dict(text=f"Species Evolution: {exp_name}", font=dict(size=14)), height=450, template="plotly_white", margin=dict(l=50, r=20, t=50, b=50))
    return fig

# =============================================================================
# Reading Results
# =============================================================================

def _find_best_phenomenological_result(output_dir: str):
    base = Path(output_dir)
    result_files = glob.glob(f"{base.as_posix()}/phenomenologic_result.json.*")
    
    best_score, best_data, best_image = -1.0, None, None
    for filepath in result_files:
        ts = re.search(r"phenomenologic_result\.json\.(\d+)$", filepath)
        if ts:
            try:
                with open(filepath, "r") as f: data = json.load(f)
                score = data.get("phenomenological_trends", {}).get("overall_score", -1)
                if score > best_score:
                    best_score, best_data = score, data
                    cand_img = base / f"phenomenological_trends_{ts.group(1)}.png"
                    if cand_img.exists(): best_image = cand_img
            except: pass
    return best_data, best_image

def format_reaction_for_display(reaction: dict) -> dict:
    formatted = {
        "Equation": reaction.get("equation", "N/A"),
        "Type": reaction.get("type", "N/A"),
        "Description": reaction.get("description", "N/A"),
    }
    if reaction.get("type") == "light":
        formatted["Fitted Parameter"] = f"QY = {reaction.get('fitted_quantum_yield', 0):.4f}"
    else:
        formatted["Fitted Parameter"] = f"k = {reaction.get('fitted_k', 0):.4e}"
    return formatted

# =============================================================================
# UI
# =============================================================================

def render_login_screen():
    st.markdown("<div style='text-align:center; padding: 2rem;'><h2>👤 Who are you?</h2><p style='color:#888;'>Enter your name to start the blind evaluation.</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        name = st.text_input("Your name", placeholder="e.g., Dr. Smith").strip()
        if st.button("🔓 Start Evaluating", use_container_width=True, type="primary"):
            if name: return name
            st.error("Please enter your name.")
    return None

def render_candidate_view(run_info: dict, rating_data: dict, all_exp_data: dict, username: str, sha: Optional[str]):
    candidate_id = run_info["candidate_id"]
    st.header(f"🔍 Inspecting: {candidate_id}")
    st.caption("Review the reaction network parameters and fits below, then submit your rating.")

    best_data, best_image = _find_best_phenomenological_result(run_info["path"])

    if not best_data:
        st.warning(f"⚠️ Simulation data missing for {candidate_id}. It might not have converged.")
        # Provide fallback rating mechanism
    else:
        # 1. Reactions Table
        reactions = best_data.get("network", {}).get("reactions",[])
        if reactions:
            df = pd.DataFrame([{**format_reaction_for_display(rxn), "#": i+1} for i, rxn in enumerate(reactions)])
            df = df[["#", "Equation", "Type", "Fitted Parameter", "Description"]]
            st.dataframe(df.style.map(lambda v: "background-color: #fff3cd" if v == "light" else "", subset=["Type"]), hide_index=True, use_container_width=True)

        # 2. Image & Plots
        col_plot, col_img = st.columns([3, 2])
        with col_img:
            if best_image:
                st.image(str(best_image), use_container_width=True)
            else:
                st.info("No phenomenological trend image available.")

        with col_plot:
            rep_exp_data = list(all_exp_data.values())[0] if all_exp_data else create_synthetic_experiment_data()
            params = best_data.get("phenomenological_trends", {}).get("global_params", {})
            for i, rxn in enumerate(reactions):
                if rxn.get("type") == "light" and "fitted_quantum_yield" in rxn: params[f"qy_{i}"] = rxn["fitted_quantum_yield"]
                elif "fitted_k" in rxn: params[f"k_{i}"] = rxn["fitted_k"]
            
            try:
                fig = create_interactive_concentration_plot(best_data["network"], params, rep_exp_data, "Sample Data")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Plot error: {e}")

    # Rating Widget
    st.markdown("---")
    st.subheader("📝 Your Rating")
    
    current_rating = rating_data.get(candidate_id, {})
    
    with st.form(key=f"form_{candidate_id}"):
        score = st.slider("Rate the chemical validity and fit quality (1 = Poor, 10 = Excellent)", 1, 10, current_rating.get("score", 5))
        comments = st.text_area("Observations (Optional)", value=current_rating.get("comments", ""), height=100)
        
        if st.form_submit_button("💾 Save Rating", type="primary"):
            rating_data[candidate_id] = {
                "score": score,
                "comments": comments,
                "true_round": run_info["round"],
                "true_type": run_info["type"],
                "true_path": run_info["path"],
                "timestamp": datetime.now().isoformat()
            }
            if save_user_ratings(username, rating_data, sha):
                st.success("✅ Saved!")
                st.rerun()

def render_admin_view():
    st.title("📊 Admin Panel: Round 1 vs Round 2 Comparison")
    
    # Load all user JSON files
    url = f"{GITHUB_API}/repos/{_github_repo()}/contents/{RATINGS_ROOT}"
    resp = requests.get(url, headers=_github_headers(), params={"ref": _github_branch()})
    
    if resp.status_code != 200:
        st.info("No rating files found or GitHub misconfigured.")
        return

    all_ratings = []
    for item in resp.json():
        if item["name"].endswith(".json"):
            data, _ = _load_json_from_github(item["path"])
            for cand_id, r_info in data.items():
                r_info["User"] = item["name"].replace(".json", "")
                all_ratings.append(r_info)
    
    if not all_ratings:
        st.info("No ratings collected yet.")
        return

    df = pd.DataFrame(all_ratings)
    
    # Pivot to compare Round 1 and Round 2
    st.subheader("Aggregate Scores Head-to-Head")
    try:
        pivot_df = df.pivot_table(
            index="true_type",
            columns="true_round",
            values="score",
            aggfunc=["mean", "count"]
        )
        
        # Flatten MultiIndex
        pivot_df.columns = [f"{col[1]} ({col[0]})" for col in pivot_df.columns]
        
        # Calculate Delta
        r1_col = "Round 1 (Before FB) (mean)"
        r2_col = "Round 2 (After FB) (mean)"
        
        if r1_col in pivot_df.columns and r2_col in pivot_df.columns:
            pivot_df["Improvement (Avg)"] = pivot_df[r2_col] - pivot_df[r1_col]
            
        st.dataframe(pivot_df.style.format(precision=2), use_container_width=True)
    except Exception as e:
        st.warning(f"Not enough data to construct comparison table: {e}")

    # Raw Data Export
    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "📥 Download CSV", 
        df.to_csv(index=False).encode('utf-8'), 
        "blind_ratings_export.csv"
    )

# =============================================================================
# Main Layout
# =============================================================================

def main():
    st.set_page_config(page_title="Blind Kinetic Rating", page_icon="⚖️", layout="wide")

    # Routing
    query_params = st.query_params
    if "admin" in query_params:
        render_admin_view()
        return

    if "current_user" not in st.session_state:
        chosen = render_login_screen()
        if chosen:
            st.session_state["current_user"] = chosen
            st.rerun()
        st.stop()

    username = st.session_state["current_user"]
    shuffled_runs = get_shuffled_runs(username)
    rating_data, sha = load_user_ratings(username)

    # Sidebar Navigation
    st.sidebar.markdown(f"### 👤 {username}")
    if st.sidebar.button("🔄 Switch User"):
        del st.session_state["current_user"]
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Candidates")
    
    # Progress check
    completed = len(rating_data.keys())
    st.sidebar.progress(completed / 12.0, text=f"{completed}/12 Evaluated")

    # Selection Menu
    options =[]
    for run in shuffled_runs:
        cid = run["candidate_id"]
        icon = "✅" if cid in rating_data else "⭕"
        options.append(f"{icon} {cid}")

    selected_option = st.sidebar.radio("Select Candidate to Evaluate", options)
    selected_idx = options.index(selected_option)
    selected_run = shuffled_runs[selected_idx]

    # Main view
    # Load basic mock data since loading full HDF5 over and over is heavy
    # (If you want real data, point create_synthetic_dataset() -> load_experimental_data_cached(DATA_PATH))
    all_exp_data = create_synthetic_dataset() 

    render_candidate_view(selected_run, rating_data, all_exp_data, username, sha)

if __name__ == "__main__":
    main()