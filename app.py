"""
Streamlit App for Collecting Chemist Feedback on Kinetic Fitting Results

OPTIMIZED VERSION with:
1. Cached ODE simulations to avoid redundant computation
2. Data downsampling for faster rendering
3. Aggregated traces instead of per-experiment traces
4. Lazy loading of visualizations
5. WebGL renderer for large datasets

Storage layout on GitHub:
    feedback/{username_slug}/{run_type_slug}.json
"""

import streamlit as st
import regex as re
import glob
import json
import base64
import warnings
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
import h5py

# =============================================================================
# Configuration
# =============================================================================

RUN_TEMPLATES = {
    "No Feedback (Claude Sonnet 4.5)": "kinetic_fitting_no_fb_claudesonnet45_*",
    "No Feedback (GPT-4o)": "kinetic_fitting_no_fb_gpt4o_feedback_*",
    "Text Feedback (Claude Sonnet 4.5)": "kinetic_fitting_with_fb_claudesonnet45_*",
    "Text+Vision Feedback (Claude Sonnet 4.5)": "kinetic_fitting_with_visionfb_claudesonnet45_*",
    "Text+Vision+Chemistry Feedback (Claude Sonnet 4.5)": "kinetic_fitting_with_visionfb_claudesonnet45_with_opus_fb_*",
    "Text+Chemistry Feedback every round (Claude Sonnet 4.5)": "kinetic_fitting_tasks_with_visionfb_and_chemistryopus_no_text_*",
}

EXCLUDE_PATTERNS = {
    "kinetic_fitting_with_visionfb_claudesonnet45_*": [
        "kinetic_fitting_with_visionfb_claudesonnet45_with_opus_fb_*"
    ],
}

BASE_DIR = Path("round1")
VIZ_DIR = Path("round_1_viz")
DATA_PATH = "data/experimental_data.h5"

# GitHub configuration
GITHUB_API = "https://api.github.com"
FEEDBACK_ROOT = "feedback"

# Performance settings
MAX_POINTS_PER_TRACE = 200  # Downsample to this many points
USE_WEBGL = True  # Use WebGL for scatter plots

# Color scheme for species
SPECIES_COLORS = {
    "O2": "#2ecc71",
    "H2O2": "#3498db",
    "Ru_Dimer": "#27ae60",
    "Ru2_dim": "#27ae60",
    "Inactive": "#9b59b6",
    "Ru_inactive": "#9b59b6",
    "Ru2_inactive": "#9b59b6",
    "RuII": "#e67e22",
    "RuIII": "#e74c3c",
    "RuII_ex": "#f1c40f",
    "S2O8": "#8e44ad",
    "SO4": "#d35400",
}

# Parameter variation colors (Viridis-like)
PARAM_COLORS = [
    "#440154", "#482878", "#3e4989", "#31688e", "#26838f",
    "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"
]


# =============================================================================
# Performance Utilities
# =============================================================================

def downsample_data(x: np.ndarray, y: np.ndarray, max_points: int = MAX_POINTS_PER_TRACE) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample data to max_points using LTTB-like algorithm for visual fidelity."""
    if len(x) <= max_points:
        return x, y
    
    # Use simple uniform sampling with preserved endpoints
    indices = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return x[indices], y[indices]


def hash_params(params: dict, conditions: dict) -> str:
    """Create a hash key for caching simulations."""
    key_str = json.dumps({"p": params, "c": conditions}, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


# =============================================================================
# Reaction Parsing
# =============================================================================

@dataclass
class Reaction:
    """Represents a chemical reaction in the kinetic network."""
    equation: str
    type: str
    reactants: List[str]
    products: List[str]
    stoichiometry: Dict[str, float]
    k_range: Optional[Tuple[float, float]] = None
    quantum_yield: Optional[Tuple[float, float]] = None
    fitted_k: Optional[float] = None
    fitted_quantum_yield: Optional[float] = None

    @classmethod
    def from_dict(cls, rxn_dict: dict) -> "Reaction":
        equation = rxn_dict["equation"]
        if "<->" in equation:
            equation = equation.replace("<->", "->")

        if "->" not in equation:
            raise ValueError(f"Invalid equation format: '{equation}'")

        left, right = equation.split("->")
        reactants = [s.strip() for s in left.split("+") if s.strip()]
        products = [s.strip() for s in right.split("+") if s.strip()]

        ignored = {"hv", "H2O", "OH", "products", "H", "H+", ""}
        stoich = {}

        def parse_species_with_coeff(species_str):
            species_str = species_str.strip()
            if not species_str:
                return 1, ""
            parts = species_str.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                return int(parts[0]), parts[1]
            return 1, species_str

        for r in reactants:
            if r and r not in ignored:
                coeff, species = parse_species_with_coeff(r)
                if species and species not in ignored:
                    stoich[species] = stoich.get(species, 0) - coeff

        for p in products:
            if p and p not in ignored:
                coeff, species = parse_species_with_coeff(p)
                if species and species not in ignored:
                    stoich[species] = stoich.get(species, 0) + coeff

        return cls(
            equation=rxn_dict["equation"],
            type=rxn_dict["type"],
            reactants=reactants,
            products=products,
            stoichiometry=stoich,
            k_range=rxn_dict.get("k_range"),
            quantum_yield=rxn_dict.get("quantum_yield"),
            fitted_k=rxn_dict.get("fitted_k"),
            fitted_quantum_yield=rxn_dict.get("fitted_quantum_yield"),
        )


# =============================================================================
# ODE System (Cached)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def create_ode_system_cached(network_json: str):
    """Create ODE system from network JSON (cached)."""
    network = json.loads(network_json)
    reactions = [Reaction.from_dict(r) for r in network.get("reactions", [])]

    all_species = set()
    for rxn in reactions:
        all_species.update(rxn.stoichiometry.keys())
    all_species = {s for s in all_species if s and s.strip()}

    species_list = sorted(all_species)
    species_idx = {sp: i for i, sp in enumerate(species_list)}

    return species_list, species_idx, network_json


def create_ode_func(reactions: List[Reaction], species_idx: Dict[str, int]):
    """Create the ODE function for integration."""
    PATHLENGTH = 2.25
    EPSILON_RU_II = 8500.0
    EPSILON_RU_III = 540.0
    AVOGADRO_NUMBER = 6.022e23
    VOLUME_L = PATHLENGTH * 1e-3

    def ode_func(y: np.ndarray, t: float, params: dict, conditions: dict) -> np.ndarray:
        dydt = np.zeros_like(y)

        c_ru_ii_M = y[species_idx["RuII"]] * 1e-6 if "RuII" in species_idx else 0
        c_ru_iii_M = y[species_idx["RuIII"]] * 1e-6 if "RuIII" in species_idx else 0

        absorbance_tot = (
            (c_ru_ii_M * EPSILON_RU_II) + (c_ru_iii_M * EPSILON_RU_III)
        ) * PATHLENGTH

        if absorbance_tot < 1e-9:
            absorptance_factor = 0
            fraction_ru_ii = 0
            fraction_ru_iii = 0
        else:
            absorptance_factor = 1 - 10 ** (-absorbance_tot)
            fraction_ru_ii = (c_ru_ii_M * EPSILON_RU_II * PATHLENGTH) / absorbance_tot
            fraction_ru_iii = (c_ru_iii_M * EPSILON_RU_III * PATHLENGTH) / absorbance_tot

        if conditions.get("photon_flux") is not None:
            photon_flux_photons_s = conditions["photon_flux"]
            photon_flux_mol_s = photon_flux_photons_s / AVOGADRO_NUMBER
            incident_flux_M_s = photon_flux_mol_s / VOLUME_L
            incident_flux = incident_flux_M_s * 1e6
        else:
            irradiance_W_m2 = conditions.get("irradiance", 1000)
            irradiance_W_cm2 = irradiance_W_m2 * 1e-4
            E_photon_J = 4.41e-19
            photon_flux_approx = irradiance_W_cm2 / E_photon_J
            photon_flux_vol_photons_s_cm3 = photon_flux_approx / PATHLENGTH
            photon_flux_vol_mol_s_cm3 = photon_flux_vol_photons_s_cm3 / AVOGADRO_NUMBER
            photon_flux_vol_M_s = photon_flux_vol_mol_s_cm3 * 1000
            incident_flux = photon_flux_vol_M_s * 1e6

        for i, rxn in enumerate(reactions):
            rate = 0.0

            if rxn.type == "light":
                is_ru_ii_absorber = "RuII" in rxn.reactants
                is_ru_iii_absorber = "RuIII" in rxn.reactants

                absorbed_flux = 0.0
                if is_ru_ii_absorber:
                    absorbed_flux = incident_flux * absorptance_factor * fraction_ru_ii
                elif is_ru_iii_absorber:
                    absorbed_flux = incident_flux * absorptance_factor * fraction_ru_iii

                qy = params.get(f"qy_{i}", rxn.quantum_yield[0] if rxn.quantum_yield else 0.1)
                rate = absorbed_flux * qy
            else:
                k = params.get(f"k_{i}", 1.0)
                rate = k

                for reactant in rxn.reactants:
                    if reactant in species_idx:
                        if rxn.equation.startswith(f"2 {reactant}"):
                            rate *= y[species_idx[reactant]] ** 2
                        else:
                            rate *= y[species_idx[reactant]]
                    elif reactant in ["H2O", "H+"]:
                        rate *= 1.0

            for species, coeff in rxn.stoichiometry.items():
                if species in species_idx:
                    dydt[species_idx[species]] += coeff * rate

        return dydt

    return ode_func


@st.cache_data(ttl=3600, show_spinner=False)
def simulate_species_evolution_cached(
    network_json: str,
    params_json: str,
    conditions_json: str,
    time_min: float,
    time_max: float,
    n_points: int = 200,
) -> Tuple[List[List[float]], List[str], Dict[str, int]]:
    """Cached simulation of species evolution."""
    network = json.loads(network_json)
    params = json.loads(params_json)
    conditions = json.loads(conditions_json)
    
    reactions = [Reaction.from_dict(r) for r in network.get("reactions", [])]
    
    all_species = set()
    for rxn in reactions:
        all_species.update(rxn.stoichiometry.keys())
    all_species = {s for s in all_species if s and s.strip()}
    species_list = sorted(all_species)
    species_idx = {sp: i for i, sp in enumerate(species_list)}
    
    ode_func = create_ode_func(reactions, species_idx)
    
    time_points = np.linspace(time_min, time_max, n_points)
    
    y0 = np.zeros(len(species_list))
    if "RuII" in species_idx:
        y0[species_idx["RuII"]] = conditions.get("c_Ru", 10)
    if "S2O8" in species_idx:
        y0[species_idx["S2O8"]] = conditions.get("c_S2O8", 6000)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_solution = odeint(
            ode_func,
            y0,
            time_points,
            args=(params, conditions),
            rtol=1e-6,
            atol=1e-8,
        )

    return y_solution.tolist(), species_list, species_idx, time_points.tolist()


def build_params_from_network(network: dict) -> dict:
    """Build parameter dictionary from fitted values in the network."""
    params = {}
    for i, rxn in enumerate(network.get("reactions", [])):
        if rxn.get("type") == "light":
            if rxn.get("fitted_quantum_yield") is not None:
                params[f"qy_{i}"] = rxn["fitted_quantum_yield"]
        else:
            if rxn.get("fitted_k") is not None:
                params[f"k_{i}"] = rxn["fitted_k"]
    return params


# =============================================================================
# Data Loading (Cached)
# =============================================================================

@dataclass
class ExperimentMetadata:
    experiment_name: str
    power_output: float
    ru_concentration: float
    oxidant_concentration: float
    buffer_concentration: float
    pH: float
    buffer_used: int
    photon_flux: Optional[float] = None
    annotations: str = ""
    color: str = "#ce1480"


@dataclass
class TimeSeriesData:
    time_reaction: np.ndarray
    data_reaction: np.ndarray
    y_fit: np.ndarray
    baseline_y: np.ndarray
    lbc_fit_y: np.ndarray
    full_x_values: np.ndarray
    full_y_corrected: np.ndarray
    x_diff: np.ndarray
    y_diff: np.ndarray
    y_diff_smoothed: np.ndarray
    y_diff_fit: np.ndarray
    time_full: np.ndarray
    data_full: np.ndarray


@dataclass
class AnalysisMetadata:
    p: np.ndarray
    max_rate: float
    max_rate_ydiff: float
    initial_state: np.ndarray
    matrix: str
    rate_constant: float
    rxn_start: int
    rxn_end: int
    residual: np.ndarray
    idx_for_fitting: int


@dataclass
class DataSets:
    data_corrected: np.ndarray


@dataclass
class ExperimentalData:
    time_series_data: TimeSeriesData
    experiment_metadata: ExperimentMetadata
    analysis_metadata: AnalysisMetadata
    datasets: DataSets


@dataclass
class ExperimentalDataset:
    experiments: Dict[str, "ExperimentalData"] = field(default_factory=dict)
    overview_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    @classmethod
    def load_from_hdf5(cls, filename: str):
        dataset = cls()

        try:
            dataset.overview_df = pd.read_hdf(filename, key="overview_df")
        except (KeyError, ValueError):
            dataset.overview_df = pd.DataFrame()

        with h5py.File(filename, "r") as f:
            for exp_name in f.keys():
                if exp_name == "overview_df":
                    continue
                try:
                    time_series_dict = dict(f[f"{exp_name}/time_series_data"].attrs)
                    time_series_data = TimeSeriesData(**time_series_dict)

                    exp_metadata_dict = dict(f[f"{exp_name}/experiment_metadata"].attrs)
                    valid_fields = ExperimentMetadata.__annotations__.keys()
                    filtered_metadata = {
                        k: v for k, v in exp_metadata_dict.items() if k in valid_fields
                    }
                    experiment_metadata = ExperimentMetadata(**filtered_metadata)

                    analysis_metadata_dict = dict(f[f"{exp_name}/analysis_metadata"].attrs)
                    analysis_metadata = AnalysisMetadata(**analysis_metadata_dict)

                    datasets_dict = dict(f[f"{exp_name}/datasets"].attrs)
                    datasets = DataSets(**datasets_dict)

                    experimental_data = ExperimentalData(
                        time_series_data,
                        experiment_metadata,
                        analysis_metadata,
                        datasets,
                    )
                    dataset.experiments[exp_name] = experimental_data

                except Exception:
                    continue

        return dataset


@st.cache_data(ttl=3600, show_spinner=False)
def load_experimental_data_cached(data_path: str) -> Dict[str, Any]:
    """Load and cache experimental data from HDF5 file."""
    try:
        dataset = ExperimentalDataset.load_from_hdf5(data_path)

        data = {}
        for exp_name, exp_data in dataset.experiments.items():
            time = exp_data.time_series_data.time_reaction
            oxygen = exp_data.time_series_data.data_reaction

            if not dataset.overview_df.empty:
                overview_row = dataset.overview_df[
                    dataset.overview_df["Experiment"] == exp_name
                ]
                if not overview_row.empty:
                    row = overview_row.iloc[0]
                    standardized_metadata = {
                        "c_Ru": row.get("c([Ru(bpy(3]Cl2) [M]", 0) * 1e6,
                        "c_S2O8": row.get("c(Na2S2O8) [M]", 0) * 1e6,
                        "power_output": row.get("Power output [W/m^2]", 1000),
                        "pH": row.get("pH [-]", 7.0),
                        "irradiance": row.get("Power output [W/m^2]", 1000),
                        "photon_flux": row.get("photon_flux", exp_data.experiment_metadata.photon_flux),
                    }
                else:
                    standardized_metadata = {
                        "c_Ru": exp_data.experiment_metadata.ru_concentration,
                        "c_S2O8": exp_data.experiment_metadata.oxidant_concentration,
                        "power_output": exp_data.experiment_metadata.power_output,
                        "pH": exp_data.experiment_metadata.pH,
                        "irradiance": exp_data.experiment_metadata.power_output,
                        "photon_flux": exp_data.experiment_metadata.photon_flux,
                    }
            else:
                standardized_metadata = {
                    "c_Ru": exp_data.experiment_metadata.ru_concentration,
                    "c_S2O8": exp_data.experiment_metadata.oxidant_concentration,
                    "power_output": exp_data.experiment_metadata.power_output,
                    "pH": exp_data.experiment_metadata.pH,
                    "irradiance": exp_data.experiment_metadata.power_output,
                    "photon_flux": exp_data.experiment_metadata.photon_flux,
                }

            data[exp_name] = {
                "time": time.tolist() if hasattr(time, "tolist") else time,
                "oxygen": oxygen.tolist() if hasattr(oxygen, "tolist") else oxygen,
                "metadata": standardized_metadata,
            }

        return data

    except Exception as e:
        raise ValueError(f"Could not load experimental data: {e}")


# =============================================================================
# Slug helper
# =============================================================================

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text


# =============================================================================
# GitHub-backed storage
# =============================================================================

def _github_headers() -> dict:
    token = st.secrets.get("GITHUB_TOKEN", "")
    if not token:
        st.error("⚠️ `GITHUB_TOKEN` is not set in Streamlit secrets.")
        st.stop()
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }


def _github_branch() -> str:
    return st.secrets.get("GITHUB_BRANCH", "main")


def _github_repo() -> str:
    return st.secrets.get("GITHUB_REPO", "")


def _feedback_file_path(username: str, run_type: str) -> str:
    return f"{FEEDBACK_ROOT}/{_slugify(username)}/{_slugify(run_type)}.json"


def _github_contents_url(repo_path: str) -> str:
    repo = _github_repo()
    return f"{GITHUB_API}/repos/{repo}/contents/{repo_path}"


def _load_json_from_github(repo_path: str) -> Tuple[list, Optional[str]]:
    url = _github_contents_url(repo_path)
    params = {"ref": _github_branch()}
    resp = requests.get(url, headers=_github_headers(), params=params)

    if resp.status_code == 200:
        payload = resp.json()
        content = base64.b64decode(payload["content"]).decode("utf-8")
        return json.loads(content), payload["sha"]
    elif resp.status_code == 404:
        return [], None
    else:
        st.error(f"GitHub API error ({resp.status_code}): {resp.text}")
        return [], None


def _save_json_to_github(repo_path: str, data: list, sha: Optional[str] = None) -> bool:
    url = _github_contents_url(repo_path)
    content_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
    encoded = base64.b64encode(content_bytes).decode("utf-8")

    body = {
        "message": f"feedback: {repo_path} @ {datetime.now().isoformat()}",
        "content": encoded,
        "branch": _github_branch(),
    }
    if sha is not None:
        body["sha"] = sha

    resp = requests.put(url, headers=_github_headers(), json=body)
    if resp.status_code not in (200, 201):
        st.error(f"Failed to save feedback ({resp.status_code}): {resp.text}")
        return False
    return True


def load_user_run_feedback(username: str, run_type: str) -> Tuple[list, Optional[str]]:
    path = _feedback_file_path(username, run_type)
    return _load_json_from_github(path)


def save_user_run_feedback(username: str, run_type: str, entries: list, sha: Optional[str] = None) -> bool:
    path = _feedback_file_path(username, run_type)
    return _save_json_to_github(path, entries, sha)


def _list_github_dir(repo_path: str) -> List[dict]:
    url = _github_contents_url(repo_path)
    params = {"ref": _github_branch()}
    resp = requests.get(url, headers=_github_headers(), params=params)
    if resp.status_code == 200:
        return resp.json()
    return []


def list_all_feedback_users() -> List[str]:
    items = _list_github_dir(FEEDBACK_ROOT)
    return sorted(item["name"] for item in items if item.get("type") == "dir")


def list_user_run_files(user_slug: str) -> List[str]:
    items = _list_github_dir(f"{FEEDBACK_ROOT}/{user_slug}")
    return sorted(
        item["name"]
        for item in items
        if item.get("type") == "file" and item["name"].endswith(".json")
    )


def load_all_feedback() -> List[dict]:
    all_entries = []
    for user_slug in list_all_feedback_users():
        for filename in list_user_run_files(user_slug):
            repo_path = f"{FEEDBACK_ROOT}/{user_slug}/{filename}"
            entries, _ = _load_json_from_github(repo_path)
            for entry in entries:
                entry.setdefault("_user_slug", user_slug)
                entry.setdefault("_file", filename)
            all_entries.extend(entries)
    return all_entries


# =============================================================================
# Result loading helpers
# =============================================================================

def _find_best_phenomenological_result(output_dir: Path) -> Tuple[Optional[Path], Optional[dict], Optional[Path]]:
    output_dir = Path(output_dir)
    pattern = f"{output_dir.as_posix()}/phenomenologic_result.json.*"
    result_files = glob.glob(pattern)

    if not result_files:
        return None, None, None

    best_score = -1.0
    best_file = None
    best_data = None
    best_timestamp = None

    for filepath in result_files:
        match = re.search(r"phenomenologic_result\.json\.(\d+)$", filepath)
        if not match:
            continue
        timestamp = match.group(1)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            overall_score = data.get("phenomenological_trends", {}).get("overall_score", -1)
            if overall_score > best_score:
                best_score = overall_score
                best_file = Path(filepath)
                best_data = data
                best_timestamp = timestamp
        except (json.JSONDecodeError, IOError):
            continue

    best_image = None
    if best_timestamp is not None:
        candidate = output_dir / f"phenomenological_trends_{best_timestamp}.png"
        if candidate.exists():
            best_image = candidate

    return best_file, best_data, best_image


def get_available_runs(template: str) -> List[Path]:
    pattern = BASE_DIR / template
    matched_dirs = set(glob.glob(str(pattern)))

    for exclude_tmpl in EXCLUDE_PATTERNS.get(template, []):
        exclude_pattern = BASE_DIR / exclude_tmpl
        exclude_dirs = set(glob.glob(str(exclude_pattern)))
        matched_dirs -= exclude_dirs

    return sorted(Path(d) for d in matched_dirs if Path(d).is_dir())


def format_reaction_for_display(reaction: dict) -> dict:
    formatted = {
        "Equation": reaction.get("equation", "N/A"),
        "Type": reaction.get("type", "N/A"),
        "Description": reaction.get("description", "N/A"),
    }

    if reaction.get("type") == "light":
        qy = reaction.get("fitted_quantum_yield")
        qy_range = reaction.get("quantum_yield", [])
        formatted["Fitted Parameter"] = f"QY = {qy:.4f}" if qy else "N/A"
        formatted["Range"] = f"[{qy_range[0]}, {qy_range[1]}]" if len(qy_range) == 2 else "N/A"
    else:
        k = reaction.get("fitted_k")
        k_range = reaction.get("k_range", [])
        formatted["Fitted Parameter"] = f"k = {k:.4e}" if k else "N/A"
        formatted["Range"] = f"[{k_range[0]:.0e}, {k_range[1]:.0e}]" if len(k_range) == 2 else "N/A"

    return formatted


def extract_run_number(name: str) -> str:
    match = re.search(r"_(\d+)$", name)
    if match:
        return f"Run {match.group(1)}"
    return name


# =============================================================================
# OPTIMIZED Interactive Plotly Visualizations
# =============================================================================

def create_interactive_concentration_plot(
    network: dict,
    params: dict,
    exp_data: dict,
    experiment_name: str = "Representative Experiment",
) -> go.Figure:
    """
    OPTIMIZED: Create interactive concentration-time plot using Plotly.
    Uses cached simulations and downsampled data.
    Each subplot has its own legend positioned below it.
    """
    time_exp = np.array(exp_data["time"])
    oxygen_exp = np.array(exp_data["oxygen"])
    conditions = exp_data["metadata"]

    # Downsample experimental data
    time_ds, oxygen_ds = downsample_data(time_exp, oxygen_exp)

    # Use cached simulation
    network_json = json.dumps(network, sort_keys=True, default=str)
    params_json = json.dumps(params, sort_keys=True, default=str)
    conditions_json = json.dumps(conditions, sort_keys=True, default=str)
    
    y_solution, species_list, species_idx, time_sim = simulate_species_evolution_cached(
        network_json, params_json, conditions_json,
        float(time_exp.min()), float(time_exp.max()), n_points=MAX_POINTS_PER_TRACE
    )
    
    y_solution = np.array(y_solution)
    time_sim = np.array(time_sim)

    # Species to exclude (bulk species)
    bulk_species = {"S2O8", "SO4"}
    panel_a_species = [s for s in species_list if s and s not in bulk_species]
    
    catalyst_species_names = {
        "RuII", "RuIII", "Ru_Dimer", "Ru2_dim", "Inactive",
        "Ru_inactive", "Ru2_inactive", "RuII_ex",
    }
    panel_b_species = [s for s in species_list if s and s in catalyst_species_names]

    # Create subplots with more space for legends
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("A: Products & Intermediates", "B: Catalyst Species"),
        horizontal_spacing=0.12,
    )

    # Use Scattergl for WebGL rendering (faster)
    ScatterClass = go.Scattergl if USE_WEBGL else go.Scatter

    # Panel A: Products and intermediates with experimental O2
    fig.add_trace(
        ScatterClass(
            x=time_ds,
            y=oxygen_ds,
            mode="markers",
            name="[O₂] Exp",
            marker=dict(color="rgba(128, 128, 128, 0.7)", size=5),
            legend="legend1",
        ),
        row=1, col=1,
    )

    for species in panel_a_species:
        if species not in species_idx:
            continue
        idx = species_idx[species]
        color = SPECIES_COLORS.get(species, "#666666")
        fig.add_trace(
            go.Scatter(
                x=time_sim,
                y=y_solution[:, idx],
                mode="lines",
                name=f"[{species}]",
                line=dict(color=color, width=2),
                legend="legend1",
            ),
            row=1, col=1,
        )

    # Panel B: Catalyst species
    for species in panel_b_species:
        if species not in species_idx:
            continue
        idx = species_idx[species]
        color = SPECIES_COLORS.get(species, "#666666")
        fig.add_trace(
            go.Scatter(
                x=time_sim,
                y=y_solution[:, idx],
                mode="lines",
                name=f"[{species}]",
                line=dict(color=color, width=2),
                legend="legend2",
            ),
            row=1, col=2,
        )

    fig.update_layout(
        title=dict(
            text=f"Species Evolution: {experiment_name}",
            font=dict(size=16),
        ),
        height=500,
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=120),
        # Legend for Panel A (left subplot)
        legend1=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.22,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            title=dict(text="Panel A", font=dict(size=10)),
        ),
        # Legend for Panel B (right subplot)
        legend2=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.78,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            title=dict(text="Panel B", font=dict(size=10)),
        ),
    )

    fig.update_xaxes(title_text="Time / s", row=1, col=1)
    fig.update_xaxes(title_text="Time / s", row=1, col=2)
    fig.update_yaxes(title_text="Concentration / µM", row=1, col=1)
    fig.update_yaxes(title_text="Concentration / µM", row=1, col=2)

    return fig


def create_interactive_rate_plots_optimized(
    network: dict,
    params: dict,
    all_exp_data: Dict[str, dict],
    selected_curves: Optional[Dict[str, List[str]]] = None,
) -> go.Figure:
    """
    OPTIMIZED: Create interactive rate-time plots using Plotly.
    Each subplot has its own legend positioned inside the plot area.
    """
    # Pre-group experiments by parameter
    groups = {
        "c_Ru": {},
        "c_S2O8": {},
        "irradiance": {},
    }

    for exp_name, exp_data in all_exp_data.items():
        meta = exp_data["metadata"]
        c_ru = meta.get("c_Ru", 0)
        c_s2o8 = meta.get("c_S2O8", 0)
        irradiance = meta.get("irradiance", 1000)

        groups["c_Ru"].setdefault(c_ru, []).append((exp_name, exp_data))
        groups["c_S2O8"].setdefault(c_s2o8, []).append((exp_name, exp_data))
        groups["irradiance"].setdefault(irradiance, []).append((exp_name, exp_data))

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "A: [Ru(bpy)₃]Cl₂ Variation",
            "B: Na₂S₂O₈ Variation",
            "C: Irradiance Variation",
            "",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )

    panel_configs = [
        ("c_Ru", "µM", (1, 1), "legend1"),
        ("c_S2O8", "µM", (1, 2), "legend2"),
        ("irradiance", "Wm⁻²", (2, 1), "legend3"),
    ]

    ScatterClass = go.Scattergl if USE_WEBGL else go.Scatter
    network_json = json.dumps(network, sort_keys=True, default=str)
    params_json = json.dumps(params, sort_keys=True, default=str)

    for param_key, unit, (row, col), legend_name in panel_configs:
        param_groups = groups[param_key]

        if not param_groups:
            continue

        sorted_values = sorted(param_groups.keys())
        n_colors = max(len(sorted_values), 2)
        colors = [PARAM_COLORS[int(i * (len(PARAM_COLORS) - 1) / (n_colors - 1))] for i in range(n_colors)]

        for i, param_val in enumerate(sorted_values):
            exp_list = param_groups[param_val]
            color = colors[i]
            label = f"{param_val:.0f} {unit}"

            # Check if this curve should be visible
            visible = True
            if selected_curves and param_key in selected_curves:
                visible = label in selected_curves[param_key]

            # OPTIMIZATION: Aggregate all experiments with same param into single traces
            all_time_exp = []
            all_rate_exp = []
            all_time_model = []
            all_rate_model = []

            for exp_name, exp_data in exp_list:
                time_exp = np.array(exp_data["time"])
                oxygen_exp = np.array(exp_data["oxygen"])
                conditions = exp_data["metadata"]

                # Downsample
                time_ds, oxygen_ds = downsample_data(time_exp, oxygen_exp, max_points=50)
                
                # Calculate experimental rate
                rate_exp = np.gradient(oxygen_ds, time_ds)
                
                all_time_exp.extend(time_ds.tolist())
                all_rate_exp.extend(rate_exp.tolist())
                # Add NaN to create gaps between experiments
                all_time_exp.append(None)
                all_rate_exp.append(None)

                # Cached simulation
                conditions_json = json.dumps(conditions, sort_keys=True, default=str)
                y_solution, species_list, species_idx, time_sim = simulate_species_evolution_cached(
                    network_json, params_json, conditions_json,
                    float(time_exp.min()), float(time_exp.max()), n_points=50
                )
                
                y_solution = np.array(y_solution)
                time_sim = np.array(time_sim)

                o2_pred = (
                    y_solution[:, species_idx["O2"]]
                    if "O2" in species_idx
                    else np.zeros_like(time_sim)
                )
                rate_pred = np.gradient(o2_pred, time_sim)
                
                all_time_model.extend(time_sim.tolist())
                all_rate_model.extend(rate_pred.tolist())
                all_time_model.append(None)
                all_rate_model.append(None)

            # Single aggregated trace for experimental points
            fig.add_trace(
                ScatterClass(
                    x=all_time_exp,
                    y=all_rate_exp,
                    mode="markers",
                    name=f"{label}",
                    marker=dict(color=color, size=4, opacity=0.6),
                    legendgroup=f"{param_key}_{param_val}",
                    showlegend=True,
                    visible=visible,
                    hovertemplate=f"{label}<br>Time: %{{x:.1f}} s<br>Rate: %{{y:.3f}} µM/s<extra></extra>",
                    legend=legend_name,
                ),
                row=row, col=col,
            )

            # Single aggregated trace for model lines
            fig.add_trace(
                go.Scatter(
                    x=all_time_model,
                    y=all_rate_model,
                    mode="lines",
                    name=f"{label} (fit)",
                    line=dict(color=color, width=2),
                    legendgroup=f"{param_key}_{param_val}",
                    showlegend=False,
                    visible=visible,
                    connectgaps=False,
                    legend=legend_name,
                ),
                row=row, col=col,
            )

    # Add zero lines
    for row, col in [(1, 1), (1, 2), (2, 1)]:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)

    # Position legends INSIDE each subplot (top-right corner)
    # Using paper coordinates based on subplot positions
    fig.update_layout(
        title=dict(
            text="Rate-Time Curves by Parameter Variation",
            font=dict(size=16),
        ),
        height=700,
        template="plotly_white",
        margin=dict(l=70, r=30, t=60, b=60),
        # Legend 1: Inside top-left subplot (Panel A)
        legend1=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.44,
            font=dict(size=8),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(100,100,100,0.3)",
            borderwidth=1,
            title=dict(text="[Ru]", font=dict(size=9, color="#333")),
            itemsizing="constant",
            tracegroupgap=1,
        ),
        # Legend 2: Inside top-right subplot (Panel B)
        legend2=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.99,
            font=dict(size=8),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(100,100,100,0.3)",
            borderwidth=1,
            title=dict(text="[S₂O₈²⁻]", font=dict(size=9, color="#333")),
            itemsizing="constant",
            tracegroupgap=1,
        ),
        # Legend 3: Inside bottom-left subplot (Panel C)
        legend3=dict(
            orientation="v",
            yanchor="top",
            y=0.42,
            xanchor="right",
            x=0.44,
            font=dict(size=8),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(100,100,100,0.3)",
            borderwidth=1,
            title=dict(text="Irradiance", font=dict(size=9, color="#333")),
            itemsizing="constant",
            tracegroupgap=1,
        ),
    )

    fig.update_xaxes(title_text="Time / s", row=1, col=1)
    fig.update_xaxes(title_text="Time / s", row=1, col=2)
    fig.update_xaxes(title_text="Time / s", row=2, col=1)
    fig.update_yaxes(title_text="Rate / µM(O₂)s⁻¹", row=1, col=1)
    fig.update_yaxes(title_text="Rate / µM(O₂)s⁻¹", row=1, col=2)
    fig.update_yaxes(title_text="Rate / µM(O₂)s⁻¹", row=2, col=1)

    return fig


def create_synthetic_experiment_data() -> dict:
    """Create synthetic experiment data for visualization when real data is unavailable."""
    time = np.linspace(0, 600, 100)  # Reduced from 200
    oxygen = 50 * (1 - np.exp(-time / 100)) + np.random.normal(0, 1, len(time))
    oxygen = np.maximum(oxygen, 0)
    
    return {
        "time": time.tolist(),
        "oxygen": oxygen.tolist(),
        "metadata": {
            "c_Ru": 50.0,
            "c_S2O8": 6000.0,
            "power_output": 1000.0,
            "pH": 7.0,
            "irradiance": 1000.0,
            "photon_flux": None,
        }
    }


def create_synthetic_dataset() -> Dict[str, dict]:
    """Create a synthetic dataset with multiple experiments for rate plot visualization."""
    dataset = {}
    
    # Vary Ru concentration
    for c_ru in [5, 10, 20, 50, 100]:
        name = f"exp_Ru_{c_ru}"
        time = np.linspace(0, 600, 100)
        scale = c_ru / 50.0
        oxygen = 50 * scale * (1 - np.exp(-time / (100 / scale))) + np.random.normal(0, 0.5, len(time))
        oxygen = np.maximum(oxygen, 0)
        dataset[name] = {
            "time": time.tolist(),
            "oxygen": oxygen.tolist(),
            "metadata": {
                "c_Ru": float(c_ru),
                "c_S2O8": 6000.0,
                "irradiance": 1000.0,
                "pH": 7.0,
                "photon_flux": None,
            }
        }
    
    # Vary S2O8 concentration
    for c_s2o8 in [1000, 3000, 6000, 10000]:
        name = f"exp_S2O8_{c_s2o8}"
        time = np.linspace(0, 600, 100)
        scale = c_s2o8 / 6000.0
        oxygen = 50 * (1 - np.exp(-time / 100)) * min(scale, 1.5) + np.random.normal(0, 0.5, len(time))
        oxygen = np.maximum(oxygen, 0)
        dataset[name] = {
            "time": time.tolist(),
            "oxygen": oxygen.tolist(),
            "metadata": {
                "c_Ru": 50.0,
                "c_S2O8": float(c_s2o8),
                "irradiance": 1000.0,
                "pH": 7.0,
                "photon_flux": None,
            }
        }
    
    # Vary irradiance
    for irr in [500, 1000, 1500, 2000, 3000]:
        name = f"exp_irr_{irr}"
        time = np.linspace(0, 600, 100)
        scale = irr / 1000.0
        oxygen = 50 * (1 - np.exp(-time / (100 / scale))) + np.random.normal(0, 0.5, len(time))
        oxygen = np.maximum(oxygen, 0)
        dataset[name] = {
            "time": time.tolist(),
            "oxygen": oxygen.tolist(),
            "metadata": {
                "c_Ru": 50.0,
                "c_S2O8": 6000.0,
                "irradiance": float(irr),
                "pH": 7.0,
                "photon_flux": None,
            }
        }
    
    return dataset


def prepare_visualization_data(
    run_results: List[dict],
    all_exp_data: Optional[Dict[str, dict]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Prepare visualization data for each run (no file I/O, just data prep).
    """
    if not all_exp_data:
        all_exp_data = create_synthetic_dataset()

    viz_data = {}

    for result in run_results:
        run_dir = result["dir"]
        best_data = result["best_data"]

        if best_data is None:
            continue

        network = best_data.get("network", {})
        phenom_trends = best_data.get("phenomenological_trends", {})
        global_params = phenom_trends.get("global_params", {})

        params = global_params.copy()
        network_params = build_params_from_network(network)
        for key, val in network_params.items():
            if key not in params:
                params[key] = val

        # Pick representative experiment
        rep_exp_name = "Synthetic"
        rep_exp_data = None
        
        if all_exp_data:
            ru_values = [
                (name, data["metadata"].get("c_Ru", 0))
                for name, data in all_exp_data.items()
            ]
            ru_values.sort(key=lambda x: x[1])
            if ru_values:
                median_idx = len(ru_values) // 2
                rep_exp_name, _ = ru_values[median_idx]
                rep_exp_data = all_exp_data[rep_exp_name]
        
        if rep_exp_data is None:
            rep_exp_data = create_synthetic_experiment_data()
            rep_exp_name = "Synthetic (no exp. data)"

        viz_data[str(run_dir)] = {
            "network": network,
            "params": params,
            "rep_exp_name": rep_exp_name,
            "rep_exp_data": rep_exp_data,
            "all_exp_data": all_exp_data,
        }

    return viz_data


# =============================================================================
# UI Components
# =============================================================================

def render_login_screen():
    """Render a login screen where the user types their name."""
    st.markdown(
        "<div style='text-align:center; padding: 2rem 0 1rem 0;'>"
        "<h2>👤 Who are you?</h2>"
        "<p style='color:#888;'>Enter your name so feedback is saved under your identity.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        name = st.text_input(
            "Your name",
            max_chars=60,
            placeholder="e.g., Kevin, Dr. Smith, Jane D.",
            label_visibility="collapsed",
        ).strip()

        st.markdown("")
        login_clicked = st.button("🔓 Continue", use_container_width=True, type="primary")

    if login_clicked:
        if name:
            return name
        else:
            st.error("Please enter your name to continue.")
            return None
    return None


def render_user_badge(username: str):
    """Show a small badge in the sidebar indicating the logged-in user."""
    st.sidebar.markdown(
        f"<div style='"
        f"background: linear-gradient(135deg, #1e3a5f, #2d5f8a);"
        f"color: white; padding: 0.6rem 1rem; border-radius: 8px;"
        f"margin-bottom: 1rem; text-align: center;"
        f"font-weight: 600; font-size: 0.95rem;"
        f"'>"
        f"👤 {username}"
        f"</div>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("🔄 Switch User", use_container_width=True):
        del st.session_state["current_user"]
        st.rerun()


def render_curve_selector(all_exp_data: Dict[str, dict], run_id: str) -> Dict[str, List[str]]:
    """Render checkboxes for selecting individual curves."""
    st.markdown("##### 🎛️ Curve Selection")

    # Group experiments
    groups = {"c_Ru": set(), "c_S2O8": set(), "irradiance": set()}

    for exp_data in all_exp_data.values():
        meta = exp_data["metadata"]
        groups["c_Ru"].add(f"{meta.get('c_Ru', 0):.0f} µM")
        groups["c_S2O8"].add(f"{meta.get('c_S2O8', 0):.0f} µM")
        groups["irradiance"].add(f"{meta.get('irradiance', 1000):.0f} Wm⁻²")

    selected = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("**[Ru(bpy)₃]Cl₂**")
        ru_options = sorted(groups["c_Ru"], key=lambda x: float(x.split()[0]))
        selected["c_Ru"] = st.multiselect(
            "Ru concentrations",
            options=ru_options,
            default=ru_options,
            label_visibility="collapsed",
            key=f"ru_select_{run_id}",
        )

    with col2:
        st.caption("**Na₂S₂O₈**")
        s2o8_options = sorted(groups["c_S2O8"], key=lambda x: float(x.split()[0]))
        selected["c_S2O8"] = st.multiselect(
            "S2O8 concentrations",
            options=s2o8_options,
            default=s2o8_options,
            label_visibility="collapsed",
            key=f"s2o8_select_{run_id}",
        )

    with col3:
        st.caption("**Irradiance**")
        irr_options = sorted(groups["irradiance"], key=lambda x: float(x.split()[0]))
        selected["irradiance"] = st.multiselect(
            "Irradiance values",
            options=irr_options,
            default=irr_options,
            label_visibility="collapsed",
            key=f"irr_select_{run_id}",
        )

    return selected


def render_run_results(run_results: list, viz_data: Dict[str, dict], all_exp_data: Dict[str, dict]):
    """Display each run's best reaction network, trend image, and interactive plots."""
    tab_labels = [r["label"] for r in run_results]
    tabs = st.tabs(tab_labels)

    for tab, result in zip(tabs, run_results):
        with tab:
            run_dir = result["dir"]
            best_file = result["best_file"]
            best_data = result["best_data"]
            best_image = result["best_image"]

            if best_data is None:
                st.warning(f"No phenomenological results found in `{run_dir.name}`")
                continue

            overall_score = best_data.get("phenomenological_trends", {}).get("overall_score", "N/A")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric(
                    "Overall Fit Score",
                    f"{overall_score:.4f}" if isinstance(overall_score, float) else overall_score,
                )
            with col2:
                if best_file:
                    st.caption(f"Source: `{best_file.name}`")

            # Reaction table and static image
            network = best_data.get("network", {})
            reactions = network.get("reactions", [])

            col_table, col_img = st.columns([3, 1])

            with col_table:
                if not reactions:
                    st.info("No reactions found in this result.")
                else:
                    reaction_data = []
                    for i, rxn in enumerate(reactions):
                        formatted = format_reaction_for_display(rxn)
                        formatted["#"] = i + 1
                        reaction_data.append(formatted)

                    df = pd.DataFrame(reaction_data)
                    df = df[["#", "Equation", "Type", "Fitted Parameter", "Range", "Description"]]

                    def highlight_type(val):
                        if val == "light":
                            return "background-color: #fff3cd"
                        return ""

                    styled_df = df.style.map(highlight_type, subset=["Type"])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)

            with col_img:
                if best_image is not None:
                    st.image(str(best_image), caption=best_image.name, use_container_width=True)
                else:
                    st.info("No trends image found.")

            # Network metadata
            metadata = network.get("metadata", {})
            if metadata:
                with st.expander("📋 Network Metadata"):
                    st.json(metadata)

            # Interactive plots section
            st.markdown("---")
            st.subheader("📊 Interactive Fit Visualizations")

            run_viz = viz_data.get(str(run_dir), {})
            exp_data_for_plots = run_viz.get("all_exp_data", all_exp_data)
            if not exp_data_for_plots:
                exp_data_for_plots = create_synthetic_dataset()

            if run_viz and run_viz.get("network"):
                run_id = _slugify(run_dir.name)
                
                # Curve selector
                selected_curves = render_curve_selector(exp_data_for_plots, run_id)

                plot_tabs = st.tabs(["📈 Concentration vs Time", "⚡ Rate vs Time"])

                with plot_tabs[0]:
                    rep_exp_data = run_viz.get("rep_exp_data")
                    if rep_exp_data is None:
                        rep_exp_data = create_synthetic_experiment_data()
                    
                    params = run_viz.get("params", {})
                    network = run_viz.get("network", {})
                    
                    if params and network.get("reactions"):
                        try:
                            conc_fig = create_interactive_concentration_plot(
                                network,
                                params,
                                rep_exp_data,
                                run_viz.get("rep_exp_name", "Representative"),
                            )
                            st.plotly_chart(conc_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not generate concentration plot: {e}")
                    else:
                        st.warning("No fitted parameters available for simulation.")
                        st.json(network.get("reactions", []))

                with plot_tabs[1]:
                    params = run_viz.get("params", {})
                    network = run_viz.get("network", {})
                    
                    if params and network.get("reactions"):
                        try:
                            rate_fig = create_interactive_rate_plots_optimized(
                                network,
                                params,
                                exp_data_for_plots,
                                selected_curves=selected_curves,
                            )
                            st.plotly_chart(rate_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not generate rate plot: {e}")
                    else:
                        st.warning("No fitted parameters available for simulation.")
            else:
                st.warning("Interactive visualizations not available for this run.")


def render_feedback_form(
    username: str,
    run_type: str,
    template: str,
    available_runs: List[Path],
    run_results: list,
):
    """Render the feedback form and handle submission."""
    st.markdown("---")
    st.header("💬 Your Feedback for This Run Type")
    st.caption(
        f"Submitting as **{username}** — one overall assessment across all "
        f"**{len(available_runs)}** run(s) of *{run_type}*."
    )

    with st.form("feedback_form"):
        st.subheader("What's Good? ✅")
        whats_good = st.text_area(
            "Describe positive aspects of the reaction networks across these runs",
            placeholder="e.g., The oxidative quenching step looks reasonable across most runs...",
            label_visibility="collapsed",
            height=100,
        )

        st.subheader("What's Bad? ❌")
        whats_bad = st.text_area(
            "Describe issues or concerns",
            placeholder="e.g., The rate constant for dimerization seems too low in several runs...",
            label_visibility="collapsed",
            height=100,
        )

        st.subheader("Suggested Action 🔧")
        action = st.text_area(
            "What changes would you recommend?",
            placeholder="e.g., Add a back-reaction for the dimer dissociation...",
            label_visibility="collapsed",
            height=100,
        )

        submitted = st.form_submit_button("📤 Submit Feedback", use_container_width=True)

    if submitted:
        if not whats_good and not whats_bad and not action:
            st.error("Please provide at least one piece of feedback.")
        else:
            entries, sha = load_user_run_feedback(username, run_type)

            per_run_scores = {}
            for result in run_results:
                if result["best_data"] is not None:
                    score = result["best_data"].get("phenomenological_trends", {}).get("overall_score", "N/A")
                    per_run_scores[str(result["dir"])] = score

            new_entry = {
                "timestamp": datetime.now().isoformat(),
                "user": username,
                "run_type": run_type,
                "glob_template": template,
                "num_runs": len(available_runs),
                "run_directories": [str(r) for r in available_runs],
                "per_run_scores": per_run_scores,
                "whats_good": whats_good,
                "whats_bad": whats_bad,
                "suggested_action": action,
                "network_snapshots": {
                    str(r["dir"]): r["best_data"].get("network", {})
                    for r in run_results
                    if r["best_data"] is not None
                },
            }

            entries.append(new_entry)

            if save_user_run_feedback(username, run_type, entries, sha):
                st.success("✅ Feedback submitted and saved to GitHub!")
                st.balloons()


def render_previous_feedback(username: str, run_type: str):
    """Show the current user's previous feedback for this run type."""
    st.markdown("---")

    with st.expander(f"📜 Your Previous Feedback for *{run_type}*", expanded=False):
        my_feedback, _ = load_user_run_feedback(username, run_type)

        if not my_feedback:
            st.info("You haven't submitted feedback for this run type yet.")
        else:
            for i, entry in enumerate(reversed(my_feedback)):
                st.markdown(
                    f"**Feedback** — {entry.get('timestamp', 'Unknown time')} "
                    f"({entry.get('num_runs', '?')} runs)"
                )
                if entry.get("whats_good"):
                    st.markdown(f"✅ **Good:** {entry['whats_good']}")
                if entry.get("whats_bad"):
                    st.markdown(f"❌ **Bad:** {entry['whats_bad']}")
                if entry.get("suggested_action"):
                    st.markdown(f"🔧 **Action:** {entry['suggested_action']}")
                if i < len(my_feedback) - 1:
                    st.markdown("---")


def render_admin_section(username: str):
    """Render the admin panel with per-user and global statistics."""
    with st.expander("📊 All Collected Feedback (Admin View)"):
        all_entries = load_all_feedback()

        if not all_entries:
            st.info("No feedback collected yet.")
        else:
            # Global stats
            st.subheader("Global Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Entries", len(all_entries))
            with col2:
                unique_types = len(set(e.get("run_type", "") for e in all_entries))
                st.metric("Run Types Covered", unique_types)
            with col3:
                unique_users = len(set(e.get("_user_slug", "unknown") for e in all_entries))
                st.metric("Unique Users", unique_users)

            # Per-user breakdown
            st.subheader("Per-User Breakdown")
            user_counts = {}
            for entry in all_entries:
                u = entry.get("user", entry.get("_user_slug", "Unknown"))
                user_counts[u] = user_counts.get(u, 0) + 1

            user_df = pd.DataFrame([
                {"User": u, "Submissions": c}
                for u, c in sorted(user_counts.items(), key=lambda x: -x[1])
            ])
            st.dataframe(user_df, use_container_width=True, hide_index=True)

            # Coverage matrix
            st.subheader("Coverage Matrix (Run Type × User)")
            matrix_data = {}
            for entry in all_entries:
                rt = entry.get("run_type", "Unknown")
                u = entry.get("user", entry.get("_user_slug", "Unknown"))
                if rt not in matrix_data:
                    matrix_data[rt] = {}
                matrix_data[rt][u] = matrix_data[rt].get(u, 0) + 1

            all_users_in_data = sorted({
                e.get("user", e.get("_user_slug", "Unknown"))
                for e in all_entries
            })
            matrix_rows = []
            for rt in sorted(matrix_data.keys()):
                row = {"Run Type": rt}
                for u in all_users_in_data:
                    row[u] = matrix_data[rt].get(u, 0)
                matrix_rows.append(row)

            if matrix_rows:
                matrix_df = pd.DataFrame(matrix_rows)
                st.dataframe(matrix_df, use_container_width=True, hide_index=True)

            # Filter by user
            st.subheader("Filter by User")
            all_user_names = sorted({
                e.get("user", e.get("_user_slug", "Unknown"))
                for e in all_entries
            })
            filter_user = st.selectbox(
                "Select a user to view their feedback",
                options=["All Users"] + all_user_names,
                key="admin_user_filter",
            )

            if filter_user == "All Users":
                filtered = all_entries
            else:
                filtered = [
                    e for e in all_entries
                    if e.get("user") == filter_user or e.get("_user_slug") == _slugify(filter_user)
                ]

            if filtered:
                for i, entry in enumerate(reversed(filtered)):
                    entry_user = entry.get("user", entry.get("_user_slug", "Unknown"))
                    st.markdown(
                        f"**{entry_user}** → *{entry.get('run_type', '?')}* — "
                        f"{entry.get('timestamp', 'Unknown time')} "
                        f"({entry.get('num_runs', '?')} runs)"
                    )
                    if entry.get("whats_good"):
                        st.markdown(f"✅ **Good:** {entry['whats_good']}")
                    if entry.get("whats_bad"):
                        st.markdown(f"❌ **Bad:** {entry['whats_bad']}")
                    if entry.get("suggested_action"):
                        st.markdown(f"🔧 **Action:** {entry['suggested_action']}")
                    if i < len(filtered) - 1:
                        st.markdown("---")
            else:
                st.info("No feedback entries match this filter.")

            # Downloads
            st.subheader("Export")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                export_all = {"feedback_entries": all_entries}
                st.download_button(
                    label="📥 Download All Feedback (JSON)",
                    data=json.dumps(export_all, indent=2, default=str),
                    file_name="chemist_feedback_all.json",
                    mime="application/json",
                )
            with col_dl2:
                my_entries = [e for e in all_entries if e.get("user") == username]
                export_mine = {"feedback_entries": my_entries}
                st.download_button(
                    label=f"📥 Download My Feedback ({username})",
                    data=json.dumps(export_mine, indent=2, default=str),
                    file_name=f"chemist_feedback_{_slugify(username)}.json",
                    mime="application/json",
                )


# =============================================================================
# Main app
# =============================================================================

def main():
    st.set_page_config(
        page_title="Kinetic Fitting Feedback",
        page_icon="⚗️",
        layout="wide",
    )

    st.title("⚗️ Kinetic Fitting - Chemistry Feedback Round")
    st.markdown("---")

    # User login gate
    if "current_user" not in st.session_state:
        chosen = render_login_screen()
        if chosen:
            st.session_state["current_user"] = chosen
            st.rerun()
        st.stop()

    username = st.session_state["current_user"]

    # Sidebar
    with st.sidebar:
        render_user_badge(username)

        st.header("🔬 Run Type Selection")
        run_type = st.selectbox(
            "Select Run Type",
            options=list(RUN_TEMPLATES.keys()),
            help="Choose the type of kinetic fitting run to review.",
        )

        template = RUN_TEMPLATES[run_type]
        available_runs = get_available_runs(template)

        if not available_runs:
            st.warning(f"No runs found for pattern: {template}")
            st.stop()

        st.markdown("---")
        st.info(
            f"📁 Found **{len(available_runs)}** run(s) for this type.\n\n"
            "Review all runs below, then submit one piece of feedback for the entire set."
        )
        st.subheader("Included Runs")
        for run_dir in available_runs:
            st.caption(f"• `{run_dir.name}`")

        # Data path configuration
        st.markdown("---")
        st.subheader("⚙️ Settings")
        data_path = st.text_input(
            "Experimental Data Path",
            value=DATA_PATH,
            help="Path to the HDF5 file containing experimental data",
        )
        
        # Performance settings
        st.caption("**Performance**")
        use_webgl = st.checkbox("Use WebGL rendering", value=True, help="Faster for large datasets")
        max_points = st.slider("Max points per trace", 50, 500, 200, 50)
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Update global settings
    # global USE_WEBGL, MAX_POINTS_PER_TRACE
    USE_WEBGL = use_webgl
    MAX_POINTS_PER_TRACE = max_points

    # Main content
    st.header(f"📊 All Runs — {run_type}")
    st.caption(
        "Browse through each run's best reaction network below. "
        "After reviewing, submit your feedback at the bottom."
    )

    # Load run results
    run_results = []
    for run_dir in available_runs:
        best_file, best_data, best_image = _find_best_phenomenological_result(run_dir)
        run_results.append({
            "dir": run_dir,
            "label": extract_run_number(run_dir.name),
            "best_file": best_file,
            "best_data": best_data,
            "best_image": best_image,
        })

    # Load experimental data (cached)
    all_exp_data = {}
    try:
        all_exp_data = load_experimental_data_cached(data_path)
        st.sidebar.success(f"✅ Loaded {len(all_exp_data)} experiments")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load experimental data: {e}")
        st.sidebar.info("Using synthetic data for visualization")

    # Prepare visualization data (lightweight, no file I/O)
    viz_data = prepare_visualization_data(run_results, all_exp_data)

    # Render results with interactive plots
    render_run_results(run_results, viz_data, all_exp_data)

    # Feedback form
    render_feedback_form(username, run_type, template, available_runs, run_results)

    # Previous feedback
    render_previous_feedback(username, run_type)

    # Admin section
    render_admin_section(username)


if __name__ == "__main__":
    main()