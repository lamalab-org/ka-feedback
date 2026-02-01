"""
Streamlit App for Collecting Chemist Feedback on Kinetic Fitting Results

Feedback is persisted to a GitHub repository via the GitHub API,
so it survives Streamlit Community Cloud restarts.
"""

import streamlit as st
import regex as re
import glob
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import requests

# =============================================================================
# Configuration
# =============================================================================

RUN_TEMPLATES = {
    "No Feedback (Claude Sonnet 4.5)": "kinetic_fitting_no_fb_claudesonnet45_*",
    "No Feedback (GPT-4o)": "kinetic_fitting_no_fb_gpt4o_feedback_*",
    "Text Feedback (Claude Sonnet 4.5)": "kinetic_fitting_with_fb_claudesonnet45_*",
    "Text+Vision Feedback (Claude Sonnet 4.5)": "kinetic_fitting_with_visionfb_claudesonnet45_*",
    "Text+Vision+Chemistry Feedback (Claude Opus 4.5) (Claude Sonnet 4.5)": "kinetic_fitting_with_visionfb_claudesonnet45_with_opus_fb_*",
}

# Patterns that should be EXCLUDED when matching a given template.
EXCLUDE_PATTERNS = {
    "kinetic_fitting_with_visionfb_claudesonnet45_*": [
        "kinetic_fitting_with_visionfb_claudesonnet45_with_opus_fb_*"
    ],
}

BASE_DIR = Path("round1")  # Adjust this to your actual base directory

# GitHub configuration — set these in Streamlit secrets.
# In your Streamlit Cloud dashboard (or .streamlit/secrets.toml locally):
#
#   GITHUB_TOKEN         = "ghp_xxxxxxxxxxxxxxxxxxxx"
#   GITHUB_REPO          = "username/repo-name"
#   GITHUB_FEEDBACK_PATH = "feedback/chemist_feedback.json"
#   GITHUB_BRANCH        = "main"

GITHUB_API = "https://api.github.com"


# =============================================================================
# GitHub-backed storage
# =============================================================================


def _github_headers() -> dict:
    """Return authorization headers for the GitHub API."""
    token = st.secrets.get("GITHUB_TOKEN", "")
    if not token:
        st.error(
            "⚠️ `GITHUB_TOKEN` is not set in Streamlit secrets. "
            "Feedback cannot be saved."
        )
        st.stop()
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }


def _github_file_url() -> str:
    """Build the GitHub Contents API URL for the feedback file."""
    repo = st.secrets.get("GITHUB_REPO", "")
    path = st.secrets.get("GITHUB_FEEDBACK_PATH", "feedback/chemist_feedback.json")
    return f"{GITHUB_API}/repos/{repo}/contents/{path}"


def _github_branch() -> str:
    return st.secrets.get("GITHUB_BRANCH", "main")


def load_feedback_data() -> Tuple[dict, Optional[str]]:
    """
    Load feedback JSON from GitHub.

    Returns:
        (data_dict, sha) — sha is needed to update the file later.
        If the file does not exist yet, returns (empty structure, None).
    """
    url = _github_file_url()
    params = {"ref": _github_branch()}
    resp = requests.get(url, headers=_github_headers(), params=params)

    if resp.status_code == 200:
        payload = resp.json()
        content = base64.b64decode(payload["content"]).decode("utf-8")
        return json.loads(content), payload["sha"]
    elif resp.status_code == 404:
        # File doesn't exist yet — will be created on first save
        return {"feedback_entries": []}, None
    else:
        st.error(f"GitHub API error ({resp.status_code}): {resp.text}")
        return {"feedback_entries": []}, None


def save_feedback_data(data: dict, sha: Optional[str] = None) -> bool:
    """
    Save (create or update) the feedback JSON on GitHub.

    Args:
        data: The full feedback dict to write.
        sha:  The current file SHA (required for updates, None for creation).
    """
    url = _github_file_url()
    content_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
    encoded = base64.b64encode(content_bytes).decode("utf-8")

    body = {
        "message": f"feedback: update {datetime.now().isoformat()}",
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


# =============================================================================
# Result loading helpers
# =============================================================================


def _find_best_phenomenological_result(
    output_dir: Path,
) -> Tuple[Optional[Path], Optional[dict]]:
    """Find the phenomenological result JSON with the best overall_score."""
    output_dir = Path(output_dir)
    pattern = f"{output_dir.as_posix()}/phenomenologic_result.json.*"
    result_files = glob.glob(pattern)

    if not result_files:
        return None, None

    best_score = -1.0
    best_file = None
    best_data = None

    for filepath in result_files:
        match = re.search(r"phenomenologic_result\.json\.(\d+)$", filepath)
        if not match:
            continue

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            overall_score = data.get("phenomenological_trends", {}).get(
                "overall_score", -1
            )
            if overall_score > best_score:
                best_score = overall_score
                best_file = Path(filepath)
                best_data = data
        except (json.JSONDecodeError, IOError):
            continue

    return best_file, best_data


def get_available_runs(template: str) -> List[Path]:
    """Get available run directories matching the template, excluding ambiguous matches."""
    pattern = BASE_DIR / template
    matched_dirs = set(glob.glob(str(pattern)))

    for exclude_tmpl in EXCLUDE_PATTERNS.get(template, []):
        exclude_pattern = BASE_DIR / exclude_tmpl
        exclude_dirs = set(glob.glob(str(exclude_pattern)))
        matched_dirs -= exclude_dirs

    return sorted(Path(d) for d in matched_dirs if Path(d).is_dir())


def format_reaction_for_display(reaction: dict) -> dict:
    """Format a reaction dictionary for cleaner display."""
    formatted = {
        "Equation": reaction.get("equation", "N/A"),
        "Type": reaction.get("type", "N/A"),
        "Description": reaction.get("description", "N/A"),
    }

    if reaction.get("type") == "light":
        qy = reaction.get("fitted_quantum_yield")
        qy_range = reaction.get("quantum_yield", [])
        formatted["Fitted Parameter"] = f"QY = {qy:.4f}" if qy else "N/A"
        formatted["Range"] = (
            f"[{qy_range[0]}, {qy_range[1]}]" if len(qy_range) == 2 else "N/A"
        )
    else:
        k = reaction.get("fitted_k")
        k_range = reaction.get("k_range", [])
        formatted["Fitted Parameter"] = f"k = {k:.4e}" if k else "N/A"
        formatted["Range"] = (
            f"[{k_range[0]:.0e}, {k_range[1]:.0e}]" if len(k_range) == 2 else "N/A"
        )

    return formatted


def extract_run_number(name: str) -> str:
    """Extract a human-readable run number from a directory name."""
    match = re.search(r"_(\d+)$", name)
    if match:
        return f"Run {match.group(1)}"
    return name


# =============================================================================
# Main app
# =============================================================================


def main():
    st.set_page_config(
        page_title="Kinetic fitting feedback", page_icon="⚗️", layout="wide"
    )

    st.title("⚗️ Kinetic fitting - chemistry feedback round")
    st.markdown("---")

    # Sidebar for run type selection
    with st.sidebar:
        st.header("🔬 Run Type Selection")

        run_type = st.selectbox(
            "Select Run Type",
            options=list(RUN_TEMPLATES.keys()),
            help="Choose the type of kinetic fitting run to review. "
            "You will see ALL runs of this type and provide one overall feedback.",
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

    # --- Main content: display ALL runs for the selected type ---

    st.header(f"📊 All Runs — {run_type}")
    st.caption(
        "Browse through each run's best reaction network below. "
        "After reviewing, submit your feedback at the bottom."
    )

    run_results = []
    for run_dir in available_runs:
        best_file, best_data = _find_best_phenomenological_result(run_dir)
        run_results.append(
            {
                "dir": run_dir,
                "label": extract_run_number(run_dir.name),
                "best_file": best_file,
                "best_data": best_data,
            }
        )

    # Display each run in its own tab
    tab_labels = [r["label"] for r in run_results]
    tabs = st.tabs(tab_labels)

    for tab, result in zip(tabs, run_results):
        with tab:
            run_dir = result["dir"]
            best_file = result["best_file"]
            best_data = result["best_data"]

            if best_data is None:
                st.warning(f"No phenomenological results found in `{run_dir.name}`")
                continue

            overall_score = best_data.get("phenomenological_trends", {}).get(
                "overall_score", "N/A"
            )

            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric(
                    "Overall Fit Score",
                    f"{overall_score:.4f}"
                    if isinstance(overall_score, float)
                    else overall_score,
                )
            with col2:
                st.caption(f"Source: `{best_file.name}`")

            network = best_data.get("network", {})
            reactions = network.get("reactions", [])

            if not reactions:
                st.info("No reactions found in this result.")
            else:
                reaction_data = []
                for i, rxn in enumerate(reactions):
                    formatted = format_reaction_for_display(rxn)
                    formatted["#"] = i + 1
                    reaction_data.append(formatted)

                df = pd.DataFrame(reaction_data)
                df = df[
                    [
                        "#",
                        "Equation",
                        "Type",
                        "Fitted Parameter",
                        "Range",
                        "Description",
                    ]
                ]

                def highlight_type(val):
                    if val == "light":
                        return "background-color: #fff3cd"
                    return ""

                styled_df = df.style.map(highlight_type, subset=["Type"])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                metadata = network.get("metadata", {})
                if metadata:
                    with st.expander("📋 Network Metadata"):
                        st.json(metadata)

    # --- Single feedback form for the entire run type ---

    st.markdown("---")
    st.header("💬 Your Feedback for This Run Type")
    st.caption(
        f"Submit **one** overall assessment across all "
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

        submitted = st.form_submit_button(
            "📤 Submit Feedback", use_container_width=True
        )

        if submitted:
            if not whats_good and not whats_bad and not action:
                st.error("Please provide at least one piece of feedback.")
            else:
                # Load current state from GitHub (gets latest SHA)
                feedback_data, sha = load_feedback_data()

                per_run_scores = {}
                for result in run_results:
                    if result["best_data"] is not None:
                        score = (
                            result["best_data"]
                            .get("phenomenological_trends", {})
                            .get("overall_score", "N/A")
                        )
                        per_run_scores[str(result["dir"])] = score

                new_entry = {
                    "timestamp": datetime.now().isoformat(),
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

                feedback_data["feedback_entries"].append(new_entry)

                if save_feedback_data(feedback_data, sha):
                    st.success("✅ Feedback submitted and saved to GitHub!")
                    st.balloons()

    # Show previous feedback for this run type
    st.markdown("---")
    with st.expander("📜 Previous Feedback for This Run Type"):
        feedback_data, _ = load_feedback_data()
        type_feedback = [
            f
            for f in feedback_data.get("feedback_entries", [])
            if f.get("run_type") == run_type
        ]

        if not type_feedback:
            st.info("No previous feedback for this run type.")
        else:
            for i, entry in enumerate(reversed(type_feedback)):
                st.markdown(
                    f"**Feedback** — {entry.get('timestamp', 'Unknown time')}  "
                    f"({entry.get('num_runs', '?')} runs)"
                )

                if entry.get("whats_good"):
                    st.markdown(f"✅ **Good:** {entry['whats_good']}")
                if entry.get("whats_bad"):
                    st.markdown(f"❌ **Bad:** {entry['whats_bad']}")
                if entry.get("suggested_action"):
                    st.markdown(f"🔧 **Action:** {entry['suggested_action']}")

                if i < len(type_feedback) - 1:
                    st.markdown("---")

    # Admin section
    with st.expander("📊 All Collected Feedback (Admin View)"):
        feedback_data, _ = load_feedback_data()
        all_entries = feedback_data.get("feedback_entries", [])

        if not all_entries:
            st.info("No feedback collected yet.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Entries", len(all_entries))
            with col2:
                unique_types = len(set(e.get("run_type", "") for e in all_entries))
                st.metric("Run Types Covered", unique_types)

            st.download_button(
                label="📥 Download All Feedback (JSON)",
                data=json.dumps(feedback_data, indent=2, default=str),
                file_name="chemist_feedback_export.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
