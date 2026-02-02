"""
Streamlit App for Collecting Chemist Feedback on Kinetic Fitting Results

Feedback is persisted to a GitHub repository via the GitHub API,
so it survives Streamlit Community Cloud restarts.

Storage layout on GitHub:
    feedback/{username_slug}/{run_type_slug}.json

Each file holds a JSON array of that user's feedback entries for one run type.
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
    "Text+Vision+Chemistry Feedback (Claude Sonnet 4.5)": "kinetic_fitting_with_visionfb_claudesonnet45_with_opus_fb_*",
    "Text+Chemistry Feedback every round (Claude Sonnet 4.5)": "kinetic_fitting_tasks_with_visionfb_and_chemistryopus_no_text_*",
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
#   GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxx"
#   GITHUB_REPO  = "username/repo-name"
#   GITHUB_BRANCH = "main"
GITHUB_API = "https://api.github.com"
FEEDBACK_ROOT = "feedback"  # top-level folder in the repo


# =============================================================================
# Slug helper
# =============================================================================


def _slugify(text: str) -> str:
    """
    Turn a human-readable string into a safe filesystem / path component.
    e.g. "No Feedback (GPT-4o)" -> "no_feedback_gpt_4o"
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text


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


def _github_branch() -> str:
    return st.secrets.get("GITHUB_BRANCH", "main")


def _github_repo() -> str:
    return st.secrets.get("GITHUB_REPO", "")


def _feedback_file_path(username: str, run_type: str) -> str:
    """Build the in-repo path: feedback/{username_slug}/{run_type_slug}.json"""
    return f"{FEEDBACK_ROOT}/{_slugify(username)}/{_slugify(run_type)}.json"


def _github_contents_url(repo_path: str) -> str:
    """Build the GitHub Contents API URL for a given in-repo path."""
    repo = _github_repo()
    return f"{GITHUB_API}/repos/{repo}/contents/{repo_path}"


# --- Generic read / write for any JSON file in the repo ---


def _load_json_from_github(repo_path: str) -> Tuple[list, Optional[str]]:
    """
    Load a JSON array from a file in the repo.

    Returns:
        (entries_list, sha) — sha is needed to update the file later.
        If the file does not exist yet, returns ([], None).
    """
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


def _save_json_to_github(
    repo_path: str, data: list, sha: Optional[str] = None
) -> bool:
    """Save (create or update) a JSON array to a file in the repo."""
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


# --- Per-user / per-run-type convenience wrappers ---


def load_user_run_feedback(
    username: str, run_type: str
) -> Tuple[list, Optional[str]]:
    """Load the feedback entries list for a specific user + run type."""
    path = _feedback_file_path(username, run_type)
    return _load_json_from_github(path)


def save_user_run_feedback(
    username: str, run_type: str, entries: list, sha: Optional[str] = None
) -> bool:
    """Save the feedback entries list for a specific user + run type."""
    path = _feedback_file_path(username, run_type)
    return _save_json_to_github(path, entries, sha)


# --- Directory listing helpers for admin view ---


def _list_github_dir(repo_path: str) -> List[dict]:
    """List contents of a directory in the repo via the GitHub API."""
    url = _github_contents_url(repo_path)
    params = {"ref": _github_branch()}
    resp = requests.get(url, headers=_github_headers(), params=params)
    if resp.status_code == 200:
        return resp.json()
    return []


def list_all_feedback_users() -> List[str]:
    """Return folder names under feedback/ — each is a slugified username."""
    items = _list_github_dir(FEEDBACK_ROOT)
    return sorted(
        item["name"] for item in items if item.get("type") == "dir"
    )


def list_user_run_files(user_slug: str) -> List[str]:
    """Return JSON filenames under feedback/{user_slug}/."""
    items = _list_github_dir(f"{FEEDBACK_ROOT}/{user_slug}")
    return sorted(
        item["name"]
        for item in items
        if item.get("type") == "file" and item["name"].endswith(".json")
    )


def load_all_feedback() -> List[dict]:
    """
    Walk the entire feedback/ tree and return every entry annotated
    with '_user_slug' and '_file'.  Used by the admin panel.
    """
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


def _find_best_phenomenological_result(
    output_dir: Path,
) -> Tuple[Optional[Path], Optional[dict], Optional[Path]]:
    """
    Find the phenomenological result JSON with the best overall_score.

    Returns:
        (best_json_path, best_data, best_image_path)
        The image is phenomenological_trends_{timestamp}.png matching the
        timestamp in phenomenologic_result.json.{timestamp}.
    """
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
            overall_score = data.get("phenomenological_trends", {}).get(
                "overall_score", -1
            )
            if overall_score > best_score:
                best_score = overall_score
                best_file = Path(filepath)
                best_data = data
                best_timestamp = timestamp
        except (json.JSONDecodeError, IOError):
            continue

    # Look up the matching image
    best_image = None
    if best_timestamp is not None:
        candidate = output_dir / f"phenomenological_trends_{best_timestamp}.png"
        if candidate.exists():
            best_image = candidate

    return best_file, best_data, best_image


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
            f"[{qy_range[0]}, {qy_range[1]}]"
            if len(qy_range) == 2
            else "N/A"
        )
    else:
        k = reaction.get("fitted_k")
        k_range = reaction.get("k_range", [])
        formatted["Fitted Parameter"] = f"k = {k:.4e}" if k else "N/A"
        formatted["Range"] = (
            f"[{k_range[0]:.0e}, {k_range[1]:.0e}]"
            if len(k_range) == 2
            else "N/A"
        )

    return formatted


def extract_run_number(name: str) -> str:
    """Extract a human-readable run number from a directory name."""
    match = re.search(r"_(\d+)$", name)
    if match:
        return f"Run {match.group(1)}"
    return name


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

        st.markdown("")  # spacer
        login_clicked = st.button(
            "🔓 Continue", use_container_width=True, type="primary"
        )

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


def render_run_results(run_results: list):
    """Display each run's best reaction network and trend image in tabs."""
    tab_labels = [r["label"] for r in run_results]
    tabs = st.tabs(tab_labels)

    for tab, result in zip(tabs, run_results):
        with tab:
            run_dir = result["dir"]
            best_file = result["best_file"]
            best_data = result["best_data"]
            best_image = result["best_image"]

            if best_data is None:
                st.warning(
                    f"No phenomenological results found in `{run_dir.name}`"
                )
                continue

            overall_score = best_data.get("phenomenological_trends", {}).get(
                "overall_score", "N/A"
            )

            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric(
                    "Overall Fit Score",
                    (
                        f"{overall_score:.4f}"
                        if isinstance(overall_score, float)
                        else overall_score
                    ),
                )
            with col2:
                st.caption(f"Source: `{best_file.name}`")

            # --- Reaction table + image side by side ---
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

                    styled_df = df.style.map(
                        highlight_type, subset=["Type"]
                    )
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        hide_index=True,
                    )

            with col_img:
                if best_image is not None:
                    st.image(
                        str(best_image),
                        caption=best_image.name,
                        use_container_width=True,
                    )
                else:
                    st.info("No trends image found.")

            metadata = network.get("metadata", {})
            if metadata:
                with st.expander("📋 Network Metadata"):
                    st.json(metadata)


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
            placeholder=(
                "e.g., The oxidative quenching step looks reasonable "
                "across most runs..."
            ),
            label_visibility="collapsed",
            height=100,
        )

        st.subheader("What's Bad? ❌")
        whats_bad = st.text_area(
            "Describe issues or concerns",
            placeholder=(
                "e.g., The rate constant for dimerization seems too low "
                "in several runs..."
            ),
            label_visibility="collapsed",
            height=100,
        )

        st.subheader("Suggested Action 🔧")
        action = st.text_area(
            "What changes would you recommend?",
            placeholder=(
                "e.g., Add a back-reaction for the dimer dissociation..."
            ),
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
            # Load current file for this user + run type
            entries, sha = load_user_run_feedback(username, run_type)

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

    with st.expander(
        f"📜 Your Previous Feedback for *{run_type}*", expanded=False
    ):
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
                    st.markdown(
                        f"🔧 **Action:** {entry['suggested_action']}"
                    )
                if i < len(my_feedback) - 1:
                    st.markdown("---")


def render_admin_section(username: str):
    """Render the admin panel with per-user and global statistics."""
    with st.expander("📊 All Collected Feedback (Admin View)"):
        all_entries = load_all_feedback()

        if not all_entries:
            st.info("No feedback collected yet.")
        else:
            # --- Global stats ---
            st.subheader("Global Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Entries", len(all_entries))
            with col2:
                unique_types = len(
                    set(e.get("run_type", "") for e in all_entries)
                )
                st.metric("Run Types Covered", unique_types)
            with col3:
                unique_users = len(
                    set(e.get("_user_slug", "unknown") for e in all_entries)
                )
                st.metric("Unique Users", unique_users)

            # --- Per-user breakdown ---
            st.subheader("Per-User Breakdown")
            user_counts = {}
            for entry in all_entries:
                u = entry.get("user", entry.get("_user_slug", "Unknown"))
                user_counts[u] = user_counts.get(u, 0) + 1

            user_df = pd.DataFrame(
                [
                    {"User": u, "Submissions": c}
                    for u, c in sorted(
                        user_counts.items(), key=lambda x: -x[1]
                    )
                ]
            )
            st.dataframe(user_df, use_container_width=True, hide_index=True)

            # --- Coverage matrix: run type × user ---
            st.subheader("Coverage Matrix (Run Type × User)")
            matrix_data = {}
            for entry in all_entries:
                rt = entry.get("run_type", "Unknown")
                u = entry.get("user", entry.get("_user_slug", "Unknown"))
                if rt not in matrix_data:
                    matrix_data[rt] = {}
                matrix_data[rt][u] = matrix_data[rt].get(u, 0) + 1

            all_users_in_data = sorted(
                {
                    e.get("user", e.get("_user_slug", "Unknown"))
                    for e in all_entries
                }
            )
            matrix_rows = []
            for rt in sorted(matrix_data.keys()):
                row = {"Run Type": rt}
                for u in all_users_in_data:
                    row[u] = matrix_data[rt].get(u, 0)
                matrix_rows.append(row)

            if matrix_rows:
                matrix_df = pd.DataFrame(matrix_rows)
                st.dataframe(
                    matrix_df, use_container_width=True, hide_index=True
                )

            # --- Filter by user ---
            st.subheader("Filter by User")
            all_user_names = sorted(
                {
                    e.get("user", e.get("_user_slug", "Unknown"))
                    for e in all_entries
                }
            )
            filter_user = st.selectbox(
                "Select a user to view their feedback",
                options=["All Users"] + all_user_names,
                key="admin_user_filter",
            )

            if filter_user == "All Users":
                filtered = all_entries
            else:
                filtered = [
                    e
                    for e in all_entries
                    if e.get("user") == filter_user
                    or e.get("_user_slug") == _slugify(filter_user)
                ]

            if filtered:
                for i, entry in enumerate(reversed(filtered)):
                    entry_user = entry.get(
                        "user", entry.get("_user_slug", "Unknown")
                    )
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
                        st.markdown(
                            f"🔧 **Action:** {entry['suggested_action']}"
                        )
                    if i < len(filtered) - 1:
                        st.markdown("---")
            else:
                st.info("No feedback entries match this filter.")

            # --- Downloads ---
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
                my_entries = [
                    e for e in all_entries if e.get("user") == username
                ]
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
        page_title="Kinetic fitting feedback",
        page_icon="⚗️",
        layout="wide",
    )

    st.title("⚗️ Kinetic fitting - chemistry feedback round")
    st.markdown("---")

    # ----- User login gate -----
    if "current_user" not in st.session_state:
        chosen = render_login_screen()
        if chosen:
            st.session_state["current_user"] = chosen
            st.rerun()
        st.stop()  # Don't render the rest until logged in

    username = st.session_state["current_user"]

    # ----- Sidebar -----
    with st.sidebar:
        render_user_badge(username)

        st.header("🔬 Run Type Selection")
        run_type = st.selectbox(
            "Select Run Type",
            options=list(RUN_TEMPLATES.keys()),
            help=(
                "Choose the type of kinetic fitting run to review. "
                "You will see ALL runs of this type and provide one "
                "overall feedback."
            ),
        )

        template = RUN_TEMPLATES[run_type]
        available_runs = get_available_runs(template)

        if not available_runs:
            st.warning(f"No runs found for pattern: {template}")
            st.stop()

        st.markdown("---")
        st.info(
            f"📁 Found **{len(available_runs)}** run(s) for this type.\n\n"
            "Review all runs below, then submit one piece of feedback "
            "for the entire set."
        )
        st.subheader("Included Runs")
        for run_dir in available_runs:
            st.caption(f"• `{run_dir.name}`")

    # ----- Main content: display ALL runs for the selected type -----
    st.header(f"📊 All Runs — {run_type}")
    st.caption(
        "Browse through each run's best reaction network below. "
        "After reviewing, submit your feedback at the bottom."
    )

    run_results = []
    for run_dir in available_runs:
        best_file, best_data, best_image = _find_best_phenomenological_result(
            run_dir
        )
        run_results.append(
            {
                "dir": run_dir,
                "label": extract_run_number(run_dir.name),
                "best_file": best_file,
                "best_data": best_data,
                "best_image": best_image,
            }
        )

    render_run_results(run_results)

    # ----- Feedback form -----
    render_feedback_form(
        username, run_type, template, available_runs, run_results
    )

    # ----- Previous feedback (current user only) -----
    render_previous_feedback(username, run_type)

    # ----- Admin section -----
    render_admin_section(username)


if __name__ == "__main__":
    main()