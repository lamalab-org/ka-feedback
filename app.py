"""
Streamlit App for Collecting Chemist Feedback on Kinetic Fitting Results
"""

import streamlit as st
import regex as re
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

# Configuration
RUN_TEMPLATES = {
    "No Feedback (Claude Sonnet 4.5)": "kinetic_fitting_no_fb_claudesonnet45_*",
    "No Feedback (GPT-4o)": "kinetic_fitting_no_fb_gpt4o_feedback_*",
    "Text Feedback (Claude Sonnet 4.5)": "kinetic_fitting_with_fb_claudesonnet45_*",
    "Text+Vision Feedback (Claude Sonnet 4.5)": "kinetic_fitting_with_visionfb_claudesonnet45_*",
    "Text+Vision+Chemistry Feedback (Claude Opus 4.5) (Claude Sonnet 4.5)": "kinetic_fitting_with_visionfb_claudesonnet45_with_opus_fb_*",
}
FEEDBACK_FILE = Path("chemist_feedback.json")
BASE_DIR = Path(".")  # Adjust this to your actual base directory


def load_feedback_data() -> dict:
    """Load existing feedback from JSON file."""
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return {"feedback_entries": []}


def save_feedback_data(data: dict):
    """Save feedback to JSON file."""
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


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


def get_available_runs(template: str) -> list:
    """Get available run directories matching the template."""
    pattern = BASE_DIR / template
    dirs = sorted(glob.glob(str(pattern)))
    return [Path(d) for d in dirs if Path(d).is_dir()]


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


def main():
    st.set_page_config(
        page_title="Kinetic fitting feedback", page_icon="⚗️", layout="wide"
    )

    st.title("⚗️ Kinetic fitting - chemistry feedback round")
    st.markdown("---")

    # Sidebar for run selection
    with st.sidebar:
        st.header("🔬 Run Selection")

        # Run type selector
        run_type = st.selectbox(
            "Select Run Type",
            options=list(RUN_TEMPLATES.keys()),
            help="Choose the type of kinetic fitting run to review",
        )

        template = RUN_TEMPLATES[run_type]
        available_runs = get_available_runs(template)

        if not available_runs:
            st.warning(f"No runs found for pattern: {template}")
            st.stop()

        # Run number selector (toggle-like with radio buttons)
        st.subheader("Select Run Number")
        run_names = [d.name for d in available_runs]

        # Extract run numbers for cleaner display
        run_options = []
        for name in run_names:
            match = re.search(r"_(\d+)$", name)
            if match:
                run_options.append(f"Run {match.group(1)}")
            else:
                run_options.append(name)

        selected_idx = st.radio(
            "Available Runs",
            range(len(run_options)),
            format_func=lambda x: run_options[x],
            horizontal=True,
        )

        selected_run_dir = available_runs[selected_idx]

        st.markdown("---")
        st.info(f"📁 Selected: `{selected_run_dir.name}`")

    # Load the best result for selected run
    best_file, best_data = _find_best_phenomenological_result(selected_run_dir)

    if best_data is None:
        st.error(f"No phenomenological results found in {selected_run_dir}")
        st.stop()

    # Display overall score if available
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

    st.markdown("---")

    # Main content area - two columns
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.header("📊 Reaction Network")

        network = best_data.get("network", {})
        reactions = network.get("reactions", [])

        if not reactions:
            st.warning("No reactions found in this result.")
        else:
            # Create a DataFrame for better display
            reaction_data = []
            for i, rxn in enumerate(reactions):
                formatted = format_reaction_for_display(rxn)
                formatted["#"] = i + 1
                reaction_data.append(formatted)

            df = pd.DataFrame(reaction_data)
            df = df[
                ["#", "Equation", "Type", "Fitted Parameter", "Range", "Description"]
            ]

            # Style the dataframe
            def highlight_type(val):
                if val == "light":
                    return "background-color: #fff3cd"  # Yellow for light
                return ""

            styled_df = df.style.map(highlight_type, subset=["Type"])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # Show metadata
            metadata = network.get("metadata", {})
            if metadata:
                with st.expander("📋 Network Metadata"):
                    st.json(metadata)

    with right_col:
        st.header("💬 Your Feedback")

        # Feedback form
        with st.form("feedback_form"):
            st.subheader("What's Good? ✅")
            whats_good = st.text_input(
                "Describe one positive aspect of this reaction network",
                placeholder="e.g., The oxidative quenching step looks reasonable...",
                label_visibility="collapsed",
            )

            st.subheader("What's Bad? ❌")
            whats_bad = st.text_input(
                "Describe one issue or concern",
                placeholder="e.g., The rate constant for dimerization seems too low...",
                label_visibility="collapsed",
            )

            st.subheader("Suggested Action 🔧")
            action = st.text_input(
                "What one change would you recommend?",
                placeholder="e.g., Add a back-reaction for the dimer dissociation...",
                label_visibility="collapsed",
            )

            # Overall rating
            # overall_rating = st.slider(
            #     "Overall Quality Rating",
            #     min_value=1,
            #     max_value=5,
            #     value=3,
            #     help="1 = Poor, 5 = Excellent",
            # )

            submitted = st.form_submit_button(
                "📤 Submit Feedback", use_container_width=True
            )

            if submitted:
                if not whats_good and not whats_bad and not action:
                    st.error("Please provide at least one piece of feedback.")
                else:
                    # Save feedback
                    feedback_data = load_feedback_data()

                    new_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "run_type": run_type,
                        "run_directory": str(selected_run_dir),
                        "source_file": str(best_file),
                        "overall_fit_score": overall_score,
                        "whats_good": whats_good,
                        "whats_bad": whats_bad,
                        "suggested_action": action,
                        # "overall_rating": overall_rating,
                        "network_snapshot": network,
                    }

                    feedback_data["feedback_entries"].append(new_entry)
                    save_feedback_data(feedback_data)

                    st.success("✅ Feedback submitted successfully!")
                    st.balloons()

    # Show previous feedback for this run
    st.markdown("---")
    with st.expander("📜 Previous Feedback for This Run"):
        feedback_data = load_feedback_data()
        run_feedback = [
            f
            for f in feedback_data.get("feedback_entries", [])
            if f.get("run_directory") == str(selected_run_dir)
        ]

        if not run_feedback:
            st.info("No previous feedback for this run.")
        else:
            for i, entry in enumerate(reversed(run_feedback)):
                st.markdown(f"**Feedback** - {entry.get('timestamp', 'Unknown time')}")
                st.markdown(f"⭐ Rating: {entry.get('overall_rating', 'N/A')}/5")

                if entry.get("whats_good"):
                    st.markdown(f"✅ **Good:** {entry['whats_good']}")
                if entry.get("whats_bad"):
                    st.markdown(f"❌ **Bad:** {entry['whats_bad']}")
                if entry.get("suggested_action"):
                    st.markdown(f"🔧 **Action:** {entry['suggested_action']}")

                if i < len(run_feedback) - 1:
                    st.markdown("---")

    # Admin section for viewing all feedback
    with st.expander("📊 All Collected Feedback (Admin View)"):
        feedback_data = load_feedback_data()
        all_entries = feedback_data.get("feedback_entries", [])

        if not all_entries:
            st.info("No feedback collected yet.")
        else:
            # Summary stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Entries", len(all_entries))
            with col2:
                avg_rating = sum(e.get("overall_rating", 0) for e in all_entries) / len(
                    all_entries
                )
                st.metric("Average Rating", f"{avg_rating:.2f}/5")

            # Download button
            st.download_button(
                label="📥 Download All Feedback (JSON)",
                data=json.dumps(feedback_data, indent=2, default=str),
                file_name="chemist_feedback_export.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
