
#!/usr/bin/env python3
"""
Minimal timeline event network tool with CSV/JSON import/export.

Dependencies:
    pip install streamlit pandas plotly

Run:
    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
import io
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------
# Data model
# -----------------------

@dataclass

class Event:
    id: str
    label: str
    start: str                    # keep as string; parse only for plotting
    end: Optional[str] = None
    actor: Optional[str] = None
    codes: List[str] = field(default_factory=list)
    y_lane: Optional[int] = None
    summary: Optional[str] = None   # ← NEW: short description (few sentences)
    notes: Optional[str] = None     # longer, detailed notes

    @staticmethod
    def from_row(row: dict) -> "Event":
        def clean(val):
            if pd.isna(val):
                return None
            return str(val).strip()

        codes_str = clean(row.get("codes", "")) or ""
        codes = [c.strip() for c in codes_str.split(";") if c.strip()]

        y_lane_raw = row.get("y_lane", None)
        if pd.isna(y_lane_raw) or y_lane_raw in ("", None):
            y_lane = None
        else:
            try:
                y_lane = int(y_lane_raw)
            except ValueError:
                y_lane = None

        return Event(
            id=clean(row.get("id", "")),
            label=clean(row.get("label", "")) or "",
            start=clean(row.get("start", "")) or "",
            end=clean(row.get("end", None)),
            actor=clean(row.get("actor", None)),
            codes=codes,
            y_lane=y_lane,
            summary=clean(row.get("summary", None)),   # ← NEW
            notes=clean(row.get("notes", None)),
        )

    def to_row(self) -> dict:
        d = asdict(self)
        d["codes"] = ";".join(self.codes)
        return d


@dataclass
class Link:
    id: str
    source: str   # event id
    target: str   # event id
    type: Optional[str] = None
    label: Optional[str] = None
    codes: List[str] = field(default_factory=list)

    @staticmethod
    def from_row(row: dict) -> "Link":
        def clean(val):
            if pd.isna(val):
                return None
            return str(val).strip()

        codes_str = clean(row.get("codes", "")) or ""
        codes = [c.strip() for c in codes_str.split(";") if c.strip()]

        return Link(
            id=clean(row.get("id", "")),
            source=clean(row.get("source", "")) or "",
            target=clean(row.get("target", "")) or "",
            type=clean(row.get("type", None)),
            label=clean(row.get("label", None)),
            codes=codes,
        )

    def to_row(self) -> dict:
        d = asdict(self)
        d["codes"] = ";".join(self.codes)
        return d


# -----------------------
# Storage helpers
# -----------------------

def load_events_from_csv(file) -> List[Event]:
    df = pd.read_csv(file)
    return [Event.from_row(row) for _, row in df.iterrows()]


def load_links_from_csv(file) -> List[Link]:
    df = pd.read_csv(file)
    return [Link.from_row(row) for _, row in df.iterrows()]


def events_to_csv_bytes(events: List[Event]) -> bytes:
    df = pd.DataFrame([e.to_row() for e in events])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def links_to_csv_bytes(links: List[Link]) -> bytes:
    df = pd.DataFrame([l.to_row() for l in links])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def load_from_json(file) -> Tuple[List[Event], List[Link]]:
    data = json.load(file)
    events_raw = data.get("events", [])
    links_raw = data.get("links", [])
    events = [
        Event(
            id=e["id"],
            label=e["label"],
            start=e["start"],
            end=e.get("end"),
            actor=e.get("actor"),
            codes=e.get("codes", []),
            y_lane=e.get("y_lane"),
            summary=e.get("summary"),      # ← NEW
            notes=e.get("notes"),
        )
        for e in events_raw
    ]
    links = [
        Link(
            id=l["id"],
            source=l["source"],
            target=l["target"],
            type=l.get("type"),
            label=l.get("label"),
            codes=l.get("codes", []),
        )
        for l in links_raw
    ]
    return events, links


def to_json_bytes(events: List[Event], links: List[Link]) -> bytes:
    data = {
        "events": [e.to_row() for e in events],
        "links": [l.to_row() for l in links],
    }
    return json.dumps(data, indent=2).encode("utf-8")

# -----------------------
# Visualization
# -----------------------

def wrap_text(text: str, width: int = 30) -> str:
    """Insert <br> every `width` characters without splitting words."""
    if not text:
        return ""
    words = text.split()
    lines = []
    current = []
    count = 0

    for w in words:
        if count + len(w) + len(current) > width:
            lines.append(" ".join(current))
            current = [w]
            count = len(w)
        else:
            current.append(w)
            count += len(w)

    if current:
        lines.append(" ".join(current))

    return "<br>".join(lines)


def plot_timeline(
    events: List[Event],
    links: List[Link],
    src_extra_offset: int = 0,
    tgt_extra_offset: int = 0,
):

    if not events:
        st.info("No events to show yet.")
        return

    # Build dataframe of events
    df = pd.DataFrame(
        [
            {
                "id": e.id,
                "label": e.label,
                "summary": e.summary or "",
                "start": e.start,
                "actor": e.actor or "Unspecified",
                "codes": ", ".join(e.codes),
                "notes": e.notes or "",
            }
            for e in events
        ]
    )

    # Parse dates
    df["start_dt"] = pd.to_datetime(df["start"], errors="coerce")
    if df["start_dt"].isna().all():
        st.warning("Could not parse any start dates. Check the 'start' column format.")
        return

    # Actors as categories → separate "lanes"
    df["actor"] = df["actor"].astype("category")

    # ---- Estimate label size for each event ----
    # Prefer summary, fallback to label
    df["raw_text"] = df.apply(
        lambda row: row["summary"] if row["summary"] else row["label"],
        axis=1,
    )
    # Wrap text
    df["display_text"] = df["raw_text"].apply(lambda t: wrap_text(t, width=30))
    # Count lines: number of <br> + 1 (for last line)
    df["n_lines"] = df["display_text"].apply(
        lambda s: (s.count("<br>") + 1) if s else 1
    )

    fig = go.Figure()

    # ---- EVENT LABELS AS FLOATING TEXT ----
    for actor in df["actor"].cat.categories:
        sub = df[df["actor"] == actor]

        hover_text = (
            "ID: " + sub["id"]
            + "<br>Label: " + sub["label"]
            + "<br>Start: "
            + sub["start_dt"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
            + "<br>Codes: " + sub["codes"]
            + "<br>Summary: " + sub["summary"]
            + "<br>Notes: " + sub["notes"]
        )

        fig.add_trace(
            go.Scatter(
                x=sub["start_dt"],
                y=[actor] * len(sub),
                mode="text",
                text=sub["display_text"],   # already wrapped
                textposition="middle center",
                hovertext=hover_text,
                hoverinfo="text",
                name=str(actor),
            )
        )



    # ---- LINK ARROWS BETWEEN EVENTS ----

    # Build lookup: event id -> (x, y, n_lines)
    positions = {}
    for _, row in df.iterrows():
        if pd.notna(row["start_dt"]):
            positions[row["id"]] = (
                row["start_dt"],
                row["actor"],
                int(row["n_lines"]),
            )

    for link in links:
        if link.source not in positions or link.target not in positions:
            continue

        x0, y0, src_lines = positions[link.source]
        x1, y1, tgt_lines = positions[link.target]

        # Base heuristic 
        per_line = 10    # extra distance per text line

        # Heuristic distances
        src_standoff_heur = per_line * src_lines
        tgt_standoff_heur = per_line * tgt_lines

        # Apply user-controlled extra offsets
        src_standoff = max(0, src_standoff_heur + src_extra_offset)
        tgt_standoff = max(0, tgt_standoff_heur + tgt_extra_offset)

        link_label = link.label or link.type or ""

        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="rgba(80,80,80,0.7)",
            opacity=0.9,
            startstandoff=src_standoff,   # distance from source text
            standoff=tgt_standoff,        # distance from target text
            text=link_label,              # or "" if you don't want labels on arrows
            align="center",
        )
       
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Actor / lane",
        yaxis=dict(type="category"),
        height=500,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# UI helpers: add / edit
# -----------------------

def add_event_ui():
    st.subheader("Add new event")

    label = st.text_input("Label", key="new_label")
    start = st.text_input("Start date/time (e.g. 2023-03-05 or ISO)", key="new_start")
    actor = st.text_input("Actor / lane", key="new_actor")
    summary = st.text_area(
        "Short description (few sentences)",
        key="new_summary"
    )                                      # ← NEW
    codes_str = st.text_input("Codes (semicolon-separated)", key="new_codes")
    notes = st.text_area("Detailed notes", key="new_notes")

    if st.button("Add event"):
        if not label or not start:
            st.error("Label and start are required.")
            return

        existing_ids = {e.id for e in st.session_state.events}
        idx = 1
        new_id = f"e{idx}"
        while new_id in existing_ids:
            idx += 1
            new_id = f"e{idx}"

        codes = [c.strip() for c in codes_str.split(";") if c.strip()]
        event = Event(
            id=new_id,
            label=label,
            start=start,
            actor=actor or None,
            codes=codes,
            summary=summary or None,       # ← NEW
            notes=notes or None,
        )
        st.session_state.events.append(event)
        st.success(f"Added event {new_id}")

def edit_event_ui():
    st.subheader("Edit / remove event")

    if not st.session_state.events:
        st.write("No events yet.")
        return

    # Create mapping for nicer labels in the dropdown
    options = {f"{e.id}: {e.label}": e.id for e in st.session_state.events}
    label_for_select = st.selectbox("Select event", list(options.keys()))
    selected_id = options[label_for_select]
    event = next(e for e in st.session_state.events if e.id == selected_id)

    # --- Edit fields ---
    new_label = st.text_input("Label", event.label, key=f"edit_label_{event.id}")
    new_start = st.text_input("Start", event.start, key=f"edit_start_{event.id}")
    new_actor = st.text_input("Actor / lane", event.actor or "",
                              key=f"edit_actor_{event.id}")
    new_summary = st.text_area(
        "Short description (few sentences)",
        event.summary or "",
        key=f"edit_summary_{event.id}",
    )
    new_codes_str = st.text_input(
        "Codes (semicolon-separated)",
        ";".join(event.codes),
        key=f"edit_codes_{event.id}",
    )
    new_notes = st.text_area(
        "Detailed notes",
        event.notes or "",
        key=f"edit_notes_{event.id}",
    )

    col_save, col_delete = st.columns(2)

    # --- Save changes ---
    with col_save:
        if st.button("Update event", key=f"update_event_{event.id}"):
            event.label = new_label
            event.start = new_start
            event.actor = new_actor or None
            event.summary = new_summary or None
            event.codes = [c.strip() for c in new_codes_str.split(";") if c.strip()]
            event.notes = new_notes or None
            st.success("Event updated.")

    # --- Delete event ---
    with col_delete:
        if st.button("Delete event", key=f"delete_event_{event.id}"):
            # Remove event
            st.session_state.events = [
                e for e in st.session_state.events if e.id != event.id
            ]
            # Remove any links that use this event as source or target
            st.session_state.links = [
                l for l in st.session_state.links
                if l.source != event.id and l.target != event.id
            ]
            st.success(
                f"Deleted event {event.id} and any links attached to it."
            )

def add_link_ui():
    st.subheader("Links")

    if not st.session_state.events:
        st.write("No events to link yet.")
        return

    st.markdown("**Add link**")

    source_id = st.selectbox(
        "Source event",
        [e.id for e in st.session_state.events],
        key="link_source",
    )
    target_id = st.selectbox(
        "Target event",
        [e.id for e in st.session_state.events],
        key="link_target",
    )
    link_type = st.text_input("Type (e.g. influence, causal)", key="link_type")
    label = st.text_input("Label", key="link_label")
    codes_str = st.text_input("Codes (semicolon-separated)", key="link_codes")

    if st.button("Add link", key="add_link_btn"):
        existing_ids = {l.id for l in st.session_state.links}
        idx = 1
        new_id = f"l{idx}"
        while new_id in existing_ids:
            idx += 1
            new_id = f"l{idx}"

        codes = [c.strip() for c in codes_str.split(";") if c.strip()]
        link = Link(
            id=new_id,
            source=source_id,
            target=target_id,
            type=link_type or None,
            label=label or None,
            codes=codes,
        )
        st.session_state.links.append(link)
        st.success(f"Added link {new_id}")

    st.markdown("---")

    # ---- Remove existing links ----
    st.markdown("**Remove link**")

    if not st.session_state.links:
        st.write("No links yet.")
        return

    # Create human-readable labels
    link_labels = {
        f"{l.id}: {l.source} → {l.target} ({l.type or ''} {l.label or ''})": l.id
        for l in st.session_state.links
    }

    link_select_label = st.selectbox(
        "Select link to remove",
        list(link_labels.keys()),
        key="delete_link_select",
    )
    selected_link_id = link_labels[link_select_label]

    if st.button("Delete link", key="delete_link_btn"):
        st.session_state.links = [
            l for l in st.session_state.links if l.id != selected_link_id
        ]
        st.success(f"Deleted link {selected_link_id}")

# -----------------------
# App
# -----------------------

def init_state():
    if "events" not in st.session_state:
        st.session_state.events: List[Event] = []
    if "links" not in st.session_state:
        st.session_state.links: List[Link] = []


def main():
    st.set_page_config(page_title="Timeline event network", layout="wide")
    init_state()

    st.title("Timeline event network")

    # Sidebar: storage mode, import/export
    st.sidebar.header("Data")

    storage_mode = st.sidebar.radio(
        "Storage format",
        ["CSV", "JSON"],
        help="CSV = events.csv + links.csv. JSON = single file with both.",
    )

    st.sidebar.subheader("Import")

    if storage_mode == "CSV":
        events_csv = st.sidebar.file_uploader("Upload events.csv", type="csv")
        links_csv = st.sidebar.file_uploader("Upload links.csv", type="csv")

        if st.sidebar.button("Load CSV data"):
            if events_csv is None:
                st.sidebar.error("Please upload at least events.csv.")
            else:
                st.session_state.events = load_events_from_csv(events_csv)
                if links_csv is not None:
                    st.session_state.links = load_links_from_csv(links_csv)
                else:
                    st.session_state.links = []
                st.sidebar.success("Loaded data from CSV.")
    else:
        json_file = st.sidebar.file_uploader("Upload data.json", type="json")
        if st.sidebar.button("Load JSON data"):
            if json_file is None:
                st.sidebar.error("Please upload a JSON file.")
            else:
                events, links = load_from_json(json_file)
                st.session_state.events = events
                st.session_state.links = links
                st.sidebar.success("Loaded data from JSON.")

    st.sidebar.subheader("Export")

    if storage_mode == "CSV":
        ev_bytes = events_to_csv_bytes(st.session_state.events)
        ln_bytes = links_to_csv_bytes(st.session_state.links)

        st.sidebar.download_button(
            "Download events.csv",
            data=ev_bytes,
            file_name="events.csv",
            mime="text/csv",
        )
        st.sidebar.download_button(
            "Download links.csv",
            data=ln_bytes,
            file_name="links.csv",
            mime="text/csv",
        )
    else:
        json_bytes = to_json_bytes(st.session_state.events, st.session_state.links)
        st.sidebar.download_button(
            "Download data.json",
            data=json_bytes,
            file_name="timeline_data.json",
            mime="application/json",
        )


    st.sidebar.subheader("Arrow spacing")

    src_extra_offset = st.sidebar.slider(
        "Extra offset at source (px)",
        min_value=-100,
        max_value=100,
        value=0,
        step=1,
        help="Increase or decrease arrow distance from source events.",
    )

    tgt_extra_offset = st.sidebar.slider(
        "Extra offset at target (px)",
        min_value=-100,
        max_value=100,
        value=0,
        step=1,
        help="Increase or decrease arrow distance from target events.",
    )
    # Layout: timeline + editor

    col1, col2 = st.columns([2, 1])

    # 1) First run the editing UI (mutates st.session_state.events)
    with col2:
        add_event_ui()
        st.markdown("---")
        edit_event_ui()
        st.markdown("---")
        add_link_ui()

    # 2) Then draw the timeline with the updated events
    with col1:
        st.subheader("Timeline")
        plot_timeline(
            st.session_state.events,
            st.session_state.links,
            src_extra_offset=src_extra_offset,
            tgt_extra_offset=tgt_extra_offset,
        )

if __name__ == "__main__":
    main()
