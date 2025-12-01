
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
from typing import IO
import io
import json
import numpy as np

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
    lane_label: Optional[str] = None
    summary: Optional[str] = None   # ← NEW: short description (few sentences)
    source: Optional[str] = None
    notes: Optional[str] = None     # longer, detailed notes

    @staticmethod
    def from_row(row: dict) -> "Event":
        def clean(val):
            if pd.isna(val):
                return None
            return str(val).strip()

        codes_str = clean(row.get("codes", "")) or ""
        codes = [c.strip() for c in codes_str.split(";") if c.strip()]

        lane_label = clean(row.get("lane_label", None))

        return Event(
                id=clean(row.get("id", "")),
                label=clean(row.get("label", "")) or "",
                start=clean(row.get("start", "")) or "",
                end=clean(row.get("end", None)),
                actor=clean(row.get("actor", None)),
                codes=codes,
                lane_label=lane_label,
                summary=clean(row.get("summary", None)),   
                source=clean(row.get("source", None)),
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

def smart_read_csv(file: IO) -> pd.DataFrame:
    """
    Robust CSV reader for uploaded files:
    - Tries multiple encodings
    - Lets pandas sniff the delimiter (comma, semicolon, tab, ...)
    """
    # encodings that realistically occur in your context
    encodings = ["utf-8", "utf-8-sig", "latin1"]

    last_error = None
    for enc in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(
                file,
                encoding=enc,
                sep=None,          # let pandas / csv.Sniffer detect delimiter
                engine="python",   # needed for sep=None
            )
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue

    # If we get here, all encodings failed with UnicodeDecodeError
    raise last_error if last_error is not None else ValueError(
        "Could not read CSV with any of the tried encodings."
    )


def load_events_from_csv(file) -> List[Event]:
    df = smart_read_csv(file)
    return [Event.from_row(row) for _, row in df.iterrows()]

def load_links_from_csv(file) -> List[Link]:
    df = smart_read_csv(file)
    return [Link.from_row(row) for _, row in df.iterrows()]

def events_template_csv_bytes() -> bytes:
    """Empty events CSV with the correct headers."""
    df = pd.DataFrame(
            columns=[
                "id",
                "label",
                "start",
                "end",
                "actor",
                "lane_label",
                "codes",
                "summary",
                "source",
                "notes",
                ]
            )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def links_template_csv_bytes() -> bytes:
    """Empty links CSV with the correct headers."""
    df = pd.DataFrame(
            columns=[
                "id",
                "source",
                "target",
                "type",
                "label",
                "codes",
                ]
            )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

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
                lane_label=e.get("lane_label"),
                summary=e.get("summary"),      
                source=e.get("source"),
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
        lane_spacing_factor: float = 1.0,
        show_link_labels: bool = True,
        node_text_mode: str = "Summary (if available)",
        node_font_size: int = 12,
        sensitivity_days: int = 60,
        max_stack: int = 3,   node_text_width: int = 30,
        ):
    """Plot events as wrapped text on a timeline, with arrows for links.

    - Vertical position is determined by a string lane_key:
        * lane_label if provided
        * otherwise actor
        * otherwise "Unspecified"

    - lane_order (in st.session_state) controls vertical order; if not set,
      lanes are ordered alphabetically.
    """
    if not events:
        st.info("No events to show yet.")
        return

    df = pd.DataFrame(
            [
                {
                    "id": e.id,
                    "label": e.label,
                    "summary": e.summary or "",
                    "start": e.start,
                    "actor": e.actor or "Unspecified",
                    "lane_label": e.lane_label,
                    "codes": ", ".join(e.codes),
                    "notes": e.notes or "",
                    "source": e.source or "",
                    }
                for e in events
                ]
            )

    # Parse dates
    df["start_dt"] = pd.to_datetime(df["start"], errors="coerce")
    if df["start_dt"].isna().all():
        st.warning("Could not parse any start dates. Check the 'start' column format.")
        return

    # Lane key = lane_label (if set) else actor
    def compute_lane_key(row):
        if row["lane_label"]:
            return row["lane_label"]
        return row["actor"] or "Unspecified"

    df["lane_key"] = df.apply(compute_lane_key, axis=1)

    # Determine lane order using st.session_state.lane_order
    order_map = getattr(st.session_state, "lane_order", {})

    lane_keys = sorted(
            df["lane_key"].unique(),
            key=lambda k: (order_map.get(k, 0), str(k).lower()),
            )

    df["lane_key"] = pd.Categorical(df["lane_key"], categories=lane_keys, ordered=True)

    # Map each lane to a numeric index for vertical geometry
    #lane_index = {lane : i for i, lane in enumerate(lane_keys)}

    # Overlap handling
    lane_keys = sorted(df["lane_key"].unique(), key=str)
    base_y = {lane: i for i, lane in enumerate(lane_keys)}

    # Sort deterministically
    df = df.sort_values(["lane_key", "start_dt", "id"])

    df["cluster"] = -1  # cluster index within a lane

    for lane, g_lane in df.groupby("lane_key"):
        if g_lane.empty:
            continue

        # walk through events in time order and start a new cluster when gap > sensitivity
        prev_time = None
        cluster_id = 0

        for idx, row in g_lane.iterrows():
            t = row["start_dt"]
            if prev_time is None:
                # first one in this lane
                df.at[idx, "cluster"] = cluster_id
                prev_time = t
                continue

            gap_days = (t - prev_time).days
            if gap_days > sensitivity_days:
                cluster_id += 1
            df.at[idx, "cluster"] = cluster_id
            prev_time = t

    # Now assign vertical offsets per (lane, cluster)
    max_total_offset = 0.8      # total vertical span we allow around base_y
    df["y_offset"] = 0.0

    for (lane, cluster), g in df.groupby(["lane_key", "cluster"]):
        n = len(g)
        if n <= 1:
            continue

        n_eff = min(n, max_stack)
        if n_eff == 1:
            continue

        # spread n_eff events between -max_total_offset/2 and +max_total_offset/2
        offsets_core = np.linspace(
            -max_total_offset / 2.0,
            max_total_offset / 2.0,
            n_eff,
            )

        # If we have more events than max_stack, reuse the topmost offset
        if n <= max_stack:
            offsets = offsets_core
        else:
            offsets = list(offsets_core)
            # extra events all get the highest offset
            offsets.extend([offsets_core[-1]] * (n - max_stack))

        for idx, off in zip(g.index, offsets):
            df.at[idx, "y_offset"] = off

    df["y_plot"] = df["base_y"] + df["y_offset"]

    # ---- Choose node text based on mode, then wrap ----
    if node_text_mode.startswith("Label"):
        # Always show the label
        df["raw_text"] = df["label"].fillna("")
    else:
        # "Summary (if available)": prefer summary, fall back to label
        df["raw_text"] = df.apply(
            lambda row: row["summary"] if row["summary"] else row["label"],
            axis=1,
        )

    df["display_text"] = df["raw_text"].apply(lambda t: wrap_text(t, width=node_text_width))
    df["n_lines"] = df["display_text"].apply(
        lambda s: (s.count("<br>") + 1) if s else 1
    )

    fig = go.Figure()


    # ---- EVENT LABELS AS FLOATING TEXT ----

    for lane in df["lane_key"].cat.categories:
        sub = df[df["lane_key"] == lane]

        hover_text = (
            "ID: " + sub["id"].astype(str)
            + "<br>Label: " + sub["label"].astype(str)
            + "<br>Start: "
            + sub["start_dt"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
            + "<br>Actor: " + sub["actor"].astype(str)
            + "<br>Lane: " + sub["lane_key"].astype(str)
            + "<br>Codes: " + sub["codes"].astype(str)
            + "<br>Source: " + sub["source"].astype(str)
            + "<br>Summary: " + sub["summary"].astype(str)
            + "<br>Notes: " + sub["notes"].astype(str)
            )

        fig.add_trace(
            go.Scatter(
                x=sub["start_dt"],
                y=sub["y_plot"],      # ← use numeric y positions with offsets
                mode="text",
                text=sub["display_text"],
                textposition="middle center",
                hovertext=hover_text,
                hoverinfo="text",
                name=str(lane),
                cliponaxis=False,
                textfont=dict(
                    size=node_font_size,
                    color="#222222",
                    ),
                )
            )
       
    # ---- LINK ARROWS BETWEEN EVENTS ----


    # event id -> (x, lane_key, lane_index, n_lines)

    positions: dict[str, tuple[pd.Timestamp, float, int, int]] = {}
    for _, row in df.iterrows():
        if pd.notna(row["start_dt"]):
            lk = row["lane_key"]
            idx = base_y[lk]  # numeric base position
            positions[row["id"]] = (
                row["start_dt"],      # x
                float(row["y_plot"]), # y (with offset)
                idx,                  # base index for midpoints
                int(row["n_lines"]),
                )

    for link in links:
        if link.source not in positions or link.target not in positions:
            continue


        x0, y0, y0_idx, src_lines = positions[link.source]
        x1, y1, y1_idx, tgt_lines = positions[link.target]

        per_line = 4  # pixels per line of text

        src_standoff_heur = per_line * src_lines
        tgt_standoff_heur = per_line * tgt_lines

        src_standoff = max(0, src_standoff_heur + src_extra_offset)
        tgt_standoff = max(0, tgt_standoff_heur + tgt_extra_offset)

        link_label = link.label or link.type or ""

        # 1) Arrow without text
        fig.add_annotation(
                x=x1,
                y=y1,   # category name
                ax=x0,
                ay=y0,  # category name
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="rgba(80, 80, 80, 0.7)",
                opacity=0.9,
                startstandoff=src_standoff,
                standoff=tgt_standoff,
                text="",  # no text on the arrow itself
                align="center",
                )

        # 2) Label at midpoint of the link, in both x and y
        if show_link_labels and link_label:
            mid_x = x0 + (x1 - x0) / 2
            mid_y = (y0 + y1) / 2.0

            fig.add_annotation(
                    x=mid_x,
                    y=mid_y,
                    xref="x",
                    yref="y",
                    showarrow=False,
                    text=link_label,
                    align="center",
                    yshift=-5,
                    font=dict(size=10),
                    )

    # Compute figure height from number of lanes and spacing factor
    n_lanes = len(lane_keys) if lane_keys else 1
    base_per_lane = 80  # px per lane at factor = 1.0
    min_height = 300
    figure_height = max(
            min_height,
            int(base_per_lane * max(1, n_lanes) * lane_spacing_factor),
            )


    fig.update_yaxes(
            type="linear",
            tickmode="array",
            tickvals=[base_y[k] for k in lane_keys],
            ticktext=[str(k) for k in lane_keys],
            title_text="Lane",
            fixedrange=True,
            )

    fig.update_layout(
            xaxis_title="Time",
            height=figure_height,
            margin=dict(l=40, r=20, t=80, b=40),  # t from 20 → 80
            showlegend=False,
            )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# UI helpers: add / edit
# -----------------------

def add_event_ui():
    st.subheader("Add event")

    label = st.text_input("Label", key="new_label")
    start = st.text_input(
            "Start date/time (e.g. 2023-03-05 or ISO)",
            key="new_start",
            )
    actor = st.text_input("Actor", key="new_actor")

    lane_label = st.text_input(
            "Lane label (optional, overrides actor for lane grouping)",
            key="new_lane_label",
            help="Examples: 'Municipality', 'Citizen initiatives', 'Province'. "
            "If left empty, the actor is used.",
            )

    summary = st.text_area(
            "Short description (few sentences)",
            key="new_summary",
            )
    codes_str = st.text_input("Codes (semicolon-separated)", key="new_codes")
    notes = st.text_area("Detailed notes", key="new_notes")

    source = st.text_input(
            "Source (e.g. interview code, document reference)",
            key="new_source",
            )

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
                lane_label=lane_label or None,
                codes=codes,
                summary=summary or None,
                notes=notes or None,
                source=source or None,
                )
        st.session_state.events.append(event)
        st.success(f"Added event {new_id}")

def edit_event_ui():
    st.subheader("Edit / remove event")

    if not st.session_state.events:
        st.write("No events yet.")
        return

    options = {f"{e.id}: {e.label}": e.id for e in st.session_state.events}
    label_for_select = st.selectbox("Select event", list(options.keys()))
    selected_id = options[label_for_select]
    event = next(e for e in st.session_state.events if e.id == selected_id)

    new_label = st.text_input(
            "Label", event.label, key=f"edit_label_{event.id}"
            )
    new_start = st.text_input(
            "Start", event.start, key=f"edit_start_{event.id}"
            )
    new_actor = st.text_input(
            "Actor",
            event.actor or "",
            key=f"edit_actor_{event.id}",
            )
    new_lane_label = st.text_input(
            "Lane label (optional, overrides actor)",
            event.lane_label or "",
            key=f"edit_lane_label_{event.id}",
            )
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
    new_source = st.text_input(
            "Source",
            event.source or "",
            key=f"edit_source_{event.id}",
            )

    col_save, col_delete = st.columns(2)

    with col_save:
        if st.button("Update event", key=f"update_event_{event.id}"):

            event.label = new_label
            event.start = new_start
            event.actor = new_actor or None
            event.lane_label = new_lane_label or None
            event.summary = new_summary or None
            event.codes = [c.strip() for c in new_codes_str.split(";") if c.strip()]
            event.notes = new_notes or None
            event.source = new_source or None

            st.success("Event updated.")

    with col_delete:
        if st.button("Delete event", key=f"delete_event_{event.id}"):

            st.session_state.events = [
                    e for e in st.session_state.events if e.id != event.id
                    ]
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
        st.session_state.events = []
    if "links" not in st.session_state:
        st.session_state.links = []
    if "lane_order" not in st.session_state:
        st.session_state.lane_order = {}   

def main():
    st.set_page_config(page_title="Timeline event network", layout="wide")
    init_state()

    st.title("Timeline event network")

    # ---- SIDEBAR UI ----
    with st.sidebar:
        st.header("Controls")

        tab_data, tab_vis, tab_events, tab_links = st.tabs(
                ["Data", "Visualization", "Events", "Links"]
                )

        # ---- DATA TAB ----
        with tab_data:
            storage_mode = st.radio(
                    "Storage format",
                    ["CSV", "JSON"],
                    help="CSV = events.csv + links.csv. JSON = single file with both.",
                    )

            st.subheader("Import")
            if storage_mode == "CSV":
                events_csv = st.file_uploader("Upload events.csv", type="csv")
                links_csv = st.file_uploader("Upload links.csv", type="csv")

                if st.button("Load CSV data"):
                    if events_csv is None:
                        st.error("Please upload at least events.csv.")
                    else:
                        st.session_state.events = load_events_from_csv(events_csv)
                        if links_csv is not None:
                            st.session_state.links = load_links_from_csv(links_csv)
                        else:
                            st.session_state.links = []
                        st.success("Loaded data from CSV.")
            else:
                json_file = st.file_uploader("Upload data.json", type="json")
                if st.button("Load JSON data"):
                    if json_file is None:
                        st.error("Please upload a JSON file.")
                    else:
                        events, links = load_from_json(json_file)
                        st.session_state.events = events
                        st.session_state.links = links
                        st.success("Loaded data from JSON.")

            st.subheader("Export")
            if storage_mode == "CSV":
                ev_bytes = events_to_csv_bytes(st.session_state.events)
                ln_bytes = links_to_csv_bytes(st.session_state.links)

                st.download_button("Download events.csv", ev_bytes, "events.csv")
                st.download_button("Download links.csv", ln_bytes, "links.csv")

                st.markdown("---")
                st.subheader("Templates")

                tmpl_ev_bytes = events_template_csv_bytes()
                tmpl_ln_bytes = links_template_csv_bytes()

                st.download_button(
                        "Download events template.csv",
                        tmpl_ev_bytes,
                        "events_template.csv",
                        )
                st.download_button(
                        "Download links template.csv",
                        tmpl_ln_bytes,
                        "links_template.csv",
                        )

            else:
                json_bytes = to_json_bytes(st.session_state.events, st.session_state.links)
                st.download_button("Download data.json", json_bytes, "timeline_data.json")

        # ---- VISUALIZATION TAB ----

        with tab_vis:

            st.subheader("Node text")

            node_text_mode = st.radio(
                    "What should be shown inside the nodes?",
                    ["Summary (if available)", "Label only"],
                    index=0,
                    help=(
                        "Summary (if available): use the short description if present, "
                        "otherwise fall back to the label.\n"
                        "Label only: always show the event label."
                        ),
                    )

            node_font_size = st.slider(
                    "Node font size",
                    min_value=1,
                    max_value=24,
                    value=12,
                    step=1,
                    help="Increase if the text is hard to read; decrease to fit more nodes.",
                    )

            node_text_width = st.slider(
                    "Node text width",
                    min_value = 20,
                    max_value = 80,
                    value = 30,
                    step = 1,
                    help="Change the width of the texts used for the event descriptions.",
                    )

            st.markdown("---")

            st.subheader("Arrow spacing")

            src_extra_offset = st.slider(
                    "Extra offset at source (px)",
                    min_value=-100,
                    max_value=100,
                    value=0,
                    step=1,
                    )

            tgt_extra_offset = st.slider(
                    "Extra offset at target (px)",
                    min_value=-100,
                    max_value=100,
                    value=0,
                    step=1,
                    )

            st.markdown("---")
            st.subheader("Lane labels (optional)")

            lane_spacing_factor = st.slider(
                    "Lane spacing factor",
                    min_value=0.5,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    help="Increase to add more vertical space between lanes; decrease to pack lanes closer.",
                    )   

            st.markdown("---")
            show_link_labels = st.checkbox(
                "Show link labels",
                value=True,
                help="Turn off if labels clutter the visualization.",
            )

            st.markdown("---")

            sensitivity_days = st.slider(
                    "Overlap sensitivity (days)",
                    min_value=0,
                    max_value=365,
                    value=60,
                    help="Events within this many days, for the same actor, are stacked vertically."
                    )

            max_stack = st.slider(
                    "Maximum vertical levels per actor",
                    min_value=1,
                    max_value=6,
                    value=3,
                    help="Upper limit on how many virtual lanes an actor can get."
                    )

            st.subheader("Lane order (optional)")

            # Collect all current lane keys (lane_label or actor)
            lane_keys = sorted(
                    {
                        (e.lane_label or e.actor or "Unspecified")
                        for e in st.session_state.events
                        }
                    )

            if not lane_keys:
                st.caption("No lanes yet. Add some events first.")
            else:
                st.caption(
                        "Lower numbers appear higher in the figure. "
                        "Leave 0 everywhere to keep alphabetical order."
                        )   
                for lane in lane_keys:
                    current = st.session_state.lane_order.get(lane, 0)
                    new_val = st.number_input(
                            f"Order for lane '{lane}'",
                            min_value=-100,
                            max_value=100,
                            value=current,
                            step=1,
                            key=f"lane_order_{lane}",
                            )
                    st.session_state.lane_order[lane] = new_val

        # ---- EVENTS TAB ----
        with tab_events:
            add_event_ui()
            st.markdown("---")
            edit_event_ui()

        # ---- LINKS TAB ----
        with tab_links:
            add_link_ui()

    # ---- MAIN AREA ----
    st.subheader("Timeline")
    plot_timeline(
            st.session_state.events,
            st.session_state.links,
            src_extra_offset=src_extra_offset,
            tgt_extra_offset=tgt_extra_offset,
            lane_spacing_factor=lane_spacing_factor,
            show_link_labels=show_link_labels,
            node_text_mode=node_text_mode,
            node_font_size=node_font_size,   
            node_text_width=node_text_width,
            sensitivity_days=sensitivity_days,
            max_stack=max_stack,
            )

if __name__ == "__main__":
    main()
