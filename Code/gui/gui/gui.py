import reflex as rx
from .state import State
from typing import Any

def sample_size_input(category: str, subkey: str) -> rx.Component:
    """Helper for nested dictionary inputs using string interpolation for placeholders."""
    label = subkey.replace("_", " ").title()
    return rx.hstack(
        rx.text(f"{category} - {label}", size="1"),
        rx.spacer(),
        rx.input(
            # Use f-string interpolation instead of .cast(str)
            placeholder=f"{State.sample_sizes[category][subkey]}",
            on_change=lambda val: State.update_sample_size(category, subkey, val),
            width="80px",
            size="1"
        ),
        width="100%"
    )

def settings_modal() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.trigger(
            rx.button(
                "Advanced Configuration",
                variant="outline",
                width="100%",
                margin_top="1em",
                size="2",
                padding_y="0.9em",
                style={"cursor": "pointer"},
            ),
        ),
        rx.dialog.content(
            rx.dialog.title("Pipeline Settings"),
            rx.scroll_area(
                rx.vstack(
                    rx.heading("Sample Sizes", size="3"),
                    rx.text("Determines the number of clean/dirty examples sent to the LLM for context.", size="1", color_scheme="gray"),
                    sample_size_input("NUMERIC", "clean_sample_size"),
                    sample_size_input("NUMERIC", "dirty_sample_size"),
                    sample_size_input("DATETIME", "clean_sample_size"),
                    sample_size_input("DATETIME", "dirty_sample_size"),
                    sample_size_input("DIRTY_NUMERIC", "random_sample_size"),
                    sample_size_input("DIRTY_NUMERIC", "unique_sample_size"),
                    sample_size_input("STRING", "random_sample_size"),
                    sample_size_input("STRING", "unique_sample_size"),
                    sample_size_input("NLT", "short_sample_size"),
                    sample_size_input("NLT", "long_sample_size"),
                    rx.hstack(rx.text("Validator sample size"), rx.spacer(), rx.input(placeholder=f"{State.sample_size_validator}", on_change=State.set_sample_size_validator, width="80px")),

                    rx.divider(margin_y="1em"),
                    rx.heading("Toggles", size="3"),
                    rx.hstack(rx.text("Verbose"), rx.spacer(), rx.switch(checked=State.verbose, on_change=State.set_verbose, style={"cursor": "pointer"}), width="100%"),
                    rx.hstack(rx.text("Enable Validation"), rx.spacer(), rx.switch(checked=State.enable_validation, on_change=State.set_enable_validation, style={"cursor": "pointer"}), width="100%"),
                    rx.hstack(rx.text("Enable multi-column cleaning"), rx.spacer(), rx.switch(checked=State.enable_multi_col_cleaning, on_change=State.set_enable_multi_col_cleaning, style={"cursor": "pointer"}), width="100%"),
                    rx.hstack(rx.text("Enable multi-column validation"), rx.spacer(), rx.switch(checked=State.enable_validation_multi, on_change=State.set_enable_validation_multi, style={"cursor": "pointer"}), width="100%"),
                    rx.hstack(rx.text("Show metadata"), rx.spacer(), rx.switch(checked=State.include_metadata, on_change=State.set_include_metadata, style={"cursor": "pointer"}), width="100%"),

                    rx.divider(margin_y="1em"),
                    rx.heading("Limits", size="3"),
                    rx.hstack(rx.text("Max. single-column cleaning attempts"), rx.spacer(), rx.input(placeholder=f"{State.max_cleaning_attempts}", on_change=State.set_max_cleaning_attempts, width="80px")),
                    rx.hstack(rx.text("Max. multi-column cleaning attempts"), rx.spacer(), rx.input(placeholder=f"{State.max_multi_col_attempts}", on_change=State.set_max_multi_col_attempts, width="80px")),
                    rx.hstack(rx.text("Max. parsing attempts"), rx.spacer(), rx.input(placeholder=f"{State.max_parse_attempts}", on_change=State.set_max_parse_attempts, width="80px")),
                    rx.hstack(rx.text("Max. coding attempts"), rx.spacer(), rx.input(placeholder=f"{State.max_coding_attempts}", on_change=State.set_max_coding_attempts, width="80px")),
                    rx.hstack(rx.text("Semaphore Limit"), rx.spacer(), rx.input(placeholder=f"{State.semaphore_limit}", on_change=State.set_semaphore_limit, width="80px")),
                    
                    spacing="2", width="100%",
                ),
                height="500px",
            ),
            rx.dialog.close(rx.button("Save", variant="soft", margin_top="1em", width="100%")),
        ),
    )

def info_modal() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.trigger(
            rx.button("Info", variant="ghost", width="100%", margin_top="0.5em", style={"cursor": "pointer"}),
        ),
        rx.dialog.content(
            rx.dialog.title("Model selection (LLM)"),
            rx.scroll_area(
                rx.vstack(
                    rx.text("How to select / add an LLM", font_weight="bold"),
                    rx.divider(margin_y="0.5em"),
                    rx.text("1) Set API keys in `.env`", font_weight="bold"),
                    rx.code(
                        "\n".join(
                            [
                                "OPENAI_API_KEY=...",
                                "GEMINI_API_KEY=...",
                                "OPENROUTER_API_KEY=...",
                            ]
                        ),
                        font_family="monospace",
                        width="100%",
                    ),
                    rx.divider(margin_y="1em"),
                    rx.text("2) Select / add an LLM model (edit code)", font_weight="bold"),
                    rx.code(
                        "\n".join(
                            [
                                "Edit registry: Code/madclean/llm/llm_registry.py",
                                "Add new client: Code/madclean/llm/llm_clients.py",
                                "Register client: Code/madclean/llm/llm_registry.py",
                                "Add api_key_name + set the key in .env",
                            ]
                        ),
                        width="100%",
                        font_family="monospace",
                    ),
                    rx.divider(margin_y="1em"),
                    rx.text("3) Select the correct model in the UI", font_weight="bold"),
                    rx.text("Use the sidebar dropdown under Model Selection.", size="1", color_scheme="gray"),
                ),
                height="420px",
                scrollbars="vertical",
            ),
            rx.dialog.close(rx.button("Close", variant="soft", margin_top="1em", width="100%", style={"cursor": "pointer"})),
        ),
    )

def sidebar() -> rx.Component:
    return rx.vstack(
        rx.heading("MADClean", size="5"),
        rx.box(
            rx.vstack(
                rx.text("Data input", font_weight="bold", size="2"),
                rx.upload(
                    rx.vstack(rx.icon(tag="upload", size=20), rx.text("Upload Data", size="1")),
                    id="upload_side",
                    border=f"1px dashed {rx.color('gray', 6)}",
                    padding="0.9em",
                    width="100%",
                    on_drop=State.handle_upload(rx.upload_files(upload_id="upload_side")),
                    style={"cursor": "pointer"},
                ),
                rx.text("Selected file", size="1", color_scheme="gray"),
                rx.text(State.file_name, size="1"),
                spacing="2",
                width="100%",
                align_items="start",
            ),
            width="100%",
            padding="0.75em",
            border=f"1px solid {rx.color('gray', 4)}",
            border_radius="10px",
            background_color=rx.color("gray", 1),
        ),
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.text("Model selection", font_weight="bold", size="2"),
                    rx.spacer(),
                    rx.text("Info", size="1", color_scheme="gray"),
                    width="100%",
                    align="center",
                ),
                rx.text("Recommender agent", size="1", color_scheme="gray"),
                rx.select(
                    State.llm_options,
                    value=State.selected_llm_key_recommender,
                    on_change=State.set_selected_llm_key_recommender,
                    width="100%",
                    style={"cursor": "pointer"},
                ),
                rx.text("Coding agent", size="1", color_scheme="gray", margin_top="0.25em"),
                rx.select(
                    State.llm_options,
                    value=State.selected_llm_key_coding,
                    on_change=State.set_selected_llm_key_coding,
                    width="100%",
                    style={"cursor": "pointer"},
                ),
                rx.text("Validation agent", size="1", color_scheme="gray", margin_top="0.25em"),
                rx.select(
                    State.llm_options,
                    value=State.selected_llm_key_validation,
                    on_change=State.set_selected_llm_key_validation,
                    width="100%",
                    style={"cursor": "pointer"},
                ),
                rx.text("API keys must be set in `.env`.", size="1", color_scheme="gray"),
                info_modal(),
                spacing="2",
                width="100%",
                align_items="start",
            ),
            width="100%",
            padding="0.75em",
            border=f"1px solid {rx.color('gray', 4)}",
            border_radius="10px",
            background_color=rx.color("gray", 1),
        ),
        rx.box(
            rx.vstack(
                rx.text("Configuration", font_weight="bold", size="2"),
                settings_modal(),
                spacing="2",
                width="100%",
                align_items="start",
            ),
            width="100%",
            padding="0.75em",
            border=f"1px solid {rx.color('gray', 4)}",
            border_radius="10px",
            background_color=rx.color("gray", 1),
        ),
        rx.box(
            rx.vstack(
                rx.text("Actions", font_weight="bold", size="2"),
                rx.button(
                    "Run Pipeline",
                    on_click=State.run_cleaning_process,
                    is_loading=State.is_cleaning,
                    is_disabled=State.is_cleaning,
                    width="100%",
                    color_scheme="blue",
                    style={"cursor": "pointer"},
                ),
                rx.button(
                    "Stop Pipeline",
                    on_click=State.stop_pipeline,
                    width="100%",
                    variant="soft",
                    is_disabled=~State.is_cleaning,
                    color_scheme="red",
                    style={"cursor": "pointer"},
                ),
                rx.button(
                    "Download CSV",
                    on_click=State.download_cleaned_file,
                    width="100%",
                    variant="soft",
                    color_scheme="green",
                    style={"cursor": "pointer"},
                ),
                spacing="2",
                width="100%",
            ),
            width="100%",
            padding="0.75em",
            border=f"1px solid {rx.color('gray', 4)}",
            border_radius="10px",
            background_color=rx.color("gray", 1),
        ),
        rx.box(
            rx.vstack(
                rx.text("Bottom panel height", font_weight="bold", size="2"),
                rx.slider(
                    min=160,
                    max=900,
                    step=10,
                    value=[State.bottom_panel_height],
                    on_change=State.set_bottom_panel_height,
                    width="100%",
                    style={"cursor": "pointer"},
                ),
                spacing="2",
                width="100%",
            ),
            width="100%",
            padding="0.75em",
            border=f"1px solid {rx.color('gray', 4)}",
            border_radius="10px",
            background_color=rx.color("gray", 1),
        ),
        height="100vh",
        width="260px",
        padding="1em",
        background_color=rx.color("gray", 2),
        border_right=f"1px solid {rx.color('gray', 5)}",
        spacing="3",
        align_items="start",
    )

def main_content() -> rx.Component:
    return rx.vstack(
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.badge(State.status_msg, color_scheme="blue", variant="surface"),
                    rx.spacer(),
                    rx.text(
                        rx.cond(
                            State.total_rows == 0,
                            "No rows loaded.",
                            "Rows "
                            + (State.page_offset + 1).to(str)
                            + "-"
                            + (State.page_offset + State.page_size).to(str)
                            + " of "
                            + State.total_rows.to(str),
                        ),
                        size="1",
                        color_scheme="gray",
                    ),
                    rx.button(
                        "Prev",
                        on_click=State.prev_page,
                        is_disabled=State.page_offset <= 0,
                        variant="soft",
                        size="1",
                        width="80px",
                        style={"cursor": "pointer"},
                    ),
                    rx.button(
                        "Next",
                        on_click=State.next_page,
                        is_disabled=(State.page_offset + State.page_size) >= State.total_rows,
                        variant="soft",
                        size="1",
                        width="80px",
                        style={"cursor": "pointer"},
                    ),
                    width="100%",
                    align="center",
                ),
                rx.cond(
                    State.is_cleaning,
                    rx.hstack(
                        rx.progress(value=State.progress_percent, width="100%"),
                        rx.text(State.progress_percent.to(str) + "%", size="1", color_scheme="gray"),
                        width="100%",
                        align="center",
                        spacing="2",
                    ),
                    rx.spacer(),
                ),
                spacing="2",
                width="100%",
            ),
            width="100%",
            padding="0.75em",
            border=f"1px solid {rx.color('gray', 4)}",
            border_radius="10px",
            background_color=rx.color("gray", 1),
        ),
        # FIXED TABLE: Manual table with CSS overrides to prevent character wrapping
        rx.box(
            rx.scroll_area(
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.foreach(
                                State.column_names.to(list[str]), 
                                lambda col: rx.table.column_header_cell(
                                    col, 
                                    white_space="nowrap", 
                                    min_width="150px",
                                    background_color=State.column_header_colors[col],
                                    color="white",
                                    border_right=f"1px solid {rx.color('gray', 4)}",
                                    padding="0.75em"
                                )
                            )
                        )
                    ),
                    rx.table.body(
                        rx.foreach(
                            State.df_preview.to(list[dict[str, Any]]),
                            lambda row: rx.table.row(
                                rx.foreach(
                                    State.column_names.to(list[str]),
                                    # Use row.get(col) or row[col] with an explicit cast
                                    lambda col, col_idx: rx.table.cell(
                                        rx.box(
                                            rx.cond(
                                                State.has_cleaned,
                                                rx.input(
                                                    value=rx.cond(
                                                        (row[col] == None)
                                                        | (row[col].to(str) == "nan")
                                                        | (row[col].to(str) == "None")
                                                        | (row[col].to(str) == "NaN"),
                                                        "",
                                                        row[col].to(str),
                                                    ),
                                                    on_change=lambda v: State.update_cell(row["__row_index"].to(int), col, v),
                                                    width="100%",
                                                    size="1",
                                                    variant="soft",
                                                ),
                                                rx.cond(
                                                    (row[col] == None) | (row[col].to(str) == "nan") | (row[col].to(str) == "None") | (row[col].to(str) == "NaN"),
                                                    rx.text(""),
                                                    rx.text(row[col].to(str)),
                                                ),
                                            ),
                                            background_color=rx.cond(
                                                row["__modified_flags"].to(list[bool])[col_idx],
                                                "rgba(34, 197, 94, 0.28)",
                                                "transparent",
                                            ),
                                            border=rx.cond(
                                                row["__modified_flags"].to(list[bool])[col_idx],
                                                "1px solid rgba(34, 197, 94, 0.55)",
                                                "1px solid transparent",
                                            ),
                                            border_radius="6px",
                                            padding_x="0.35em",
                                            padding_y="0.15em",
                                            display="block",
                                            width="100%",
                                        ),
                                        white_space="nowrap",
                                        min_width="180px",
                                        border_right=f"1px solid {rx.color('gray', 4)}",  # Vertical line
                                        border_bottom=f"1px solid {rx.color('gray', 4)}", # Horizontal line
                                        overflow="hidden",
                                        text_overflow="ellipsis",
                                        padding="0.5em",
                                    )
                                ),
                                style={
                                    "_hover": {
                                        "backgroundColor": rx.color("gray", 2),
                                    }
                                },
                            ),
                        )
                    ),
                    width="100%",
                    variant="surface",
                    border_left=f"1px solid {rx.color('gray', 4)}",
                    border_top=f"1px solid {rx.color('gray', 4)}"
                ),
                scrollbars="both",
                style={"width": "100%", "height": "100%"},
            ),
            flex="1", width="100%", border=f"1px solid {rx.color('gray', 4)}", border_radius="8px",
            overflow="hidden",
            background_color=rx.color("gray", 1)
        ),
        rx.box(
            rx.scroll_area(
                rx.tabs.root(
                    rx.box(
                        rx.tabs.list(
                            rx.tabs.trigger(
                                "Logs",
                                value="logs",
                                style={"cursor": "pointer", "padding": "0.35em 0.6em"},
                            ),
                            rx.tabs.trigger(
                                "Profile",
                                value="profile",
                                style={"cursor": "pointer", "padding": "0.35em 0.6em"},
                            ),
                            rx.tabs.trigger(
                                "Usage",
                                value="usage",
                                style={"cursor": "pointer", "padding": "0.35em 0.6em"},
                            ),
                            rx.tabs.trigger(
                                "Report",
                                value="report",
                                style={"cursor": "pointer", "padding": "0.35em 0.6em"},
                            ),
                            rx.tabs.trigger(
                                "Instructions",
                                value="instructions",
                                style={"cursor": "pointer", "padding": "0.35em 0.6em"},
                            ),
                        ),
                        style={
                            "position": "sticky",
                            "top": "0px",
                            "zIndex": "10",
                            "background": rx.color("gray", 1),
                            "paddingTop": "0.35rem",
                            "paddingBottom": "0.35rem",
                            "paddingLeft": "0.5rem",
                            "paddingRight": "0.5rem",
                            "borderBottom": f"1px solid {rx.color('gray', 4)}",
                        },
                    ),
                    rx.tabs.content(
                        rx.vstack(
                            rx.text(State.profiling_status, font_weight="bold"),
                            rx.cond(
                                State.is_cleaning,
                                rx.progress(value=State.progress_percent, width="100%"),
                                rx.spacer(),
                            ),
                            rx.cond(
                                State.last_run_status == "cancelled",
                                rx.box(
                                    rx.text(
                                        "Pipeline stopped. No cleaning result was generated.",
                                        font_family="monospace",
                                        size="1",
                                    ),
                                    width="100%",
                                    padding="0.6em",
                                    border=f"1px solid {rx.color('gray', 4)}",
                                    border_radius="10px",
                                    background_color=rx.color("gray", 2),
                                ),
                                rx.foreach(State.logs, lambda log: rx.text(log, font_family="monospace", size="1")),
                            ),
                            spacing="1",
                            width="100%",
                        ),
                        value="logs",
                    ),
                    rx.tabs.content(
                        rx.vstack(
                            rx.hstack(
                                rx.text(State.profiling_status, font_weight="bold"),
                                rx.spacer(),
                            ),
                            rx.cond(
                                State.is_profiling,
                                rx.text("Profiling running...", color_scheme="gray"),
                                rx.spacer(),
                            ),
                            rx.divider(margin_y="0.5em"),
                            rx.heading("Column semantic types", size="3"),
                            rx.vstack(
                                rx.foreach(
                                    State.column_profile_rows.to(list[dict[str, str]]),
                                    lambda r: rx.hstack(
                                        rx.badge(
                                            r["name"],
                                            background_color=r["color"],
                                            color="white",
                                        ),
                                        rx.text(r["semantic_type"], size="1", color_scheme="gray"),
                                    ),
                                ),
                                align_items="start",
                            ),
                            rx.divider(margin_y="0.5em"),
                            rx.heading("Functional dependencies (issues)", size="3"),
                            rx.table.root(
                                rx.table.header(
                                    rx.table.row(
                                        rx.table.column_header_cell("LHS", padding="0.5em"),
                                        rx.table.column_header_cell("RHS", padding="0.5em"),
                                        rx.table.column_header_cell("Score", padding="0.5em"),
                                        rx.table.column_header_cell("Violations", padding="0.5em"),
                                        rx.table.column_header_cell("Imputables", padding="0.5em"),
                                    )
                                ),
                                rx.table.body(
                                    rx.foreach(
                                        State.fd_results.to(list[dict[str, Any]]),
                                        lambda fd: rx.table.row(
                                            rx.table.cell(rx.text(fd["lhs"])),
                                            rx.table.cell(rx.text(fd["rhs"])),
                                            rx.table.cell(rx.text(rx.cond(fd["score"] == None, "", fd["score"].to(str)))),
                                            rx.table.cell(rx.text(rx.cond(fd["violations_count"] == None, "", fd["violations_count"].to(str)))),
                                            rx.table.cell(rx.text(rx.cond(fd["imputables_count"] == None, "", fd["imputables_count"].to(str)))),
                                        ),
                                    )
                                ),
                                variant="surface",
                            ),
                            spacing="3",
                            width="100%",
                        ),
                        value="profile",
                    ),
                    rx.tabs.content(
                        rx.vstack(
                            rx.hstack(
                                rx.text("Runtime:", font_weight="bold"),
                                rx.text(State.runtime_seconds_display + "s", color_scheme="gray"),
                            ),
                            rx.text("Token usage:", font_weight="bold"),
                            rx.cond(
                                State.last_run_status == "cancelled",
                                rx.box(
                                    rx.text(
                                        "Pipeline stopped. No usage available for this run.",
                                        size="1",
                                        font_family="monospace",
                                        color_scheme="gray",
                                    ),
                                    width="100%",
                                    padding="0.6em",
                                    border=f"1px solid {rx.color('gray', 4)}",
                                    border_radius="10px",
                                    background_color=rx.color("gray", 2),
                                ),
                                rx.vstack(
                                    rx.text(
                                        rx.cond(
                                            State.enable_validation,
                                            "",
                                            "Note: Validation is disabled, so Validation agent token usage can be 0.",
                                        ),
                                        size="1",
                                        color_scheme="gray",
                                    ),
                                    rx.box(
                                        rx.text(
                                            State.token_usage_block,
                                            font_family="monospace",
                                            size="1",
                                            white_space="pre-wrap",
                                        ),
                                        width="100%",
                                        padding="0.6em",
                                        border=f"1px solid {rx.color('gray', 4)}",
                                        border_radius="8px",
                                        background_color=rx.color("gray", 2),
                                    ),
                                    spacing="1",
                                    width="100%",
                                ),
                            ),
                            spacing="2",
                            width="100%",
                        ),
                        value="usage",
                    ),
                    rx.tabs.content(
                        rx.vstack(
                            rx.text("Cleaning code report", font_weight="bold"),
                            rx.text(
                                "Select a column to view the generated Python cleaning code and metadata.",
                                size="1",
                                color_scheme="gray",
                            ),
                            rx.divider(margin_y="0.5em"),
                            rx.cond(
                                State.last_run_status == "cancelled",
                                rx.box(
                                    rx.text(
                                        "Pipeline stopped. No report generated for this run.",
                                        size="1",
                                        font_family="monospace",
                                        color_scheme="gray",
                                    ),
                                    width="100%",
                                    padding="0.6em",
                                    border=f"1px solid {rx.color('gray', 4)}",
                                    border_radius="10px",
                                    background_color=rx.color("gray", 2),
                                ),
                                rx.spacer(),
                            ),
                            rx.select(
                                State.report_code_keys.to(list[str]),
                                value=State.selected_report_code_key,
                                on_change=State.set_selected_report_code_key,
                                width="100%",
                                style={"cursor": "pointer"},
                            ),
                            rx.text(State.selected_report_status, color_scheme="gray", size="1"),
                            rx.text("Metadata:", font_weight="bold", margin_top="0.5em"),
                            rx.vstack(
                                rx.foreach(
                                    State.selected_report_meta_lines,
                                    lambda line: rx.text(line, font_family="monospace", size="1"),
                                ),
                                align_items="start",
                            ),
                            rx.text("Generated Python code:", font_weight="bold", margin_top="0.5em"),
                            rx.box(
                                rx.vstack(
                                    rx.foreach(
                                        State.generated_code_lines_for_selected,
                                        lambda line: rx.text(line, font_family="monospace", size="1"),
                                    ),
                                    align_items="start",
                                ),
                                width="100%",
                                padding="0.6em",
                                border=f"1px solid {rx.color('gray', 4)}",
                                border_radius="8px",
                                background_color=rx.color("gray", 2),
                            ),
                            spacing="2",
                            width="100%",
                        ),
                        value="report",
                    ),
                    rx.tabs.content(
                        rx.vstack(
                            rx.text("Instructions", font_weight="bold"),
                            rx.text("End-to-end checklist to run MADClean.", size="1", color_scheme="gray"),
                            rx.divider(margin_y="0.5em"),
                            rx.text("1) Add API keys to `.env`", font_weight="bold"),
                            rx.code(
                                "\n".join(
                                    [
                                        "OPENAI_API_KEY=...",
                                        "GEMINI_API_KEY=...",
                                        "OPENROUTER_API_KEY=...",
                                    ]
                                ),
                                width="100%",
                                font_family="monospace",
                            ),
                            rx.text("2) Configure model", font_weight="bold"),
                            rx.text(
                                "Configure model using instructions in Model Selection (left sidebar → Info).",
                                size="1",
                                color_scheme="gray",
                            ),
                            rx.text("3) Use the UI", font_weight="bold"),
                            rx.text("- Upload CSV → wait for profiling", size="1"),
                            rx.text("- Pick LLM client → set config (Advanced Configuration)", size="1"),
                            rx.text("- Run Pipeline → watch Logs/Progress", size="1"),
                            rx.text("- Review Usage/Report → optionally edit table cells", size="1"),
                            rx.text("- Download CSV (exports current cleaned+edited table)", size="1"),
                            spacing="3",
                            width="100%",
                        ),
                        value="instructions",
                    ),
                    width="100%",
                    background_color=rx.color("gray", 1),
                    border_top=f"1px solid {rx.color('gray', 5)}",
                ),
                scrollbars="vertical",
                style={"height": "100%", "width": "100%"},
            ),
            width="100%",
            padding="1em",
            border_top=f"2px solid {rx.color('gray', 6)}",
            style={"height": State.bottom_panel_height.to(str) + "px"},
        ),
        padding="1.5em", width="100%", height="100vh", spacing="4",
    )

def index() -> rx.Component:
    return rx.hstack(sidebar(), main_content(), width="100%", spacing="0")

app = rx.App(theme=rx.theme(accent_color="indigo"))
app.add_page(index)