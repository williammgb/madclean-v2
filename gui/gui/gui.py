import reflex as rx
from .state import State
from typing import Any

def sample_size_input(category: str, subkey: str) -> rx.Component:
    """Compact helper for nested dictionary sample-size inputs."""
    label = subkey.replace("_", " ").title()
    return rx.hstack(
        rx.text(label, size="1", color_scheme="gray"),
        rx.input(
            placeholder=f"{State.sample_sizes[category][subkey]}",
            on_change=lambda val: State.update_sample_size(category, subkey, val),
            width="80px",
            size="1",
        ),
        spacing="2",
        align="center",
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
                    rx.heading("Per-agent models", size="2"),
                    rx.grid(
                        rx.text("Recommender", size="1", color_scheme="gray"),
                        rx.text("Coding", size="1", color_scheme="gray"),
                        rx.text("Validation", size="1", color_scheme="gray"),
                        columns="3",
                        spacing="2",
                        width="100%",
                    ),
                    rx.grid(
                        rx.select(
                            State.llm_options,
                            value=State.selected_llm_key_recommender,
                            on_change=State.set_selected_llm_key_recommender,
                            width="100%",
                            size="1",
                            style={"cursor": "pointer"},
                        ),
                        rx.select(
                            State.llm_options,
                            value=State.selected_llm_key_coding,
                            on_change=State.set_selected_llm_key_coding,
                            width="100%",
                            size="1",
                            style={"cursor": "pointer"},
                        ),
                        rx.select(
                            State.validation_llm_options,
                            value=State.selected_llm_key_validation,
                            on_change=State.set_selected_llm_key_validation,
                            width="100%",
                            size="1",
                            style={"cursor": "pointer"},
                        ),
                        columns="3",
                        spacing="2",
                        width="100%",
                    ),

                    rx.divider(margin_y="1em"),
                    rx.heading("Toggles", size="2"),
                    rx.hstack(rx.text("Verbose (show raw logs)", size="1"), rx.spacer(), rx.switch(checked=State.verbose, on_change=State.set_verbose, size="1", style={"cursor": "pointer"}), width="100%"),
                    rx.hstack(rx.text("Enable validation of cleaning operations", size="1"), rx.spacer(), rx.switch(checked=State.enable_validation, on_change=State.set_enable_validation, size="1", style={"cursor": "pointer"}), width="100%"),
                    rx.hstack(rx.text("Enable multi-column cleaning", size="1"), rx.spacer(), rx.switch(checked=State.enable_multi_col_cleaning, on_change=State.set_enable_multi_col_cleaning, size="1", style={"cursor": "pointer"}), width="100%"),
                    rx.hstack(rx.text("Enable multi-column validation", size="1"), rx.spacer(), rx.switch(checked=State.enable_validation_multi, on_change=State.set_enable_validation_multi, size="1", style={"cursor": "pointer"}), width="100%"),

                    rx.divider(margin_y="1em"),
                    rx.heading("Limits", size="2"),
                    rx.hstack(rx.text("Max. single-column cleaning attempts", size="1"), rx.spacer(), rx.input(placeholder=f"{State.max_cleaning_attempts}", on_change=State.set_max_cleaning_attempts, width="80px", size="1")),
                    rx.hstack(rx.text("Max. multi-column cleaning attempts", size="1"), rx.spacer(), rx.input(placeholder=f"{State.max_multi_col_attempts}", on_change=State.set_max_multi_col_attempts, width="80px", size="1")),
                    rx.hstack(rx.text("Max. parsing attempts", size="1"), rx.spacer(), rx.input(placeholder=f"{State.max_parse_attempts}", on_change=State.set_max_parse_attempts, width="80px", size="1")),
                    rx.hstack(rx.text("Max. coding attempts", size="1"), rx.spacer(), rx.input(placeholder=f"{State.max_coding_attempts}", on_change=State.set_max_coding_attempts, width="80px", size="1")),
                    rx.hstack(rx.text("Semaphore Limit", size="1"), rx.spacer(), rx.input(placeholder=f"{State.semaphore_limit}", on_change=State.set_semaphore_limit, width="80px", size="1")),
                    rx.hstack(
                        rx.text("Max labeled cells per column", size="1"),
                        rx.spacer(),
                        rx.input(
                            placeholder=f"{State.max_labeled_cells_per_column}",
                            on_change=State.set_max_labeled_cells_per_column,
                            width="80px",
                        ),
                    ),
                    rx.hstack(
                        rx.text("If validation fails after max attempts", size="1", flex="1"),
                        rx.select(
                            ["accept_cleaned", "leave_uncleaned", "ask_user"],
                            value=State.validator_failure_strategy,
                            on_change=State.set_validator_failure_strategy,
                            width="200px",
                        ),
                        width="100%",
                        align="center",
                        spacing="2",
                    ),
                    rx.text(
                        "accept_cleaned: use the last attempt; leave_uncleaned: revert; ask_user: pause for your decision.",
                        size="1",
                        color_scheme="gray",
                    ),

                    rx.divider(margin_y="1em"),
                    rx.heading("LLM settings", size="3"),
                    rx.text("Leave empty to use each provider’s defaults. Unsupported values are ignored automatically.", size="1", color_scheme="gray"),
                    rx.hstack(rx.text("temperature", size="1"), rx.spacer(), rx.input(placeholder="default", on_change=State.set_llm_temperature_input, width="100px", size="1")),
                    rx.hstack(rx.text("top_p", size="1"), rx.spacer(), rx.input(placeholder="default", on_change=State.set_llm_top_p_input, width="100px", size="1")),

                    rx.divider(margin_y="1em"),
                    rx.heading("Sample sizes", size="2"),
                    rx.text("Determines the number of clean/dirty examples sent to the LLM for context.", size="1", color_scheme="gray"),
                    rx.hstack(
                        rx.text("NUMERIC", size="1", min_width="74px"),
                        sample_size_input("NUMERIC", "clean_sample_size"),
                        sample_size_input("NUMERIC", "dirty_sample_size"),
                        width="100%",
                        align="center",
                        spacing="2",
                    ),
                    rx.hstack(
                        rx.text("DATETIME", size="1", min_width="74px"),
                        sample_size_input("DATETIME", "clean_sample_size"),
                        sample_size_input("DATETIME", "dirty_sample_size"),
                        width="100%",
                        align="center",
                        spacing="2",
                    ),
                    rx.hstack(
                        rx.text("DIRTY_NUMERIC", size="1", min_width="74px"),
                        sample_size_input("DIRTY_NUMERIC", "random_sample_size"),
                        sample_size_input("DIRTY_NUMERIC", "unique_sample_size"),
                        width="100%",
                        align="center",
                        spacing="2",
                    ),
                    rx.hstack(
                        rx.text("STRING", size="1", min_width="74px"),
                        sample_size_input("STRING", "random_sample_size"),
                        sample_size_input("STRING", "unique_sample_size"),
                        width="100%",
                        align="center",
                        spacing="2",
                    ),
                    rx.hstack(
                        rx.text("NLT", size="1", min_width="74px"),
                        sample_size_input("NLT", "short_sample_size"),
                        sample_size_input("NLT", "long_sample_size"),
                        width="100%",
                        align="center",
                        spacing="2",
                    ),
                    rx.hstack(rx.text("Validator sample size"), rx.spacer(), rx.input(placeholder=f"{State.sample_size_validator}", on_change=State.set_sample_size_validator, width="80px")),
                    rx.hstack(rx.text("Validator random sample"), rx.spacer(), rx.input(placeholder=f"{State.sample_size_validator_random}", on_change=State.set_sample_size_validator_random, width="80px")),
                    rx.hstack(rx.text("Validator modified sample"), rx.spacer(), rx.input(placeholder=f"{State.sample_size_validator_changed}", on_change=State.set_sample_size_validator_changed, width="80px")),

                    spacing="2", width="100%",
                ),
                height="420px",
            ),
            rx.dialog.close(rx.button("Save", variant="soft", margin_top="1em", width="100%")),
        ),
    )

def evaluation_panel() -> rx.Component:
    """Ground-truth upload and cleaning quality metrics (vs pipeline output)."""
    return rx.vstack(
        rx.heading("Evaluation", size="4"),
        rx.hstack(
            rx.text("Ground truth file:", font_weight="bold", size="2"),
            rx.text(State.gt_file_name, size="2"),
            width="100%",
            align="center",
            spacing="2",
        ),
        rx.upload(
            rx.vstack(rx.icon(tag="upload", size=20), rx.text("Upload ground-truth CSV or Excel", size="1")),
            id="upload_gt",
            border=f"1px dashed {rx.color('gray', 6)}",
            padding="0.9em",
            width="100%",
            on_drop=State.handle_gt_upload(rx.upload_files(upload_id="upload_gt")),
            style={"cursor": "pointer"},
        ),
        rx.button(
            "Clear ground truth",
            on_click=State.clear_ground_truth,
            variant="soft",
            size="2",
            style={"cursor": "pointer"},
        ),
        rx.cond(
            State.evaluation_error != "",
            rx.box(
                rx.text(State.evaluation_error, size="2", color_scheme="red"),
                width="100%",
                padding="0.65em",
                border=f"1px solid {rx.color('red', 6)}",
                border_radius="8px",
                background_color=rx.color("red", 2),
            ),
            rx.fragment(),
        ),
        rx.cond(
            State.evaluation_status != "",
            rx.text(State.evaluation_status, size="2", color_scheme="gray"),
            rx.fragment(),
        ),
        rx.cond(
            State.evaluation_ready,
            rx.vstack(
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("TP", padding="0.5em"),
                            rx.table.column_header_cell("FP", padding="0.5em"),
                            rx.table.column_header_cell("TN", padding="0.5em"),
                            rx.table.column_header_cell("FN", padding="0.5em"),
                            rx.table.column_header_cell("Precision", padding="0.5em"),
                            rx.table.column_header_cell("Recall", padding="0.5em"),
                            rx.table.column_header_cell("F1", padding="0.5em"),
                        )
                    ),
                    rx.table.body(
                        rx.foreach(
                            State.evaluation_overall_rows.to(list[dict[str, str]]),
                            lambda r: rx.table.row(
                                rx.table.cell(rx.text(r["tp"], size="1")),
                                rx.table.cell(rx.text(r["fp"], size="1")),
                                rx.table.cell(rx.text(r["tn"], size="1")),
                                rx.table.cell(rx.text(r["fn"], size="1")),
                                rx.table.cell(rx.text(r["precision"], size="1", font_weight="bold")),
                                rx.table.cell(rx.text(r["recall"], size="1", font_weight="bold")),
                                rx.table.cell(rx.text(r["f1"], size="1", font_weight="bold")),
                            ),
                        )
                    ),
                    variant="surface",
                    width="100%",
                ),
                rx.heading("Per column", size="3", margin_top="0.75em"),
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Column", padding="0.35em"),
                            rx.table.column_header_cell("TP", padding="0.35em"),
                            rx.table.column_header_cell("FP", padding="0.35em"),
                            rx.table.column_header_cell("TN", padding="0.35em"),
                            rx.table.column_header_cell("FN", padding="0.35em"),
                            rx.table.column_header_cell("Precision", padding="0.35em"),
                            rx.table.column_header_cell("Recall", padding="0.35em"),
                            rx.table.column_header_cell("F1", padding="0.35em"),
                        )
                    ),
                    rx.table.body(
                        rx.foreach(
                            State.evaluation_column_rows.to(list[dict[str, str]]),
                            lambda r: rx.table.row(
                                rx.table.cell(rx.text(r["column"], size="1")),
                                rx.table.cell(rx.text(r["tp"], size="1")),
                                rx.table.cell(rx.text(r["fp"], size="1")),
                                rx.table.cell(rx.text(r["tn"], size="1")),
                                rx.table.cell(rx.text(r["fn"], size="1")),
                                rx.table.cell(rx.text(r["precision"], size="1")),
                                rx.table.cell(rx.text(r["recall"], size="1")),
                                rx.table.cell(rx.text(r["f1"], size="1")),
                            ),
                        )
                    ),
                    variant="surface",
                    width="100%",
                ),
                spacing="2",
                width="100%",
                align_items="start",
            ),
            rx.fragment(),
        ),
        spacing="3",
        width="100%",
        align_items="start",
        padding_bottom="1em",
    )

def usage_panel() -> rx.Component:
    return rx.cond(
        State.has_cleaned,
        rx.vstack(
            rx.text("Runtime: ", State.runtime_seconds_display + "s"),
            rx.flex(
                rx.foreach(
                    State.token_usage_donut_rows.to(list[dict[str, Any]]),
                    lambda row: rx.box(
                        rx.vstack(
                            rx.box(
                                rx.box(
                                    rx.text(row["center_text"], font_weight="bold", size="2", color="#111827"),
                                    width="56px",
                                    height="56px",
                                    border_radius="9999px",
                                    background_color="white",
                                    display="flex",
                                    align_items="center",
                                    justify_content="center",
                                    border=f"1px solid {rx.color('gray', 5)}",
                                ),
                                width="108px",
                                height="108px",
                                border_radius="9999px",
                                style={"background": row["bg"]},
                                display="flex",
                                align_items="center",
                                justify_content="center",
                            ),
                            rx.text(row["title"], font_weight="bold", size="1"),
                            rx.text(row["subtitle"], size="1", color_scheme="gray"),
                            rx.vstack(
                                rx.foreach(
                                    row["legend_items"].to(list[dict[str, str]]),
                                    lambda li: rx.hstack(
                                        rx.box(
                                            width="10px",
                                            height="10px",
                                            border_radius="2px",
                                            background_color=li["color"],
                                            flex_shrink="0",
                                        ),
                                        rx.text(li["label"], size="1", white_space="pre-wrap"),
                                        spacing="2",
                                        align="center",
                                        width="100%",
                                    ),
                                ),
                                spacing="1",
                                width="100%",
                                align_items="start",
                            ),
                            spacing="1",
                            width="100%",
                            align_items="center",
                        ),
                        min_width="180px",
                        padding="0.6em",
                        border=f"1px solid {rx.color('gray', 4)}",
                        border_radius="10px",
                        background_color=rx.color("gray", 1),
                    ),
                ),
                wrap="wrap",
                gap="0.75rem",
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        rx.text(
            "Runtime and token usage will be shown here after cleaning is finished.",
            size="1",
            color_scheme="gray",
        ),
    )

def sidebar() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.image(src="/icon.png", width="28px", height="28px"),
            rx.heading("MADClean", size="5"),
            spacing="2",
            align="center",
            width="100%",
        ),
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
                rx.text("LLM model", font_weight="bold", size="2"),
                rx.select(
                    State.llm_options,
                    value=State.selected_llm_key_recommender,
                    on_change=State.set_selected_llm_key_all_agents,
                    width="100%",
                    size="2",
                    style={"cursor": "pointer"},
                ),
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
                    rx.text("Human-in-the-loop", font_weight="bold", size="2"),
                    rx.spacer(),
                    rx.switch(checked=State.human_in_the_loop, on_change=State.set_human_in_the_loop, style={"cursor": "pointer"}),
                    width="100%",
                    align="center",
                ),
                rx.cond(
                    State.human_in_the_loop,
                    rx.vstack(
                        rx.hstack(
                            rx.text("All columns"),
                            rx.spacer(),
                            rx.switch(checked=State.hitl_apply_to_all_columns, on_change=State.set_hitl_apply_to_all_columns, style={"cursor": "pointer"}),
                            width="100%",
                        ),
                        rx.cond(
                            ~State.hitl_apply_to_all_columns,
                            rx.cond(
                                State.total_rows > 0,
                                rx.vstack(
                                    rx.text("Columns for HITL", size="1", color_scheme="gray"),
                                    rx.box(
                                        rx.scroll_area(
                                            rx.vstack(
                                                rx.foreach(
                                                    State.hitl_column_checkbox_rows.to(list[dict[str, str]]),
                                                    lambda row: rx.hstack(
                                                        rx.checkbox(
                                                            checked=row["active"] == "1",
                                                            on_change=lambda _: State.toggle_hitl_column_pick(
                                                                row["column"]
                                                            ),
                                                            style={"cursor": "pointer"},
                                                        ),
                                                        rx.text(row["column"], size="1"),
                                                        width="100%",
                                                        align="center",
                                                        spacing="2",
                                                    ),
                                                ),
                                                spacing="1",
                                                width="100%",
                                                align_items="start",
                                            ),
                                            type="hover",
                                            scrollbars="vertical",
                                            style={"maxHeight": "125px"},
                                        ),
                                        width="100%",
                                        padding="0.35em",
                                        border=f"1px solid {rx.color('gray', 6)}",
                                        border_radius="8px",
                                        background_color=rx.color("gray", 1),
                                    ),
                                    spacing="1",
                                    width="100%",
                                    align_items="start",
                                ),
                                rx.text("Upload a dataset to choose columns.", size="1", color_scheme="gray"),
                            ),
                            rx.spacer(),
                        ),
                        spacing="2",
                        width="100%",
                        align_items="start",
                    ),
                    rx.spacer(),
                ),
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
        rx.box(settings_modal(), width="100%", padding_x="0.75em"),
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
                rx.button(
                    "Download Cleaning Code",
                    on_click=State.download_cleaning_code,
                    width="100%",
                    variant="soft",
                    color_scheme="indigo",
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
        height="100%",
        min_height="100vh",
        width="100%",
        max_width="340px",
        flex="1 1 240px",
        min_width="min(100%, 220px)",
        flex_shrink="1",
        overflow_y="auto",
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
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.button(
                        "Table",
                        on_click=lambda: State.set_selected_main_tab("table"),
                        variant=rx.cond(State.selected_main_tab == "table", "solid", "soft"),
                        size="1",
                        style={"cursor": "pointer"},
                    ),
                    rx.button(
                        "Data Profiling",
                        on_click=lambda: State.set_selected_main_tab("dashboard"),
                        variant=rx.cond(State.selected_main_tab == "dashboard", "solid", "soft"),
                        size="1",
                        style={"cursor": "pointer"},
                    ),
                    rx.button(
                        "Pipeline",
                        on_click=lambda: State.set_selected_main_tab("pipeline"),
                        variant=rx.cond(State.selected_main_tab == "pipeline", "solid", "soft"),
                        size="1",
                        style={"cursor": "pointer"},
                    ),
                    rx.button(
                        "Report",
                        on_click=lambda: State.set_selected_main_tab("report"),
                        variant=rx.cond(State.selected_main_tab == "report", "solid", "soft"),
                        size="1",
                        style={"cursor": "pointer"},
                    ),
                    rx.button(
                        "Logs",
                        on_click=lambda: State.set_selected_main_tab("logs"),
                        variant=rx.cond(State.selected_main_tab == "logs", "solid", "soft"),
                        size="1",
                        style={"cursor": "pointer"},
                    ),
                    rx.button(
                        "Evaluation",
                        on_click=lambda: State.set_selected_main_tab("evaluation"),
                        variant=rx.cond(State.selected_main_tab == "evaluation", "solid", "soft"),
                        size="1",
                        style={"cursor": "pointer"},
                    ),
                    rx.button(
                        "Instructions",
                        on_click=lambda: State.set_selected_main_tab("instructions"),
                        variant=rx.cond(State.selected_main_tab == "instructions", "solid", "soft"),
                        size="1",
                        style={"cursor": "pointer"},
                    ),
                    rx.spacer(),
                    width="100%",
                    align="center",
                    flex_wrap="wrap",
                    spacing="2",
                ),
                rx.cond(
                    State.selected_main_tab == "table",
                    rx.box(
                        rx.vstack(
                            rx.cond(
                                State.recommender_hint_dialog_open,
                                rx.box(
                                    rx.box(
                                        rx.vstack(
                                            rx.text("Recommender instructions", font_weight="bold", size="4"),
                                            rx.text(State.recommender_hint_editing_column, size="2"),
                                            rx.text(
                                                "Appended to the recommender LLM prompt for this column only (see Instructions tab).",
                                                size="1",
                                                color_scheme="gray",
                                            ),
                                            rx.text_area(
                                                value=State.recommender_hint_editor_text,
                                                on_change=State.set_recommender_hint_editor_text,
                                                placeholder="Optional: e.g. replace sentinel -999 with NaN.",
                                                min_height="140px",
                                                width="100%",
                                                size="2",
                                            ),
                                            rx.hstack(
                                                rx.button(
                                                    "Cancel",
                                                    variant="soft",
                                                    on_click=State.cancel_recommender_hint_editor,
                                                    style={"cursor": "pointer"},
                                                ),
                                                rx.button(
                                                    "Save",
                                                    color_scheme="indigo",
                                                    on_click=State.save_recommender_hint_and_close,
                                                    style={"cursor": "pointer"},
                                                ),
                                                spacing="3",
                                                width="100%",
                                                justify="end",
                                            ),
                                            spacing="3",
                                            width="100%",
                                            max_width="520px",
                                            align_items="start",
                                        ),
                                        padding="1.25em",
                                        border_radius="12px",
                                        background_color=rx.color("gray", 1),
                                        box_shadow="lg",
                                        border=f"1px solid {rx.color('gray', 5)}",
                                    ),
                                    position="fixed",
                                    top="0",
                                    left="0",
                                    right="0",
                                    bottom="0",
                                    z_index="55",
                                    background_color="rgba(15, 23, 42, 0.45)",
                                    display="flex",
                                    align_items="center",
                                    justify_content="center",
                                    padding="1rem",
                                    overflow_y="auto",
                                ),
                                rx.fragment(),
                            ),
                            rx.cond(
                                State.labeling_dialog_open,
                                rx.box(
                                    rx.box(
                                        rx.vstack(
                                            rx.text("Label cell", font_weight="bold", size="4"),
                                            rx.text("Column", size="1", color_scheme="gray"),
                                            rx.text(State.labeling_target_col, size="2"),
                                            rx.text("Value", size="1", color_scheme="gray"),
                                            rx.code(
                                                rx.cond(
                                                    (State.labeling_current_value_preview == "")
                                                    | (State.labeling_current_value_preview == "nan")
                                                    | (State.labeling_current_value_preview == "None")
                                                    | (State.labeling_current_value_preview == "NaN"),
                                                    "(empty)",
                                                    State.labeling_current_value_preview,
                                                ),
                                                width="100%",
                                                white_space="pre-wrap",
                                                font_size="13px",
                                            ),
                                            rx.text("Label", size="1", color_scheme="gray"),
                                            rx.vstack(
                                                rx.button(
                                                    "Clean",
                                                    on_click=lambda: State.set_labeling_kind("clean"),
                                                    variant=rx.cond(State.labeling_kind == "clean", "solid", "soft"),
                                                    color_scheme="green",
                                                    width="100%",
                                                    style={"cursor": "pointer"},
                                                ),
                                                rx.button(
                                                    "Dirty",
                                                    on_click=lambda: State.set_labeling_kind("dirty"),
                                                    variant=rx.cond(State.labeling_kind == "dirty", "solid", "soft"),
                                                    color_scheme="red",
                                                    width="100%",
                                                    style={"cursor": "pointer"},
                                                ),
                                                spacing="2",
                                                width="100%",
                                            ),
                                            rx.cond(
                                                State.labeling_kind == "dirty",
                                                rx.vstack(
                                                    rx.text_area(
                                                        value=State.labeling_expected_value,
                                                        on_change=State.set_labeling_expected_value,
                                                        placeholder="Correct value for this cell",
                                                        min_height="100px",
                                                        width="100%",
                                                        size="2",
                                                    ),
                                                    spacing="2",
                                                    align_items="start",
                                                    width="100%",
                                                ),
                                                rx.fragment(),
                                            ),
                                            rx.hstack(
                                                rx.button(
                                                    "Remove label",
                                                    variant="outline",
                                                    color_scheme="orange",
                                                    on_click=State.clear_current_cell_label,
                                                    style={"cursor": "pointer"},
                                                ),
                                                rx.spacer(),
                                                rx.button(
                                                    "Cancel",
                                                    variant="soft",
                                                    on_click=State.cancel_cell_label_dialog,
                                                    style={"cursor": "pointer"},
                                                ),
                                                rx.button(
                                                    "Save",
                                                    color_scheme="indigo",
                                                    on_click=State.save_cell_label,
                                                    style={"cursor": "pointer"},
                                                ),
                                                spacing="3",
                                                width="100%",
                                                justify="end",
                                            ),
                                            spacing="3",
                                            width="100%",
                                            max_width="520px",
                                            align_items="start",
                                        ),
                                        padding="1.25em",
                                        border_radius="12px",
                                        background_color=rx.color("gray", 1),
                                        box_shadow="lg",
                                        border=f"1px solid {rx.color('gray', 5)}",
                                    ),
                                    position="fixed",
                                    top="0",
                                    left="0",
                                    right="0",
                                    bottom="0",
                                    z_index="56",
                                    background_color="rgba(15, 23, 42, 0.45)",
                                    display="flex",
                                    align_items="center",
                                    justify_content="center",
                                    padding="1rem",
                                    overflow_y="auto",
                                ),
                                rx.fragment(),
                            ),
                            rx.hstack(
                                rx.button(
                                    "Show FDs",
                                    on_click=lambda: State.set_show_fds(~State.show_fds),
                                    variant=rx.cond(State.show_fds, "solid", "soft"),
                                    color_scheme="violet",
                                    size="1",
                                    style={"cursor": "pointer"},
                                ),
                                rx.hstack(
                                    rx.text("Label cells", size="1", color_scheme="gray"),
                                    rx.switch(
                                        checked=State.label_cells_mode,
                                        on_change=State.set_label_cells_mode,
                                        size="1",
                                        style={"cursor": "pointer"},
                                    ),
                                    spacing="2",
                                    align="center",
                                ),
                                rx.spacer(),
                                width="100%",
                                align="center",
                            ),
                            rx.box(
                                rx.vstack(
                                    rx.cond(
                                        State.show_fds,
                                        rx.box(
                                            rx.vstack(
                                                rx.flex(
                                                    rx.foreach(
                                                        State.fd_button_rows.to(list[dict[str, str]]),
                                                        lambda fd: rx.button(
                                                            fd["lhs"],
                                                            " \u2192 ",
                                                            fd["rhs"],
                                                            on_click=lambda: State.toggle_fd_selection(fd["lhs"], fd["rhs"]),
                                                            variant=rx.cond(fd["active"] == "1", "solid", "soft"),
                                                            color_scheme=rx.cond(fd["active"] == "1", "gray", "violet"),
                                                            size="1",
                                                            style={"cursor": "pointer"},
                                                        ),
                                                    ),
                                                    wrap="wrap",
                                                    gap="0.45rem",
                                                    width="100%",
                                                ),
                                                width="100%",
                                                spacing="2",
                                                align_items="start",
                                            ),
                                            width="100%",
                                            padding="0.5em",
                                            border=f"1px solid {rx.color('gray', 4)}",
                                            border_radius="8px",
                                            background_color=rx.color("gray", 1),
                                        ),
                                        rx.fragment(),
                                    ),
                                    rx.scroll_area(
                                        rx.vstack(
                                            rx.table.root(
                                                rx.table.header(
                                                    rx.table.row(
                                                        rx.foreach(
                                                            State.column_names.to(list[str]),
                                                            lambda col: rx.table.column_header_cell(
                                                                rx.box(
                                                                    rx.vstack(
                                                                        rx.cond(
                                                                            State.column_fd_markers[col] != "",
                                                                            rx.hstack(
                                                                                rx.cond(
                                                                                    State.column_fd_markers.get(col, "").contains("->"),
                                                                                    rx.badge("LHS", size="1", color_scheme="gray", variant="surface"),
                                                                                    rx.fragment(),
                                                                                ),
                                                                                rx.cond(
                                                                                    State.column_fd_markers.get(col, "").contains("<-"),
                                                                                    rx.badge("RHS", size="1", color_scheme="gray", variant="surface"),
                                                                                    rx.fragment(),
                                                                                ),
                                                                                spacing="1",
                                                                                align="center",
                                                                            ),
                                                                            rx.fragment(),
                                                                        ),
                                                                        rx.hstack(
                                                                            rx.text(col, font_weight="bold"),
                                                                            rx.icon(
                                                                                tag=rx.cond(
                                                                                    State.user_marked_clean_columns.get(col, False),
                                                                                    "check-check",
                                                                                    "check",
                                                                                ),
                                                                                size=14,
                                                                                flex_shrink="0",
                                                                                opacity="0.95",
                                                                                on_click=State.toggle_user_marked_column_clean(col),
                                                                                style={"cursor": "pointer"},
                                                                                title="Mark column as already clean",
                                                                            ),
                                                                            rx.icon(
                                                                                tag="cog",
                                                                                size=14,
                                                                                flex_shrink="0",
                                                                                opacity="0.95",
                                                                                on_click=State.open_recommender_hint_editor(col),
                                                                                style={"cursor": "pointer"},
                                                                                title="Add column-specific instructions",
                                                                            ),
                                                                            spacing="2",
                                                                            align="center",
                                                                            width="100%",
                                                                        ),
                                                                        width="100%",
                                                                        spacing="1",
                                                                        align_items="start",
                                                                    ),
                                                                    width="100%",
                                                                ),
                                                                white_space="nowrap",
                                                                min_width=State.column_min_width[col],
                                                                width=State.column_min_width[col],
                                                                max_width="none",
                                                                box_sizing="border-box",
                                                                background_color=State.display_column_header_colors[col],
                                                                color="white",
                                                                border_right=f"1px solid {rx.color('gray', 4)}",
                                                                border_bottom=f"1px solid {rx.color('gray', 4)}",
                                                                padding="0.75em",
                                                                style={
                                                                    "position": "sticky",
                                                                    "top": "0px",
                                                                    "zIndex": "8",
                                                                },
                                                            ),
                                                        ),
                                                    )
                                                ),
                                                rx.table.body(
                                                    rx.foreach(
                                                        State.df_preview.to(list[dict[str, Any]]),
                                                        lambda row, row_idx: rx.table.row(
                                                            rx.foreach(
                                                                State.column_names.to(list[str]),
                                                                lambda col, col_idx: rx.table.cell(
                                                                    rx.box(
                                                                        rx.cond(
                                                                            State.label_cells_mode,
                                                                            rx.box(
                                                                                rx.vstack(
                                                                                    rx.cond(
                                                                                        State.has_cleaned,
                                                                                        rx.text(
                                                                                            rx.cond(
                                                                                                (row[col] == None)
                                                                                                | (row[col].to(str) == "nan")
                                                                                                | (row[col].to(str) == "None")
                                                                                                | (row[col].to(str) == "NaN"),
                                                                                                "(empty)",
                                                                                                row[col].to(str),
                                                                                            ),
                                                                                            size="1",
                                                                                            white_space="pre-wrap",
                                                                                        ),
                                                                                        rx.cond(
                                                                                            (row[col] == None)
                                                                                            | (row[col].to(str) == "nan")
                                                                                            | (row[col].to(str) == "None")
                                                                                            | (row[col].to(str) == "NaN"),
                                                                                            rx.text("(empty)", size="1", color_scheme="gray"),
                                                                                            rx.text(row[col].to(str), size="1", white_space="pre-wrap"),
                                                                                        ),
                                                                                    ),
                                                                                    rx.cond(
                                                                                        row["__label_flags"].to(list[bool])[col_idx],
                                                                                        rx.badge(
                                                                                            "Labeled",
                                                                                            size="1",
                                                                                            color_scheme=rx.cond(
                                                                                                row["__label_kinds"].to(list[str])[col_idx] == "dirty",
                                                                                                "red",
                                                                                                "green",
                                                                                            ),
                                                                                            variant="surface",
                                                                                        ),
                                                                                        rx.fragment(),
                                                                                    ),
                                                                                    spacing="1",
                                                                                    width="100%",
                                                                                    align_items="start",
                                                                                ),
                                                                                on_click=lambda _evt, row_idx=row_idx, col_idx=col_idx: State.open_cell_label_dialog_by_position(
                                                                                    f"{row_idx}:{col_idx}",
                                                                                ),
                                                                                cursor="pointer",
                                                                                width="100%",
                                                                                padding_x="0.35em",
                                                                                padding_y="0.15em",
                                                                                border_radius="6px",
                                                                                display="block",
                                                                                border=rx.cond(
                                                                                    row["__label_flags"].to(list[bool])[col_idx],
                                                                                    rx.cond(
                                                                                        row["__label_kinds"].to(list[str])[col_idx] == "dirty",
                                                                                        "2px solid rgba(220, 38, 38, 0.95)",
                                                                                        "2px solid rgba(22, 163, 74, 0.95)",
                                                                                    ),
                                                                                    rx.cond(
                                                                                        row["__modified_flags"].to(list[bool])[col_idx],
                                                                                        "1px solid rgba(34, 197, 94, 0.55)",
                                                                                        "1px solid transparent",
                                                                                    ),
                                                                                ),
                                                                                background_color=rx.cond(
                                                                                    row["__label_flags"].to(list[bool])[col_idx],
                                                                                    rx.cond(
                                                                                        row["__label_kinds"].to(list[str])[col_idx] == "dirty",
                                                                                        "rgba(220, 38, 38, 0.18)",
                                                                                        "rgba(22, 163, 74, 0.18)",
                                                                                    ),
                                                                                    rx.cond(
                                                                                        row["__modified_flags"].to(list[bool])[col_idx],
                                                                                        "rgba(34, 197, 94, 0.28)",
                                                                                        "transparent",
                                                                                    ),
                                                                                ),
                                                                                title="Click to label this cell",
                                                                            ),
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
                                                                                        on_change=lambda v: State.update_cell(
                                                                                            row["__row_index"].to(int), col, v
                                                                                        ),
                                                                                        width="100%",
                                                                                        min_width="0",
                                                                                        size="1",
                                                                                        variant="soft",
                                                                                    ),
                                                                                    rx.cond(
                                                                                        (row[col] == None)
                                                                                        | (row[col].to(str) == "nan")
                                                                                        | (row[col].to(str) == "None")
                                                                                        | (row[col].to(str) == "NaN"),
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
                                                                                title=rx.cond(
                                                                                    row["__modified_flags"].to(list[bool])[col_idx],
                                                                                    row["__original_values"].to(list[str])[col_idx],
                                                                                    "",
                                                                                ),
                                                                            ),
                                                                        ),
                                                                    ),
                                                                    white_space="pre-wrap",
                                                                    overflow_wrap="anywhere",
                                                                    word_break="break-word",
                                                                    vertical_align="top",
                                                                    min_width=State.column_min_width[col],
                                                                    width=State.column_min_width[col],
                                                                    max_width="none",
                                                                    box_sizing="border-box",
                                                                    border_right=f"1px solid {rx.color('gray', 4)}",
                                                                    border_bottom=f"1px solid {rx.color('gray', 4)}",
                                                                    overflow="visible",
                                                                    padding="0.5em",
                                                                ),
                                                            ),
                                                        ),
                                                    )
                                                ),
                                                width="max-content",
                                                min_width="100%",
                                                variant="surface",
                                                border_left=f"1px solid {rx.color('gray', 4)}",
                                                border_top=f"1px solid {rx.color('gray', 4)}",
                                            ),
                                            spacing="0",
                                            width="max-content",
                                            min_width="100%",
                                        ),
                                        scrollbars="both",
                                        style={"width": "100%", "height": "100%"},
                                    ),
                                    width="100%",
                                    height="100%",
                                    spacing="0",
                                ),
                                width="100%",
                                height="100%",
                                min_width="0",
                            ),
                            width="100%",
                            height="100%",
                            spacing="2",
                        ),
                        width="100%",
                        height="100%",
                    ),
                    rx.cond(
                        State.selected_main_tab == "pipeline",
                        rx.vstack(
                        rx.hstack(
                            rx.text(State.selected_trace_status, size="1", color_scheme="gray"),
                            rx.spacer(),
                            width="100%",
                            align="center",
                        ),
                        rx.scroll_area(
                            rx.vstack(
                                rx.foreach(
                                    State.pipeline_flow_rows.to(list[dict[str, Any]]),
                                    lambda flow: rx.box(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.badge(
                                                    flow["column"],
                                                    variant="surface",
                                                    color_scheme=rx.cond(flow["column_done"] == "1", "green", "gray"),
                                                ),
                                                rx.hstack(
                                                    rx.foreach(
                                                        flow["steps"].to(list[dict[str, str]]),
                                                    lambda step: rx.hstack(
                                                            rx.cond(
                                                                step["needs_blink"] == "1",
                                                                rx.box(
                                                                    rx.icon(tag="bell", size=16, color="#dc2626"),
                                                                    title="Needs your attention",
                                                                    display="inline-flex",
                                                                    align_items="center",
                                                                ),
                                                                rx.fragment(),
                                                            ),
                                                            rx.button(
                                                                step["title"],
                                                                on_click=lambda: State.select_pipeline_flow_step(
                                                                    flow["column"], step["id"]
                                                                ),
                                                                variant=rx.cond(
                                                                    (step["is_active"] == "1") & (flow["column_done"] == "0"),
                                                                    "solid",
                                                                    "soft",
                                                                ),
                                                                color_scheme=rx.cond(
                                                                    step["is_error"] == "1",
                                                                    "red",
                                                                    rx.cond(
                                                                        ((step["is_active"] == "1") & (flow["column_done"] == "0")) | (step["is_success"] == "1"),
                                                                        "green",
                                                                        "gray",
                                                                    ),
                                                                ),
                                                                size="1",
                                                                style={
                                                                    "cursor": "pointer",
                                                                    "fontWeight": rx.cond(
                                                                        (step["is_active"] == "1") | (step["is_expanded"] == "1"),
                                                                        "600",
                                                                        "400",
                                                                    ),
                                                                },
                                                            ),
                                                            rx.cond(
                                                                step["show_arrow_after"] == "1",
                                                                rx.icon(tag="arrow-right", size=14, color=rx.color("gray", 8)),
                                                                rx.spacer(),
                                                            ),
                                                            align="center",
                                                            spacing="1",
                                                        ),
                                                    ),
                                                    wrap="wrap",
                                                    spacing="1",
                                                    width="100%",
                                                ),
                                                width="100%",
                                                align="center",
                                                spacing="2",
                                            ),
                                            rx.hstack(
                                                rx.text("Status:", size="1", color_scheme="gray"),
                                                rx.text(flow["active_status"], size="1", color_scheme="gray"),
                                                width="100%",
                                                align="center",
                                            ),
                                            rx.cond(
                                                flow["show_details"] == "1",
                                                rx.box(
                                                    rx.vstack(
                                                        rx.text(flow["active_title"], font_weight="bold", size="1"),
                                                        rx.cond(
                                                            (flow["hitl_request_id"] != "")
                                                            & (flow["hitl_kind"] == "code_review")
                                                            & (flow["expanded_is_coder_step"] == "1"),
                                                            rx.fragment(),
                                                            rx.foreach(
                                                                flow["active_output_lines"].to(list[str]),
                                                                lambda line: rx.text(line, font_family="monospace", size="1"),
                                                            ),
                                                        ),
                                                        rx.cond(
                                                            State.pending_user_validation
                                                            & (State.pending_validation_column == flow["column"]),
                                                            rx.box(
                                                                rx.vstack(
                                                                    rx.text("User Validation Required", font_weight="bold"),
                                                                    rx.text(
                                                                        "Review sample values and decide whether correction is needed.",
                                                                        size="1",
                                                                        color_scheme="gray",
                                                                    ),
                                                                    rx.scroll_area(
                                                                        rx.table.root(
                                                                            rx.table.header(
                                                                                rx.table.row(
                                                                                    rx.table.column_header_cell(
                                                                                        "Original",
                                                                                        padding="0.35em",
                                                                                        white_space="nowrap",
                                                                                        min_width=State.pending_validation_width_original,
                                                                                        width=State.pending_validation_width_original,
                                                                                        max_width="none",
                                                                                    ),
                                                                                    rx.table.column_header_cell(
                                                                                        "Cleaned",
                                                                                        padding="0.35em",
                                                                                        white_space="nowrap",
                                                                                        min_width=State.pending_validation_width_cleaned,
                                                                                        width=State.pending_validation_width_cleaned,
                                                                                        max_width="none",
                                                                                    ),
                                                                                )
                                                                            ),
                                                                            rx.table.body(
                                                                                rx.foreach(
                                                                                    State.pending_validation_sample_rows.to(list[dict[str, str]]),
                                                                                    lambda r: rx.table.row(
                                                                                        rx.table.cell(
                                                                                            rx.text(
                                                                                                r["original"],
                                                                                                size="1",
                                                                                                style={
                                                                                                    "color": rx.cond(
                                                                                                        r["changed"] == "1",
                                                                                                        "#b91c1c",
                                                                                                        "inherit",
                                                                                                    ),
                                                                                                },
                                                                                            ),
                                                                                            min_width=State.pending_validation_width_original,
                                                                                            width=State.pending_validation_width_original,
                                                                                            max_width="none",
                                                                                            white_space="pre-wrap",
                                                                                            overflow_wrap="anywhere",
                                                                                            word_break="break-word",
                                                                                            vertical_align="top",
                                                                                            overflow="visible",
                                                                                        ),
                                                                                        rx.table.cell(
                                                                                            rx.text(
                                                                                                r["cleaned"],
                                                                                                size="1",
                                                                                                style={
                                                                                                    "color": rx.cond(
                                                                                                        r["changed"] == "1",
                                                                                                        "#15803d",
                                                                                                        "inherit",
                                                                                                    ),
                                                                                                },
                                                                                            ),
                                                                                            min_width=State.pending_validation_width_cleaned,
                                                                                            width=State.pending_validation_width_cleaned,
                                                                                            max_width="none",
                                                                                            white_space="pre-wrap",
                                                                                            overflow_wrap="anywhere",
                                                                                            word_break="break-word",
                                                                                            vertical_align="top",
                                                                                            overflow="visible",
                                                                                        ),
                                                                                    ),
                                                                                )
                                                                            ),
                                                                            variant="surface",
                                                                            width="max-content",
                                                                            min_width="100%",
                                                                        ),
                                                                        style={"width": "100%", "maxWidth": "100%", "maxHeight": "260px"},
                                                                        scrollbars="both",
                                                                    ),
                                                                    rx.text("Showing sampled rows used for validation.", size="1", color_scheme="gray"),
                                                                    rx.text(
                                                                        "Modified cells in column: ",
                                                                        State.pending_validation_modified_count.to(str),
                                                                        size="1",
                                                                        color_scheme="gray",
                                                                    ),
                                                                    rx.hstack(
                                                                        rx.text("Needs correction"),
                                                                        rx.switch(
                                                                            checked=State.user_validation_needs_correction,
                                                                            on_change=State.set_user_validation_needs_correction,
                                                                        ),
                                                                        width="100%",
                                                                    ),
                                                                    rx.hstack(
                                                                        rx.text("Feedback target"),
                                                                        rx.select(
                                                                            ["RECOMMENDER", "CODER"],
                                                                            value=State.user_validation_feedback_target,
                                                                            on_change=State.set_user_validation_feedback_target,
                                                                            width="190px",
                                                                        ),
                                                                        width="100%",
                                                                    ),
                                                                    rx.text_area(
                                                                        value=State.user_validation_feedback_message,
                                                                        on_change=State.set_user_validation_feedback_message,
                                                                        placeholder="Provide detailed feedback for the selected agent...",
                                                                        min_height="110px",
                                                                        width="80%",
                                                                    ),
                                                                    rx.button(
                                                                        "Submit User Validation",
                                                                        on_click=lambda: State.submit_user_validation(flow["column"]),
                                                                        color_scheme="red",
                                                                        width="100%",
                                                                        style={"cursor": "pointer"},
                                                                    ),
                                                                    spacing="2",
                                                                    width="100%",
                                                                    align_items="start",
                                                                ),
                                                                width="100%",
                                                                padding="0.6em",
                                                                border=f"1px solid {rx.color('red', 6)}",
                                                                border_radius="8px",
                                                                background_color=rx.color("red", 2),
                                                            ),
                                                            rx.spacer(),
                                                        ),
                                                        rx.cond(
                                                            (flow["hitl_request_id"] != "")
                                                            & (flow["hitl_kind"] == "code_review")
                                                            & (flow["expanded_is_coder_step"] == "1"),
                                                            rx.box(
                                                                rx.vstack(
                                                                    rx.text("Human review: cleaning code", font_weight="bold", size="2"),
                                                                    rx.text(
                                                                        "Edit the code if needed, then submit. Expand the Coder step above.",
                                                                        size="1",
                                                                        color_scheme="gray",
                                                                    ),
                                                                    rx.text_area(
                                                                        value=flow["hitl_code"],
                                                                        on_change=lambda val: State.set_hitl_code_column(
                                                                            flow["column"], val
                                                                        ),
                                                                        font_family="monospace",
                                                                        min_height="220px",
                                                                        width="100%",
                                                                        size="2",
                                                                    ),
                                                                    rx.button(
                                                                        "Use this code",
                                                                        on_click=State.submit_hitl_code_review(
                                                                            flow["hitl_request_id"], flow["column"]
                                                                        ),
                                                                        color_scheme="indigo",
                                                                        style={"cursor": "pointer"},
                                                                    ),
                                                                    spacing="2",
                                                                    width="100%",
                                                                    align_items="start",
                                                                ),
                                                                width="100%",
                                                                padding="0.6em",
                                                                margin_top="0.5em",
                                                                border=f"1px solid {rx.color('indigo', 6)}",
                                                                border_radius="8px",
                                                                background_color=rx.color("indigo", 2),
                                                            ),
                                                            rx.fragment(),
                                                        ),
                                                        rx.cond(
                                                            (flow["hitl_request_id"] != "")
                                                            & (flow["hitl_kind"] == "validation_review")
                                                            & (flow["expanded_is_validator_step"] == "1"),
                                                            rx.box(
                                                                rx.vstack(
                                                                            rx.text("Human review: validation decision", font_weight="bold", size="2"),
                                                                            rx.foreach(
                                                                                flow["hitl_validator_summary_lines"].to(
                                                                                    list[str]
                                                                                ),
                                                                                lambda line: rx.text(line, size="1"),
                                                                            ),
                                                                            rx.scroll_area(
                                                                                rx.table.root(
                                                                                    rx.table.header(
                                                                                        rx.table.row(
                                                                                            rx.table.column_header_cell(
                                                                                                "Original",
                                                                                                min_width=flow["hitl_width_original"],
                                                                                                width=flow["hitl_width_original"],
                                                                                            ),
                                                                                            rx.table.column_header_cell(
                                                                                                "Cleaned",
                                                                                                min_width=flow["hitl_width_cleaned"],
                                                                                                width=flow["hitl_width_cleaned"],
                                                                                            ),
                                                                                        )
                                                                                    ),
                                                                                    rx.table.body(
                                                                                        rx.foreach(
                                                                                            flow["hitl_sample_rows"].to(
                                                                                                list[dict[str, str]]
                                                                                            ),
                                                                                            lambda r: rx.table.row(
                                                                                                rx.table.cell(
                                                                                                    rx.text(
                                                                                                        r["original"],
                                                                                                        size="1",
                                                                                                        style={
                                                                                                            "color": rx.cond(
                                                                                                                r["changed"] == "1",
                                                                                                                "#b91c1c",
                                                                                                                "inherit",
                                                                                                            ),
                                                                                                        },
                                                                                                    ),
                                                                                                    white_space="pre-wrap",
                                                                                                ),
                                                                                                rx.table.cell(
                                                                                                    rx.text(
                                                                                                        r["cleaned"],
                                                                                                        size="1",
                                                                                                        style={
                                                                                                            "color": rx.cond(
                                                                                                                r["changed"] == "1",
                                                                                                                "#15803d",
                                                                                                                "inherit",
                                                                                                            ),
                                                                                                        },
                                                                                                    ),
                                                                                                    white_space="pre-wrap",
                                                                                                ),
                                                                                            ),
                                                                                        )
                                                                                    ),
                                                                                    variant="surface",
                                                                                    width="max-content",
                                                                                    min_width="100%",
                                                                                ),
                                                                                style={"width": "100%", "maxWidth": "100%", "maxHeight": "220px"},
                                                                                scrollbars="both",
                                                                            ),
                                                                            rx.text(
                                                                                "Modified cells (approx.): ",
                                                                                flow["hitl_modified_count"],
                                                                                size="1",
                                                                                color_scheme="gray",
                                                                            ),
                                                                            rx.cond(
                                                                                flow["hitl_llm_needs_correction"] == "0",
                                                                                rx.vstack(
                                                                                    rx.text(
                                                                                        "Do you agree with the validation agent that cleaning is valid?",
                                                                                        size="1",
                                                                                        font_weight="bold",
                                                                                    ),
                                                                                    rx.hstack(
                                                                                        rx.button(
                                                                                            "Yes — cleaning is valid",
                                                                                            on_click=State.submit_hitl_validation_validator_ok_agree(
                                                                                                flow["hitl_request_id"], flow["column"]
                                                                                            ),
                                                                                            color_scheme="green",
                                                                                            style={"cursor": "pointer"},
                                                                                        ),
                                                                                        rx.button(
                                                                                            "No — correction is needed",
                                                                                            on_click=State.submit_hitl_validation_validator_ok_disagree(
                                                                                                flow["hitl_request_id"], flow["column"]
                                                                                            ),
                                                                                            color_scheme="orange",
                                                                                            variant="soft",
                                                                                            style={"cursor": "pointer"},
                                                                                        ),
                                                                                        spacing="2",
                                                                                        flex_wrap="wrap",
                                                                                    ),
                                                                                    spacing="2",
                                                                                    width="100%",
                                                                                    align_items="start",
                                                                                ),
                                                                                rx.vstack(
                                                                                    rx.text(
                                                                                        "The validation agent suggested corrections. What should happen next?",
                                                                                        size="1",
                                                                                        font_weight="bold",
                                                                                    ),
                                                                                    rx.hstack(
                                                                                        rx.button(
                                                                                            "Yes — accept this feedback",
                                                                                            on_click=State.submit_hitl_validation_feedback_accept(
                                                                                                flow["hitl_request_id"], flow["column"]
                                                                                            ),
                                                                                            color_scheme="green",
                                                                                            style={"cursor": "pointer"},
                                                                                        ),
                                                                                        rx.button(
                                                                                            "No — cleaning is already acceptable",
                                                                                            on_click=State.submit_hitl_validation_feedback_reject_cleaning_valid(
                                                                                                flow["hitl_request_id"], flow["column"]
                                                                                            ),
                                                                                            color_scheme="blue",
                                                                                            variant="soft",
                                                                                            style={"cursor": "pointer"},
                                                                                        ),
                                                                                        rx.button(
                                                                                            "No — revise correction instructions",
                                                                                            on_click=State.submit_hitl_validation_feedback_reject_revise(
                                                                                                flow["hitl_request_id"], flow["column"]
                                                                                            ),
                                                                                            color_scheme="orange",
                                                                                            variant="soft",
                                                                                            style={"cursor": "pointer"},
                                                                                        ),
                                                                                        spacing="2",
                                                                                        flex_wrap="wrap",
                                                                                    ),
                                                                                    spacing="2",
                                                                                    width="100%",
                                                                                    align_items="start",
                                                                                ),
                                                                            ),
                                                                            rx.text("Feedback target (when you disagree or revise)", size="1", color_scheme="gray"),
                                                                            rx.hstack(
                                                                                rx.select(
                                                                                    ["RECOMMENDER", "CODER"],
                                                                                    value=flow["hitl_feedback_target"],
                                                                                    on_change=lambda val: State.set_hitl_validation_feedback_target_column(
                                                                                        flow["column"], val
                                                                                    ),
                                                                                    width="160px",
                                                                                ),
                                                                                width="100%",
                                                                            ),
                                                                            rx.text_area(
                                                                                value=flow["hitl_feedback_message"],
                                                                                on_change=lambda val: State.set_hitl_validation_feedback_message_column(
                                                                                    flow["column"], val
                                                                                ),
                                                                                placeholder="Your instructions when you disagree with the agent or want different correction guidance.",
                                                                                min_height="100px",
                                                                                width="100%",
                                                                            ),
                                                                            spacing="2",
                                                                            width="100%",
                                                                            align_items="start",
                                                                        ),
                                                                width="100%",
                                                                padding="0.6em",
                                                                margin_top="0.5em",
                                                                border=f"1px solid {rx.color('indigo', 6)}",
                                                                border_radius="8px",
                                                                background_color=rx.color("indigo", 2),
                                                            ),
                                                            rx.fragment(),
                                                        ),
                                                        rx.cond(
                                                            (flow["hitl_request_id"] != "")
                                                            & (flow["hitl_kind"] == "already_clean_review")
                                                            & (flow["expanded_is_already_clean_step"] == "1"),
                                                            rx.box(
                                                                rx.vstack(
                                                                    rx.text("Already clean — confirm or reject", font_weight="bold", size="2"),
                                                                    rx.foreach(
                                                                        flow["hitl_validator_summary_lines"].to(
                                                                            list[str]
                                                                        ),
                                                                        lambda line: rx.text(line, size="1"),
                                                                    ),
                                                                    rx.scroll_area(
                                                                        rx.table.root(
                                                                            rx.table.header(
                                                                                rx.table.row(
                                                                                    rx.table.column_header_cell(
                                                                                        "Sample values",
                                                                                        min_width=flow["hitl_width_original"],
                                                                                        width=flow["hitl_width_original"],
                                                                                    ),
                                                                                    rx.table.column_header_cell(
                                                                                        "Same (no change yet)",
                                                                                        min_width=flow["hitl_width_cleaned"],
                                                                                        width=flow["hitl_width_cleaned"],
                                                                                    ),
                                                                                )
                                                                            ),
                                                                            rx.table.body(
                                                                                rx.foreach(
                                                                                    flow["hitl_sample_rows"].to(
                                                                                        list[dict[str, str]]
                                                                                    ),
                                                                                    lambda r: rx.table.row(
                                                                                        rx.table.cell(
                                                                                            rx.text(
                                                                                                r["original"],
                                                                                                size="1",
                                                                                                style={"color": "#57534e"},
                                                                                            ),
                                                                                            white_space="pre-wrap",
                                                                                        ),
                                                                                        rx.table.cell(
                                                                                            rx.text(
                                                                                                r["cleaned"],
                                                                                                size="1",
                                                                                                style={"color": "#57534e"},
                                                                                            ),
                                                                                            white_space="pre-wrap",
                                                                                        ),
                                                                                    ),
                                                                                )
                                                                            ),
                                                                            variant="surface",
                                                                            width="max-content",
                                                                            min_width="100%",
                                                                        ),
                                                                        style={"width": "100%", "maxWidth": "100%", "maxHeight": "200px"},
                                                                        scrollbars="both",
                                                                    ),
                                                                    rx.text(
                                                                        "If you reject, describe what is wrong (required).",
                                                                        size="1",
                                                                        color_scheme="gray",
                                                                    ),
                                                                    rx.text_area(
                                                                        value=flow["hitl_feedback_message"],
                                                                        on_change=lambda val: State.set_hitl_validation_feedback_message_column(
                                                                            flow["column"], val
                                                                        ),
                                                                        placeholder="e.g. null-like sentinels, inconsistent formats, …",
                                                                        min_height="90px",
                                                                        width="100%",
                                                                    ),
                                                                    rx.hstack(
                                                                        rx.button(
                                                                            "Accept (column is clean)",
                                                                            on_click=State.submit_hitl_already_clean_confirm(
                                                                                flow["hitl_request_id"], flow["column"]
                                                                            ),
                                                                            color_scheme="green",
                                                                            style={"cursor": "pointer"},
                                                                        ),
                                                                        rx.button(
                                                                            "Reject (needs cleaning)",
                                                                            on_click=State.submit_hitl_already_clean_reject(
                                                                                flow["hitl_request_id"], flow["column"]
                                                                            ),
                                                                            color_scheme="orange",
                                                                            variant="soft",
                                                                            style={"cursor": "pointer"},
                                                                        ),
                                                                        spacing="2",
                                                                        flex_wrap="wrap",
                                                                    ),
                                                                    spacing="2",
                                                                    width="100%",
                                                                    align_items="start",
                                                                ),
                                                                width="100%",
                                                                padding="0.6em",
                                                                margin_top="0.5em",
                                                                border=f"1px solid {rx.color('indigo', 6)}",
                                                                border_radius="8px",
                                                                background_color=rx.color("indigo", 2),
                                                            ),
                                                            rx.fragment(),
                                                        ),
                                                        align_items="start",
                                                        spacing="1",
                                                    ),
                                                    width="100%",
                                                    padding="0.5em",
                                                    border=f"1px solid {rx.color('gray', 4)}",
                                                    border_radius="8px",
                                                    background_color=rx.color("gray", 2),
                                                ),
                                                rx.spacer(),
                                            ),
                                            spacing="2",
                                            width="100%",
                                        ),
                                        padding="0.6em",
                                        border=f"1px solid {rx.color('gray', 4)}",
                                        border_radius="10px",
                                        background_color=rx.color("gray", 1),
                                        width="100%",
                                    ),
                                ),
                                spacing="2",
                                width="100%",
                            ),
                            width="100%",
                            flex="1",
                            min_height="0",
                            style={"width": "100%", "height": "100%", "paddingBottom": "0.75rem"},
                        ),
                        width="100%",
                        height="100%",
                        min_height="0",
                        spacing="2",
                    ),
                        rx.scroll_area(
                            rx.cond(
                                State.selected_main_tab == "dashboard",
                                rx.vstack(
                                    rx.hstack(
                                        rx.heading("Data Profiling Dashboard", size="4"),
                                        rx.spacer(),
                                        rx.text(State.profiling_status, size="1", color_scheme="gray"),
                                        width="100%",
                                    ),
                                    rx.foreach(
                                        State.profiling_dashboard_rows.to(list[dict[str, Any]]),
                                        lambda r: rx.box(
                                            rx.vstack(
                                                rx.hstack(
                                                    rx.badge(r["column"], background_color=r["color"], color="white"),
                                                    rx.text(r["semantic_type"], size="1", color_scheme="gray"),
                                                    rx.spacer(),
                                                    rx.vstack(
                                                        rx.text("Labeled", size="1", color_scheme="gray", font_weight="bold"),
                                                        rx.text(
                                                            "Clean ",
                                                            r["labeled_clean_count"],
                                                            " | Dirty ",
                                                            r["labeled_dirty_count"],
                                                            size="1",
                                                        ),
                                                        rx.cond(
                                                            (r["labeled_clean_count"] != "0") | (r["labeled_dirty_count"] != "0"),
                                                            rx.vstack(
                                                                rx.text(
                                                                    "Labeled examples",
                                                                    size="1",
                                                                    color_scheme="gray",
                                                                    font_weight="bold",
                                                                ),
                                                                rx.cond(
                                                                    r["labeled_clean_count"] != "0",
                                                                    rx.vstack(
                                                                        rx.foreach(
                                                                            r["labeled_clean_rows"].to(list[dict[str, str]]),
                                                                            lambda ex: rx.hstack(
                                                                                rx.badge("CLEAN", size="1", color_scheme="green", variant="surface"),
                                                                                rx.code(ex["value"], size="1"),
                                                                                spacing="2",
                                                                                align="center",
                                                                                wrap="wrap",
                                                                            ),
                                                                        ),
                                                                        spacing="1",
                                                                        width="100%",
                                                                        align_items="start",
                                                                    ),
                                                                    rx.fragment(),
                                                                ),
                                                                rx.cond(
                                                                    r["labeled_dirty_count"] != "0",
                                                                    rx.vstack(
                                                                        rx.foreach(
                                                                            r["labeled_dirty_rows"].to(list[dict[str, str]]),
                                                                            lambda ex: rx.hstack(
                                                                                rx.badge("DIRTY", size="1", color_scheme="red", variant="surface"),
                                                                                rx.code(ex["value"], size="1"),
                                                                                rx.text("->", size="1", color_scheme="gray"),
                                                                                rx.code(ex["expected"], size="1"),
                                                                                spacing="2",
                                                                                align="center",
                                                                                wrap="wrap",
                                                                            ),
                                                                        ),
                                                                        spacing="1",
                                                                        width="100%",
                                                                        align_items="start",
                                                                    ),
                                                                    rx.fragment(),
                                                                ),
                                                                spacing="1",
                                                                width="100%",
                                                                align_items="end",
                                                            ),
                                                            rx.fragment(),
                                                        ),
                                                        rx.cond(
                                                            r["user_marked_clean"] == "1",
                                                            rx.badge("User marked clean", color_scheme="green", variant="surface", size="1"),
                                                            rx.fragment(),
                                                        ),
                                                        spacing="0",
                                                        align_items="end",
                                                    ),
                                                    width="100%",
                                                    align="center",
                                                ),
                                                rx.vstack(
                                                    rx.hstack(
                                                        rx.text("Missing:", size="1", font_weight="bold"),
                                                        rx.text(r["missing_count"], " (", r["missing_pct"], "%)", size="1"),
                                                    ),
                                                    rx.hstack(
                                                        rx.text("Unique:", size="1", font_weight="bold"),
                                                        rx.text(r["unique_count"], " (", r["unique_pct"], "%)", size="1"),
                                                    ),
                                                    spacing="1",
                                                    align_items="start",
                                                ),
                                                rx.cond(
                                                    r["is_numeric"] == "1",
                                                    rx.box(
                                                        rx.hstack(
                                                            rx.vstack(
                                                                rx.text("Min", size="1", color_scheme="gray", font_weight="bold"),
                                                                rx.text(r["numeric_min"], size="2"),
                                                                rx.text("Max", size="1", color_scheme="gray", font_weight="bold", margin_top="0.35em"),
                                                                rx.text(r["numeric_max"], size="2"),
                                                                rx.text("Mean", size="1", color_scheme="gray", font_weight="bold", margin_top="0.35em"),
                                                                rx.text(r["numeric_mean"], size="2"),
                                                                rx.text("Median", size="1", color_scheme="gray", font_weight="bold", margin_top="0.35em"),
                                                                rx.text(r["numeric_median"], size="2"),
                                                                rx.text("Std", size="1", color_scheme="gray", font_weight="bold", margin_top="0.35em"),
                                                                rx.text(r["numeric_std"], size="2"),
                                                                rx.text("Range", size="1", color_scheme="gray", font_weight="bold", margin_top="0.35em"),
                                                                rx.text(r["numeric_range"], size="2"),
                                                                spacing="0",
                                                                width="100%",
                                                                min_width="140px",
                                                                max_width="200px",
                                                                align_items="start",
                                                            ),
                                                            rx.box(
                                                                rx.vstack(
                                                                    rx.text(
                                                                        "Distribution",
                                                                        size="1",
                                                                        font_weight="bold",
                                                                        color_scheme="gray",
                                                                    ),
                                                                    rx.box(
                                                                        rx.hstack(
                                                                            rx.foreach(
                                                                                r["histogram"].to(list[dict[str, Any]]),
                                                                                lambda b: rx.vstack(
                                                                                    rx.spacer(),
                                                                                    rx.box(
                                                                                        width="100%",
                                                                                        height=(b["height_pct"].to(str) + "%"),
                                                                                        background_color=rx.color("indigo", 8),
                                                                                        border_radius="3px 3px 0 0",
                                                                                        title=(b["bin"].to(str) + ": " + b["count"].to(str)),
                                                                                    ),
                                                                                    spacing="0",
                                                                                    height="100%",
                                                                                    justify="end",
                                                                                    align_items="stretch",
                                                                                    flex="1",
                                                                                    min_width="0",
                                                                                ),
                                                                            ),
                                                                            spacing="1",
                                                                            width="100%",
                                                                            height="100%",
                                                                            align="end",
                                                                        ),
                                                                        width="100%",
                                                                        height="120px",
                                                                        border=f"1px solid {rx.color('gray', 4)}",
                                                                        border_radius="6px",
                                                                        padding="4px",
                                                                        box_sizing="border-box",
                                                                        overflow="hidden",
                                                                    ),
                                                                    rx.hstack(
                                                                        rx.text(r["hist_x_min"], size="1", color_scheme="gray"),
                                                                        rx.text(r["hist_x_max"], size="1", color_scheme="gray"),
                                                                        width="100%",
                                                                        justify="between",
                                                                        box_sizing="border-box",
                                                                    ),
                                                                    spacing="1",
                                                                    width="100%",
                                                                    align_items="stretch",
                                                                ),
                                                                flex="0",
                                                                min_width="220px",
                                                                width="100%",
                                                                max_width="320px",
                                                                padding="0.65em",
                                                                border=f"1px solid {rx.color('gray', 5)}",
                                                                border_radius="8px",
                                                                background_color=rx.color("gray", 2),
                                                                box_sizing="border-box",
                                                            ),
                                                            width="100%",
                                                            align="start",
                                                            spacing="5",
                                                            flex_wrap="wrap",
                                                        ),
                                                        width="100%",
                                                    ),
                                                    rx.vstack(
                                                        rx.text("Top 5 values", size="1", color_scheme="gray", font_weight="bold"),
                                                        rx.vstack(
                                                            rx.foreach(
                                                                r["top_values"].to(list[dict[str, str]]),
                                                                lambda t: rx.text(
                                                                    "- ", t["value"], " (", t["pct"], "%)",
                                                                    size="1",
                                                                ),
                                                            ),
                                                            width="100%",
                                                            align_items="start",
                                                            spacing="1",
                                                        ),
                                                        spacing="1",
                                                        width="100%",
                                                    ),
                                                ),
                                                rx.cond(
                                                    r["has_top_regex"] == "1",
                                                    rx.vstack(
                                                        rx.text("Top regex patterns", size="1", color_scheme="gray", font_weight="bold"),
                                                        rx.hstack(
                                                            rx.foreach(
                                                                r["top_regex"].to(list[dict[str, str]]),
                                                                lambda rgx: rx.badge(
                                                                    rx.text(rgx["pattern"], " ", rgx["pct"], "%"),
                                                                    variant="surface",
                                                                    color_scheme="gray",
                                                                ),
                                                            ),
                                                            wrap="wrap",
                                                            width="100%",
                                                        ),
                                                        spacing="1",
                                                        width="100%",
                                                    ),
                                                    rx.spacer(),
                                                ),
                                                spacing="2",
                                                width="100%",
                                                align_items="start",
                                            ),
                                            width="100%",
                                            padding="0.8em",
                                            border=f"1px solid {rx.color('gray', 4)}",
                                            border_radius="10px",
                                            background_color=rx.color("gray", 1),
                                        ),
                                    ),
                                    rx.divider(margin_y="0.5em"),
                                    rx.heading("Functional dependencies", size="3"),
                                    rx.cond(
                                        State.fd_results.length() > 0,
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
                                        rx.text("No FDs detected.", size="1", color_scheme="gray"),
                                    ),
                                    spacing="2",
                                    width="100%",
                                    align_items="start",
                                ),
                                rx.cond(
                                    State.selected_main_tab == "logs",
                                    rx.vstack(
                                        rx.text(State.profiling_status, font_weight="bold"),
                                        rx.cond(
                                            State.verbose,
                                            rx.cond(
                                                State.logs.length() > 0,
                                                rx.foreach(State.logs, lambda log: rx.text(log, font_family="monospace", size="1")),
                                                rx.text("Logs will be shown here.", size="1", color_scheme="gray"),
                                            ),
                                            rx.text("Verbose for logs is off.", size="1", color_scheme="gray"),
                                        ),
                                        spacing="1",
                                        width="100%",
                                        align_items="start",
                                    ),
                                    rx.cond(
                                        State.selected_main_tab == "evaluation",
                                        evaluation_panel(),
                                        rx.cond(
                                            State.selected_main_tab == "report",
                                            rx.cond(
                                                State.has_cleaned,
                                                rx.vstack(
                                                    usage_panel(),
                                                    rx.divider(),
                                                    rx.text("Cleaning code report", font_weight="bold"),
                                                    rx.select(
                                                        State.report_code_keys.to(list[str]),
                                                        value=State.selected_report_code_key,
                                                        on_change=State.set_selected_report_code_key,
                                                        width="100%",
                                                    ),
                                                    rx.text(State.selected_report_status, color_scheme="gray", size="1"),
                                                    rx.box(
                                                        rx.text(
                                                            State.generated_code_for_selected_pretty,
                                                            width="100%",
                                                            white_space="pre-wrap",
                                                            font_family="monospace",
                                                            font_size="12px",
                                                        ),
                                                        width="100%",
                                                        padding="0.6em",
                                                        border=f"1px solid {rx.color('gray', 4)}",
                                                        border_radius="8px",
                                                        background_color="transparent",
                                                    ),
                                                    spacing="2",
                                                    width="100%",
                                                ),
                                                rx.text(
                                                    "Cleaning report will be shown here after cleaning.",
                                                    size="1",
                                                    color_scheme="gray",
                                                ),
                                            ),
                                            rx.vstack(
                                                rx.text("Instructions & model setup", font_weight="bold"),
                                                rx.text("1) Add API keys to .env", font_weight="bold"),
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
                                                rx.text("2) LLM selection", font_weight="bold"),
                                                rx.text("The sidebar dropdown sets one default model for the Recommender, Coding and Validation Agents.", size="1", color_scheme="gray"),
                                                rx.text("Advanced Configuration overrides that per agent (e.g. select different model for Coding Agent).", size="1", color_scheme="gray"),
                                                rx.text("Set Validation to USER for fully manual validation.", size="1", color_scheme="gray"),
                                                rx.text("3) Add or customize models", font_weight="bold"),
                                                rx.text("Clients: madclean/llm/llm_clients.py", size="1", color_scheme="gray"),
                                                rx.text("Registry: madclean/llm/llm_registry.py", size="1", color_scheme="gray"),
                                                rx.text("Match api_key_name and add the key in .env", size="1", color_scheme="gray"),
                                                rx.text("4) LLM settings (temperature / top_p)", font_weight="bold"),
                                                rx.text(
                                                    "Under Advanced Configuration → LLM settings. Leave blank for provider defaults; unsupported models ignore overrides.",
                                                    size="1",
                                                    color_scheme="gray",
                                                ),
                                                rx.text("5) Column-specific user-provided instructions", font_weight="bold"),
                                                rx.text("On the Table tab, click a column header (gear icon) to provide custom cleaning instructions.", size="1", color_scheme="gray"),
                                                rx.text("Your text is merged into the Recommender Agent prompt.", size="1", color_scheme="gray"),
                                                rx.text("Click the check mark icon to flag the column as already clean. The column will not be cleaned by the system.", size="1", color_scheme="gray"),
                                                rx.text("Optionally, cells can be labeled as examples for the Recommender Agent.", size="1", color_scheme="gray"),
                                                rx.text(
                                                    "On the Table tab, enable “Label cells” next to Show FDs, then click a cell: mark CLEAN or DIRTY and (if dirty) the expected value.",
                                                    size="1",
                                                    color_scheme="gray",
                                                ),
                                                rx.text("6) Human-in-the-loop (HITL)", font_weight="bold"),
                                                rx.text("Enable in the sidebar and pick columns.", size="1", color_scheme="gray"),
                                                rx.text("Open the Pipeline tab, a red notification button appears when the system requires your input.", size="1", color_scheme="gray"),
                                                rx.text("Use the step buttons; code review and validation decisions appear inline.", size="1", color_scheme="gray"),
                                                rx.text("Respond to the validation agent verdict (agree / disagree / accept feedback / override).", size="1", color_scheme="gray"),
                                                rx.text("7) Full workflow", font_weight="bold"),
                                                rx.text("- Upload CSV and review profiling + FD detection", size="1"),
                                                rx.text("- Adjust Advanced settings (sampling, retries, validation fallback)", size="1"),
                                                rx.text("- Run pipeline and follow the Pipeline tab", size="1"),
                                                rx.text("- If validation is USER, review samples and send feedback to Recommender/Coder", size="1"),
                                                rx.text("- Review Report and Evaluation; download cleaned CSV or cleaning code", size="1"),
                                                rx.text("- Optional: upload a matching ground-truth file on the Evaluation tab to measure TP/FP/TN/FN, precision, recall and F1 after cleaning.", size="1"),
                                                spacing="2",
                                                width="100%",
                                                align_items="start",
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            style={"width": "100%", "height": "100%"},
                        ),
                    ),
                ),
                width="100%",
                height="100%",
            ),
            flex="1",
            width="100%",
            border=f"1px solid {rx.color('gray', 4)}",
            border_radius="8px",
            overflow="hidden",
            background_color=rx.color("gray", 1),
        ),
        padding="1em",
        width="100%",
        min_height="100vh",
        spacing="3",
        font_size="14px",
        flex="2 1 360px",
        min_width="0",
    )

@rx.page(title="MADClean")
def index() -> rx.Component:
    return rx.fragment(
        rx.script(
            "(() => { const existing = document.querySelector(\"link[rel~='icon']\") || document.createElement('link'); "
            "existing.setAttribute('rel', 'icon'); existing.setAttribute('href', '/icon.png'); document.head.appendChild(existing); })();"
        ),
        rx.box(
            rx.hstack(
                sidebar(),
                main_content(),
                width="100%",
                spacing="0",
                align="start",
                flex_wrap="wrap",
            ),
            width="100%",
            min_height="100vh",
        ),
    )

app = rx.App(
    theme=rx.theme(accent_color="indigo", radius="small"),
    stylesheets=["/custom.css"],
)
app.add_page(index)