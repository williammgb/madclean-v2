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
            rx.button("Advanced Configuration", variant="outline", width="100%", margin_top="1em"),
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
                    rx.hstack(rx.text("Verbose"), rx.spacer(), rx.switch(checked=State.verbose, on_change=State.set_verbose), width="100%"),
                    rx.hstack(rx.text("Enable Validation"), rx.spacer(), rx.switch(checked=State.enable_validation, on_change=State.set_enable_validation), width="100%"),
                    rx.hstack(rx.text("Enable multi-column cleaning"), rx.spacer(), rx.switch(checked=State.enable_multi_col_cleaning, on_change=State.set_enable_multi_col_cleaning), width="100%"),
                    rx.hstack(rx.text("Enable multi-column validation"), rx.spacer(), rx.switch(checked=State.enable_validation_multi, on_change=State.set_enable_validation_multi), width="100%"),
                    rx.hstack(rx.text("Show metadata"), rx.spacer(), rx.switch(checked=State.include_metadata, on_change=State.set_include_metadata), width="100%"),

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

def sidebar() -> rx.Component:
    return rx.vstack(
        rx.heading("MADClean", size="6"),
        rx.text("1. Data Input", font_weight="bold", size="2"),
        rx.upload(
            rx.vstack(rx.icon(tag="upload", size=20), rx.text("Upload Data", size="1")),
            id="upload_side", border="1px dashed #ccc", padding="1em", width="100%",
            on_drop=State.handle_upload(rx.upload_files(upload_id="upload_side")),
        ),
        rx.text(State.file_name, size="1", color_scheme="gray"),
        rx.divider(margin_y="1.5em"),
        rx.text("2. Model Selection", font_weight="bold", size="2"),
        rx.select(State.llm_options, value=State.selected_llm_key, on_change=State.set_selected_llm_key, width="100%"),
        rx.text("API Keys must be set in .env", size="1", color_scheme="gray"),
        rx.divider(margin_y="1.5em"),
        rx.text("3. Configuration", font_weight="bold", size="2"),
        settings_modal(),
        rx.spacer(),
        rx.button("Run Pipeline", on_click=State.run_cleaning_process, is_loading=State.is_cleaning, width="100%", color_scheme="blue"),
        rx.button("Download CSV", on_click=State.download_cleaned_file, width="100%", variant="soft", color_scheme="green", margin_top="0.5em"),
        height="100vh", width="260px", padding="1.5em", background_color=rx.color("gray", 2), border_right=f"1px solid {rx.color('gray', 5)}",
    )

def main_content() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.badge(State.status_msg, color_scheme="blue", variant="surface"),
            rx.spacer(),
            rx.cond(State.is_cleaning, rx.progress(value=State.progress_percent, width="250px")),
            width="100%",
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
                                    border_right=f"1px solid {rx.color('gray', 4)}",
                                    padding="1em"
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
                                    lambda col: rx.table.cell(
                                        rx.cond(
                                            (row[col] == None) | (row[col].to(str) == "nan") | (row[col].to(str) == "None") | (row[col].to(str) == "NaN"),
                                            rx.text(""), 
                                            rx.text(row[col].to(str))
                                        ), 
                                        white_space="nowrap", 
                                        min_width="180px",
                                        border_right=f"1px solid {rx.color('gray', 4)}",  # Vertical line
                                        border_bottom=f"1px solid {rx.color('gray', 4)}", # Horizontal line
                                        overflow="hidden",
                                        text_overflow="ellipsis"
                                    )
                                )
                            )
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
            background_color="white"
        ),
        rx.tabs.root(
            rx.tabs.list(rx.tabs.trigger("Logs", value="logs"), rx.tabs.trigger("Usage", value="tokens")),
            rx.tabs.content(rx.scroll_area(rx.vstack(rx.foreach(State.logs, lambda log: rx.text(log, font_family="monospace", size="1")), padding="1em"), height="120px"), value="logs"),
            rx.tabs.content(rx.scroll_area(rx.code(State.token_usage.to(str)), padding="1em", height="120px"), value="tokens"),
            width="100%", background_color=rx.color("gray", 1), border_top=f"1px solid {rx.color('gray', 5)}",
        ),
        padding="1.5em", width="100%", height="100vh", spacing="4",
    )

def index() -> rx.Component:
    return rx.hstack(sidebar(), main_content(), width="100%", spacing="0")

app = rx.App(theme=rx.theme(accent_color="indigo"))
app.add_page(index)