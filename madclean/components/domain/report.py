"""The records a cleaning run produces.

The thesis built these as plain dictionaries, spelled out at every point in the agent loop.
They are the same records here, with the field names and the field order of those dictionaries,
so `to_dict()` still writes exactly what the GUI and the evaluation scripts read.
"""

from dataclasses import asdict, dataclass, field


@dataclass
class TraceStep:
    """One step of an agent loop, as the report keeps it."""
    id: str
    title: str
    status: str
    output: str = ""


@dataclass
class TraceEvent:
    """The same step on its way to the GUI, which needs to know which column it belongs to."""
    column: str
    step_id: str
    title: str
    status: str
    output: str = ""


@dataclass
class TokenUsage:
    """Tokens spent, either by one agent or by the whole run."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens


@dataclass
class AgentTokenUsage:
    """Tokens spent per agent. The field names are the agent names the loop counts under."""
    recommender: TokenUsage = field(default_factory=TokenUsage)
    coding: TokenUsage = field(default_factory=TokenUsage)
    validation: TokenUsage = field(default_factory=TokenUsage)

    def total(self) -> TokenUsage:
        agents = (self.recommender, self.coding, self.validation)
        return TokenUsage(
            input_tokens=sum(agent.input_tokens for agent in agents),
            output_tokens=sum(agent.output_tokens for agent in agents),
            total_tokens=sum(agent.total_tokens for agent in agents),
        )


@dataclass
class _Entry:
    """One line of the report. `reason` is only written when the run has one to give."""

    def to_dict(self) -> dict:
        data = asdict(self)
        if data.get("reason") is None:
            data.pop("reason", None)
        return data


@dataclass
class ColumnReport(_Entry):
    """A column the agent loop worked on."""
    datatype: str
    already_clean: bool
    cleaned: bool
    attempts: int
    generated_code: str | None
    cleaning_validated: bool
    trace_steps: list[TraceStep] = field(default_factory=list)
    reason: str | None = None


@dataclass
class FDReport(_Entry):
    """A dependency task the agent loop worked on."""
    target_columns: list[str]
    cleaned: bool
    attempts: int
    cleaning_validated: bool
    generated_code: str | None = None
    trace_steps: list[TraceStep] = field(default_factory=list)
    reason: str | None = None


@dataclass
class SkippedColumnReport(_Entry):
    """A column the loop never saw: empty, or of a type the profiler could not name."""
    datatype: str
    already_clean: bool
    cleaned: bool
    attempts: int
    cleaning_validated: bool
    reason: str


@dataclass
class CleaningReport:
    """Everything one run of the pipeline produced."""
    entries: dict[str, ColumnReport | FDReport | SkippedColumnReport] = field(default_factory=dict)
    token_usage: AgentTokenUsage = field(default_factory=AgentTokenUsage)
    cancelled: bool = False
    runtime_seconds: float | None = None
    total_usage: TokenUsage | None = None

    def to_dict(self) -> dict:
        """The flat dictionary the thesis returned: one key per entry, then the run's own fields.

        An entry named like one of those fields is overwritten by it, exactly as before.
        """
        data: dict = {key: entry.to_dict() for key, entry in self.entries.items()}
        data["token_usage"] = asdict(self.token_usage)
        if self.cancelled:
            data["cancelled"] = True
        if self.runtime_seconds is not None:
            data["runtime_seconds"] = self.runtime_seconds
        if self.total_usage is not None:
            data["total_usage"] = asdict(self.total_usage)
        return data
