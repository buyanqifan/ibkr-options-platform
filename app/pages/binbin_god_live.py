"""Binbin God live paper-trading control page."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from app.components.connection_status import connection_badge
from app.components.tables import create_data_table, metric_card
from app.pages.binbin_god import ALL_FORM_FIELDS, QC_UI_DEFAULTS, _build_rows, _section_card
from app.services import get_services
from core.live_trading.binbin_god.defaults import build_live_defaults
from core.live_trading.binbin_god.models import ControlCommandType
from core.live_trading.binbin_god.repository import BinbinGodLiveRepository
from core.live_trading.binbin_god.service import BinbinGodLiveService

dash.register_page(__name__, path="/binbin-god-live", name="Binbin God Live", icon="bi bi-broadcast")

LIVE_RUNTIME_FIELDS = [
    {"id": "bbg-live-poll-interval-seconds", "label": "Poll Interval Seconds", "type": "number", "default": "poll_interval_seconds", "step": 5, "min": 5},
    {"id": "bbg-live-allow-new-entries", "label": "Allow New Entries", "type": "switch", "default": "allow_new_entries"},
    {"id": "bbg-live-max-parallel-open-orders", "label": "Max Parallel Open Orders", "type": "number", "default": "max_parallel_open_orders", "step": 1, "min": 1},
    {"id": "bbg-live-enable-emergency-controls", "label": "Enable Emergency Controls", "type": "switch", "default": "enable_emergency_controls"},
]
LIVE_FORM_FIELDS = ALL_FORM_FIELDS + LIVE_RUNTIME_FIELDS
LIVE_FORM_DEFAULTS = {**QC_UI_DEFAULTS, **build_live_defaults()}


def _get_live_service() -> BinbinGodLiveService:
    services = get_services() or {}
    if "binbin_god_live_service" in services:
        return services["binbin_god_live_service"]
    return BinbinGodLiveService(BinbinGodLiveRepository())


def _risk_summary_cards(detail: dict):
    phase_counts = detail.get("phase_counts", {})
    return dbc.Row(
        [
            dbc.Col(metric_card("Cash", str(phase_counts.get("cash", 0)), "secondary"), md=2),
            dbc.Col(metric_card("Short Put", str(phase_counts.get("short_put", 0)), "warning"), md=2),
            dbc.Col(metric_card("Assigned Stock", str(phase_counts.get("assigned_stock", 0)), "info"), md=2),
            dbc.Col(metric_card("Covered Call", str(phase_counts.get("covered_call", 0)), "primary"), md=2),
            dbc.Col(metric_card("Repair Call", str(phase_counts.get("repair_call", 0)), "danger"), md=2),
            dbc.Col(metric_card("Anomalies", str(detail.get("anomaly_count", 0)), "warning"), md=2),
        ],
        className="g-3",
    )


def _build_live_form_payload(*values):
    payload = {}
    for field, value in zip(LIVE_FORM_FIELDS, values):
        key = field["default"]
        if key == "stock_pool_text":
            symbols = [item.strip().upper() for item in str(value or "").split(",") if item.strip()]
            payload["stock_pool_text"] = ",".join(symbols)
            payload["stock_pool"] = symbols or LIVE_FORM_DEFAULTS["stock_pool"]
        else:
            payload[key] = value
    payload.setdefault("stock_pool", LIVE_FORM_DEFAULTS["stock_pool"])
    payload.setdefault("stock_pool_text", ",".join(payload["stock_pool"]))
    return payload


def _form_values_from_live_params(params: dict):
    merged = {**LIVE_FORM_DEFAULTS, **(params or {})}
    stock_pool = merged.get("stock_pool") or LIVE_FORM_DEFAULTS["stock_pool"]
    merged["stock_pool_text"] = ",".join(stock_pool)
    return tuple(merged.get(field["default"], LIVE_FORM_DEFAULTS[field["default"]]) for field in LIVE_FORM_FIELDS)


layout = html.Div(
    [
        dcc.Store(id="bbg-live-config-store"),
        dcc.Interval(id="bbg-live-interval", interval=5000, n_intervals=0),
        html.H3("Binbin God Live", className="mb-3"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Worker Status"),
                            dbc.CardBody(
                                [
                                    html.Div(id="bbg-live-status"),
                                    html.Div(id="bbg-live-worker-meta", className="mt-2 text-muted small"),
                                    html.Div(id="bbg-live-last-error", className="mt-2"),
                                ]
                            ),
                        ]
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Recovery"),
                            dbc.CardBody([html.Div(id="bbg-live-recovery-summary")]),
                        ]
                    ),
                    md=8,
                ),
            ],
            className="mb-4 g-3",
        ),
        _section_card(
            "Live Controls",
            [
                dbc.ButtonGroup(
                    [
                        dbc.Button("Start", id="bbg-live-start-btn", color="success"),
                        dbc.Button("Pause", id="bbg-live-pause-btn", color="warning"),
                        dbc.Button("Emergency Stop", id="bbg-live-emergency-stop-btn", color="danger"),
                        dbc.Button("Resume", id="bbg-live-resume-btn", color="primary"),
                        dbc.Button("Block Entries", id="bbg-live-block-entries-btn", color="secondary"),
                        dbc.Button("Allow Entries", id="bbg-live-allow-entries-btn", color="info"),
                        dbc.Button("Cancel Entry Orders", id="bbg-live-cancel-entry-orders-btn", color="dark"),
                    ],
                    className="flex-wrap gap-2",
                ),
                html.Div(id="bbg-live-command-feedback", className="mt-3"),
            ],
            color="danger",
        ),
        _section_card("Live Risk Summary", html.Div(id="bbg-live-risk-summary"), color="warning"),
        _section_card(
            "Live Parameters",
            [
                dbc.ButtonGroup(
                    [
                        dbc.Button("Save", id="bbg-live-save-config-btn", color="success"),
                        dbc.Button("Load", id="bbg-live-load-config-btn", color="secondary"),
                        dbc.Button("Reset", id="bbg-live-reset-config-btn", color="warning"),
                        dbc.Button("Apply", id="bbg-live-apply-config-btn", color="primary"),
                    ],
                    className="mb-3 gap-2 flex-wrap",
                ),
                html.Div(id="bbg-live-config-feedback", className="mb-3"),
                *_build_rows(LIVE_FORM_FIELDS, LIVE_FORM_DEFAULTS),
            ],
            color="secondary",
        ),
        _section_card("Open Orders", html.Div(id="bbg-live-orders-table"), color="primary"),
        _section_card("Recent Fills", html.Div(id="bbg-live-fills-table"), color="info"),
        _section_card("Strategy Positions", html.Div(id="bbg-live-positions-table"), color="secondary"),
        _section_card("Execution Events", html.Div(id="bbg-live-events"), color="dark"),
    ]
)


@callback(
    Output("bbg-live-config-store", "data"),
    Output("bbg-live-status", "children"),
    Output("bbg-live-worker-meta", "children"),
    Output("bbg-live-last-error", "children"),
    Output("bbg-live-recovery-summary", "children"),
    Output("bbg-live-risk-summary", "children"),
    Output("bbg-live-orders-table", "children"),
    Output("bbg-live-fills-table", "children"),
    Output("bbg-live-positions-table", "children"),
    Output("bbg-live-events", "children"),
    Input("bbg-live-interval", "n_intervals"),
)
def refresh_live_dashboard(_n):
    service = _get_live_service()
    data = service.get_dashboard_data()
    state = data["state"]
    detail = state.get("detail", {})
    recovery = state.get("recovery_summary", {})
    status_badge = connection_badge(state.get("status", "disconnected"), state.get("last_error") or "")

    orders = create_data_table(
        data["orders"],
        [
            {"headerName": "Symbol", "field": "symbol"},
            {"headerName": "Action", "field": "action"},
            {"headerName": "Right", "field": "right"},
            {"headerName": "Strike", "field": "strike"},
            {"headerName": "Expiry", "field": "expiry"},
            {"headerName": "Qty", "field": "quantity"},
            {"headerName": "Status", "field": "status"},
        ],
        table_id="bbg-live-orders-grid",
        height=250,
    ) if data["orders"] else html.P("No order audit records yet.", className="text-muted")

    fills = create_data_table(
        data["fills"],
        [
            {"headerName": "Symbol", "field": "symbol"},
            {"headerName": "Side", "field": "side"},
            {"headerName": "Shares", "field": "shares"},
            {"headerName": "Price", "field": "price"},
            {"headerName": "Time", "field": "time"},
        ],
        table_id="bbg-live-fills-grid",
        height=220,
    ) if data["fills"] else html.P("No fills synced yet.", className="text-muted")

    positions = create_data_table(
        data["positions"],
        [
            {"headerName": "Symbol", "field": "symbol"},
            {"headerName": "Type", "field": "secType"},
            {"headerName": "Expiry", "field": "expiry"},
            {"headerName": "Strike", "field": "strike"},
            {"headerName": "Right", "field": "right"},
            {"headerName": "Qty", "field": "position"},
        ],
        table_id="bbg-live-positions-grid",
        height=260,
    ) if data["positions"] else html.P("No broker positions synced yet.", className="text-muted")

    events = html.Ul(
        [
            html.Li(f"{item['created_at']} [{item['severity']}] {item['message']}")
            for item in data["events"]
        ]
    ) if data["events"] else html.P("No live events yet.", className="text-muted")

    recovery_view = html.Div(
        [
            html.P(f"Result: {recovery.get('result', 'N/A')}"),
            html.P(f"Recovered At: {recovery.get('recovered_at', 'N/A')}"),
            html.P(f"Open Orders: {recovery.get('open_orders_count', 0)}"),
            html.P(f"Positions: {recovery.get('positions_count', 0)}"),
            html.P(f"Anomalies: {recovery.get('anomaly_count', 0)}"),
        ]
    )

    worker_meta = html.Div(
        [
            html.P(f"Account: {state.get('account_id') or 'N/A'}"),
            html.P(f"Heartbeat: {state.get('heartbeat_at') or 'N/A'}"),
            html.P(f"Allow New Entries: {'Yes' if data.get('allow_new_entries') else 'No'}"),
        ]
    )
    last_error = (
        dbc.Alert(state["last_error"], color="danger", className="mb-0")
        if state.get("last_error")
        else html.P("No active errors", className="text-muted mb-0")
    )

    return (
        data["config"],
        status_badge,
        worker_meta,
        last_error,
        recovery_view,
        _risk_summary_cards(detail),
        orders,
        fills,
        positions,
        events,
    )


@callback(
    Output("bbg-live-command-feedback", "children"),
    Input("bbg-live-start-btn", "n_clicks"),
    Input("bbg-live-pause-btn", "n_clicks"),
    Input("bbg-live-emergency-stop-btn", "n_clicks"),
    Input("bbg-live-resume-btn", "n_clicks"),
    Input("bbg-live-block-entries-btn", "n_clicks"),
    Input("bbg-live-allow-entries-btn", "n_clicks"),
    Input("bbg-live-cancel-entry-orders-btn", "n_clicks"),
    prevent_initial_call=True,
)
def issue_live_command(*_args):
    service = _get_live_service()
    command_map = {
        "bbg-live-start-btn": ControlCommandType.START,
        "bbg-live-pause-btn": ControlCommandType.PAUSE,
        "bbg-live-emergency-stop-btn": ControlCommandType.EMERGENCY_STOP,
        "bbg-live-resume-btn": ControlCommandType.RESUME,
        "bbg-live-block-entries-btn": ControlCommandType.BLOCK_NEW_ENTRIES,
        "bbg-live-allow-entries-btn": ControlCommandType.ALLOW_NEW_ENTRIES,
        "bbg-live-cancel-entry-orders-btn": ControlCommandType.CANCEL_OPEN_ENTRY_ORDERS,
    }
    command_type = command_map.get(ctx.triggered_id)
    if command_type is None:
        return no_update
    service.enqueue_command(command_type)
    return dbc.Alert(f"Queued command: {command_type.value}", color="success", duration=2500)


@callback(
    Output("bbg-live-command-feedback", "children", allow_duplicate=True),
    Input("bbg-live-config-store", "data"),
    prevent_initial_call=True,
)
def acknowledge_loaded_config(_config):
    return no_update


@callback(
    Output("bbg-live-config-feedback", "children"),
    Output("bbg-live-config-store", "data", allow_duplicate=True),
    *[Output(field["id"], "value") for field in LIVE_FORM_FIELDS],
    Input("bbg-live-save-config-btn", "n_clicks"),
    Input("bbg-live-load-config-btn", "n_clicks"),
    Input("bbg-live-reset-config-btn", "n_clicks"),
    Input("bbg-live-apply-config-btn", "n_clicks"),
    *[State(field["id"], "value") for field in LIVE_FORM_FIELDS],
    prevent_initial_call=True,
)
def manage_live_config(_save, _load, _reset, _apply, *values):
    service = _get_live_service()
    form_values = values[: len(LIVE_FORM_FIELDS)]
    triggered = ctx.triggered_id

    if triggered == "bbg-live-reset-config-btn":
        defaults = _form_values_from_live_params(LIVE_FORM_DEFAULTS)
        return (
            dbc.Alert("Reset to live defaults", color="warning", duration=2500),
            LIVE_FORM_DEFAULTS,
            *defaults,
        )

    if triggered == "bbg-live-load-config-btn":
        params = service.get_dashboard_data()["config"]
        restored = _form_values_from_live_params(params)
        return (
            dbc.Alert("Loaded saved live config", color="info", duration=2500),
            params,
            *restored,
        )

    payload = _build_live_form_payload(*form_values)
    if triggered == "bbg-live-save-config-btn":
        saved = service.save_config(payload)
        return (
            dbc.Alert("Saved live config", color="success", duration=2500),
            saved,
            *_form_values_from_live_params(saved),
        )

    if triggered == "bbg-live-apply-config-btn":
        service.enqueue_command(ControlCommandType.APPLY_CONFIG, payload)
        saved = service.save_config(payload)
        return (
            dbc.Alert("Queued config apply for worker reload", color="primary", duration=2500),
            saved,
            *_form_values_from_live_params(saved),
        )

    return no_update, no_update, *([no_update] * len(LIVE_FORM_FIELDS))
