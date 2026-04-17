"""Interactive step-by-step setup wizard for PROBOT.

Launched automatically when ``python3 trader.py`` is called with no
arguments in an interactive terminal.  Returns a fully-populated
:class:`~trader_app.config.Settings` object that is passed directly to
:func:`~trader_app.bot.run_bot`.
"""
from __future__ import annotations

import getpass
import os
import select
import sys
import tty
import termios
from typing import Any, Callable

from trader_app.config import Settings

# ── ANSI palette ──────────────────────────────────────────────────────────────
_R   = "\033[0m"
_B   = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GRN = "\033[32m"
_YLW = "\033[33m"
_MAG = "\033[35m"
_CYN = "\033[36m"
_WHT = "\033[37m"
_CLR = "\033[2J\033[H"

# Cursor control
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_UP1         = "\033[1A"
_CLEAR_LINE  = "\033[2K\r"

_W = 62  # inner box width


def _c(text: str, *codes: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _R


def _clear() -> None:
    if sys.stdout.isatty():
        sys.stdout.write(_CLR)
        sys.stdout.flush()


# ── Box-drawing helpers ───────────────────────────────────────────────────────

def _box_top(title: str = "") -> None:
    if title:
        t = f"═ {title} "
        fill = "═" * (_W - len(t) + 1)
        print(_c("╔" + t + fill + "╗", _B, _CYN))
    else:
        print(_c("╔" + "═" * _W + "╗", _B, _CYN))


def _box_row(text: str = "", color: str = _WHT) -> None:
    raw = text[:_W - 1].ljust(_W - 1)
    print(_c("║ ", _CYN) + _c(raw, color) + _c("║", _CYN))


def _box_sep() -> None:
    print(_c("╠" + "═" * _W + "╣", _CYN))


def _box_bot() -> None:
    print(_c("╚" + "═" * _W + "╝", _B, _CYN))


# ── Splash ────────────────────────────────────────────────────────────────────

_ART = [
    "██████╗ ██████╗  ██████╗ ██████╗  ██████╗ ████████╗",
    "██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝",
    "██████╔╝██████╔╝██║   ██║██████╔╝██║   ██║   ██║   ",
    "██╔═══╝ ██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║   ",
    "██║     ██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║   ",
    "╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝   ╚═╝   ",
]


def _splash() -> None:
    _clear()
    print()
    for line in _ART:
        print(_c("  " + line, _B, _MAG))
    print()
    print(_c("  Automated Trading Bot", _B, _CYN))
    print(_c("  Interactive Setup Wizard  ·  Ctrl+C to abort at any time", _DIM, _CYN))
    print()


# ── Progress bar ──────────────────────────────────────────────────────────────

_TOTAL = 7


def _progress(step: int, label: str) -> None:
    bar_w = _W - 12
    filled = round((step / _TOTAL) * bar_w)
    bar = _c("━" * filled, _B, _GRN) + _c("╌" * (bar_w - filled), _DIM, _CYN)
    tag = _c(f" [{step}/{_TOTAL}]", _B, _YLW) + _c(f" {label}", _B, _WHT)
    print("  " + bar + tag)
    print()


# ── Raw keystroke reader ──────────────────────────────────────────────────────

def _read_key() -> str:
    """Read a single keypress from stdin in raw mode.

    Uses os.read() directly (bypasses Python's buffered IO) and a
    select() timeout to distinguish a bare Escape from an arrow sequence.
    Returns one of: "up", "down", "left", "right", "enter", "ctrl_c",
    or a character.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1).decode("utf-8", errors="replace")
        if ch == "\x03":                         # Ctrl+C
            return "ctrl_c"
        if ch == "\r" or ch == "\n":             # Enter
            return "enter"
        if ch == "\x1b":                         # possible escape sequence
            # Use select with a short timeout so a bare Escape doesn't block
            ready, _, _ = select.select([fd], [], [], 0.05)
            if ready:
                nxt = os.read(fd, 1).decode("utf-8", errors="replace")
                if nxt in ("[", "O"):
                    ready2, _, _ = select.select([fd], [], [], 0.05)
                    if ready2:
                        code = os.read(fd, 1).decode("utf-8", errors="replace")
                        if code == "A":
                            return "up"
                        if code == "B":
                            return "down"
                        if code == "C":
                            return "right"
                        if code == "D":
                            return "left"
                        # Drain extended sequences like Page-Up (~)
                        if code.isdigit():
                            select.select([fd], [], [], 0.05)
                            try:
                                os.read(fd, 1)
                            except OSError:
                                pass
            return "escape"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Arrow-key menu ────────────────────────────────────────────────────────────

Option = tuple[str, str]  # (value, display label)


def _menu(options: list[Option], default: str = "") -> str:
    """Render an arrow-key navigable menu and return the chosen value.

    Navigation: ↑ / ↓ arrows or k / j  ·  Enter to confirm.
    The cursor is hidden while the menu is active.
    """
    n = len(options)
    # Start at the default item, falling back to 0
    cursor = 0
    for i, (val, _) in enumerate(options):
        if val == default:
            cursor = i
            break

    def _render(redraw: bool = False) -> None:
        """Print (or re-print) the menu lines."""
        if redraw:
            # We're at the end of the hint line (no trailing \n).
            # Clear hint line first, then move up and clear each option line.
            sys.stdout.write(_CLEAR_LINE + (_UP1 + _CLEAR_LINE) * n)
            sys.stdout.flush()

        lines: list[str] = []
        for i, (val, label) in enumerate(options):
            selected = i == cursor
            is_def   = val == default
            if selected:
                def_tag = _c("  ← default", _DIM, _YLW) if is_def else ""
                line = _c("  ❯ ", _B, _GRN) + _c(label, _B, _GRN) + def_tag
            else:
                def_tag = _c("  ← default", _DIM, _YLW) if is_def else ""
                line = "    " + _c(label, _WHT) + def_tag
            lines.append(line)

        hint = _c("  ↑↓ navigate  ·  Enter to select", _DIM, _CYN)
        lines.append(hint)

        # Write everything in one flush to avoid partial redraws
        sys.stdout.write("\n".join(lines))
        sys.stdout.flush()

    sys.stdout.write(_HIDE_CURSOR)
    sys.stdout.flush()
    try:
        _render(redraw=False)
        while True:
            key = _read_key()
            if key == "ctrl_c":
                raise KeyboardInterrupt
            elif key in ("up", "k"):
                cursor = (cursor - 1) % n
                _render(redraw=True)
            elif key in ("down", "j"):
                cursor = (cursor + 1) % n
                _render(redraw=True)
            elif key == "enter":
                # Finalize: move past hint line, print confirmation
                sys.stdout.write("\n")
                chosen_val, chosen_label = options[cursor]
                print(
                    _c("  ✔ ", _B, _GRN)
                    + _c(chosen_label, _B, _WHT),
                    flush=True,
                )
                print()
                return chosen_val
    finally:
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()


# ── Input helper ──────────────────────────────────────────────────────────────

def _ask(
    prompt: str,
    options: list[Option] | None = None,
    default: str = "",
    secret: bool = False,
    validate: Callable[[str], Any] | None = None,
) -> str:
    """Render a prompt (arrow-key menu for options, text input otherwise)."""
    if options:
        print(_c(f"  {prompt}", _B, _WHT))
        print()
        return _menu(options, default=default)

    while True:
        hint = f" [{default}]" if default else ""
        sys.stdout.write(
            _c("  ❯ ", _B, _GRN)
            + _c(prompt + hint + ":  ", _WHT)
        )
        sys.stdout.flush()
        try:
            raw = getpass.getpass("") if secret else input()
        except (EOFError, KeyboardInterrupt):
            raise

        raw = raw.strip()

        if raw == "" and default:
            return default

        if validate is not None:
            try:
                validate(raw)
            except Exception as exc:
                print(_c(f"  {exc}", _RED))
                continue

        return raw


def _yn(prompt: str, default: bool) -> bool:
    return _ask(
        prompt,
        options=[("y", "Yes"), ("n", "No")],
        default="y" if default else "n",
    ) == "y"


# ── Step screens ──────────────────────────────────────────────────────────────

def _step1_exchange() -> tuple[str, bool, bool]:
    """Step 1 — Exchange & Environment."""
    _clear()
    _progress(1, "Exchange & Environment")
    _box_top("STEP 1 · Exchange")
    _box_row("  Select the exchange and trading environment.")
    _box_bot()
    print()

    exchange = _ask(
        "Exchange",
        options=[
            ("bybit",   "Bybit       (demo + unified margin)"),
            ("binance", "Binance     (spot)"),
            ("okx",     "OKX"),
            ("kraken",  "Kraken"),
            ("other",   "Other — enter a custom CCXT id"),
        ],
        default="bybit",
    )
    if exchange == "other":
        exchange = _ask("Custom CCXT exchange id", default="bybit")

    print()

    # Only Bybit supports demo mode; other exchanges use sandbox or mainnet.
    _demo_supported = exchange == "bybit"
    if not _demo_supported:
        print(_c(f"  ℹ  Demo mode is only available on Bybit.", _YLW))
        print(_c(f"     For {exchange}, use Sandbox (testnet) or Mainnet.", _DIM, _WHT))
        print()
        env_options = [
            ("sandbox",  "Sandbox / testnet     (no real funds, recommended for testing)"),
            ("mainnet",  "Mainnet — real funds, real risk"),
        ]
        env_default = "sandbox"
    else:
        env_options = [
            ("demo",     "Demo / paper trading  (no real funds, recommended for testing)"),
            ("sandbox",  "Sandbox / testnet"),
            ("mainnet",  "Mainnet — real funds, real risk"),
        ]
        env_default = "demo"

    env = _ask("Environment", options=env_options, default=env_default)
    return exchange, env == "demo", env == "sandbox"


def _step2_market() -> tuple[str, str, int, int]:
    """Step 2 — Market & Timeframe."""
    _clear()
    _progress(2, "Market & Timeframe")
    _box_top("STEP 2 · Market")
    _box_row("  Which asset and candle timeframe should the bot trade?")
    _box_bot()
    print()

    symbol = _ask(
        "Symbol",
        options=[
            ("BTC/USDT", "Bitcoin  / USDT  (most liquid)"),
            ("ETH/USDT", "Ethereum / USDT"),
            ("SOL/USDT", "Solana   / USDT"),
            ("other",    "Custom symbol"),
        ],
        default="BTC/USDT",
    )
    if symbol == "other":
        symbol = _ask("Custom symbol (e.g. ADA/USDT)", default="BTC/USDT")

    print()
    timeframe = _ask(
        "Timeframe",
        options=[
            ("1m",  "1 minute    (very noisy, high churn)"),
            ("5m",  "5 minutes"),
            ("15m", "15 minutes"),
            ("1h",  "1 hour"),
            ("4h",  "4 hours    (recommended — fewer false signals)"),
            ("1d",  "1 day      (slow, big-picture trend only)"),
        ],
        default="4h",
    )

    _windows = {
        "1m": (5, 20), "5m": (10, 30), "15m": (14, 40),
        "1h": (20, 50), "4h": (20, 50), "1d": (10, 30),
    }
    short_w, long_w = _windows.get(timeframe, (20, 50))
    return symbol, timeframe, short_w, long_w


# ── Strategy presets ──────────────────────────────────────────────────────────

_PRESETS: dict[str, dict] = {
    "conservative": dict(
        confluence_threshold=3, min_adx=25.0, rsi_filter=True,
        volume_confirmation=True, use_atr_stops=True,
        atr_sl_multiplier=2.0, atr_tp_multiplier=3.0,
        use_trailing_stop=True, trail_atr_multiplier=2.0,
        use_atr_sizing=True, atr_risk_pct=0.01,
        allow_short=False, loss_cooldown=300,
        max_drawdown=0.05, max_daily_loss=0.03,
        stop_loss=0.01, take_profit=0.02,
    ),
    "balanced": dict(
        confluence_threshold=2, min_adx=20.0, rsi_filter=True,
        volume_confirmation=False, use_atr_stops=True,
        atr_sl_multiplier=1.5, atr_tp_multiplier=2.5,
        use_trailing_stop=False, trail_atr_multiplier=2.0,
        use_atr_sizing=True, atr_risk_pct=0.015,
        allow_short=False, loss_cooldown=120,
        max_drawdown=0.08, max_daily_loss=0.05,
        stop_loss=0.01, take_profit=0.02,
    ),
    "aggressive": dict(
        confluence_threshold=1, min_adx=0.0, rsi_filter=False,
        volume_confirmation=False, use_atr_stops=False,
        atr_sl_multiplier=2.0, atr_tp_multiplier=3.0,
        use_trailing_stop=False, trail_atr_multiplier=2.0,
        use_atr_sizing=False, atr_risk_pct=0.02,
        allow_short=True, loss_cooldown=0,
        max_drawdown=0.15, max_daily_loss=0.0,
        stop_loss=0.015, take_profit=0.03,
    ),
}


def _step3_profile() -> dict:
    """Step 3 — Strategy Profile."""
    _clear()
    _progress(3, "Strategy Profile")
    _box_top("STEP 3 · Strategy Profile")
    _box_row("  Conservative → safer, slower, fewer trades.")
    _box_row("  Balanced     → moderate risk and frequency.")
    _box_row("  Aggressive   → minimal filters, shorts enabled.")
    _box_row("  Custom       → configure every option yourself.")
    _box_bot()
    print()

    choice = _ask(
        "Profile",
        options=[
            ("conservative", "Conservative"),
            ("balanced",     "Balanced"),
            ("aggressive",   "Aggressive"),
            ("custom",       "Custom"),
        ],
        default="conservative",
    )

    if choice != "custom":
        return dict(_PRESETS[choice])

    # ── Custom sub-wizard ─────────────────────────────────────────────────────
    print()
    print(_c("  Custom mode — press Enter to accept each default.\n", _YLW))

    def _int_validate(s: str) -> None:
        if int(s) < 0:
            raise ValueError("must be ≥ 0")

    def _float_validate(lo: float, hi: float) -> Callable[[str], None]:
        def v(s: str) -> None:
            f = float(s)
            if not (lo <= f <= hi):
                raise ValueError(f"must be {lo}–{hi}")
        return v

    ct = int(_ask("Confluence threshold (0=off, 2-3 recommended)", default="3",
                  validate=_int_validate))
    adx = float(_ask("Minimum ADX (0=off, 25 recommended)", default="25",
                     validate=_float_validate(0, 100)))
    rsi = _yn("RSI filter (block overbought BUY / oversold SELL entries)?", True)
    vol = _yn("Volume confirmation (apply -1 confluence penalty on low volume)?", True)
    atr_stops = _yn("ATR-based stops (replaces fixed SL/TP)?", True)

    sl_mult, tp_mult = 2.0, 3.0
    if atr_stops:
        sl_mult = float(_ask("ATR SL multiplier", default="2.0",
                             validate=_float_validate(0.1, 20)))
        tp_mult = float(_ask("ATR TP multiplier", default="3.0",
                             validate=_float_validate(0.1, 20)))

    trailing, trail_mult = False, 2.0
    if atr_stops:
        trailing = _yn("Trailing stop?", True)
        if trailing:
            trail_mult = float(_ask("Trailing stop ATR multiplier", default="2.0",
                                    validate=_float_validate(0.1, 20)))

    atr_sizing = _yn("ATR-based position sizing?", True)
    risk_pct = 0.01
    if atr_sizing:
        risk_pct = float(_ask("Risk fraction per trade (0.01 = 1%)", default="0.01",
                              validate=_float_validate(0.001, 0.5)))

    shorts  = _yn("Allow short trades?", False)
    cd      = int(_ask("Loss cooldown seconds (0=off)", default="300",
                       validate=_int_validate))
    max_dd  = float(_ask("Max drawdown fraction to stop (0=off, 0.05=5%)", default="0.05",
                         validate=_float_validate(0, 1)))
    max_dl  = float(_ask("Max daily loss fraction (0=off, 0.03=3%)", default="0.03",
                         validate=_float_validate(0, 1)))

    return dict(
        confluence_threshold=ct, min_adx=adx, rsi_filter=rsi,
        volume_confirmation=vol, use_atr_stops=atr_stops,
        atr_sl_multiplier=sl_mult, atr_tp_multiplier=tp_mult,
        use_trailing_stop=trailing, trail_atr_multiplier=trail_mult,
        use_atr_sizing=atr_sizing, atr_risk_pct=risk_pct,
        allow_short=shorts, loss_cooldown=cd,
        max_drawdown=max_dd, max_daily_loss=max_dl,
        stop_loss=0.01, take_profit=0.02,
    )


def _step4_execution(demo: bool) -> tuple[bool, float, int]:
    """Step 4 — Execution mode, order size, poll interval."""
    _clear()
    _progress(4, "Execution")
    _box_top("STEP 4 · Execution")
    _box_row("  How should the bot place trades?")
    _box_bot()
    print()

    mode = _ask(
        "Execution mode",
        options=[
            ("dry",  "Dry-run  — simulate only, no real orders (safe)"),
            ("live", "Live     — place real market orders"),
        ],
        default="dry",
    )
    execute = mode == "live"

    print()

    def _pos_float(s: str) -> None:
        if float(s) <= 0:
            raise ValueError("must be > 0")

    def _non_neg_int(s: str) -> None:
        if int(s) < 0:
            raise ValueError("must be ≥ 0")

    amt = float(_ask(
        "Order amount (base asset, e.g. 0.001 for 0.001 BTC)",
        default="0.001",
        validate=_pos_float,
    ))

    print()
    poll = int(_ask(
        "Poll interval in seconds (60 recommended; 0 = run once)",
        default="60",
        validate=_non_neg_int,
    ))

    return execute, amt, poll


def _step5_files() -> tuple[str, str | None]:
    """Step 5 — State file + optional record file."""
    _clear()
    _progress(5, "Files")
    _box_top("STEP 5 · State & Record Files")
    _box_row("  State file persists your open position across restarts.")
    _box_row("  Record file logs every cycle to CSV for review.")
    _box_bot()
    print()

    state = _ask("State file path", default="state/conservative.json")
    state_file = state or "state/conservative.json"

    print()
    rec = _ask(
        "Record file (CSV path, or Enter to skip)",
        default="logs/conservative.csv",
    )
    record_file: str | None = rec if rec else None

    return state_file, record_file


def _step6_extras(exchange: str, master_pw: str | None) -> tuple[bool, str | None, str | None]:
    """Step 6 — XGBoost + API credentials.

    *master_pw* is the session master password established at startup.
    All vault operations use it directly — no per-operation password prompts.
    """
    _clear()
    _progress(6, "ML Filter & API Keys")
    _box_top("STEP 6 · Extras")
    _box_row("  XGBoost adds an ML bias layer on top of the MA signal.")
    _box_row("  API keys are managed via the encrypted vault (~/.probot/vault.enc)")
    _box_row("  or read from TRADER_API_KEY / TRADER_API_SECRET env vars.")
    _box_bot()
    print()

    use_xgb = _yn("Enable XGBoost ML filter?", False)
    print()

    from trader_app.credentials import (
        is_available as _vault_available,
        vault_exists, load_vault, add_credential, clear_vault,
        DEFAULT_VAULT_PATH,
    )

    # ── 1. Env vars take priority ─────────────────────────────────────────────
    env_key    = os.getenv("TRADER_API_KEY")
    env_secret = os.getenv("TRADER_API_SECRET")
    if env_key and env_secret:
        print(_c("  API key found in environment (TRADER_API_KEY) — will be used.", _GRN))
        print()
        return use_xgb, env_key, env_secret

    # ── 2. No crypto or no master password → manual entry ────────────────────
    if not _vault_available() or master_pw is None:
        if not _vault_available():
            print(_c("  ⚠  'cryptography' package not installed — vault unavailable.", _YLW))
            print(_c("     Run: pip install cryptography", _DIM, _WHT))
        else:
            print(_c("  Vault access unavailable (master password not set).", _YLW))
        print()
        key_in = _ask("API key (Enter to skip)", default="")
        sec_in = _ask("API secret (Enter to skip, hidden)", default="", secret=True)
        return use_xgb, key_in or None, sec_in or None

    # ── 3. Vault flow (master_pw already verified at startup) ─────────────────
    print(_c("  Credential Vault  (~/.probot/vault.enc)", _B, _CYN))
    print()

    vault_action = _ask(
        "API credentials source",
        options=[
            ("load",  "Load from vault      (use a saved key)"),
            ("save",  "Save new keys        (add to vault)"),
            ("clear", "Clear vault          (delete all stored credentials)"),
            ("skip",  "Enter without saving (dry-run friendly)"),
        ],
        default="load" if vault_exists(DEFAULT_VAULT_PATH) else "save",
    )

    # ── Clear vault ───────────────────────────────────────────────────────────
    if vault_action == "clear":
        print()
        _box_top("CLEAR VAULT")
        _box_row("  This will permanently delete ALL stored API credentials.", _RED)
        _box_bot()
        print()
        try:
            clear_vault(master_pw, DEFAULT_VAULT_PATH)
            print(_c("  ✔  Vault cleared — all credentials have been deleted.", _GRN))
        except Exception as exc:
            print(_c(f"  ✗  Could not clear vault: {exc}", _RED))
        print()
        key_in = _ask("API key (Enter to skip)", default="")
        sec_in = _ask("API secret (Enter to skip, hidden)", default="", secret=True)
        return use_xgb, key_in or None, sec_in or None

    # ── Load from vault ───────────────────────────────────────────────────────
    if vault_action == "load":
        try:
            creds = load_vault(master_pw, DEFAULT_VAULT_PATH)
        except Exception as exc:
            print(_c(f"  ✗  Could not open vault: {exc}", _RED))
            print()
            key_in = _ask("API key (Enter to skip)", default="")
            sec_in = _ask("API secret (Enter to skip, hidden)", default="", secret=True)
            return use_xgb, key_in or None, sec_in or None

        matching = [c for c in creds if c["exchange"] == exchange]
        all_entries = creds
        display = (
            [(c["label"], f"{c['label']}  ({c['exchange']})") for c in matching]
            if matching else
            [(c["label"], f"{c['label']}  ({c['exchange']})") for c in all_entries]
        )

        if not creds:
            print(_c("  Vault is empty — switch to 'Save new keys' to add credentials.", _YLW))
            print()
            key_in = _ask("API key (Enter to skip)", default="")
            sec_in = _ask("API secret (Enter to skip, hidden)", default="", secret=True)
            return use_xgb, key_in or None, sec_in or None

        if not matching:
            print(_c(f"  No credentials stored for {exchange}. Showing all.", _YLW))
        print()

        chosen_label = _ask(
            "Select credential",
            options=display,
            default=display[0][0],
        )
        chosen = next((c for c in creds if c["label"] == chosen_label), None)
        if chosen:
            print(_c(f"  ✔  Loaded credentials for '{chosen_label}'", _GRN))
            print()
            return use_xgb, chosen["key"] or None, chosen["secret"] or None

    # ── Save new keys to vault ────────────────────────────────────────────────
    if vault_action == "save":
        print()
        print(_c("  Enter new API credentials to save to the vault.", _WHT))
        print()
        label   = _ask("Label for this credential (e.g. bybit-demo)", default=exchange)
        key_in  = _ask("API key", default="")
        sec_in  = _ask("API secret (hidden)", default="", secret=True)
        pass_in = _ask("API password / passphrase (Enter to skip)", default="")
        if key_in and sec_in:
            try:
                add_credential(
                    label=label, exchange=exchange,
                    key=key_in, secret=sec_in, password=pass_in,
                    master_password=master_pw, vault_path=DEFAULT_VAULT_PATH,
                )
                print(_c(f"  ✔  Saved '{label}' to vault.", _GRN))
            except Exception as exc:
                print(_c(f"  ✗  Could not save to vault: {exc}", _RED))
        print()
        return use_xgb, key_in or None, sec_in or None

    # ── Skip / manual entry ───────────────────────────────────────────────────
    print()
    key_in = _ask("API key (Enter to skip)", default="")
    sec_in = _ask("API secret (Enter to skip, hidden)", default="", secret=True)
    return use_xgb, key_in or None, sec_in or None


def _step7_summary(cfg: dict) -> bool:
    """Step 7 — Summary + launch confirmation."""
    _clear()
    _progress(7, "Summary & Launch")
    _box_top("CONFIGURATION SUMMARY")
    _box_sep()

    rows: list[tuple[str, str]] = [
        ("Exchange",       cfg.get("exchange_id", "")),
        ("Environment",    "demo" if cfg.get("demo") else "sandbox" if cfg.get("sandbox") else "mainnet"),
        ("Symbol",         cfg.get("symbol", "")),
        ("Timeframe",      cfg.get("timeframe", "")),
        ("MA windows",     f"short={cfg.get('short_window')}  long={cfg.get('long_window')}"),
        ("Execution",      "LIVE ⚠" if cfg.get("execute_orders") else "dry-run"),
        ("Order amount",   str(cfg.get("order_amount", ""))),
        ("Poll seconds",   str(cfg.get("poll_seconds", ""))),
        ("Strategy",       _describe_profile(cfg)),
        ("Confluence",     str(cfg.get("confluence_threshold", 0))),
        ("ADX filter",     str(cfg.get("min_adx", 0))),
        ("RSI filter",     "yes" if cfg.get("rsi_filter") else "no"),
        ("Volume penalty", "yes" if cfg.get("volume_confirmation") else "no"),
        ("ATR stops",      "yes" if cfg.get("use_atr_stops") else "no"),
        ("Trailing stop",  "yes" if cfg.get("use_trailing_stop") else "no"),
        ("ATR sizing",     "yes" if cfg.get("use_atr_sizing") else "no"),
        ("Shorts",         "yes" if cfg.get("allow_short") else "no"),
        ("XGBoost ML",     "yes" if cfg.get("use_xgboost") else "no"),
        ("Max drawdown",   f"{cfg.get('max_drawdown', 0)*100:.1f}%"),
        ("Max daily loss", f"{cfg.get('max_daily_loss', 0)*100:.1f}%"),
        ("State file",     cfg.get("state_file", "")),
        ("Record file",    cfg.get("record_file") or "(none)"),
    ]

    for label, value in rows:
        line = f"  {label:<18} {value}"
        _box_row(line)

    _box_sep()
    if cfg.get("execute_orders"):
        _box_row("  ⚠  LIVE MODE — real orders will be placed", _RED)
    else:
        _box_row("  DRY-RUN — no real orders will be placed", _GRN)
    _box_bot()
    print()

    return _ask(
        "Launch PROBOT",
        options=[
            ("y", "Yes — start the bot now"),
            ("n", "No  — abort"),
        ],
        default="y",
    ) == "y"


def _describe_profile(cfg: dict) -> str:
    ct  = cfg.get("confluence_threshold", 0)
    adx = cfg.get("min_adx", 0)
    if ct >= 3 and adx >= 25:
        return "conservative"
    if ct >= 2 and adx >= 20:
        return "balanced"
    return "aggressive"


# ── Master password gate ─────────────────────────────────────────────────────

def _unlock_master_password() -> str | None:
    """Prompt for the session master password at wizard startup.

    * If the vault already exists  → verify the password against it.
    * If no vault exists yet       → ask for a new password + confirmation.
    * If the cryptography package is missing → skip silently (returns None).
    """
    import getpass as _gp
    from trader_app.credentials import (
        is_available as _vault_available,
        vault_exists, load_vault, save_vault, DEFAULT_VAULT_PATH,
    )

    if not _vault_available():
        return None

    _clear()
    _box_top("MASTER PASSWORD")
    if vault_exists(DEFAULT_VAULT_PATH):
        _box_row("  Enter your master password to unlock the credential vault.", _WHT)
        _box_row("  This password will be used for all vault operations this session.", _DIM)
    else:
        _box_row("  No vault found. Set a master password to create one.", _YLW)
        _box_row("  This password protects all your stored API credentials.", _DIM)
        _box_row("  Choose something strong — it cannot be recovered if lost.", _DIM)
    _box_bot()
    print()

    if vault_exists(DEFAULT_VAULT_PATH):
        for attempt in range(3):
            sys.stdout.write(_c("  Master password: ", _WHT))
            sys.stdout.flush()
            pw = _gp.getpass("")
            try:
                load_vault(pw, DEFAULT_VAULT_PATH)  # verify
                print(_c("  ✔  Vault unlocked.", _GRN))
                print()
                return pw
            except ValueError:
                remaining = 2 - attempt
                if remaining:
                    print(_c(f"  ✗  Wrong password. {remaining} attempt(s) remaining.", _RED))
                else:
                    print(_c("  ✗  Too many failed attempts — vault access skipped.", _RED))
        print()
        return None
    else:
        # New vault — set password with confirmation
        while True:
            sys.stdout.write(_c("  New master password: ", _WHT))
            sys.stdout.flush()
            pw = _gp.getpass("")
            if not pw:
                print(_c("  Password cannot be empty.", _RED))
                continue
            sys.stdout.write(_c("  Confirm master password: ", _WHT))
            sys.stdout.flush()
            pw2 = _gp.getpass("")
            if pw != pw2:
                print(_c("  ✗  Passwords do not match — try again.", _RED))
                continue
            # Initialise vault with empty list so future runs know it exists
            save_vault([], pw, DEFAULT_VAULT_PATH)
            print(_c("  ✔  Master password set. Vault created.", _GRN))
            print()
            return pw


# ── Public entry point ────────────────────────────────────────────────────────

def run_wizard() -> Settings | None:
    """Run the interactive setup wizard and return a :class:`Settings` object.

    Returns ``None`` if the user aborts or if stdin/stdout are not a TTY.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None

    try:
        _splash()
        input(_c("  Press Enter to begin  ·  Ctrl+C to abort", _B, _YLW))
        print()

        master_pw = _unlock_master_password()

        exchange, demo, sandbox   = _step1_exchange()
        symbol, tf, sw, lw        = _step2_market()
        profile                   = _step3_profile()
        execute, amount, poll     = _step4_execution(demo)
        state_file, record_file   = _step5_files()
        use_xgb, api_key, api_sec = _step6_extras(exchange, master_pw)

        cfg: dict = dict(
            exchange_id=exchange,
            symbol=symbol,
            timeframe=tf,
            short_window=sw,
            long_window=lw,
            demo=demo,
            sandbox=sandbox,
            execute_orders=execute,
            order_amount=amount,
            poll_seconds=poll,
            state_file=state_file,
            record_file=record_file,
            api_key=api_key,
            api_secret=api_sec,
            api_password=os.getenv("TRADER_API_PASSWORD"),
            use_xgboost=use_xgb,
            **profile,
        )

        if not _step7_summary(cfg):
            print(_c("\n  Aborted.\n", _YLW))
            return None

        print()
        print(_c("  Launching PROBOT…\n", _B, _GRN))
        return Settings(**cfg)

    except KeyboardInterrupt:
        print(_c("\n\n  Wizard aborted.\n", _YLW))
        return None
