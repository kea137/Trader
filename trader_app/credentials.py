"""Encrypted API credential vault for PROBOT.

Credentials for multiple exchanges are stored in a single encrypted file
(default: ``~/.probot/vault.enc``).  The file is encrypted with AES-128
via Fernet (from the ``cryptography`` package) using a key derived from a
master password with PBKDF2-HMAC-SHA256.

Vault file format (after decryption): JSON list of objects::

    [
      {
        "label":    "bybit-demo",
        "exchange": "bybit",
        "key":      "...",
        "secret":   "...",
        "password": ""
      },
      ...
    ]
"""
from __future__ import annotations

import base64
import getpass
import json
import os
from pathlib import Path
from typing import TypedDict

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    _CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CRYPTO_AVAILABLE = False

# Default vault location — stored in user's home directory, outside the project
DEFAULT_VAULT_PATH = Path.home() / ".probot" / "vault.enc"

# PBKDF2 parameters
_ITERATIONS = 480_000
_SALT_LEN   = 16


class Credential(TypedDict):
    label:    str
    exchange: str
    key:      str
    secret:   str
    password: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 32-byte Fernet key from *password* and *salt*."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def _encrypt(plaintext: bytes, password: str) -> bytes:
    """Return ``salt || ciphertext`` encrypted with *password*."""
    salt = os.urandom(_SALT_LEN)
    key  = _derive_key(password, salt)
    return salt + Fernet(key).encrypt(plaintext)


def _decrypt(data: bytes, password: str) -> bytes:
    """Decrypt ``salt || ciphertext`` produced by :func:`_encrypt`."""
    salt, ciphertext = data[:_SALT_LEN], data[_SALT_LEN:]
    key = _derive_key(password, salt)
    return Fernet(key).decrypt(ciphertext)  # raises InvalidToken on bad password


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Return True if the ``cryptography`` package is installed."""
    return _CRYPTO_AVAILABLE


def vault_exists(vault_path: Path = DEFAULT_VAULT_PATH) -> bool:
    return vault_path.exists() and vault_path.stat().st_size > 0


def load_vault(password: str, vault_path: Path = DEFAULT_VAULT_PATH) -> list[Credential]:
    """Load and decrypt the vault.  Returns an empty list if the vault does not exist.

    Raises ``ValueError`` on wrong password.
    """
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError("Install the 'cryptography' package to use the credential vault.")
    if not vault_exists(vault_path):
        return []
    try:
        plaintext = _decrypt(vault_path.read_bytes(), password)
    except InvalidToken:
        raise ValueError("Wrong master password — could not decrypt the vault.")
    return json.loads(plaintext)


def save_vault(
    credentials: list[Credential],
    password: str,
    vault_path: Path = DEFAULT_VAULT_PATH,
) -> None:
    """Encrypt and save *credentials* to *vault_path*."""
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError("Install the 'cryptography' package to use the credential vault.")
    vault_path.parent.mkdir(parents=True, exist_ok=True)
    vault_path.write_bytes(_encrypt(json.dumps(credentials).encode(), password))
    # Restrict permissions to owner-only
    try:
        vault_path.chmod(0o600)
    except OSError:
        pass


def add_credential(
    label: str,
    exchange: str,
    key: str,
    secret: str,
    password: str,
    master_password: str,
    vault_path: Path = DEFAULT_VAULT_PATH,
) -> None:
    """Add or overwrite a credential entry identified by *label*."""
    creds = load_vault(master_password, vault_path)
    creds = [c for c in creds if c["label"] != label]
    creds.append(Credential(
        label=label,
        exchange=exchange,
        key=key,
        secret=secret,
        password=password,
    ))
    save_vault(creds, master_password, vault_path)


def delete_credential(
    label: str,
    master_password: str,
    vault_path: Path = DEFAULT_VAULT_PATH,
) -> bool:
    """Remove the entry with *label*.  Returns True if it existed."""
    creds = load_vault(master_password, vault_path)
    new_creds = [c for c in creds if c["label"] != label]
    if len(new_creds) == len(creds):
        return False
    save_vault(new_creds, master_password, vault_path)
    return True


def get_credential(
    label: str,
    master_password: str,
    vault_path: Path = DEFAULT_VAULT_PATH,
) -> Credential | None:
    """Return the credential with *label*, or None if not found."""
    for c in load_vault(master_password, vault_path):
        if c["label"] == label:
            return c
    return None


def list_labels(
    master_password: str,
    vault_path: Path = DEFAULT_VAULT_PATH,
) -> list[str]:
    """Return the labels of all stored credentials."""
    return [c["label"] for c in load_vault(master_password, vault_path)]


def clear_vault(
    master_password: str,
    vault_path: Path = DEFAULT_VAULT_PATH,
) -> None:
    """Delete all credentials from the vault after verifying *master_password*.

    The vault file is overwritten with an empty list (not deleted), so the
    master password remains valid for future use.

    Raises ``ValueError`` on wrong password.
    """
    # load_vault will raise ValueError if the password is wrong
    load_vault(master_password, vault_path)
    save_vault([], master_password, vault_path)


def prompt_master_password(confirm: bool = False) -> str:
    """Interactively prompt for the master password (hidden input)."""
    pw = getpass.getpass("  Vault master password: ")
    if confirm:
        pw2 = getpass.getpass("  Confirm master password: ")
        if pw != pw2:
            raise ValueError("Passwords do not match.")
    return pw
