from __future__ import annotations

import sys
from pathlib import Path


def apply_owner_only_permissions(path: Path) -> None:
    """Apply private POSIX modes or a protected owner-only Windows DACL."""

    candidate = Path(path)
    if sys.platform != "win32":
        candidate.chmod(0o700 if candidate.is_dir() else 0o600)
        return
    try:
        import ntsecuritycon
        import win32security
    except ImportError as exc:  # pragma: no cover - Windows dependency contract
        raise OSError("Owner-only Windows permissions require pywin32.") from exc
    security = win32security.GetFileSecurity(
        str(candidate), win32security.OWNER_SECURITY_INFORMATION
    )
    owner = security.GetSecurityDescriptorOwner()
    dacl = win32security.ACL()
    dacl.AddAccessAllowedAce(
        win32security.ACL_REVISION,
        ntsecuritycon.FILE_ALL_ACCESS,
        owner,
    )
    win32security.SetNamedSecurityInfo(
        str(candidate),
        win32security.SE_FILE_OBJECT,
        win32security.DACL_SECURITY_INFORMATION
        | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        None,
        None,
        dacl,
        None,
    )
