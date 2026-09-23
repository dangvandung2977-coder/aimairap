"""OPT-IN real Drive upload test. Skipped unless GDRIVE_REAL_TEST=1.

Usage (VPS, after `--setup` succeeded):
  GDRIVE_REAL_TEST=1 python tools/test_google_drive_upload.py
"""
from __future__ import annotations

import os
import sys

if os.environ.get("GDRIVE_REAL_TEST") != "1":
    print("SKIP: set GDRIVE_REAL_TEST=1 to run the real upload test")
    raise SystemExit(0)

from tools import google_drive_auth as gda
from tools import google_drive_reporter as gdr

service = gda.get_drive_service()
assert service is not None, "no authorized token; run --setup first"
cfg = gdr.load_config()
assert gdr.ensure_folders(service, cfg), "folder setup failed"
jp, _ = gdr.write_local_reports(gdr.collect_report())
ok, fid = gdr.upload_file(service, cfg["reports_folder_id"], jp)
print("REAL_UPLOAD:", "OK id=" + fid if ok else "FAILED")
sys.exit(0 if ok else 1)
