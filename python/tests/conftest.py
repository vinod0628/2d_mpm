import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("USE_PYGEOS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# preferences (recent projects, language, theme) go to a throw-away .ini, never to the user's registry
import tempfile  # noqa: E402
os.environ["MPM_SETTINGS_FILE"] = os.path.join(tempfile.mkdtemp(prefix="mpm_settings_"), "settings.ini")
if os.name == "nt":   # the offscreen platform has no font database of its own
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

DEMO = os.path.abspath(os.path.join(ROOT, "..", "demo_data"))
CARLIN = os.path.join(DEMO, "nevada_carlin_au", "processed")
CURNAMONA = os.path.join(DEMO, "curnamona_ree", "processed")
