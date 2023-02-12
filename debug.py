# debug.py
from __future__ import annotations

from singing_transcription import TranscribeSinging
# from transpose_track import TransposeTrack
from tuneflow_devkit import Debugger

if __name__ == "__main__":
    # Debugger(plugin_class=TransposeTrack).start()
    Debugger(plugin_class=TranscribeSinging).start()