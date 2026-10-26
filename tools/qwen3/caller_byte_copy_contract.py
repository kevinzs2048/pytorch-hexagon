"""Import the shared authenticated caller transfer contract."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'qwen3/runtime/arm'))
from hexagon_caller_byte_copy import CONTRACT, rewrite_caller
