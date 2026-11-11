"""Verify the fixed _canonical path with the REAL pynput Listener (no OS hook).

The combo state machine must recognize every plausible key-event form the
Windows hook can deliver for ctrl/alt/space.
"""
import sys

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from pynput import keyboard
from pynput.keyboard import HotKey, KeyCode, Key

listener = keyboard.Listener(on_press=lambda k: None, on_release=lambda k: None)
combo = frozenset(HotKey.parse("<ctrl>+<alt>+<space>"))
print(f"combo: {sorted(repr(k) for k in combo)}")

forms = {
    "ctrl": [Key.ctrl, KeyCode.from_vk(17), KeyCode.from_vk(0xA2)],
    "alt": [Key.alt, KeyCode.from_vk(18), KeyCode.from_vk(0xA4)],
    "space": [KeyCode.from_vk(32), KeyCode(char=" ", vk=32), KeyCode.from_char(" ")],
}

ok = True
for name, keys in forms.items():
    results = []
    for k in keys:
        try:
            c = listener.canonical(k)
            hit = c in combo
        except Exception as e:
            c, hit = f"RAISED {e}", False
        results.append(f"{type(k).__name__}({getattr(k, 'vk', None)},{getattr(k, 'char', None)}) -> in_combo={hit}")
        ok = ok and hit
    print(f"{name}: {results}")

print("\nALL EVENT FORMS RECOGNIZED" if ok else "SOME FORMS MISS THE COMBO - CHECK")
