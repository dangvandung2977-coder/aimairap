import os
me = os.getpid()
found = None
for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) == me:
        continue
    try:
        cl = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\x00", b" ").decode()
    except OSError:
        continue
    if "train_7day.py" in cl and "python" in cl:
        found = int(pid)
        break
lock = "checkpoints/final_7day/controller.lock"
if found:
    with open(lock, "w") as f:
        f.write(str(found))
    print("wrote lock pid", found)
else:
    print("controller process not found")
