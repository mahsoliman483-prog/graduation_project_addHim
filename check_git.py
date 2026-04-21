import subprocess

def run(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip(), res.stderr.strip()

out, err = run(["git", "status", "-s"])
print("--- STATUS REPR ---")
print(repr(out))
