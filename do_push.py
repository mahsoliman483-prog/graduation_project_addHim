import subprocess
import os
import sys

def run(cmd, capture=True):
    print("Running:", " ".join(cmd))
    if capture:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("Error:", res.stderr.strip())
            return False, res.stdout.strip()
        return True, res.stdout.strip()
    else:
        res = subprocess.run(cmd)
        return res.returncode == 0, ""

def get_files():
    success, out = run(["git", "ls-files", "-m", "-o", "--exclude-standard"])
    if not success or not out:
        return []
    lines = out.strip().split("\n")
    return [line.strip() for line in lines if line.strip()]

def main():
    print("Unstaged files list...")
    # Now for uncommitted files
    files = get_files()
    if not files:
        print("No files left to commit.")
        return

    for file in files:
        if file in ['do_push.py', 'check_git.py', 'status.txt', 'files.json']:
            continue
            
        print(f"\nProcessing file: {file}")
        
        success, _ = run(["git", "add", file])
        if not success:
            continue
            
        # Check if anything was added
        success, out = run(["git", "diff", "--cached", "--quiet"])
        if success: # If quiet returns 0, no changes were staged
            print(f"No changes staged for {file}")
            continue

        success, _ = run(["git", "commit", "-m", f"Automated commit: Update {os.path.basename(file)}"])
        if not success:
            print("Commit failed.")
            continue
            
        success, _ = run(["git", "push", "origin", "HEAD:refs/heads/dev"], capture=False)
        if not success:
            print(f"Failed to push after committing {file}. Stopping.")
            sys.exit(1)
            
        print(f"Successfully pushed {file}")
    
    print("Done!")

if __name__ == "__main__":
    main()
