# CS4248-Group-30-Project
This repository hosts the code used for our CS4248 Natural Language Processing Project

Authors: Adrian, Billy, Kenji, Nick, Norbert, Russell 

Mentor: Yisong

## Remote cluster workflow (send code, setup, run)

This project uses three scripts for remote usage:

- `deploy_and_submit.sh`: sync local code to the cluster (deploy only).
- `remote_setup.sh`: create `.venv` and install Python dependencies on the cluster.
- `remote_run.sh`: run inference on GPU (uses `srun` automatically if needed).

### 1) Set up passwordless SSH from WSL (one time)

To avoid entering your password every time, configure SSH key-based auth first.

1. Generate an SSH key (run in WSL):

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "wsl@$(hostname)" -f ~/.ssh/id_ed25519
```

2. Copy your public key to the remote host (replace user/host):

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <name>@xlogin.comp.nus.edu.sg
```

If `ssh-copy-id` is not available, run:

```bash
cat ~/.ssh/id_ed25519.pub | ssh <name>@xlogin.comp.nus.edu.sg 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
```

3. (Optional) Use `ssh-agent` to cache your key passphrase:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

4. Configure a convenient SSH host entry in `~/.ssh/config`:

```
Host xlogin
	HostName xlogin.comp.nus.edu.sg
	User <name>
	IdentityFile ~/.ssh/id_ed25519
	AddKeysToAgent yes
```

### 2) Send code to cluster

From your local project root:

```bash
./deploy_and_submit.sh
```

This syncs code to `~/CS4248/<project-folder>` on the remote and excludes `.venv`, `venv`, `.env`, and `models`.

### 3) Setup environment on cluster (first time, or when deps change)

```bash
ssh <name>@xlogin.comp.nus.edu.sg
cd ~/CS4248/<project-folder>
chmod +x remote_setup.sh remote_run.sh
./remote_setup.sh
```

### 4) Run inference on GPU

Interactive run (recommended for quick testing):

```bash
ssh <name>@xlogin.comp.nus.edu.sg
cd ~/CS4248/<project-folder>
./remote_run.sh --prompt "Hello"
```

Batch run with Slurm:

```bash
ssh <name>@xlogin.comp.nus.edu.sg
cd ~/CS4248/<project-folder>
sbatch remote_job.sbatch
```

Check jobs/logs:

```bash
squeue -u $(whoami)
tail -f slurm-<jobid>.out
```

Notes:
- Ensure remote `~/.ssh` permissions are `700` and `authorized_keys` is `600`.
- If your institution requires additional authentication (2FA, LDAP), follow local instructions or contact admins.
- If `remote_job.sbatch` is used, run `./remote_setup.sh` first so `.venv` exists before the batch job starts.
