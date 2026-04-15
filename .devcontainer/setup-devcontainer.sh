#!/usr/bin/env bash

# Create a virtual environment at $VIRTUAL_ENV (set in devcontainer.json)
# and install requirements. The pip cache is shared with the host via
# PIP_CACHE_DIR (also set in devcontainer.json), so rebuilds are fast.
python -m venv "$VIRTUAL_ENV"
pip install -r requirements.txt

# Configure Git
git config pull.rebase true
git config --type bool push.autoSetupRemote true
cat <<EOF >> ~/.gitconfig
[user]
    name = AISB Account
    email = 26707046+aisb-account@users.noreply.github.com
EOF

# Set up SSH for GitHub
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Create SSH config for GitHub
cat > ~/.ssh/config << EOL
Host github.com
    HostName github.com
    User git
    IdentityFile /mnt/ssh/aisb_ssh
    IdentitiesOnly yes
EOL

chmod 600 ~/.ssh/config

# Add GitHub to known hosts to avoid prompts
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

echo "Setup finished."
