#!/bin/sh
# Download the pinned Firecracker binary and guest kernel (x86_64 only) and verify sha256.
# Usage: fetch-assets.sh DEST_DIR
# Prints `export ITERANT_FIRECRACKER_BIN=... ITERANT_FIRECRACKER_KERNEL=...` on success.
set -eu
dest=${1:?usage: fetch-assets.sh DEST_DIR}

FC_VERSION=1.12.1
FC_SHA256=0a75e67ef6e4c540a2cf248b06822b0be9820cbba9fe19f9e0321200fe76ff6b
FC_URL="https://github.com/firecracker-microvm/firecracker/releases/download/v${FC_VERSION}/firecracker-v${FC_VERSION}-x86_64.tgz"
KERNEL_VERSION=6.1.128
KERNEL_SHA256=27a8310b9a727517e9eb02044524b6ceb77de5728e3491b6974d5c846227ecc8
KERNEL_URL="https://s3.amazonaws.com/spec.ccfc.min/firecracker-ci/v1.12/x86_64/vmlinux-${KERNEL_VERSION}"

[ "$(uname -m)" = x86_64 ] || { echo "only x86_64 is pinned" >&2; exit 1; }
mkdir -p "$dest"
curl -fsSL -o "$dest/firecracker.tgz" "$FC_URL"
curl -fsSL -o "$dest/vmlinux" "$KERNEL_URL"
echo "$FC_SHA256  $dest/firecracker.tgz" | sha256sum -c - >&2
echo "$KERNEL_SHA256  $dest/vmlinux" | sha256sum -c - >&2
tar -xzf "$dest/firecracker.tgz" -C "$dest" --strip-components=1 \
  "release-v${FC_VERSION}-x86_64/firecracker-v${FC_VERSION}-x86_64"
mv "$dest/firecracker-v${FC_VERSION}-x86_64" "$dest/firecracker"
chmod +x "$dest/firecracker"
echo "export ITERANT_FIRECRACKER_BIN=$dest/firecracker ITERANT_FIRECRACKER_KERNEL=$dest/vmlinux"
