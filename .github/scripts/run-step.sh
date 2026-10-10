#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# The shell of the CI steps that build and test.
# It runs one step and, when the step fails, publishes the last lines of its output as an annotation, which anyone can read without signing in to GitHub (the job logs need a sign-in).
log="$(mktemp)"
set -o pipefail
bash -e "$1" 2>&1 | tee "$log"
status=$?
if [ "$status" -ne 0 ]; then
  tail -n 80 "$log" | python3 -c 'import sys; t = sys.stdin.read().replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"); print("::error title=Last lines of the failed step::" + t)'
fi
exit "$status"
