# Pipeline Verification Checklist

## Local Development

- [ ] Clone repo: `git clone https://github.com/YOUR_ORG/Q_Drop_Intergration.git`
- [ ] Install deps: `pip install -r requirements.txt`
- [ ] Run locally: `dvc repro --force`
- [ ] Check output: `cat metrics.json && ls src/qd_hqgc_mnist_training.png`

## Runner Setup (A6000 Server)

- [ ] OS is Linux x86_64: `uname -m` → `x86_64`
- [ ] Python 3.10 installed: `python3.10 --version`
- [ ] NVIDIA driver installed: `nvidia-smi` → shows GPU
- [ ] Runner directory created: `ls -la ~/actions-runner/`
- [ ] Runner registered: Check GitHub Repo → Settings → Actions → Runners
- [ ] Runner labels include `A6000`: Verify in GitHub UI
- [ ] Runner is "Idle" (not "Offline")
- [ ] Systemd service running: `sudo systemctl status actions.runner.* --no-pager`

## DVC Remote (Optional)

- [ ] Remote configured: `dvc remote list`
- [ ] Remote credentials valid: `dvc status` (no permission errors)
- [ ] Can write to remote: `echo "test" > /tmp/test.txt && dvc add /tmp/test.txt -o`

## Workflow Files

- [ ] `.github/workflows/train.yml` exists
- [ ] `dvc.yaml` exists with train stage
- [ ] `params.yaml` exists
- [ ] `baseline.json` exists
- [ ] `requirements.txt` exists
- [ ] `src/train_mnist.py` has JSON metrics output

## Git & GitHub

- [ ] Repo is on GitHub: `git remote -v`
- [ ] Main branch exists: `git branch | grep main`
- [ ] All files committed: `git status` → clean
- [ ] Ready to push: `git log --oneline -3`

## First Run

```bash
# From local machine
git checkout -b ci/test-pipeline
git add .github/ dvc.yaml params.yaml baseline.json requirements.txt PIPELINE_SETUP.md setup-runner.sh
git commit -m "ci: add CML + DVC training pipeline"
git push origin ci/test-pipeline

# Create PR or push to main to trigger
git checkout main
git merge ci/test-pipeline
git push origin main

# Monitor on GitHub
# Your Repo → Actions → Q-Drop QGN Integration Training Pipeline → [Running...]
```

## Post-First-Run Verification

- [ ] Workflow triggered automatically
- [ ] All steps passed (green checkmarks)
- [ ] Metrics saved: `cat metrics.json`
- [ ] Baseline updated: `git log --oneline baseline.json | head -1`
- [ ] Artifacts on server: `ls -lah /tmp/q-drop-training-*/`
- [ ] Plot generated: `ls -lah */qd_hqgc_mnist_training.png`

## Troubleshooting Commands

```bash
# Check runner logs (on server)
sudo journalctl -u actions.runner.* -f --no-pager

# Check disk space
df -h /tmp /

# Check GPU status
nvidia-smi

# Manual test of training
cd ~/Q_Drop_Intergration
pip install -q -r requirements.txt
dvc repro --force

# Check metrics
cat metrics.json | python -m json.tool
```

## Success Criteria

✅ **Pipeline succeeds when:**
1. Runner picks up job automatically
2. Code pulled (git) in ~30s
3. Datasets pulled (DVC) — may skip if no remote
4. Environment verified (Python, TF, GPU, PennyLane)
5. Training completes (may take 10-30 min depending on epochs)
6. JSON metrics generated
7. Baseline compared
8. Artifacts saved to `/tmp/q-drop-training-{RUN_ID}`
9. Baseline git commit created (on main push)

---

**Questions?** Check `PIPELINE_SETUP.md` or GitHub Actions runner logs.
