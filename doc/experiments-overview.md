# Overview of Experiments 


## Scripts at a glance

| Prefix | Directory | Plain / Encrypted | Dataset scope | Purpose |
|--------|---|---|---|---|
| **1_** | `1_training_plain` | Plain only | Full dataset, all classes | Baselines — plain PyTorch backprop vs. ReBoot's own local-loss algorithm, isolating the *algorithm* change from encryption entirely |
| **2_** | `2_logreg_comparison` | Both (plain + encrypted, side-by-side per batch) | Binary subset (2 classes) | First plain-vs-encrypted cross-check, on a simplified binary logistic-regression-style task |
| **3_** | `3_training_encrypted` | Both, and encrypted-only | Full dataset, all classes | Generalizes the cross-check to multi-class, adds CKKS-packing-specific diagnostics, and includes the actual real end-to-end encrypted training run (no plaintext shadow model) |
| **4_** | `4_performance_analysis` | Encrypted only | Full dataset | Not about accuracy — measures wall-clock time per stage (model/data encryption, forward, backward, bootstrap) |

The progression is roughly: **1** proves the algorithm works in plaintext → **2** proves plain-vs-encrypted agree on an easy binary case → **3** proves it on the full multi-class task and runs it for real → **4** measures how fast it actually runs.


## Scripts at a glance

| Script | Algorithm | Plain / Encrypted | Data scope | What it's for |
|---|---|---|---|---|
| `1_training_plain/backprop_plain.py` | Standard PyTorch backprop (`BackpropMLP`, `torch.autograd`) | Plain only | Full dataset, all classes | Baseline: plain backprop training, no ReBoot algorithm at all |
| `1_training_plain/reboot_plain.py` | ReBoot local-loss (`LocalLossMLP`) | Plain only | Full dataset, all classes | Same task as above, but with ReBoot's own local-loss algorithm instead of backprop — isolates the *algorithm* change from encryption |
| `2_logreg_comparison/reboot_plain.py` | ReBoot local-loss (`LocalLossMLP`) | Plain only | Binary subset (2 classes) | Plaintext side of a binary logistic-regression-style comparison |
| `2_logreg_comparison/cross_training.py` | ReBoot local-loss, plain + encrypted trained side-by-side per batch | Both | Binary subset (2 classes) | Cross-checks plain-vs-encrypted divergence (weight/pred differences, precision) on the same binary task as the `reboot_plain.py` sibling above |
| `3_training_encrypted/cross_training.py` | ReBoot local-loss, plain + encrypted side-by-side per batch | Both | Full dataset, all classes | Same cross-check as `2_logreg_comparison/cross_training.py`, generalized to multi-class instead of the binary logreg case; also logs precisions to CSV |
| `3_training_encrypted/precision_analysis.py` | ReBoot local-loss, plain + encrypted side-by-side per batch | Both | Full dataset (fixed `resize=14`) | Adds CKKS-packing-specific diagnostics on top of `cross_training.py` (zero-padding norms, repeated-value checks) — packing/precision correctness, not just accuracy |
| `3_training_encrypted/training_encrypted.py` | ReBoot local-loss, encrypted only | Encrypted only | Full dataset | The actual end-to-end encrypted training run — no plaintext shadow model, no diff computation, just trains + logs the encrypted model |
| `4_performance_analysis/performance_analysis.py` | ReBoot local-loss, encrypted only | Encrypted only | Full dataset | Not about accuracy at all — measures wall-clock time per stage (model encryption, data encryption, forward, backward, bootstrap) |

**The differences boil down to 4 independent axes, mixed and matched per script:**
1. **Algorithm**: plain PyTorch backprop (only `1_training_plain/backprop_plain.py`) vs. ReBoot's local-loss algorithm (every other script).
2. **Plain vs. encrypted**: plaintext-only, encrypted-only, or both trained side-by-side every batch purely to diff against each other.
3. **Dataset scope**: full multi-class dataset vs. a filtered binary-class subset (the `2_logreg_comparison/*` scripts).
4. **What's being measured**: model accuracy (`*_plain.py`, `training_encrypted.py`), plaintext/encrypted numerical divergence (`cross_training.py`, `precision_analysis.py`), or raw wall-clock performance (`performance_analysis.py`).




## backprop_plain.py
``` 
backprop_plain.py (module level)
├─ get_parser_args()                          [reboot.parser]      → argparse CLI parsing
├─ wandb_init(configs)                        [reboot.utils.train] → no-op here (wandb=False)
└─ for i in range(num_runs):
   ├─ set_seed(...)
   ├─ load_float_dataset(configs, ...)        [reboot.utils.data]  → load + preprocess dataset → numpy arrays
   ├─ BackpropMLP(...)                        [reboot.models.backprop_models]
   │  └─ __init__ → nn.ModuleList([_make_linear_block(...) for ...]) + final nn.Linear
   ├─ torch.optim.SGD(...) + CosineAnnealingWarmRestarts(...)
   ├─ TensorDataset / DataLoader (train, test)
   └─ for epoch in range(num_epochs):
      ├─ model.train()
      │  └─ for batch in train_dataloader:
      │     ├─ lr_scheduler.step()
      │     ├─ optimizer.zero_grad()
      │     ├─ model(X)                       → BackpropMLP.forward()
      │     │  └─ for layer in self.layers: x = layer(x)   [Linear → activation, per block]
      │     │     └─ x = self.layer_out(x)                 [final Linear]
      │     ├─ loss_fn(y_pred, y)              [CrossEntropyLoss]
      │     ├─ accuracy(y_pred.detach(), y)    [reboot.utils.nn]
      │     ├─ loss_batch.backward()           → autograd backward through the whole graph
      │     └─ optimizer.step()                → SGD w/ momentum+nesterov+weight_decay
      ├─ model.eval() + torch.inference_mode()
      │  └─ for batch in test_dataloader: forward-only, accumulate val_loss/val_acc
      └─ epoch_log(...)                        [reboot.utils.train] → prints epoch stats

```