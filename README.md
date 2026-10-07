# LEADER V7

Current released research version: **V7 — LWV-trained FAR weights with the original V6 self-verifying inference solver**.

- [V7 implementation, protocol, running instructions and limitations](experiments/v7/README.md)
- [V7 frozen baseline](experiments/v7/baseline.json)
- [V7 execution validation](experiments/v7/results/)
- [V5 implementation and historical results](experiments/v5/README.md)
- [Original LEADER project](UPSTREAM_README.md)

V7 keeps the frozen LEADER/SAM frontend, uses LWV for backend training and the unchanged original V6 cubic-volume solver for inference. This snapshot includes validated seven-replica training acceleration, four-frame parallel inference and strict NCLT reference-policy scoring. Checkpoints and prepared data are external assets.

The reused 1,200-frame development result is MPE **0.118944 m**, MOE **1.411591°**. Full-data continuation and evaluation were ongoing at publication; these numbers are not final full-Test results. Public path/prepared-frame adapters have not been rerun on a fresh GPU environment.
