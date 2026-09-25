49.1 phase 4/
├── main.py
├── config.py
├── market.py
├── provider.py
├── portfolio.py
├── dataset.py
├── binance_vision.py
├── external_history.py
├── primary_features.py
├── feature_isolation.py
├── fractional.py
├── calibration.py
├── meta_model.py
├── hmm.py
├── XGBoost.py
├── TripleBarrier.py
├── OFI.py
├── VPIN.py
├── Sarparast.py
├── senate.py
├── telemetry_utils.py
├── analytics/
│   ├── build_external_history.py
│   ├── calibrate_meta.py
│   ├── research.py
│   ├── select_features.py
│   ├── train_meta.py
│   ├── train_XGBoost.py
│   └── ablation81/
│       ├── __init__.py
│       ├── a4.py
│       ├── adjudication.py
│       ├── adjudication_platt.py
│       ├── adjudication_restore.py
│       ├── cscv.py
│       ├── feature_ablation.py
│       ├── integrity.py
│       ├── lookahead.py
│       ├── metrics.py
│       ├── nested.py
│       ├── phase0.py
│       ├── phase1.py
│       ├── phase2.py
│       ├── phase3.py
│       ├── phase3_complete.py
│       ├── phase3_gates.py
│       ├── phase4.py
│       ├── placebo.py
│       ├── postmortem.py
│       └── quarantine.py
├── tests/
│   ├── conftest.py
│   ├── hmm_walkforward.py
│   ├── test_ablation81.py
│   ├── test_calibration.py
│   ├── test_dataset.py
│   ├── test_dataset_enrichment.py
│   ├── test_external_history.py
│   ├── test_hmm_contract.py
│   ├── test_import_discipline.py
│   ├── test_meta_model.py
│   ├── test_micro_history.py
│   ├── test_OFI.py
│   ├── test_primary_features.py
│   ├── test_sarparast.py
│   ├── test_triple_barrier.py
│   ├── test_VPIN.py
│   ├── test_XGBoost.py
│   └── scripts/
│       └── live_microstructure_smoke.py
├── data/
│   ├── binance_vision/
│   ├── datasets/
│   ├── datasets_4h/
│   ├── datasets_h48/
│   ├── external/
│   ├── models/
│   │   └── ablation81/
│   ├── models_4h/
│   ├── models_h24/
│   ├── models_h48/
│   ├── models_meta_dryrun/
│   └── 7 JSON files for state and history
└── log/
    ├── feature_isolation/
    ├── hmm_walkforward/
    ├── hmm_walkforward_fwd_long/
    ├── hmm_walkforward_fwd_long_backup_20260715_1343/
    ├── hmm_walkforward_fwd_smoke/
    ├── screener/
    ├── screener_runs/
    ├── senate/
    ├── senate_transcripts/
    └── application run logs
