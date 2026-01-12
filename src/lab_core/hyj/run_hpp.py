from __future__ import annotations

from lab_core.hyj.core.rf_lgbm_runner import ModelParams, RfLgbmRunner
from lab_core.hyj.hpp_dataset_builder import HPPDatasetBuilder


def main() -> None:
    # 여기서 옵션을 직접 바꿔가며 실행한다.
    model_mode = "ensemble"  # "lgbm" | "rf" | "ensemble"
    build_only = False
    seed = 42
    use_cache = True
    split_policy = {
        "kind": "time_holdout",
        "ym_col": "계약년월",
        "train_until": 202303,
        "valid_from": 202304,
        "valid_until": 202306,
    }

    params = ModelParams(
        lgbm={
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 255,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "reg_alpha": 0.05,
            "reg_lambda": 0.05,
        },
        rf={
            "n_estimators": 300,
            "max_depth": 24,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "n_jobs": -1,
        },
    )

    builder = HPPDatasetBuilder(target_transform="log1p")
    runner = RfLgbmRunner(builder, params=params)

    if build_only:
        builder.build(
            model="lgbm",
            use_cache=use_cache,
            split_policy=split_policy,
        )
        return

    if model_mode == "ensemble":
        runner.run_ensemble(
            weights={"lgbm": 0.7, "rf": 0.3},
            seed=seed,
            use_cache=use_cache,
            split_policy=split_policy,
        )
        return

    runner.run(
        model=model_mode,
        seed=seed,
        use_cache=use_cache,
        split_policy=split_policy,
    )


if __name__ == "__main__":
    main()
