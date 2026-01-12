from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from lab_core.util.path import out_dir
from lab_core.util.time_ids import make_run_id

from .dataset.builder import DatasetBundle, ModelKind
from .runner import BaseRunner, RunResult


@dataclass(frozen=True)
class ModelParams:
    lgbm: dict[str, Any]
    rf: dict[str, Any]


class RfLgbmRunner(BaseRunner):
    """
    RF/LGBM 전용 러너.
    """

    def __init__(self, builder, *, params: ModelParams) -> None:
        super().__init__(builder)
        self.params = params

    def train_predict(
        self,
        bundle: DatasetBundle,
        *,
        model: Optional[ModelKind],
        seed: int,
    ) -> tuple[pd.Series, dict[str, Any]]:
        if model not in {"lgbm", "rf"}:
            raise ValueError(f"unknown model: {model}")

        X_train = bundle.X_train
        y_train = bundle.y_train
        X_test = bundle.X_test
        X_valid = bundle.X_valid
        y_valid = bundle.y_valid

        target_transform = bundle.meta.get("target_transform", "none")
        target_params = bundle.meta.get("target_trans_params", {})

        artifacts: dict[str, Any] = {"model_tag": model}
        if model == "lgbm":
            params = dict(self.params.lgbm)
            params["random_state"] = seed
            print("[runner] train lgbm")
            model_obj = LGBMRegressor(**params)
            best_iter = None
            if X_valid is not None and y_valid is not None:
                model_obj.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric="rmse",
                    callbacks=[
                        early_stopping(stopping_rounds=50),
                        log_evaluation(period=100),
                    ],
                )
                best_iter = getattr(model_obj, "best_iteration_", None)
            else:
                model_obj.fit(X_train, y_train)
        else:
            params = dict(self.params.rf)
            params["random_state"] = seed
            print("[runner] train rf")
            model_obj = RandomForestRegressor(**params)
            model_obj.fit(X_train, y_train)

        if X_valid is not None and y_valid is not None:
            val_pred = np.asarray(model_obj.predict(X_valid))
            val_pred_series = pd.Series(val_pred)
            val_pred_out = self.builder.inverse_target(
                val_pred_series, transform=target_transform, params=target_params
            )
            y_valid_out = self.builder.inverse_target(
                y_valid, transform=target_transform, params=target_params
            )
            rmse = float(
                mean_squared_error(
                    np.asarray(y_valid_out), np.asarray(val_pred_out), squared=False
                )
            )
            artifacts["valid_rmse"] = rmse
            artifacts["valid_pred"] = val_pred_out
            artifacts["valid_y"] = y_valid_out
            print(f"[runner] valid rmse: {rmse:,.0f}")

        # valid가 있으면 train+valid로 재학습 후 test 예측
        if X_valid is not None and y_valid is not None:
            print("[runner] refit with train+valid")
            X_full = pd.concat([X_train, X_valid], axis=0)
            y_full = pd.concat([y_train, y_valid], axis=0)
            if model == "lgbm" and best_iter:
                params_full = dict(self.params.lgbm)
                params_full["random_state"] = seed
                params_full["n_estimators"] = int(best_iter)
                model_obj = LGBMRegressor(**params_full)
            elif model == "lgbm":
                params_full = dict(self.params.lgbm)
                params_full["random_state"] = seed
                model_obj = LGBMRegressor(**params_full)
            else:
                params_full = dict(self.params.rf)
                params_full["random_state"] = seed
                model_obj = RandomForestRegressor(**params_full)
            model_obj.fit(X_full, y_full)

        pred = np.asarray(model_obj.predict(X_test))
        pred_series = pd.Series(pred)
        pred_out = self.builder.inverse_target(
            pred_series, transform=target_transform, params=target_params
        )

        return pred_out, artifacts

    def run_ensemble(
        self,
        *,
        weights: dict[str, float],
        seed: int = 42,
        use_cache: bool = True,
        split_policy: dict[str, Any] | None = None,
        run_prefix: str = "submission",
        out_subdir: str = "subs",
    ) -> RunResult:
        print(f"[runner] ensemble start: seed={seed}")
        bundle_lgbm = self.build_dataset(
            model="lgbm", use_cache=use_cache, split_policy=split_policy
        )
        bundle_rf = self.build_dataset(
            model="rf", use_cache=use_cache, split_policy=split_policy
        )
        print("[runner] ensemble dataset ready")
        pred_lgbm, art_lgbm = self.train_predict(bundle_lgbm, model="lgbm", seed=seed)
        pred_rf, art_rf = self.train_predict(bundle_rf, model="rf", seed=seed)

        w_lgbm = float(weights.get("lgbm", 0.5))
        w_rf = float(weights.get("rf", 0.5))
        total = w_lgbm + w_rf
        if total == 0:
            raise ValueError("ensemble weights sum to zero")
        w_lgbm /= total
        w_rf /= total
        weight_tag = f"w{int(round(w_lgbm * 100))}-{int(round(w_rf * 100))}"
        lgbm_tag = self._model_param_tag("lgbm", self.params.lgbm)
        rf_tag = self._model_param_tag("rf", self.params.rf)

        ensemble = (w_lgbm * pred_lgbm) + (w_rf * pred_rf)
        artifacts = {
            "model_tag": "ensemble",
            "weights": {"lgbm": w_lgbm, "rf": w_rf},
            "component": {
                "lgbm": art_lgbm,
                "rf": art_rf,
            },
        }

        valid_pred_lgbm = art_lgbm.get("valid_pred")
        valid_pred_rf = art_rf.get("valid_pred")
        valid_y = art_lgbm.get("valid_y")
        if (
            isinstance(valid_pred_lgbm, pd.Series)
            and isinstance(valid_pred_rf, pd.Series)
            and isinstance(valid_y, pd.Series)
            and len(valid_pred_lgbm) == len(valid_pred_rf) == len(valid_y)
        ):
            valid_ens = (w_lgbm * valid_pred_lgbm) + (w_rf * valid_pred_rf)
            valid_rmse = float(
                mean_squared_error(
                    np.asarray(valid_y), np.asarray(valid_ens), squared=False
                )
            )
            artifacts["valid_rmse"] = valid_rmse
            print(f"[runner] ensemble valid rmse: {valid_rmse:,.0f}")

        run_id = make_run_id(run_prefix, mid_id=self.__class__.__name__)
        run_dir = out_dir(out_subdir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pred_path = run_dir / f"{run_id}_ens_{weight_tag}.csv"
        pred_lgbm_path = run_dir / f"{run_id}_lgbm{lgbm_tag}.csv"
        pred_rf_path = run_dir / f"{run_id}_rf{rf_tag}.csv"
        meta_path = run_dir / "meta.json"

        pd.DataFrame({"target": BaseRunner._to_int_pred(ensemble)}).to_csv(
            pred_path, index=False
        )
        pd.DataFrame({"target": BaseRunner._to_int_pred(pred_lgbm)}).to_csv(
            pred_lgbm_path, index=False
        )
        pd.DataFrame({"target": BaseRunner._to_int_pred(pred_rf)}).to_csv(
            pred_rf_path, index=False
        )
        meta = dict(bundle_lgbm.meta)
        meta["ensemble"] = artifacts["weights"]
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[runner] ensemble saved: {pred_path}")

        return RunResult(
            pred=ensemble,
            meta=meta,
            artifacts=artifacts,
            run_id=run_id,
            run_dir=run_dir,
            pred_path=pred_path,
            meta_path=meta_path,
        )

    @staticmethod
    def _model_param_tag(model: str, params: dict[str, Any]) -> str:
        if model == "lgbm":
            mapping = {
                "n_estimators": "n",
                "learning_rate": "lr",
                "num_leaves": "l",
                "min_data_in_leaf": "m",
                "feature_fraction": "ff",
                "bagging_fraction": "bf",
                "reg_alpha": "ra",
                "reg_lambda": "rl",
            }
        else:
            mapping = {
                "n_estimators": "n",
                "max_depth": "d",
                "min_samples_split": "s",
                "min_samples_leaf": "l",
                "max_features": "f",
            }
        parts = []
        for k, short in mapping.items():
            if k not in params:
                continue
            v = params[k]
            if isinstance(v, float):
                v_str = f"{v:.3g}".replace(".", "")
            else:
                v_str = str(v)
            parts.append(f"_{short}{v_str}")
        return "".join(parts)
