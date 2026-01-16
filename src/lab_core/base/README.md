# Base 구조 사용 예시

아래는 `base` 패키지의 최소 사용 흐름 예시입니다. 실제 도메인 구현에서는 `BaseDataSource`, `Pipeline`, `BaseRunner`를 구체 클래스로 확장합니다.

## 1) 데이터셋 정의

```python
from lab_core.base.dataset import BaseDataSource, DataSpec

class MyDataSource(BaseDataSource):
    def load_train(self):  # pd.DataFrame
        return train_df

    def load_test(self):  # pd.DataFrame
        return test_df

    def spec(self) -> DataSpec:
        return DataSpec(target_col="target")
```

## 2) 파이프라인 정의

```python
from lab_core.base.pipeline import BaseBlock, Pipeline

class MyBlock(BaseBlock):
    def transform(self, X, *, is_train):
        out = X.copy()
        # 전처리 로직
        return out

    def describe(self) -> dict[str, str]:
        return {"desc": "example"}

pipeline = Pipeline([
    MyBlock(meta_cols=("_org_idx",)),
])
```

## 3) 러너 정의

```python
from lab_core.base.runner import BaseRunner

class MyRunner(BaseRunner):
    def __init__(self, *, name=None, data_source: MyDataSource, pipeline: Pipeline):
        super().__init__(name=name)
        self.data_source = data_source
        self.pipeline = pipeline

    def run(self, **kwargs):
        train = self.data_source.get_train()
        test = self.data_source.get_test()
        spec = self.data_source.get_spec()
        # 여기서 split/fit/transform/학습/예측을 구성
        return {"ok": True}
```

## 4) 오케스트레이터

```python
from lab_core.base.runner import RunOrchestrator

runner = MyRunner(name="model-a", data_source=data_source, pipeline=pipeline)
orch = RunOrchestrator([runner])
results = orch.run_all()
```
