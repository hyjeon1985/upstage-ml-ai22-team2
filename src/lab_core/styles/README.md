# Styles

이 도구 패키지는 본 프로젝트의 시각화(그래프) 스타일 설정을 위한 보조 구성 요소입니다.

모델 학습/추론 로직에는 영향을 주지 않으며, **데이터 시각화의 가독성 향상**만을 다룹니다.

전역 스타일 설정에서는 **단순 한국어 표시 오류**만 바로 잡고 있으며,
프리셋 스타일 설정을 컨텍스트 매니저(context manager) 방식으로 제공합니다.

## 사용 방법

### 폰트 설치

```bash
bash script/install_korean_fonts.sh
```

- `bootstrap.sh` 스크립트 실행 과정에 포함되어 있음.

### 전역 한국어 폰트 설정 (권장: 노트북 상단 1회)

```python
from lab_.styles.viz import setup_global
setup_global()
```

- 한글 깨짐 방지
- 마이너스 기호(-) 깨짐 방지
- 폰트 패밀리 자동 선택(OS fallback 지원)

이 설정은 커널 전역에 적용해도 안전합니다.

### 프리셋 스타일 적용

```python
import matplotlib.pyplot as plt
from lab_.styles.viz import use_style, Preset

with use_style(preset=Preset.LIGHT):
    plt.figure()
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.title("Light Style Example")
    plt.show()
```

또는 Preset.DARK:

```python
with use_style(preset=Preset.DARK):
    plt.figure()
    plt.plot([1, 2, 3], [3, 1, 5])
    plt.title("Dark Style Example")
    plt.show()
```

- `with` 컨텍스트 블록 내부에서만 스타일 적용
- 블록 종료 시 matplotlib 스타일은 원래 상태로 복구
- 다른 노트북/셀의 작업에 영향 없음

### 주의사항

- `use_style()` 외부에서는 matplotlib 설정이 변경되지 않습니다.
- 그래프 스타일을 적용하려면 반드시 `with use_style(...):` 블록 안에서
  `plt.plot`, `plt.show()` 등을 호출해야 합니다.
