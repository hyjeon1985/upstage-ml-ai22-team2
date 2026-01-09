# data_beautify.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, TypedDict, Union

import pandas as pd

__all__ = [
    "ColumnSpec",
    "ColumnGroup",
    "ColDict",
    "GroupDict",
    "parse_groups",
    "flatten_groups",
    "beautify_columns",
]


# -----------------------------
# 1) JSON / 문서형 선언을 위한 TypedDict
# -----------------------------
class ColDict(TypedDict, total=False):
    name: str
    desc: str
    alias: str


ColsLike = Union[str, ColDict]
ColsListLike = List[ColsLike]


class GroupDict(TypedDict, total=False):
    name: str
    desc: str
    cols: ColsListLike


# -----------------------------
# 2) Dataclass 스펙
# -----------------------------
@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """
    단일 컬럼에 대한 스펙.

    name  : 원본 DF의 실제 컬럼명
    desc  : EDA 문서용 설명(선택)
    alias : 반환 DF에서 사용할 컬럼명(선택)
    """

    name: str
    desc: str = ""
    alias: Optional[str] = None

    def out_name(self) -> str:
        return self.alias if (self.alias and self.alias.strip()) else self.name

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON에서 불필요한 null/빈값을 줄이고 싶으면 아래처럼 정리 가능
        if not d.get("desc"):
            d.pop("desc", None)
        if not d.get("alias"):
            d.pop("alias", None)
        return d


@dataclass(frozen=True, slots=True)
class ColumnGroup:
    """
    컬럼 그룹 스펙.

    name: 그룹명(또는 문서 상의 섹션명)
    desc: 그룹 설명(선택)
    cols: ColumnSpec 리스트(순서가 곧 출력 순서)
    """

    name: str
    desc: str = ""
    cols: List[ColumnSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "cols": [c.to_dict() for c in self.cols],
        }
        if self.desc:
            d["desc"] = self.desc
        return d


# -----------------------------
# 3) dict -> dataclass 변환 (문서형 선언 지원)
# -----------------------------
GroupLike = Union[ColumnGroup, Mapping[str, object]]


def _parse_col_like(item: ColsLike, *, group_name: str) -> ColumnSpec:
    # 단축 표기: "colname"
    if isinstance(item, str):
        col_name = item.strip()
        if not col_name:
            raise ValueError(
                f"Group '{group_name}': empty column name string is not allowed."
            )
        return ColumnSpec(name=col_name)

    # 상세 표기: {"name": "...", "desc": "...", "alias": "..."}
    if isinstance(item, Mapping):
        col_name = str(item.get("name", "")).strip()
        if not col_name:
            raise ValueError(
                f"Group '{group_name}': each col dict must have non-empty 'name'."
            )

        col_desc = str(item.get("desc", "") or "")

        alias = item.get("alias", None)
        alias_str = str(alias).strip() if alias is not None else None
        alias_final = alias_str or None

        return ColumnSpec(name=col_name, desc=col_desc, alias=alias_final)

    raise TypeError(
        f"Group '{group_name}': each col must be either a string column name "
        f"or a dict like {{'name': '...'}}."
    )


def parse_groups(groups: Iterable[GroupLike]) -> List[ColumnGroup]:
    """
    ColumnGroup | dict(GroupDict) 리스트를 ColumnGroup 리스트로 변환.

    cols는 아래 표기를 모두 지원:
    - ["train_dtype", "apt_dtype"]
    - [{"name": "train_dtype"}, {"name": "apt_dtype", "alias": "APT_DTYPE"}]
    - 혼합: ["a", {"name": "b", "desc": "..."}]
    """
    parsed: List[ColumnGroup] = []
    for g in groups:
        if isinstance(g, ColumnGroup):
            parsed.append(g)
            continue

        name = str(g.get("name", "")).strip()
        if not name:
            raise ValueError("Group dict must have non-empty 'name'.")

        desc = str(g.get("desc", "") or "")

        cols_raw = g.get("cols", [])
        if not isinstance(cols_raw, list):
            raise TypeError(f"Group '{name}': 'cols' must be a list.")

        cols: List[ColumnSpec] = []
        for item in cols_raw:
            cols.append(_parse_col_like(item, group_name=name))

        parsed.append(ColumnGroup(name=name, desc=desc, cols=cols))

    return parsed


# -----------------------------
# 4) Flatten + 검증 + Reorder
# -----------------------------
def flatten_groups(
    groups: Sequence[ColumnGroup],
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    그룹 스펙을 (원본컬럼명 리스트, 출력컬럼명 리스트, (src, out) 매핑)으로 펼칩니다.
    """
    src_cols: List[str] = []
    out_cols: List[str] = []
    mapping: List[Tuple[str, str]] = []

    for g in groups:
        for c in g.cols:
            src = c.name
            out = c.out_name()
            src_cols.append(src)
            out_cols.append(out)
            mapping.append((src, out))

    return src_cols, out_cols, mapping


def beautify_columns(
    df: pd.DataFrame,
    groups: Iterable[GroupLike],
    *,
    reset_index: bool = False,
    copy: bool = True,
) -> pd.DataFrame:
    """
    요구 조건에 맞게 DataFrame을 재구성합니다.

    - groups 순서대로 컬럼을 재배치
    - df에 없는 컬럼이 스펙에 있으면 예외
    - 스펙에 없는 df 컬럼은 무시(드랍)
    - 스펙에 지정된 컬럼만으로 구성된 새 DF 반환
    - alias가 있으면 반환 DF에서 컬럼명을 alias로 변경
    - reset_index=True면 index를 drop하고 0..N-1로 재설정
    """
    parsed = parse_groups(groups)
    src_cols, out_cols, mapping = flatten_groups(parsed)

    # 1) 스펙 중복(원본명) 검출: 정책 모호성 제거
    src_dupes = _find_duplicates(src_cols)
    if src_dupes:
        raise ValueError(f"Duplicated source column names in spec: {src_dupes}")

    # 2) 결과 컬럼명(alias 적용 후) 중복 검출
    out_dupes = _find_duplicates(out_cols)
    if out_dupes:
        raise ValueError(f"Duplicated output column names (after alias): {out_dupes}")

    # 3) 존재하지 않는 컬럼 검증
    df_col_set = set(df.columns)
    missing = [c for c in src_cols if c not in df_col_set]
    if missing:
        raise KeyError(
            f"Missing columns in DataFrame (declared but not found): {missing}"
        )

    # 4) 선택 + rename
    out = df.loc[:, src_cols]
    if copy:
        out = out.copy()

    rename_map = {src: out_name for (src, out_name) in mapping if src != out_name}
    if rename_map:
        out = out.rename(columns=rename_map)

    if reset_index:
        out = out.reset_index(drop=True)

    return out


def _find_duplicates(items: Sequence[str]) -> List[str]:
    seen = set()
    dupes: List[str] = []
    for x in items:
        if x in seen and x not in dupes:
            dupes.append(x)
        seen.add(x)
    return dupes
