"""
Common Data Preprocessing for Code Corpus
공통 - 주석 제거, 변수명 익명화, 포맷팅, 중복 제거

Usage:
    python preprocessing/preprocess_corpus.py
"""

import io, ast, re, hashlib, tokenize, builtins, warnings
import pandas as pd
from typing import Dict, List, Set
from tqdm.auto import tqdm

try:
    import black
except ImportError:
    print("⚠️ black 패키지가 없습니다. pip install black 권장")
    black = None

warnings.filterwarnings("ignore", category=SyntaxWarning)

# =============================
# 설정
# =============================
INPUT_PARQUET = "./data/code_corpus.parquet"
OUTPUT_PARQUET = "./data/code_corpus_processed.parquet"
TEXT_COL = "text"


# =========================================
# 1) 주석 제거
# =========================================
def remove_comments(code: str) -> str:
    """Python 코드에서 주석 제거"""
    try:
        tokgen = tokenize.generate_tokens(io.StringIO(code).readline)
        out_tokens = [(tt, ts) for tt, ts, *_ in tokgen if tt != tokenize.COMMENT]
        return tokenize.untokenize(out_tokens)
    except Exception:
        # fallback: 단순 # 제거
        stripped_lines = []
        for ln in code.splitlines():
            idx = ln.find("#")
            if idx != -1:
                prefix = ln[:idx]
                if prefix.count('"') % 2 == 0 and prefix.count("'") % 2 == 0:
                    ln = prefix.rstrip()
            stripped_lines.append(ln)
        return "\n".join(stripped_lines)


# =========================================
# 2) AST 기반 스코프 일관 치환 (Alpha Renaming)
# =========================================
_BUILTINS: Set[str] = set(dir(builtins))


class Scope:
    """변수 스코프 관리"""
    
    def __init__(self, kind: str):
        self.kind = kind
        self.map: Dict[str, str] = {}
        self.counter_var = 0
        self.counter_global = 0
        self.protected: Set[str] = set()

    def new_var(self):
        self.counter_var += 1
        return f"v{self.counter_var}"

    def new_global(self):
        self.counter_global += 1
        return f"g{self.counter_global}"


class AlphaRenamer(ast.NodeTransformer):
    """
    변수명/함수명 익명화
    - 함수: func1, func2, ...
    - 클래스: Cls1, Cls2, ...
    - 변수: v1, v2, ... (로컬), g1, g2, ... (글로벌)
    """
    
    def __init__(self):
        super().__init__()
        self.scopes: List[Scope] = [Scope("module")]
        self.func_counter = 0
        self.cls_counter = 0

    @property
    def scope(self):
        return self.scopes[-1]

    def push(self, kind):
        self.scopes.append(Scope(kind))

    def pop(self):
        self.scopes.pop()

    def _protect_name(self, name):
        self.scope.protected.add(name)

    def _is_protected(self, name):
        if name in _BUILTINS:
            return True
        return any(name in sc.protected for sc in reversed(self.scopes))

    def _lookup(self, name):
        for sc in reversed(self.scopes):
            if name in sc.map:
                return sc.map[name]
        return name

    def _ensure_binding(self, name, is_module_level=False, allow_protected=False):
        if not allow_protected and self._is_protected(name):
            return name
        if name in self.scope.map:
            return self.scope.map[name]
        alias = (
            self.scope.new_global()
            if (self.scope.kind == "module" and is_module_level)
            else self.scope.new_var()
        )
        self.scope.map[name] = alias
        return alias

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            alias = self._ensure_binding(
                node.id,
                is_module_level=(self.scope.kind == "module"),
                allow_protected=True,
            )
            return ast.copy_location(ast.Name(id=alias, ctx=node.ctx), node)
        else:
            if self._is_protected(node.id):
                return node
            alias = self._lookup(node.id)
            return ast.copy_location(ast.Name(id=alias, ctx=node.ctx), node)

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self._protect_name(name)
        return node

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self._protect_name(name)
        return node

    def visit_FunctionDef(self, node):
        self.func_counter += 1
        node.name = f"func{self.func_counter}"
        self.push("function")
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            arg.arg = self._ensure_binding(arg.arg, allow_protected=True)
        if node.args.vararg:
            node.args.vararg.arg = self._ensure_binding(
                node.args.vararg.arg, allow_protected=True
            )
        if node.args.kwarg:
            node.args.kwarg.arg = self._ensure_binding(
                node.args.kwarg.arg, allow_protected=True
            )
        self.generic_visit(node)
        self.pop()
        return node

    def visit_ClassDef(self, node):
        self.cls_counter += 1
        node.name = f"Cls{self.cls_counter}"
        self.push("class")
        self.generic_visit(node)
        self.pop()
        return node

    def _bind_target(self, target, is_module_level=False):
        if isinstance(target, ast.Name):
            self._ensure_binding(
                target.id, is_module_level=is_module_level, allow_protected=True
            )
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind_target(elt, is_module_level=is_module_level)

    def visit_Assign(self, node):
        self.generic_visit(node.value)
        for t in node.targets:
            self._bind_target(t, is_module_level=(self.scope.kind == "module"))
            self.visit(t)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node.value)
        return node


def alpha_rename(code: str) -> str:
    """AST 기반 변수명/함수명 익명화"""
    try:
        tree = ast.parse(code)
        tree = AlphaRenamer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return code


# =========================================
# 3) 전처리 파이프라인
# =========================================
def normalize_identifiers(code: str) -> str:
    """
    전처리 파이프라인:
    1. 주석 제거
    2. 변수명/함수명 익명화
    3. 연속 개행 정리
    4. Black 포맷팅 (optional)
    """
    # 1. 주석 제거
    code = remove_comments(code)
    
    # 2. 변수명/함수명 익명화
    code = alpha_rename(code)
    
    # 3. 연속 개행을 1개로 줄임
    code = re.sub(r"\n\s*\n+", "\n\n", code)
    
    # 4. Black 포맷팅 (마지막에 적용)
    if black is not None:
        try:
            code = black.format_str(code, mode=black.Mode(line_length=88))
        except Exception:
            pass
    
    return code


# =========================================
# 4) DataFrame 단위 처리 + 중복 제거
# =========================================
def process_corpus(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    코퍼스 전처리:
    - text_norm: 정규화된 코드
    - text_norm_sha1: 중복 체크용 해시
    - n_chars_norm: 문자 수
    - n_lines_norm: 라인 수
    """
    print(f"\n{'=' * 70}")
    print("코퍼스 전처리")
    print(f"{'=' * 70}")
    print(f"입력 샘플 수: {len(df):,}개")
    
    out = df.copy()
    
    # 정규화
    print(f"\n정규화 진행 중...")
    tqdm.pandas(desc="Normalizing")
    out["text_norm"] = out[text_col].astype(str).progress_apply(normalize_identifiers)
    
    # 메타 정보
    print(f"\n메타 정보 생성 중...")
    out["text_norm_sha1"] = out["text_norm"].apply(
        lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()
    )
    out["n_chars_norm"] = out["text_norm"].str.len()
    out["n_lines_norm"] = out["text_norm"].apply(
        lambda s: s.count("\n") + 1 if s else 0
    )

    # 중복 제거 (정규화 후 동일 코드 제거)
    print(f"\n중복 제거 중...")
    before = len(out)
    out = out.drop_duplicates(subset=["text_norm_sha1"]).reset_index(drop=True)
    after = len(out)
    
    print(f"\n✓ 중복 제거 완료")
    print(f"  제거된 중복: {before - after:,}개")
    print(f"  최종 샘플: {after:,}개 ({after/before*100:.1f}%)")

    return out


# =========================================
# Main
# =========================================
def main():
    print("=" * 70)
    print("Code Corpus Preprocessing")
    print("공통 - 주석 제거, 익명화, 포맷팅, 중복 제거")
    print("=" * 70)
    
    print(f"\n설정:")
    print(f"  입력: {INPUT_PARQUET}")
    print(f"  출력: {OUTPUT_PARQUET}")
    print(f"  텍스트 컬럼: {TEXT_COL}")
    
    # 데이터 로드
    print(f"\n{'=' * 70}")
    print("데이터 로드")
    print(f"{'=' * 70}")
    
    try:
        corpus = pd.read_parquet(INPUT_PARQUET)
        print(f"✓ 로드 완료: {len(corpus):,}개 샘플")
        print(f"  컬럼: {corpus.columns.tolist()}")
    except FileNotFoundError:
        print(f"✗ 파일을 찾을 수 없습니다: {INPUT_PARQUET}")
        return
    except Exception as e:
        print(f"✗ 로드 실패: {e}")
        return
    
    # 전처리
    corpus_processed = process_corpus(corpus, text_col=TEXT_COL)
    
    # 저장
    print(f"\n{'=' * 70}")
    print("결과 저장")
    print(f"{'=' * 70}")
    
    corpus_processed.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✓ 저장 완료: {OUTPUT_PARQUET}")
    
    # 샘플 출력
    print(f"\n{'=' * 70}")
    print("샘플 미리보기")
    print(f"{'=' * 70}")
    
    if len(corpus_processed) > 0:
        sample = corpus_processed.iloc[0]
        print(f"\n원본 (처음 200자):")
        print(sample[TEXT_COL][:200])
        print(f"\n정규화 후 (처음 200자):")
        print(sample["text_norm"][:200])
        print(f"\nSHA1: {sample['text_norm_sha1']}")
        print(f"문자 수: {sample['n_chars_norm']:,}")
        print(f"라인 수: {sample['n_lines_norm']:,}")
    
    print(f"\n{'=' * 70}")
    print("✓ 모든 작업 완료!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
