# -*- coding: utf-8 -*-
import io, re
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

st.set_page_config(page_title="국민건강보험공단 비급여보고 코드 매핑도우미", layout="wide")

# ===================== 공통 유틸 =====================
def read_excel(file, sheet_name=0):
    return pd.read_excel(file, sheet_name=sheet_name, dtype=str).fillna("")

def normalize_code(s: str) -> str:
    if s is None: return ""
    return re.sub(r"\s+", "", str(s).strip().upper())

def normalize_ko(s: str) -> str:
    if s is None: return ""
    return re.sub(r"[\s\-/_,.:;()\[\]{}<>·•＋+※★☆]", "", str(s))

def to_excel_bytes(df: pd.DataFrame, sheet="result"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name=sheet)
    buf.seek(0)
    return buf

def find_col_case_insensitive(df: pd.DataFrame, wanted: str):
    wanted_norm = wanted.strip().lower()
    for c in df.columns:
        if c.strip().lower() == wanted_norm:
            return c
    return None

# ===================== 단계 1: (원내 edi ⊂ 공단 코드) 연속 포함 =====================
def stage1_map(wonnae: pd.DataFrame, gongdan: pd.DataFrame, edi_col: str):
    wn = wonnae.copy()
    gd = gongdan.copy()
    if "_row_id" not in wn.columns:
        wn["_row_id"] = range(len(wn))
    wn["edi_norm"] = wn[edi_col].apply(normalize_code)
    gd["코드_norm"] = gd["코드"].apply(normalize_code)

    out = []
    gd_view = gd[["코드", "코드_norm", "명칭(가이드)"]].values.tolist()
    for _, r in wn.iterrows():
        edi = r["edi_norm"]
        if not edi:
            continue
        matches = [(g_code, g_name) for (g_code, g_code_norm, g_name) in gd_view if edi in g_code_norm]
        if matches:
            for g_code, g_name in matches:
                base = r.drop(labels=["edi_norm"], errors="ignore").to_dict()
                base.update({
                    "매핑단계": "1단계(연속포함)",
                    "1단계_매핑코드": g_code,
                    "1단계_명칭(가이드)": g_name,
                    "제외": False,
                })
                out.append(base)

    cols_front = ["매핑단계", "1단계_매핑코드", "1단계_명칭(가이드)", "제외"]
    helper_cols = {"edi_norm"}
    if not out:
        ordered = cols_front + [c for c in wn.columns if c not in helper_cols]
        return pd.DataFrame(columns=ordered).loc[:, ordered].copy()

    res = pd.DataFrame(out)
    tail_cols = [c for c in wn.columns if c not in cols_front and c not in helper_cols and c in res.columns]
    res = res[cols_front + tail_cols + [c for c in res.columns if c not in cols_front + tail_cols]]
    return res

# ===================== 단계 2: (원내 한글명칭 ↔ 공단 명칭(가이드)) 유사도 =====================
def stage2_map(unmatched_wn: pd.DataFrame, gongdan: pd.DataFrame, score_cut: int, top_k: int, prev_best: dict|None):
    wn = unmatched_wn.copy()
    gd = gongdan.copy()
    if "_row_id" not in wn.columns:
        wn["_row_id"] = range(len(wn))

    wn["q_norm"] = wn["한글명칭"].apply(normalize_ko)
    gd["cand_norm"] = gd["명칭(가이드)"].apply(normalize_ko)

    lookup = {}
    for _, r in gd[["cand_norm", "코드", "명칭(가이드)"]].iterrows():
        lookup.setdefault(r["cand_norm"], []).append({"코드": r["코드"], "명칭(가이드)": r["명칭(가이드)"]})
    choices = list(lookup.keys())

    out = []
    for _, r in wn.iterrows():
        q = r["q_norm"]
        if not q:
            continue
        extracted = process.extract(q, choices, scorer=fuzz.token_set_ratio, limit=top_k)
        for cand_norm, score, _ in extracted:
            if score >= score_cut:
                for info in lookup.get(cand_norm, []):
                    base = r.drop(labels=["q_norm"], errors="ignore").to_dict()
                    base.update({
                        "매핑단계": "2단계(유사도)",
                        "2단계_매핑코드": info["코드"],
                        "2단계_명칭(가이드)": info["명칭(가이드)"],
                        "2단계_유사도": int(score),
                        "제외": False,
                    })
                    out.append(base)

    cols_front = ["매핑단계", "2단계_매핑코드", "2단계_명칭(가이드)", "2단계_유사도", "제외"]
    helper_cols = {"q_norm"}
    if not out:
        empty = pd.DataFrame(columns=cols_front + [c for c in wn.columns if c not in helper_cols])
        empty_styler = empty.style
        return empty, set(), {}, empty_styler

    res = pd.DataFrame(out)
    tail_cols = [c for c in wn.columns if c not in cols_front and c not in helper_cols and c in res.columns]
    res = res[cols_front + tail_cols + [c for c in res.columns if c not in cols_front + tail_cols]]

    res_best = res.sort_values(["_row_id", "2단계_유사도"], ascending=[True, False]).drop_duplicates("_row_id")
    current_best = {int(r["_row_id"]): (r["2단계_매핑코드"], int(r["2단계_유사도"])) for _, r in res_best.iterrows()}

    changed = set()
    if prev_best is not None:
        for k, cur in current_best.items():
            pre = prev_best.get(k)
            if pre is None or pre != cur:
                changed.add(k)
    else:
        changed = set(current_best.keys())

    def highlight_changed(row):
        key = int(row["_row_id"]) if "_row_id" in row else None
        color = "background-color: #fff3b0" if key in changed else ""
        return [color] * len(row)

    styled = res.style.apply(highlight_changed, axis=1)
    return res, changed, current_best, styled

# ===================== 단계 3: (원내 EDI 앞4자리 ⊂ 공단 코드) 연속 포함 =====================
def stage3_map(unmatched_after2: pd.DataFrame, gongdan: pd.DataFrame, edi_col: str):
    wn = unmatched_after2.copy()
    gd = gongdan.copy()
    if "_row_id" not in wn.columns:
        wn["_row_id"] = range(len(wn))

    wn["edi4_norm"] = wn[edi_col].apply(lambda x: normalize_code(x)[:4] if normalize_code(x) else "")
    gd["코드_norm"] = gd["코드"].apply(normalize_code)

    out = []
    gd_view = gd[["코드", "코드_norm", "명칭(가이드)"]].values.tolist()
    for _, r in wn.iterrows():
        k4 = r["edi4_norm"]
        if not k4:
            continue
        matches = [(g_code, g_name) for (g_code, g_code_norm, g_name) in gd_view if k4 in g_code_norm]
        if matches:
            for g_code, g_name in matches:
                base = r.drop(labels=["edi4_norm"], errors="ignore").to_dict()
                base.update({
                    "매핑단계": "3단계(EDI앞4 포함)",
                    "3단계_매핑코드": g_code,
                    "3단계_명칭(가이드)": g_name,
                    "제외": False,
                })
                out.append(base)

    cols_front = ["매핑단계", "3단계_매핑코드", "3단계_명칭(가이드)", "제외"]
    helper_cols = {"edi4_norm"}
    if not out:
        ordered = cols_front + [c for c in wn.columns if c not in helper_cols]
        return pd.DataFrame(columns=ordered).loc[:, ordered].copy()

    res = pd.DataFrame(out)
    tail_cols = [c for c in wn.columns if c not in cols_front and c not in helper_cols and c in res.columns]
    res = res[cols_front + tail_cols + [c for c in res.columns if c not in cols_front + tail_cols]]
    return res

# ===================== 세션 초기화 =====================
def init_session():
    defaults = {
        "s1_saved": None, "s2_saved": None, "s3_saved": None,  # 각 단계 저장본
        "s2_df": None, "s3_df": None,
        "unmatched2_from_stage2": None,
        "prev_best_map_stage2": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ===================== 헤더/사이드바 =====================
st.title("국민건강보험공단 비급여보고 코드 매핑도우미")
st.caption("체크만 하면 적용되지 않아요. 각 단계 표 위의 **[💾 (n단계) 제외처리 저장]** 버튼을 눌러야 다음 단계/최종결과에 반영됩니다.")
st.caption("공단파일은 그대로 업로드. 원내파일은 'EDI' / '한글명칭' / '수가코드' 컬럼 필요.")

with st.sidebar:
    st.image("ryoryo.png", width=120)
    st.markdown("<div style='text-align:center; font-weight:700;'>RYORYO</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("① 파일 업로드 (파일명 자유)")
    f_wonnae = st.file_uploader("원내 엑셀 업로드", type=["xlsx"])
    f_gongdan = st.file_uploader("공단 엑셀 업로드", type=["xlsx"])

    st.markdown("---")
    st.subheader("② 2단계 옵션")
    score_cut = st.slider("유사도 임계치(%)", 70, 100, 86, 1)
    top_k = st.slider("Top-K(후보 수)", 1, 5, 1, 1)
    run_stage2 = st.button("▶ 2단계 유사도 매핑 실행")

    st.markdown("---")
    st.subheader("③ 3단계 옵션")
    run_stage3 = st.button("▶ 3단계 (EDI 앞4 포함) 매핑 실행")

if not (f_wonnae and f_gongdan):
    st.info("좌측에서 '원내'와 '공단' 파일을 업로드 해주세요. (원내: EDI/한글명칭/수가코드, 공단: 코드/명칭(가이드))")
    st.stop()

# ===================== 파일 읽기/검증 =====================
df_wn = read_excel(f_wonnae)
if "수가코드" not in df_wn.columns:
    st.error("❌ 원내 파일에 '수가코드' 컬럼이 필요합니다.")
    st.stop()
df_gd = read_excel(f_gongdan)

edi_col = find_col_case_insensitive(df_wn, "edi")
if edi_col is None:
    st.error("❌ 원내 파일에 'edi' 컬럼이 필요합니다. (대소문자 무관)")
    st.stop()
if "한글명칭" not in df_wn.columns:
    df_wn["한글명칭"] = ""
if "코드" not in df_gd.columns or "명칭(가이드)" not in df_gd.columns:
    st.error("❌ 공단 파일에는 '코드'와 '명칭(가이드)' 컬럼이 필요합니다.")
    st.stop()
if "_row_id" not in df_wn.columns:
    df_wn["_row_id"] = range(len(df_wn))

# ===================== 1단계 =====================
with st.spinner("1단계 매핑 중..."):
    s1 = stage1_map(df_wn, df_gd, edi_col)

st.subheader("1단계 결과 — (원내 EDI ⊂ 공단 코드) 연속포함")
st.caption("여러 건 매칭되면 행 폭발. 체크 후 반드시 **[💾 (1단계) 제외처리 저장]**을 누르세요.")
height1 = st.slider("1단계 표 높이(px)", 300, 1200, 520, 20)
col_s1a, col_s1b, col_s1c = st.columns([1,1,5])
with col_s1a:
    s1_select_all = st.button("✅(1단계) 전체 제외")
with col_s1b:
    s1_clear_all  = st.button("❌(1단계) 제외 해제")
if len(s1):
    if s1_select_all: s1["제외"] = True
    if s1_clear_all:  s1["제외"] = False

s1_edit = st.data_editor(
    s1, use_container_width=True, height=height1, hide_index=True,
    column_config={"제외": st.column_config.CheckboxColumn(default=False)},
    key="s1_editor"
)

col_s1_save, _ = st.columns([1,5])
with col_s1_save:
    if st.button("💾 (1단계) 제외처리 저장"):
        st.session_state["s1_saved"] = s1_edit.copy()
        st.success("1단계 제외 저장 완료!")

st.download_button("⬇️ 1단계 결과 다운로드", to_excel_bytes(s1_edit, "stage1"), "stage1.xlsx")

# --- 2단계 대상 (1단계 저장본 기준) ---
s1_base_for_next = st.session_state["s1_saved"] if st.session_state["s1_saved"] is not None else None
if s1_base_for_next is None:
    st.info("👉 2단계로 넘어가려면 먼저 1단계 표에서 제외 체크 후 **[💾 (1단계) 제외처리 저장]**을 눌러주세요.")
    st.stop()

mapped_keys_stage1 = set()
if len(s1_base_for_next):
    mapped_keys_stage1 = set(s1_base_for_next.loc[s1_base_for_next["제외"] == False, "수가코드"].astype(str).unique())

unmatched1 = df_wn.loc[~df_wn["수가코드"].astype(str).isin(mapped_keys_stage1)].copy()
st.subheader("2단계 대상 — 1단계 미매핑(저장본 기준)")
st.write(f"행 수: {len(unmatched1)}")
height2t = st.slider("2단계 대상 표 높이(px)", 300, 1200, 420, 20, key="h2t")
st.dataframe(unmatched1, use_container_width=True, height=height2t)

# ===================== 2단계 실행/표시 =====================
if run_stage2:
    with st.spinner("2단계(유사도) 매핑 중..."):
        s2, changed_ids, current_best, _ = stage2_map(
            unmatched1, df_gd, score_cut=score_cut, top_k=top_k, prev_best=st.session_state.get("prev_best_map_stage2")
        )
    st.session_state["s2_df"] = s2.copy()
    st.session_state["prev_best_map_stage2"] = current_best

st.subheader("2단계 결과 — (원내 한글명칭 ↔ 공단 명칭(가이드)) 유사도")
s2 = st.session_state["s2_df"]
if s2 is None or len(s2) == 0:
    st.info("아직 2단계 결과가 없습니다. 사이드바에서 **[▶ 2단계 유사도 매핑 실행]**을 눌러주세요.")
    st.stop()
else:
    height2 = st.slider("2단계 표 높이(px)", 300, 1200, 520, 20, key="h2_show")
    col_s2a, col_s2b, col_s2c = st.columns([1,1,5])
    with col_s2a:
        s2_select_all = st.button("✅(2단계) 전체 제외")
    with col_s2b:
        s2_clear_all  = st.button("❌(2단계) 제외 해제")
    if s2_select_all: s2["제외"] = True
    if s2_clear_all:  s2["제외"] = False

    s2_edit = st.data_editor(
        s2, use_container_width=True, height=height2, hide_index=True,
        column_config={"제외": st.column_config.CheckboxColumn(default=False)},
        key="s2_editor"
    )

    col_s2_save, _ = st.columns([1,5])
    with col_s2_save:
        if st.button("💾 (2단계) 제외처리 저장"):
            st.session_state["s2_saved"] = s2_edit.copy()
            st.success("2단계 제외 저장 완료!")

    st.download_button("⬇️ 2단계 결과 다운로드", to_excel_bytes(s2_edit, "stage2"), "stage2.xlsx")

# --- 3단계 대상 (1·2단계 저장본 기준) ---
if st.session_state["s2_saved"] is None:
    st.info("👉 3단계로 넘어가려면 2단계 표에서 제외 체크 후 **[💾 (2단계) 제외처리 저장]**을 눌러주세요.")
    st.stop()

mapped_keys_stage2 = set()
tmp2_saved = st.session_state["s2_saved"]
if len(tmp2_saved):
    tmp2 = (tmp2_saved.loc[tmp2_saved["제외"] == False]
            .sort_values(["수가코드","2단계_유사도"], ascending=[True, False])
            .drop_duplicates("수가코드"))
    mapped_keys_stage2 = set(tmp2["수가코드"].astype(str).unique())

base_unmatched2 = df_wn.loc[
    ~df_wn["수가코드"].astype(str).isin(mapped_keys_stage1) &
    ~df_wn["수가코드"].astype(str).isin(mapped_keys_stage2)
].copy()

st.subheader("3단계 대상 — 저장된 1·2단계 결과 기준")
st.write(f"행 수: {len(base_unmatched2)}")
height3t = st.slider("3단계 대상 표 높이(px)", 300, 1200, 420, 20, key="h3_target")
st.dataframe(base_unmatched2, use_container_width=True, height=height3t)

# ===================== 3단계 실행/표시 =====================
if run_stage3:
    with st.spinner("3단계(EDI 앞4 포함) 매핑 중..."):
        s3 = stage3_map(base_unmatched2, df_gd, edi_col)
    st.session_state["s3_df"] = s3.copy()

st.subheader("3단계 결과 — (원내 EDI 앞4 ⊂ 공단 코드) 연속포함")
s3 = st.session_state["s3_df"]
if s3 is None or len(s3) == 0:
    st.info("아직 3단계 결과가 없습니다. 사이드바에서 **[▶ 3단계 (EDI 앞4 포함)]**을 눌러주세요.")
    st.stop()
else:
    height3 = st.slider("3단계 표 높이(px)", 300, 1200, 520, 20, key="h3_show")
    col_s3a, col_s3b, col_s3c = st.columns([1,1,5])
    with col_s3a:
        s3_select_all = st.button("✅(3단계) 전체 제외")
    with col_s3b:
        s3_clear_all  = st.button("❌(3단계) 제외 해제")
    if s3_select_all: s3["제외"] = True
    if s3_clear_all:  s3["제외"] = False

    s3_edit = st.data_editor(
        s3, use_container_width=True, height=height3, hide_index=True,
        column_config={"제외": st.column_config.CheckboxColumn(default=False)},
        key="s3_editor"
    )

    col_s3_save, _ = st.columns([1,5])
    with col_s3_save:
        if st.button("💾 (3단계) 제외처리 저장"):
            st.session_state["s3_saved"] = s3_edit.copy()
            st.success("3단계 제외 저장 완료!")

    st.download_button("⬇️ 3단계 결과 다운로드", to_excel_bytes(s3_edit, "stage3"), "stage3.xlsx")

# ===================== 최종 합본/미매핑 (저장본만 반영) =====================
if st.session_state["s3_saved"] is None:
    st.info("👉 최종 결과를 보려면 3단계 표에서 제외 체크 후 **[💾 (3단계) 제외처리 저장]**을 눌러주세요.")
    st.stop()

def best_stage1(saved_df: pd.DataFrame|None):
    if saved_df is None or len(saved_df)==0: return pd.DataFrame(columns=["수가코드","1단계_매핑코드","1단계_명칭(가이드)"])
    tmp = saved_df.loc[saved_df["제외"] == False]
    if len(tmp)==0: return pd.DataFrame(columns=["수가코드","1단계_매핑코드","1단계_명칭(가이드)"])
    return tmp.sort_values(["수가코드"]).drop_duplicates("수가코드")[["수가코드","1단계_매핑코드","1단계_명칭(가이드)"]]

def best_stage2(saved_df: pd.DataFrame|None):
    if saved_df is None or len(saved_df)==0: return pd.DataFrame(columns=["수가코드","2단계_매핑코드","2단계_명칭(가이드)","2단계_유사도"])
    tmp = (saved_df.loc[saved_df["제외"] == False]
                 .sort_values(["수가코드","2단계_유사도"], ascending=[True, False])
                 .drop_duplicates("수가코드"))
    if len(tmp)==0: return pd.DataFrame(columns=["수가코드","2단계_매핑코드","2단계_명칭(가이드)","2단계_유사도"])
    return tmp[["수가코드","2단계_매핑코드","2단계_명칭(가이드)","2단계_유사도"]]

def best_stage3(saved_df: pd.DataFrame|None):
    if saved_df is None or len(saved_df)==0: return pd.DataFrame(columns=["수가코드","3단계_매핑코드","3단계_명칭(가이드)"])
    tmp = saved_df.loc[saved_df["제외"] == False]
    if len(tmp)==0: return pd.DataFrame(columns=["수가코드","3단계_매핑코드","3단계_명칭(가이드)"])
    return tmp.sort_values(["수가코드"]).drop_duplicates("수가코드")[["수가코드","3단계_매핑코드","3단계_명칭(가이드)"]]

s1_best = best_stage1(st.session_state["s1_saved"])
s2_best = best_stage2(st.session_state["s2_saved"])
s3_best = best_stage3(st.session_state["s3_saved"])

final = df_wn.copy()
final["수가코드"] = final["수가코드"].astype(str)
for dfb in (s1_best, s2_best, s3_best):
    if "수가코드" in dfb.columns:
        dfb["수가코드"] = dfb["수가코드"].astype(str)

final = final.merge(s1_best, on="수가코드", how="left")
final = final.merge(s2_best, on="수가코드", how="left")
final = final.merge(s3_best, on="수가코드", how="left")

front = [
    "수가코드",
    "1단계_매핑코드", "1단계_명칭(가이드)",
    "2단계_매핑코드", "2단계_명칭(가이드)", "2단계_유사도",
    "3단계_매핑코드", "3단계_명칭(가이드)"
]
front_present = [c for c in front if c in final.columns]
final = final[front_present + [c for c in final.columns if c not in front_present]]

st.subheader("최종 결과 — (저장된 제외 기준) 원내 전체판 + 1/2/3단계 결과")
st.dataframe(final.head(30), use_container_width=True, height=520)
st.download_button("⬇️ 최종 결과 다운로드", to_excel_bytes(final, "final"), "final.xlsx")

# 미매핑 산출 (빈값/NaN 모두 비어있음으로 처리)
for c in ["1단계_매핑코드", "2단계_매핑코드", "3단계_매핑코드"]:
    if c not in final.columns: final[c] = ""
unmapped_final = final[
    (final["1단계_매핑코드"].fillna("") == "") &
    (final["2단계_매핑코드"].fillna("") == "") &
    (final["3단계_매핑코드"].fillna("") == "")
].copy()

st.subheader("미매핑 목록 — (세 단계 모두 미매핑, 저장본 기준)")
st.write(f"행 수: {len(unmapped_final)}")
st.dataframe(unmapped_final, use_container_width=True, height=480)
st.download_button("⬇️ 미매핑만 다운로드", to_excel_bytes(unmapped_final, "unmapped"), "unmapped.xlsx")

# ===================== 공단 미사용 목록만 표시 =====================
# 최종 결과에서 사용된 공단코드 집합 만들기
mapped_codes = set()
for col in ["1단계_매핑코드", "2단계_매핑코드", "3단계_매핑코드"]:
    if col in final.columns:
        mapped_codes |= set(final[col].dropna().astype(str).str.strip())

# 공백 제거 후 빈값 제거
mapped_codes = {c for c in mapped_codes if c != ""}

# 공단 파일에서 '코드'가 한번도 쓰이지 않은 행만 추출
unused_gd = df_gd[~df_gd["코드"].astype(str).str.strip().isin(mapped_codes)].copy()

st.subheader("공단 미사용 목록 — (최종 매핑에 쓰이지 않은 공단코드)")
st.write(f"행 수: {len(unused_gd)} / 전체: {len(df_gd)}")
height_unused = st.slider("공단 미사용 표 높이(px)", 300, 1200, 480, 20, key="h_gd_unused")
st.dataframe(unused_gd, use_container_width=True, height=height_unused)
st.download_button("⬇️ 공단 미사용만 다운로드", to_excel_bytes(unused_gd, "gongdan_unused"), "gongdan_unused.xlsx")

