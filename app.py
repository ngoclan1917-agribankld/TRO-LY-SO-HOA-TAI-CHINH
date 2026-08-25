import io
import json
import re
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

APP_TITLE = "TRỢ LÝ TÀI CHÍNH NHỎ"
MODEL_NAME = "gemini-2.5-flash"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CSS / GIAO DIỆN
# =========================
st.markdown(
    """
<style>
:root {
    --red: #b71c1c;
    --red2: #d32f2f;
    --green: #1b5e20;
    --blue: #1565c0;
    --gold: #f9a825;
    --line: #d7dce1;
    --soft: #f7f8fa;
    --text: #202124;
}
.block-container { padding-top: 0.08rem; max-width: 1450px; }
.app-title {
    color: var(--red); text-align: center; font-size: 2.05rem;
    line-height: 1.15;
    font-weight: 900; margin: 0 0 0.65rem 0; letter-spacing: .3px;
}
.app-note { text-align:center; color:#60656b; font-size:.92rem; margin-bottom:1rem; }
.section-header {
    font-size: 1.45rem; font-weight: 900; color: var(--red);
    border-bottom: 2px solid var(--red); padding-bottom:.35rem; margin:.35rem 0 .8rem;
}
.input-title, .stTextInput label, .stNumberInput label, .stDateInput label,
.stSelectbox label, .stFileUploader label, .stButton button, .stDownloadButton button {
    font-weight: 800 !important;
}
.card {
    border: 1px solid var(--line); border-radius: 14px; padding: 1rem;
    background: #fff; box-shadow: 0 4px 12px rgba(0,0,0,.07); height:100%;
}
.card:hover { transform: translateY(-2px); box-shadow:0 8px 20px rgba(0,0,0,.12); }
.card-title { color:var(--red); font-size:1.28rem; font-weight:900; margin-bottom:.5rem; }
.card-text { line-height:1.55; }
.rule-box {
    border:1px solid #e0bdbd; border-left:5px solid var(--red); border-radius:12px;
    padding:1rem 1.1rem; background:#fff8f8; margin:.8rem 0 1rem;
}
.data-box { border:1px solid var(--line); border-radius:12px; padding:.7rem; background:var(--soft); }
.metric-box { border:1px solid var(--line); border-radius:12px; padding:.75rem .9rem; background:#fff; min-height:90px; }
.metric-label { color:#656b73; font-size:.88rem; font-weight:800; }
.metric-value { color:#111827; font-size:1.25rem; font-weight:900; margin-top:.2rem; }
.term-title { font-size:1.1rem; font-weight:900; color:var(--red); }
.term-desc { font-size:.92rem; line-height:1.5; margin:.25rem 0 .65rem; }
.small-note { color:#666; font-size:.83rem; }
.ai-box { border:1px solid #c8d7eb; border-left:5px solid var(--blue); padding:.9rem 1rem; border-radius:10px; background:#f6f9ff; }
.sidebar-title { font-size:1.45rem !important; font-weight:900 !important; color:var(--red) !important; }
.sidebar-help { font-size:.88rem; line-height:1.55; color:#555; }
div[data-testid="stVerticalBlock"] .stMetric { background:transparent; }
/* Ẩn dấu +/- của number_input nếu còn component cũ dùng */
button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] { display:none !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)

# =========================
# HÀM TIỆN ÍCH
# =========================
def money(v):
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v):,.0f}".replace(",", ".") + " đ"


def compact_money(v):
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v):,.0f}".replace(",", ".") + " đ"


def percent(v):
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v) * 100:.2f}%"


def clean_money_text(value):
    """Nhận số tiền nhập thủ công không dấu; cũng chấp nhận dữ liệu cũ có . , hoặc khoảng trắng."""
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    s = s.replace("đ", "").replace("Đ", "").replace(" ", "")
    # Chỉ nhận số nguyên dương trong giao diện; loại bỏ mọi ký tự không phải số.
    s = re.sub(r"[^0-9]", "", s)
    return float(s) if s else 0.0


def fmt_money_input(value):
    value = int(round(float(value))) if value else 0
    return f"{value:,}".replace(",", ".")


def _normalize_money_widget(key):
    """Chuẩn hóa nội dung ô tiền sau mỗi lần người dùng kết thúc lượt nhập."""
    raw = st.session_state.get(key, "")
    digits = re.sub(r"\D", "", str(raw))
    digits = digits.lstrip("0") or "0"
    st.session_state[key] = f"{int(digits):,}".replace(",", ".")


def money_input(label, key, default=0):
    """Ô nhập tiền an toàn:
    - Mặc định hiển thị 0.
    - Người dùng chỉ cần nhập chữ số, không cần tự gõ dấu phân cách.
    - Khi kết thúc lượt nhập, hệ thống tự bỏ số 0 thừa ở đầu và định dạng
      thành 1.000, 1.000.000, 1.000.000.000...
    - Không có nút +/- và không dùng number_input format, tránh
      StreamlitInvalidNumberFormatError.
    """
    if key not in st.session_state:
        st.session_state[key] = fmt_money_input(default)

    st.text_input(
        label,
        key=key,
        on_change=_normalize_money_widget,
        args=(key,),
    )
    return clean_money_text(st.session_state.get(key, "0"))


def get_gemini_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def ai_available():
    return genai is not None and bool(get_gemini_key())


@st.cache_resource
def get_gemini_client():
    key = get_gemini_key()
    if not key or genai is None:
        return None
    return genai.Client(api_key=key)


def call_gemini(instruction, data):
    client = get_gemini_client()
    if client is None:
        return "Chưa cấu hình Gemini API. Bạn vẫn có thể sử dụng toàn bộ phần tính toán."
    system_instruction = """
Bạn là trợ lý tài chính bình dân cho tiểu thương, hộ kinh doanh, nông hộ,
cơ sở sản xuất nhỏ và hợp tác xã.
Chỉ sử dụng số liệu được cung cấp; không tự tạo số liệu. Phân biệt rõ Kết quả tính toán,
Nhận xét và Cảnh báo. Giải thích thuật ngữ bằng tiếng Việt dễ hiểu. Không khẳng định
chắc chắn lợi nhuận tương lai và không thay thế kế toán, kiểm toán hoặc thẩm định chuyên môn.
"""
    prompt = instruction + "\n\nDỮ LIỆU:\n" + json.dumps(data, ensure_ascii=False, indent=2, default=str)
    try:
        cfg = types.GenerateContentConfig(system_instruction=system_instruction) if types else None
        res = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=cfg)
        return res.text or "AI không trả về nội dung."
    except Exception as exc:
        return f"Không thể gọi Gemini API: {exc}"


def dataframe_to_excel_bytes(sheets):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=str(name)[:31], index=False)
    return out.getvalue()


def npv(rate, cashflows):
    return float(sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows)))


def irr_bisection(cashflows):
    vals = np.asarray(cashflows, dtype=float)
    if len(vals) < 2 or not (np.any(vals > 0) and np.any(vals < 0)):
        return None
    grid = np.concatenate([np.linspace(-0.99, -0.01, 100), np.linspace(0, 5, 300)])
    prev_r = float(grid[0]); prev_v = npv(prev_r, cashflows)
    for r in grid[1:]:
        r = float(r); curr_v = npv(r, cashflows)
        if np.isfinite(prev_v) and np.isfinite(curr_v) and prev_v * curr_v <= 0:
            lo, hi = prev_r, r
            flo = prev_v
            for _ in range(120):
                mid = (lo + hi) / 2
                fm = npv(mid, cashflows)
                if abs(fm) < 1e-9:
                    return float(mid)
                if flo * fm <= 0:
                    hi = mid
                else:
                    lo, flo = mid, fm
            return float((lo + hi) / 2)
        prev_r, prev_v = r, curr_v
    return None


def payback_period(cashflows):
    cum = float(cashflows[0])
    if cum >= 0:
        return 0.0
    for i in range(1, len(cashflows)):
        prev = cum
        cum += cashflows[i]
        if cum >= 0:
            step = cashflows[i]
            return float(i) if step == 0 else (i - 1) + min(max((-prev) / step, 0), 1)
    return None


def compute_wacc(e, d, ke, kd, tax):
    total = e + d
    if total <= 0:
        return None
    e, d = e / total, d / total
    return e * ke + d * kd * (1 - tax)


def navigate(page_name):
    st.session_state["page"] = page_name
    st.rerun()

# =========================
# DANH MỤC
# =========================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Tổng quan"

with st.sidebar:
    st.markdown('<div class="sidebar-title">DANH MỤC</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-help"><b>Hướng dẫn nhanh:</b><br>'
        'Chọn một mục → nhập dữ liệu → xem kết quả.<br><br>'
        'Nút AI chỉ dùng khi cần giải thích hoặc nhận xét.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    pages = {
        "🏠 Tổng quan": "TỔNG QUAN",
        "💰 Sổ tay Dòng tiền": "SỔ TAY DÒNG TIỀN",
        "⚙️ Tính Khấu hao": "TÍNH KHẤU HAO",
        "📈 Đánh giá Hiệu quả Đầu tư": "ĐÁNH GIÁ HIỆU QUẢ ĐẦU TƯ",
    }
    selected = st.radio(
        "", list(pages.keys()), index=list(pages.keys()).index(st.session_state.page),
        label_visibility="collapsed",
    )
    st.session_state.page = selected

# =========================
# TỔNG QUAN
# =========================
if st.session_state.page == "🏠 Tổng quan":
    st.markdown('<div class="section-header">TỔNG QUAN CHƯƠNG TRÌNH</div>', unsafe_allow_html=True)
    cards = st.columns(3)
    card_data = [
        ("💰 SỔ TAY DÒNG TIỀN", "Ghi lại từng khoản thu và chi, theo dõi tổng tiền vào, tổng tiền ra và dòng tiền ròng", "💰 Sổ tay Dòng tiền"),
        ("⚙️ TÍNH KHẤU HAO", "Tính khấu hao đường thẳng cho máy móc, thiết bị và tài sản.", "⚙️ Tính Khấu hao"),
        ("📈 ĐÁNH GIÁ ĐẦU TƯ", "Tính NPV- Giá trị hiện tại ròng, IRR - Tỷ suất hoàn vốn nội bộ, thời gian hoàn vốn và WACC- Chi phí sử dụng vốn bình quân gia quyền khi đủ dữ liệu", "📈 Đánh giá Hiệu quả Đầu tư"),
    ]
    for col, (title, desc, target) in zip(cards, card_data):
        with col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if st.button(title, key=f"card_{target}", use_container_width=True):
                navigate(target)
            st.markdown(f'<div class="card-title">{title}</div><div class="card-text">{desc}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rule-box"><b>Nguyên tắc sử dụng:</b> dữ liệu nhập vào là cơ sở của mọi kết quả. '
        'Cần kiểm tra số liệu trước khi sử dụng để báo cáo hoặc ra quyết định.<br><br>'
        '<b>AI:</b> chỉ hỗ trợ giải thích và nhận xét từ dữ liệu đã tính, không thay thế kế toán, kiểm toán hoặc thẩm định chuyên môn.</div>',
        unsafe_allow_html=True,
    )
    if ai_available():
        st.success("Gemini AI đã sẵn sàng.")
    else:
        st.info("Gemini AI chưa được cấu hình; các chức năng tính toán vẫn sử dụng bình thường.")

# =========================
# DÒNG TIỀN
# =========================
elif st.session_state.page == "💰 Sổ tay Dòng tiền":
    st.markdown('<div class="section-header">SỔ TAY DÒNG TIỀN</div>', unsafe_allow_html=True)
    if "cashflows" not in st.session_state:
        st.session_state.cashflows = pd.DataFrame(columns=["Ngày", "Loại", "Nhóm", "Nội dung", "Số tiền"])

    st.markdown('<div class="section-header">NHẬP DỮ LIỆU</div>', unsafe_allow_html=True)
    cols = st.columns([1.15, 1.0, 1.25, 1.7, 1.25, .75])
    with cols[0]:
        d = date_input_vn("Ngày", "cash_date")
    with cols[1]:
        typ = st.selectbox("Loại giao dịch", ["Thu", "Chi"], key="cash_type")
    with cols[2]:
        group = st.selectbox("Nhóm", ["Bán hàng", "Nguyên liệu", "Lương", "Điện/nước", "Vận chuyển", "Thuê mặt bằng", "Mua tài sản", "Khác"], key="cash_group")
    with cols[3]:
        content = st.text_input("Nội dung", key="cash_content")
    with cols[4]:
        amount = money_input("Số tiền", "cash_amount", 0)
    with cols[5]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Thêm", key="cash_add", use_container_width=True):
            row = pd.DataFrame([[pd.Timestamp(d), typ, group, content, amount]], columns=st.session_state.cashflows.columns)
            st.session_state.cashflows = pd.concat([st.session_state.cashflows, row], ignore_index=True)
            st.session_state.cash_amount = "0"
            st.session_state.cash_content = ""
            st.success("Đã thêm giao dịch.")

    up_cols = st.columns([1, 1])
    with up_cols[0]:
        uploaded = st.file_uploader("Tải dữ liệu Excel/CSV", type=["xlsx", "csv"], key="cash_upload")
    with up_cols[1]:
        st.caption("File nên có các cột: Ngày, Loại, Nhóm, Nội dung, Số tiền.")
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            required = {"Ngày", "Loại", "Số tiền"}
            if not required.issubset(df_up.columns):
                st.error("File thiếu một hoặc nhiều cột bắt buộc: Ngày, Loại, Số tiền.")
            else:
                for c in ["Nhóm", "Nội dung"]:
                    if c not in df_up.columns:
                        df_up[c] = ""
                st.session_state.cashflows = df_up[["Ngày", "Loại", "Nhóm", "Nội dung", "Số tiền"]].copy()
                st.success("Đã tải dữ liệu thành công.")
        except Exception as exc:
            st.error(f"Không thể đọc file: {exc}")

    df = st.session_state.cashflows.copy()
    if not df.empty:
        df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce", dayfirst=True)
        df["Số tiền"] = pd.to_numeric(df["Số tiền"].apply(clean_money_text), errors="coerce").fillna(0)
        df = df.dropna(subset=["Ngày"])

    st.markdown('<div class="section-header">KẾT QUẢ</div>', unsafe_allow_html=True)
    total_in = float(df.loc[df["Loại"].eq("Thu"), "Số tiền"].sum()) if not df.empty else 0
    total_out = float(df.loc[df["Loại"].eq("Chi"), "Số tiền"].sum()) if not df.empty else 0
    net = total_in - total_out
    mcols = st.columns(3)
    for c, lab, val in zip(mcols, ["TỔNG TIỀN VÀO", "TỔNG TIỀN RA", "DÒNG TIỀN RÒNG"], [total_in, total_out, net]):
        with c:
            st.markdown(f'<div class="metric-box"><div class="metric-label">{lab}</div><div class="metric-value">{money(val)}</div></div>', unsafe_allow_html=True)
    if net > 0:
        st.success("Dòng tiền đang dương. Tiền thu vào lớn hơn tiền chi ra, kết quả kinh doanh có dấu hiệu sinh lời")
    elif net < 0:
        st.warning("Dòng tiền đang âm. Tiền thu vào đang ít hơn tiền chi ra, có thể xem xét lại các khoản chi phí và kết quả kinh doanh")
    else:
        st.info("Dòng tiền đang cân bằng. Tiền thu vào bằng tiền chi ra.")

    if not df.empty:
        monthly = df.assign(Tháng=df["Ngày"].dt.to_period("M").astype(str)).groupby(["Tháng", "Loại"], as_index=False)["Số tiền"].sum()
        fig = px.bar(monthly, x="Tháng", y="Số tiền", color="Loại", barmode="group", title="Tiền vào – tiền ra theo tháng", text="Số tiền")
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", hovertemplate="%{y:,.0f} đ<extra></extra>")
        fig.update_yaxes(tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

        net_month = df.assign(Tháng=df["Ngày"].dt.to_period("M").astype(str), signed=np.where(df["Loại"].eq("Thu"), df["Số tiền"], -df["Số tiền"])).groupby("Tháng", as_index=False)["signed"].sum().rename(columns={"signed":"Dòng tiền ròng"})
        fig2 = px.line(net_month, x="Tháng", y="Dòng tiền ròng", markers=True, title="Dòng tiền ròng theo tháng", text="Dòng tiền ròng")
        fig2.update_traces(texttemplate="%{text:,.0f}", textposition="top center", hovertemplate="%{y:,.0f} đ<extra></extra>")
        fig2.update_yaxes(tickformat=",.0f")
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Xem bảng giao dịch"):
            show = df.copy()
            show["Ngày"] = show["Ngày"].dt.strftime("%d/%m/%Y")
            show["Số tiền"] = show["Số tiền"].map(money)
            st.dataframe(show, use_container_width=True, hide_index=True)

        a, b = st.columns(2)
        with a:
            export = dataframe_to_excel_bytes({"So_tay_dong_tien": df})
            st.download_button("Tải Excel", export, "so_tay_dong_tien.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with b:
            if st.button("Phân tích bằng AI", key="cash_ai", use_container_width=True):
                st.markdown('<div class="ai-box">'+call_gemini("Phân tích tình hình dòng tiền, nêu điểm đáng chú ý và cảnh báo nếu có.", {"tong_tien_vao":total_in,"tong_tien_ra":total_out,"dong_tien_rong":net,"so_giao_dich":len(df)})+'</div>', unsafe_allow_html=True)

# =========================
# KHẤU HAO
# =========================
elif st.session_state.page == "⚙️ Tính Khấu hao":
    st.markdown('<div class="section-header">TÍNH KHẤU HAO</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">NHẬP THÔNG TIN TÀI SẢN</div>', unsafe_allow_html=True)
    if "assets" not in st.session_state:
        st.session_state.assets = []
    if "dep_row" not in st.session_state:
        st.session_state.dep_row = 0

    cols = st.columns([1.35, 1.2, 1.2, 1.25, 1.25, 1.1, .7])
    with cols[0]: name = st.text_input("Tên tài sản", key="dep_name")
    with cols[1]: cost = money_input("Nguyên giá", "dep_cost", 0)
    with cols[2]: salvage = money_input("Giá trị thu hồi", "dep_salvage", 0)
    with cols[3]: years = st.number_input("Thời gian sử dụng (năm)", min_value=1, max_value=100, value=5, step=1, key="dep_years")
    with cols[4]: purchase = date_input_vn("Ngày mua", "dep_purchase")
    with cols[5]:
        st.markdown("<br>", unsafe_allow_html=True)
        add_asset = st.button("Thêm", key="dep_add", use_container_width=True)
    with cols[6]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Xóa", key="dep_clear", use_container_width=True):
            st.session_state.assets = []
            st.rerun()

    if add_asset:
        if cost <= 0 or salvage > cost:
            st.error("Kiểm tra nguyên giá và giá trị thu hồi.")
        else:
            st.session_state.assets.append({"Tên tài sản": name or f"Tài sản {len(st.session_state.assets)+1}", "Nguyên giá": cost, "Giá trị thu hồi": salvage, "Thời gian sử dụng (năm)": years, "Ngày mua": purchase.strftime("%d/%m/%Y")})
            st.session_state.dep_name = ""
            st.session_state.dep_cost = "0"
            st.session_state.dep_salvage = "0"
            st.success("Đã thêm tài sản.")

    if st.session_state.assets:
        rows = []
        today = date.today()
        for a in st.session_state.assets:
            pdate = datetime.strptime(a["Ngày mua"], "%d/%m/%Y").date()
            months_used = max(0, (today.year - pdate.year) * 12 + today.month - pdate.month - (today.day < pdate.day))
            total_months = int(a["Thời gian sử dụng (năm)"] * 12)
            used_months = min(months_used, total_months)
            annual_dep = (a["Nguyên giá"] - a["Giá trị thu hồi"]) / a["Thời gian sử dụng (năm)"]
            monthly_dep = annual_dep / 12
            incurred = min(used_months * monthly_dep, a["Nguyên giá"] - a["Giá trị thu hồi"])
            remaining_dep = max(0, (a["Nguyên giá"] - a["Giá trị thu hồi"]) - incurred)
            remaining_months = max(0, total_months - used_months)
            book_value = max(a["Giá trị thu hồi"], a["Nguyên giá"] - incurred)
            rows.append({**a, "Đã sử dụng": f"{used_months/12:.1f} năm", "Thời gian còn lại": f"{remaining_months/12:.1f} năm", "Khấu hao năm": annual_dep, "Khấu hao tháng": monthly_dep, "Khấu hao đã phát sinh": incurred, "Khấu hao còn lại": remaining_dep, "Giá trị còn lại": book_value})
        dep_df = pd.DataFrame(rows)
        st.markdown('<div class="section-header">KẾT QUẢ KHẤU HAO</div>', unsafe_allow_html=True)
        disp = dep_df.copy()
        for c in ["Nguyên giá", "Giá trị thu hồi", "Khấu hao năm", "Khấu hao tháng", "Khấu hao đã phát sinh", "Khấu hao còn lại", "Giá trị còn lại"]:
            disp[c] = disp[c].map(money)
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.caption(f"Ngày tính: {date.today().strftime('%d/%m/%Y')}")

        metric_cols = st.columns(3)
        total_incurred = float(dep_df["Khấu hao đã phát sinh"].sum())
        total_remaining = float(dep_df["Khấu hao còn lại"].sum())
        total_book = float(dep_df["Giá trị còn lại"].sum())
        for c, lab, val in zip(metric_cols, ["KHẤU HAO ĐÃ PHÁT SINH", "KHẤU HAO CÒN LẠI", "GIÁ TRỊ CÒN LẠI"], [total_incurred, total_remaining, total_book]):
            with c:
                st.markdown(f'<div class="metric-box"><div class="metric-label">{lab}</div><div class="metric-value">{money(val)}</div></div>', unsafe_allow_html=True)

        a, b = st.columns(2)
        with a:
            export = dataframe_to_excel_bytes({"Khau_hao": dep_df})
            st.download_button("Tải Excel", export, "tinh_khau_hao.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with b:
            if st.button("Phân tích bằng AI", key="dep_ai", use_container_width=True):
                st.markdown('<div class="ai-box">'+call_gemini("Giải thích ngắn gọn khấu hao và ảnh hưởng của phần khấu hao đến chi phí kinh doanh.", {"tai_san":rows})+'</div>', unsafe_allow_html=True)
    else:
        st.info("Chưa có tài sản. Nhập thông tin và bấm Thêm.")

# =========================
# ĐẦU TƯ
# =========================
elif st.session_state.page == "📈 Đánh giá Hiệu quả Đầu tư":
    st.markdown('<div class="section-header">ĐÁNH GIÁ HIỆU QUẢ ĐẦU TƯ</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">THÔNG TIN DỰ ÁN</div>', unsafe_allow_html=True)
    pcols = st.columns([1.8, 1.25, 1.25])
    with pcols[0]: project = st.text_input("Tên dự án", key="project")
    with pcols[1]: initial = money_input("Vốn đầu tư ban đầu", "initial", 0)
    with pcols[2]: periods = st.number_input("Số kỳ dự kiến", min_value=1, max_value=50, value=5, step=1, key="periods")

    st.markdown('<div class="section-header">DÒNG TIỀN TỪNG KỲ</div>', unsafe_allow_html=True)
    cf_cols = st.columns(int(periods) + 1)
    for i in range(1, int(periods) + 1):
        with cf_cols[i-1]:
            money_input(f"Kỳ {i}", f"cf_{i}", 0)
    with cf_cols[-1]:
        discount = st.number_input("Tỷ lệ chiết khấu (%)", min_value=-99.0, max_value=500.0, value=10.0, step=0.5, key="discount") / 100.0

    st.markdown('<div class="section-header">DỮ LIỆU WACC — NẾU CÓ</div>', unsafe_allow_html=True)
    wcols = st.columns(5)
    with wcols[0]: e = st.number_input("Vốn chủ sở hữu (%)", 0.0, 100.0, 100.0, 1.0) / 100.0
    with wcols[1]: d = st.number_input("Vốn vay (%)", 0.0, 100.0, 0.0, 1.0) / 100.0
    with wcols[2]: ke = st.number_input("Chi phí vốn chủ (%)", 0.0, 100.0, 10.0, 0.5) / 100.0
    with wcols[3]: kd = st.number_input("Chi phí vốn vay (%)", 0.0, 100.0, 8.0, 0.5) / 100.0
    with wcols[4]: tax = st.number_input("Thuế suất (%)", 0.0, 100.0, 20.0, 0.5) / 100.0

    cashflows = [-initial] + [clean_money_text(st.session_state.get(f"cf_{i}", "0")) for i in range(1, int(periods)+1)]
    npv_val = npv(discount, cashflows)
    irr_val = irr_bisection(cashflows)
    pb = payback_period(cashflows)
    wacc = compute_wacc(e, d, ke, kd, tax)

    st.markdown('<div class="section-header">KẾT QUẢ VÀ Ý NGHĨA</div>', unsafe_allow_html=True)
    concepts = [
        ("NPV — GIÁ TRỊ HIỆN TẠI RÒNG", "Là tổng chênh lệch giữa dòng tiền thu vào và dòng tiền chi ra, được quy đổi tất cả về giá trị tại thời điểm hiện tại theo một tỷ suất chiết khấu nhất định.", money(npv_val)),
        ("IRR — TỶ SUẤT HOÀN VỐN NỘI BỘ", "Là mức tỷ suất làm cho NPV của dự án bằng 0. Có thể hiểu đơn giản là mức sinh lời nội tại của chuỗi dòng tiền.", percent(irr_val)),
        ("PAYBACK — THỜI GIAN HOÀN VỐN", "Là khoảng thời gian dự kiến để dòng tiền tích lũy thu hồi đủ số vốn đầu tư ban đầu.", f"{pb:.1f} năm" if pb is not None else "Chưa hoàn vốn"),
        ("WACC — CHI PHÍ SỬ DỤNG VỐN BÌNH QUÂN GIA QUYỀN", "Là chi phí bình quân của các nguồn vốn tài trợ cho dự án, có xét tỷ trọng vốn chủ sở hữu và vốn vay.", percent(wacc)),
    ]
    c = st.columns(4)
    for col, (title, desc, val) in zip(c, concepts):
        with col:
            st.markdown(f'<div class="metric-box"><div class="term-title">{title}</div><div class="term-desc">{desc}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

    if npv_val > 0:
        st.success("NPV dương: dự án đang có tín hiệu tạo thêm giá trị theo tỷ suất chiết khấu đã nhập.")
    elif npv_val < 0:
        st.warning("NPV âm: dự án chưa đạt mức sinh lời yêu cầu theo tỷ suất chiết khấu đã nhập.")
    else:
        st.info("NPV bằng 0: dự án vừa đạt mức sinh lời yêu cầu theo giả định hiện tại.")

    cum = np.cumsum(cashflows)
    chart_df = pd.DataFrame({"Kỳ": range(len(cum)), "Dòng tiền tích lũy": cum})
    fig = px.line(chart_df, x="Kỳ", y="Dòng tiền tích lũy", markers=True, text="Dòng tiền tích lũy", title="Dòng tiền tích lũy")
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="top center", hovertemplate="%{y:,.0f} đ<extra></extra>")
    fig.update_yaxes(tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    with a:
        summary = pd.DataFrame({"Chỉ tiêu":["NPV","IRR","Thời gian hoàn vốn","WACC","Tỷ lệ chiết khấu"], "Giá trị":[npv_val, irr_val if irr_val is not None else np.nan, pb if pb is not None else np.nan, wacc if wacc is not None else np.nan, discount]})
        details = pd.DataFrame({"Kỳ":range(len(cashflows)), "Dòng tiền":cashflows, "Dòng tiền tích lũy":cum})
        export = dataframe_to_excel_bytes({"Ket_qua":summary, "Dong_tien":details})
        st.download_button("Tải báo cáo Excel", export, "danh_gia_hieu_qua_dau_tu.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with b:
        if st.button("Phân tích bằng AI", key="investment_ai", use_container_width=True):
            payload = {"du_an":project, "von_dau_tu_ban_dau":initial, "dong_tien":cashflows, "ty_le_chiet_khau":discount, "NPV":npv_val, "IRR":irr_val, "Payback":pb, "WACC":wacc}
            st.markdown('<div class="ai-box">'+call_gemini("Phân tích hiệu quả dự án theo 3 phần: kết quả, nhận xét dễ hiểu và cảnh báo/điểm cần kiểm tra.", payload)+'</div>', unsafe_allow_html=True)
