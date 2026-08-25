import io
import json
from datetime import date
from typing import Optional

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
MODEL_NAME = "gemini-3.7-flash"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: Arial, Helvetica, sans-serif; }
    .main-title { color:#C00000; font-size:2.35rem; font-weight:900; text-align:center; margin:.05rem 0 .15rem; }
    .sub-title { text-align:center; color:#555; font-size:1rem; margin-bottom:1rem; }
    .section-head { color:#A00000; font-size:1.42rem; font-weight:900; margin:.25rem 0 .9rem; padding-bottom:.4rem; border-bottom:2px solid #D9A3A3; }
    .sub-head { color:#7A0000; font-size:1.05rem; font-weight:800; margin:.25rem 0 .55rem; }
    .guide-box { border-left:4px solid #C00000; background:#FFF7F7; padding:10px 13px; border-radius:8px; margin-bottom:12px; color:#333; }
    .overview-card { background:#fff; border:1px solid #E2CACA; border-radius:16px; padding:18px 18px 16px; min-height:250px; box-shadow:0 5px 14px rgba(0,0,0,.08); transition:.15s; }
    .overview-card:hover { transform:translateY(-2px); box-shadow:0 8px 20px rgba(192,0,0,.16); }
    .overview-title { color:#C00000; font-size:1.22rem; font-weight:900; margin-bottom:10px; text-align:center; }
    .overview-text { color:#333; line-height:1.55; font-size:.96rem; }
    .policy-box { border:1px solid #D7DDE5; background:#F8FAFC; border-radius:12px; padding:14px 16px; margin-top:14px; line-height:1.6; }
    .panel { border:1px solid #D5D9E0; background:#fff; border-radius:12px; padding:14px 14px 10px; box-shadow:0 2px 6px rgba(0,0,0,.04); }
    .result-panel { border:1px solid #C9D8C9; background:#F8FCF8; border-radius:12px; padding:14px; box-shadow:0 2px 6px rgba(0,0,0,.04); }
    .metric-card { border:1px solid #CDD3DB; background:#fff; border-radius:12px; padding:11px 12px; min-height:102px; box-shadow:0 2px 7px rgba(0,0,0,.05); }
    .metric-name { color:#555; font-size:.86rem; font-weight:800; line-height:1.25; }
    .metric-value { color:#1F2937; font-size:1.30rem; font-weight:900; margin-top:4px; }
    .meaning { color:#666; font-size:.79rem; margin-top:4px; line-height:1.4; }
    .positive-box { border:1px solid #B8D7B8; background:#F3FBF3; color:#215A21; padding:12px 14px; border-radius:10px; font-weight:700; }
    .negative-box { border:1px solid #E2B6B6; background:#FFF5F5; color:#8A1C1C; padding:12px 14px; border-radius:10px; font-weight:700; }
    .neutral-box { border:1px solid #D8D8B2; background:#FFFDF1; color:#665B00; padding:12px 14px; border-radius:10px; font-weight:700; }
    [data-testid="stSidebar"] { border-right:1px solid #E1E1E1; }
    [data-testid="stSidebar"] h2 { color:#C00000; }
    .sidebar-menu-title { font-size:1.08rem; font-weight:900; color:#222; margin:.4rem 0 .25rem; }
    .sidebar-guide { font-size:.87rem; color:#555; line-height:1.45; }
    .small-note { color:#666; font-size:.82rem; }
    .stButton > button { border-radius:9px; font-weight:800; }
    div[data-testid="stForm"] { border:0 !important; padding:0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================
def money(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.0f} đ".replace(",", ".")


def percent(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.2f}%"


def get_secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name, default)
        return str(v) if v else default
    except Exception:
        return default


def gemini_available() -> bool:
    return genai is not None and bool(get_secret("GEMINI_API_KEY"))


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    key = get_secret("GEMINI_API_KEY")
    if not key or genai is None:
        return None
    try:
        http_options = types.HttpOptions(timeout=60000) if types else None
        return genai.Client(api_key=key, http_options=http_options)
    except Exception:
        return None


def call_gemini(instruction: str, data: dict) -> str:
    client = get_gemini_client()
    if client is None or types is None:
        return "Chưa cấu hình Gemini API. Kết quả tính toán vẫn hoạt động bình thường. Hãy thêm GEMINI_API_KEY vào Streamlit Secrets."

    system_instruction = (
        "Bạn là Trợ lý Tài chính Nhỏ, hỗ trợ tiểu thương, hộ kinh doanh, nông hộ, "
        "cơ sở sản xuất nhỏ và hợp tác xã. Chỉ dùng số liệu được cung cấp; không tự "
        "tạo hoặc thay đổi số liệu; phân biệt KẾT QUẢ, NHẬN XÉT, CẢNH BÁO; giải thích "
        "thuật ngữ bằng tiếng Việt dễ hiểu; không khẳng định chắc chắn lợi nhuận tương lai; "
        "không thay thế kế toán, kiểm toán hoặc thẩm định chuyên môn."
    )
    prompt = instruction + "\n\nDỮ LIỆU ĐÃ ĐƯỢC HỆ THỐNG TÍNH TOÁN:\n" + json.dumps(data, ensure_ascii=False, indent=2, default=str)
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=800,
            ),
        )
        return response.text or "AI không trả về nội dung."
    except Exception as exc:
        return f"Không thể gọi Gemini API lúc này: {exc}"


def dataframe_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=str(name)[:31], index=False)
    return buffer.getvalue()


def npv(rate: float, cashflows: list[float]) -> float:
    if rate <= -1:
        return float("nan")
    return float(sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows)))


def irr_bisection(cashflows: list[float]) -> Optional[float]:
    if len(cashflows) < 2:
        return None
    arr = np.asarray(cashflows, dtype=float)
    if not (np.any(arr > 0) and np.any(arr < 0)):
        return None

    def f(rate):
        try:
            return npv(rate, cashflows)
        except Exception:
            return np.nan

    grid = np.concatenate([np.linspace(-0.99, -0.01, 120), np.linspace(0.0, 5.0, 400)])
    prev_r, prev_v = grid[0], f(grid[0])
    for current_r in grid[1:]:
        current_v = f(current_r)
        if np.isfinite(prev_v) and np.isfinite(current_v):
            if prev_v == 0:
                return float(prev_r)
            if current_v == 0 or prev_v * current_v < 0:
                low, high = prev_r, current_r
                flow = prev_v
                for _ in range(120):
                    mid = (low + high) / 2
                    fmid = f(mid)
                    if not np.isfinite(fmid):
                        break
                    if abs(fmid) < 1e-9:
                        return float(mid)
                    if flow * fmid <= 0:
                        high = mid
                    else:
                        low, flow = mid, fmid
                return float((low + high) / 2)
        prev_r, prev_v = current_r, current_v
    return None


def payback_period(cashflows: list[float]) -> Optional[float]:
    cumulative = float(cashflows[0])
    if cumulative >= 0:
        return 0.0
    for i in range(1, len(cashflows)):
        previous = cumulative
        cumulative += cashflows[i]
        if cumulative >= 0:
            step = cashflows[i]
            if step == 0:
                return float(i)
            return (i - 1) + float(np.clip((-previous) / step, 0, 1))
    return None


def calc_wacc(equity_weight, debt_weight, cost_equity, cost_debt, tax_rate):
    total = equity_weight + debt_weight
    if total <= 0:
        return None
    e = equity_weight / total
    d = debt_weight / total
    return e * cost_equity + d * cost_debt * (1 - tax_rate)


def money_input(label: str, value: float = 0.0, key: Optional[str] = None, step: int = 100000):
    # Streamlit's number_input accepts printf-style formats such as %.0f.
    # Do NOT use format="%,.0f": that triggers StreamlitInvalidNumberFormatError.
    return st.number_input(
        label,
        min_value=0.0,
        value=float(value),
        step=float(step),
        format="%.0f",
        key=key,
        help="Nhập số tiền. Giá trị được hiển thị dạng số nguyên; kết quả bên dưới có phân cách hàng nghìn.",
    )


def metric_card(title: str, value: str, meaning: str = ""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-name">{title}</div><div class="metric-value">{value}</div><div class="meaning">{meaning}</div></div>',
        unsafe_allow_html=True,
    )


def navigate(page_name: str):
    st.session_state.page = page_name
    st.rerun()


# ============================================================
# STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Tổng quan"
if "cashflows" not in st.session_state:
    st.session_state.cashflows = pd.DataFrame(columns=["Ngày", "Loại", "Nhóm", "Nội dung", "Số tiền"])

# ============================================================
# HEADER
# ============================================================
st.markdown(f'<div class="main-title">{APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Số hóa dòng tiền • Tính khấu hao • Đánh giá hiệu quả đầu tư</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## DANH MỤC")
st.sidebar.markdown('<div class="sidebar-guide"><b>Hướng dẫn nhanh</b><br>Chọn một mục → nhập dữ liệu → xem kết quả. Nút AI chỉ dùng khi cần giải thích hoặc nhận xét.</div>', unsafe_allow_html=True)

menu_items = [
    ("🏠 Tổng quan", "Tổng quan"),
    ("💰 Sổ tay Dòng tiền", "Dòng tiền"),
    ("⚙️ Tính Khấu hao", "Khấu hao"),
    ("📈 Đánh giá Đầu tư", "Đầu tư"),
]
for label, value in menu_items:
    st.sidebar.markdown(f'<div class="sidebar-menu-title">{label}</div>', unsafe_allow_html=True)
    if st.sidebar.button("Mở chức năng", key=f"side_{value}", use_container_width=True):
        navigate(label)

st.sidebar.divider()
st.sidebar.info("Gemini AI: Đã sẵn sàng" if gemini_available() else "Gemini AI: Chưa cấu hình")
page = st.session_state.page

# ============================================================
# TỔNG QUAN
# ============================================================
if page == "🏠 Tổng quan":
    st.markdown('<div class="section-head">TỔNG QUAN CHƯƠNG TRÌNH</div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-box">Chương trình giúp số hóa việc theo dõi tiền, phân bổ chi phí tài sản và đánh giá một phương án đầu tư bằng cách nhập dữ liệu đơn giản rồi xem kết quả trực quan.</div>', unsafe_allow_html=True)

    cols = st.columns(3, gap="large")
    cards = [
        ("💰 SỔ TAY DÒNG TIỀN", "💰 Sổ tay Dòng tiền", "Ghi lại từng khoản thu và chi, theo dõi tổng tiền vào, tổng tiền ra và dòng tiền ròng. Có biểu đồ theo tháng, hỗ trợ tải Excel/CSV và AI nhận xét xu hướng.", "Cách dùng: nhập giao dịch hoặc tải file, sau đó xem dòng tiền và các khoản chi đáng chú ý."),
        ("⚙️ TÍNH KHẤU HAO", "⚙️ Tính Khấu hao", "Tính khấu hao đường thẳng cho máy móc, thiết bị và tài sản. Có ngày mua để ước tính thời gian đã sử dụng, khấu hao đã phát sinh, thời gian còn lại và giá trị còn lại.", "Cách dùng: nhập tài sản, nguyên giá, giá trị thu hồi, ngày mua và thời gian sử dụng."),
        ("📈 ĐÁNH GIÁ ĐẦU TƯ", "📈 Đánh giá Đầu tư", "Tính NPV, IRR, thời gian hoàn vốn và WACC khi đủ dữ liệu. Mỗi chỉ tiêu có nghĩa tiếng Việt, ý nghĩa ngắn gọn và kết luận dễ hiểu.", "Cách dùng: nhập vốn đầu tư, dòng tiền từng kỳ, tỷ lệ chiết khấu và thông tin nguồn vốn nếu muốn tính WACC."),
    ]
    for c, (title, target, text, guide) in zip(cols, cards):
        with c:
            st.markdown(f'<div class="overview-card"><div class="overview-title">{title}</div><div class="overview-text"><b>Công dụng:</b> {text}<br><br><b>{guide}</b></div></div>', unsafe_allow_html=True)
            if st.button(f"MỞ {title.replace('💰 ','').replace('⚙️ ','').replace('📈 ','')}", key=title, use_container_width=True):
                navigate(target)

    st.markdown('<div class="policy-box"><b>Nguyên tắc sử dụng:</b> dữ liệu nhập vào là cơ sở của mọi kết quả. Cần kiểm tra số liệu trước khi sử dụng để báo cáo hoặc ra quyết định.<br><br><b>AI:</b> chỉ hỗ trợ giải thích và nhận xét từ dữ liệu đã tính, không thay thế kế toán, kiểm toán hoặc thẩm định chuyên môn.</div>', unsafe_allow_html=True)

# ============================================================
# DÒNG TIỀN
# ============================================================
elif page == "💰 Sổ tay Dòng tiền":
    st.markdown('<div class="section-head">SỔ TAY DÒNG TIỀN</div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-box"><b>Cách dùng:</b> nhập một giao dịch trên một hàng, chọn Thu/Chi, nhập số tiền rồi bấm <b>Thêm giao dịch</b>. Có thể tải Excel/CSV nếu đã có dữ liệu.</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="sub-head">NHẬP DỮ LIỆU — CÁC TRƯỜNG XẾP NGANG</div>', unsafe_allow_html=True)
    with st.form("cash_form", clear_on_submit=True):
        r1, r2, r3, r4, r5, r6 = st.columns([1.0, .9, 1.2, 1.7, 1.15, .9])
        with r1:
            d = st.date_input("Ngày", value=date.today())
        with r2:
            typ = st.selectbox("Loại", ["Thu", "Chi"])
        with r3:
            group = st.selectbox("Nhóm", ["Bán hàng", "Nguyên liệu", "Lương", "Điện/nước", "Vận chuyển", "Thuê mặt bằng", "Mua tài sản", "Khác"])
        with r4:
            content = st.text_input("Nội dung")
        with r5:
            amount = money_input("Số tiền (đồng)", key="cash_amount", step=10000)
        with r6:
            submitted = st.form_submit_button("➕ THÊM", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    upcol1, upcol2 = st.columns([2, 3])
    with upcol1:
        uploaded = st.file_uploader("Tải Excel/CSV", type=["xlsx", "csv"], key="cash_upload")
    with upcol2:
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
                required = {"Ngày", "Loại", "Nhóm", "Nội dung", "Số tiền"}
                missing = required - set(df_up.columns)
                if missing:
                    st.error("File thiếu cột: " + ", ".join(sorted(missing)))
                else:
                    st.session_state.cashflows = df_up.copy()
                    st.success("Đã nạp dữ liệu từ file.")
            except Exception as exc:
                st.error(f"Không đọc được file: {exc}")

    if submitted and amount > 0:
        new_row = pd.DataFrame([[pd.Timestamp(d), typ, group, content, amount]], columns=st.session_state.cashflows.columns)
        st.session_state.cashflows = pd.concat([st.session_state.cashflows, new_row], ignore_index=True)
        st.success("Đã thêm giao dịch.")

    df = st.session_state.cashflows.copy()
    if not df.empty:
        df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
        df["Số tiền"] = pd.to_numeric(df["Số tiền"], errors="coerce").fillna(0)

    if df.empty:
        total_in = total_out = net = 0.0
    else:
        total_in = float(df.loc[df["Loại"].astype(str).eq("Thu"), "Số tiền"].sum())
        total_out = float(df.loc[df["Loại"].astype(str).eq("Chi"), "Số tiền"].sum())
        net = total_in - total_out

    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    st.markdown('<div class="sub-head">KẾT QUẢ DÒNG TIỀN</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1: metric_card("TỔNG TIỀN VÀO", money(total_in), "Tổng các khoản Thu")
    with m2: metric_card("TỔNG TIỀN RA", money(total_out), "Tổng các khoản Chi")
    with m3: metric_card("DÒNG TIỀN RÒNG", money(net), "Tiền vào trừ tiền ra")
    if net > 0:
        st.markdown('<div class="positive-box">Dòng tiền đang dương. Tiền thu vào lớn hơn tiền chi ra, kết quả kinh doanh có dấu hiệu sinh lời</div>', unsafe_allow_html=True)
    elif net < 0:
        st.markdown('<div class="negative-box">Dòng tiền đang âm. Tiền thu vào đang ít hơn tiền chi ra, có thể xem xét lại các khoản chi phí và kết quả kinh doanh</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="neutral-box">Dòng tiền đang cân bằng. Tiền thu vào và tiền chi ra hiện bằng nhau.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not df.empty:
        charts = st.columns(2)
        monthly = df.assign(Tháng=df["Ngày"].dt.to_period("M").astype(str)).groupby(["Tháng", "Loại"], as_index=False)["Số tiền"].sum()
        with charts[0]:
            st.plotly_chart(px.bar(monthly, x="Tháng", y="Số tiền", color="Loại", barmode="group", title="Tiền vào – tiền ra theo tháng"), use_container_width=True)
        net_month = df.assign(Tháng=df["Ngày"].dt.to_period("M").astype(str), signed=np.where(df["Loại"].eq("Thu"), df["Số tiền"], -df["Số tiền"])).groupby("Tháng", as_index=False)["signed"].sum().rename(columns={"signed":"Dòng tiền ròng"})
        with charts[1]:
            st.plotly_chart(px.line(net_month, x="Tháng", y="Dòng tiền ròng", markers=True, title="Dòng tiền ròng theo tháng"), use_container_width=True)

        a, b = st.columns(2)
        with a:
            if st.button("🤖 PHÂN TÍCH BẰNG AI", key="cash_ai", use_container_width=True):
                st.markdown(call_gemini("Phân tích tình hình dòng tiền; nêu các điểm đáng chú ý và cảnh báo nếu có, bằng ngôn ngữ dễ hiểu.", {"tong_tien_vao":total_in,"tong_tien_ra":total_out,"dong_tien_rong":net,"so_giao_dich":len(df)}))
        with b:
            st.download_button("⬇️ XUẤT EXCEL", dataframe_to_excel_bytes({"So_tay_dong_tien":df}), "so_tay_dong_tien.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ============================================================
# KHẤU HAO
# ============================================================
elif page == "⚙️ Tính Khấu hao":
    st.markdown('<div class="section-head">TÍNH KHẤU HAO</div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-box"><b>Cách dùng:</b> nhập nguyên giá, giá trị thu hồi, ngày mua và thời gian sử dụng. Hệ thống tính theo phương pháp đường thẳng.</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="sub-head">NHẬP THÔNG TIN TÀI SẢN — XẾP NGANG</div>', unsafe_allow_html=True)
    i1,i2,i3,i4,i5 = st.columns([1.3,1.15,1.15,1.1,.9])
    with i1: asset = st.text_input("Tên tài sản", placeholder="Ví dụ: Máy cày")
    with i2: cost = money_input("Nguyên giá (đồng)", key="dep_cost", step=1000000)
    with i3: salvage = money_input("Giá trị thu hồi (đồng)", key="dep_salvage", step=1000000)
    with i4: purchase_date = st.date_input("Ngày mua tài sản", value=date.today(), key="dep_purchase_date")
    with i5: years = st.number_input("Thời gian sử dụng (năm)", min_value=1, max_value=100, value=5, step=1, format="%d", key="dep_years")
    st.markdown('</div>', unsafe_allow_html=True)

    valid = cost > 0 and 0 <= salvage <= cost and purchase_date <= date.today()
    if not valid:
        if purchase_date > date.today(): st.error("Ngày mua tài sản không được lớn hơn ngày hiện tại.")
        elif salvage > cost: st.error("Giá trị thu hồi không được lớn hơn nguyên giá.")
    else:
        annual = (cost - salvage) / years
        monthly = annual / 12
        elapsed_days = max((date.today() - purchase_date).days, 0)
        elapsed_months = min(elapsed_days / 30.4375, years * 12)
        accumulated = min(annual * elapsed_months / 12, cost - salvage)
        remaining_depr = max((cost - salvage) - accumulated, 0)
        book_value = max(cost - accumulated, salvage)
        remaining_months = max(years * 12 - elapsed_months, 0)

        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        st.markdown('<div class="sub-head">KẾT QUẢ KHẤU HAO</div>', unsafe_allow_html=True)
        r = st.columns(5)
        vals = [
            ("KHẤU HAO NĂM", money(annual), "Mức khấu hao trung bình mỗi năm"),
            ("KHẤU HAO THÁNG", money(monthly), "Mức khấu hao trung bình mỗi tháng"),
            ("ĐÃ KHẤU HAO", money(accumulated), "Giá trị khấu hao ước tính đã phát sinh"),
            ("KHẤU HAO CÒN LẠI", money(remaining_depr), "Phần khấu hao chưa phát sinh"),
            ("GIÁ TRỊ CÒN LẠI", money(book_value), "Giá trị tài sản sau phần khấu hao đã phát sinh"),
        ]
        for c, (t,v,m) in zip(r, vals):
            with c: metric_card(t,v,m)
        s1,s2 = st.columns(2)
        with s1: metric_card("THỜI GIAN ĐÃ SỬ DỤNG", f"{elapsed_months:.1f} tháng", f"Từ ngày mua {purchase_date.strftime('%d/%m/%Y')} đến hôm nay")
        with s2: metric_card("THỜI GIAN SỬ DỤNG CÒN LẠI", f"{remaining_months:.1f} tháng", "Theo vòng đời tài sản đã nhập")
        st.markdown('</div>', unsafe_allow_html=True)

        schedule = pd.DataFrame([{"Năm":y,"Khấu hao năm":annual,"Giá trị còn lại cuối năm":max(cost-annual*y,salvage)} for y in range(1,int(years)+1)])
        st.dataframe(schedule, use_container_width=True, hide_index=True, column_config={"Khấu hao năm":st.column_config.NumberColumn(format="%.0f"),"Giá trị còn lại cuối năm":st.column_config.NumberColumn(format="%.0f")})
        st.plotly_chart(px.line(schedule, x="Năm", y="Giá trị còn lại cuối năm", markers=True, title="Giá trị còn lại theo thời gian"), use_container_width=True)

        a,b = st.columns(2)
        with a:
            if st.button("🤖 PHÂN TÍCH BẰNG AI", key="dep_ai", use_container_width=True):
                st.markdown(call_gemini("Giải thích khấu hao, thời gian đã sử dụng, khấu hao đã phát sinh và giá trị còn lại bằng ngôn ngữ dễ hiểu.", {"tai_san":asset or "Chưa đặt tên","nguyen_gia":cost,"gia_tri_thu_hoi":salvage,"ngay_mua":str(purchase_date),"thoi_gian_su_dung_nam":years,"da_su_dung_thang":round(elapsed_months,1),"thoi_gian_con_lai_thang":round(remaining_months,1),"khau_hao_da_phat_sinh":accumulated,"khau_hao_con_lai":remaining_depr,"gia_tri_con_lai":book_value}))
        with b:
            st.download_button("⬇️ XUẤT EXCEL", dataframe_to_excel_bytes({"Tai_san":pd.DataFrame([{"Tên tài sản":asset,"Nguyên giá":cost,"Giá trị thu hồi":salvage,"Ngày mua":purchase_date,"Thời gian sử dụng (năm)":years,"Đã sử dụng (tháng)":round(elapsed_months,1),"Khấu hao đã phát sinh":accumulated,"Khấu hao còn lại":remaining_depr,"Giá trị còn lại":book_value}]),"Phan_bo_khau_hao":schedule}), "tinh_khau_hao.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

# ============================================================
# ĐẦU TƯ
# ============================================================
elif page == "📈 Đánh giá Đầu tư":
    st.markdown('<div class="section-head">ĐÁNH GIÁ HIỆU QUẢ ĐẦU TƯ</div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-box"><b>Cách dùng:</b> nhập vốn đầu tư, dòng tiền từng kỳ và tỷ lệ chiết khấu. Nếu có dữ liệu nguồn vốn, nhập thêm để tính WACC.</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    top = st.columns([1.5,1.0,.8,1.0])
    with top[0]: project = st.text_input("Tên dự án", placeholder="Ví dụ: Đầu tư máy sấy nông sản")
    with top[1]: initial = money_input("Vốn đầu tư ban đầu (đồng)", key="inv_initial", step=1000000)
    with top[2]: n_periods = st.number_input("Số kỳ dự kiến", min_value=1, max_value=50, value=5, step=1, format="%d", key="inv_periods")
    with top[3]: discount = st.number_input("Tỷ lệ chiết khấu (%)", min_value=-99.0, max_value=500.0, value=10.0, step=.5, format="%.2f", key="inv_discount") / 100
    st.markdown('<div class="sub-head">DÒNG TIỀN TỪNG KỲ — XẾP NGANG</div>', unsafe_allow_html=True)
    cf_cols = st.columns(min(int(n_periods), 6))
    cashflows = [-initial]
    for i in range(1, int(n_periods)+1):
        with cf_cols[(i-1)%len(cf_cols)]:
            cashflows.append(money_input(f"Kỳ {i}", key=f"inv_cf_{i}", step=1000000))
    st.markdown('<div class="sub-head">THÔNG TIN WACC — CHỈ NHẬP KHI CÓ NHU CẦU</div>', unsafe_allow_html=True)
    wcols = st.columns(5)
    with wcols[0]: ew = st.number_input("Vốn chủ (%)",0.0,100.0,100.0,1.0,format="%.1f")/100
    with wcols[1]: dw = st.number_input("Vốn vay (%)",0.0,100.0,0.0,1.0,format="%.1f")/100
    with wcols[2]: ke = st.number_input("Chi phí vốn chủ (%)",0.0,100.0,10.0,.5,format="%.2f")/100
    with wcols[3]: kd = st.number_input("Chi phí vốn vay (%)",0.0,100.0,8.0,.5,format="%.2f")/100
    with wcols[4]: tax = st.number_input("Thuế suất (%)",0.0,100.0,20.0,.5,format="%.2f")/100
    st.markdown('</div>', unsafe_allow_html=True)

    npv_value = npv(discount, cashflows)
    irr_value = irr_bisection(cashflows)
    payback = payback_period(cashflows)
    wacc_value = calc_wacc(ew,dw,ke,kd,tax)

    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    st.markdown('<div class="sub-head">KẾT QUẢ VÀ Ý NGHĨA</div>', unsafe_allow_html=True)
    terms = st.columns(4)
    with terms[0]: metric_card("NPV — GIÁ TRỊ HIỆN TẠI RÒNG", money(npv_value), "Tổng chênh lệch giữa dòng tiền thu vào và chi ra, quy đổi về giá trị hiện tại theo tỷ suất chiết khấu.")
    with terms[1]: metric_card("IRR — TỶ SUẤT HOÀN VỐN NỘI BỘ", percent(irr_value), "Mức tỷ suất làm NPV bằng 0; dùng để so sánh với mức sinh lời yêu cầu.")
    with terms[2]: metric_card("PAYBACK — THỜI GIAN HOÀN VỐN", f"{payback:.1f} năm" if payback is not None else "Chưa hoàn vốn", "Khoảng thời gian để dòng tiền tích lũy bù đắp vốn đầu tư ban đầu.")
    with terms[3]: metric_card("WACC — CHI PHÍ SỬ DỤNG VỐN BÌNH QUÂN GIA QUYỀN", percent(wacc_value), "Chi phí vốn bình quân theo tỷ trọng vốn chủ và vốn vay, có điều chỉnh thuế đối với nợ vay.")
    if npv_value > 0:
        st.markdown('<div class="positive-box"><b>NPV dương:</b> Dự án đang có tín hiệu tạo thêm giá trị theo mức chiết khấu đã nhập.</div>', unsafe_allow_html=True)
    elif npv_value < 0:
        st.markdown('<div class="negative-box"><b>NPV âm:</b> Dự án chưa đạt mức sinh lời yêu cầu theo mức chiết khấu đã nhập.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="neutral-box"><b>NPV bằng 0:</b> Dự án vừa đạt mức sinh lời yêu cầu theo giả định hiện tại.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    chart = pd.DataFrame({"Kỳ":range(len(cashflows)),"Dòng tiền tích lũy":np.cumsum(cashflows)})
    st.plotly_chart(px.line(chart,x="Kỳ",y="Dòng tiền tích lũy",markers=True,title="Dòng tiền tích lũy"), use_container_width=True)
    a,b = st.columns(2)
    with a:
        if st.button("🤖 PHÂN TÍCH BẰNG AI", key="inv_ai", use_container_width=True):
            st.markdown(call_gemini("Phân tích dự án theo 3 phần: Kết quả tính toán, Nhận xét dễ hiểu, Cảnh báo/điểm cần kiểm tra.", {"du_an":project or "Chưa đặt tên","von_dau_tu_ban_dau":initial,"dong_tien":cashflows,"ty_le_chiet_khau":discount,"NPV":npv_value,"IRR":irr_value,"thoi_gian_hoan_von_nam":payback,"WACC":wacc_value}))
    with b:
        export_df = pd.DataFrame({"Kỳ":range(len(cashflows)),"Dòng tiền":cashflows,"Dòng tiền tích lũy":np.cumsum(cashflows)})
        summary = pd.DataFrame([["NPV",npv_value],["IRR",irr_value],["Thời gian hoàn vốn (năm)",payback],["WACC",wacc_value],["Tỷ lệ chiết khấu",discount]],columns=["Chỉ tiêu","Giá trị"])
        st.download_button("⬇️ XUẤT BÁO CÁO EXCEL", dataframe_to_excel_bytes({"Dong_tien":export_df,"Ket_qua":summary}), "danh_gia_hieu_qua_dau_tu.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
