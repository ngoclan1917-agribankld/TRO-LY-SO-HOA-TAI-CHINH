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


# ============================================================
# CẤU HÌNH CHUNG
# ============================================================
APP_TITLE = "TRỢ LÝ TÀI CHÍNH NHỎ"
MODEL_NAME = "gemini-3.7-flash"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS - GIAO DIỆN
# ============================================================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: Arial, Helvetica, sans-serif;
    }

    .main-title {
        color: #C00000;
        font-size: 2.25rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: .4px;
        margin: .1rem 0 .15rem 0;
    }

    .sub-title {
        text-align: center;
        color: #555555;
        font-size: 0.98rem;
        margin-bottom: 1rem;
    }

    .section-head {
        color: #A00000;
        font-size: 1.35rem;
        font-weight: 800;
        margin: .4rem 0 .9rem 0;
        padding-bottom: .45rem;
        border-bottom: 2px solid #D9A3A3;
    }

    .sub-head {
        color: #7A0000;
        font-size: 1.05rem;
        font-weight: 700;
        margin: .35rem 0 .55rem 0;
    }

    .info-box {
        border: 1px solid #D9DDE3;
        background: #F8F9FA;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 12px;
        min-height: 106px;
    }

    .input-box {
        border: 1px solid #D8DDE6;
        background: #FCFCFD;
        border-radius: 12px;
        padding: 12px 14px 8px 14px;
    }

    .result-box {
        border: 1px solid #C9D8C9;
        background: #F8FCF8;
        border-radius: 12px;
        padding: 12px 14px 8px 14px;
        min-height: 120px;
    }

    .metric-card {
        border: 1px solid #D4D8DE;
        background: white;
        border-radius: 12px;
        padding: 12px 14px;
        min-height: 90px;
        box-shadow: 0 1px 3px rgba(0,0,0,.05);
    }

    .metric-name {
        color: #555;
        font-size: 0.88rem;
        font-weight: 600;
    }

    .metric-value {
        color: #1F2937;
        font-size: 1.32rem;
        font-weight: 800;
        margin-top: 2px;
    }

    .meaning {
        color: #666;
        font-size: 0.86rem;
        margin-top: 3px;
        line-height: 1.35;
    }

    .guide-box {
        border-left: 4px solid #C00000;
        background: #FFF7F7;
        padding: 10px 12px;
        border-radius: 7px;
        margin-bottom: 12px;
    }

    .positive-box {
        border: 1px solid #B8D7B8;
        background: #F3FBF3;
        color: #215A21;
        padding: 12px 14px;
        border-radius: 10px;
        font-weight: 700;
    }

    .negative-box {
        border: 1px solid #E2B6B6;
        background: #FFF5F5;
        color: #8A1C1C;
        padding: 12px 14px;
        border-radius: 10px;
        font-weight: 700;
    }

    .neutral-box {
        border: 1px solid #D8D8B2;
        background: #FFFDF1;
        color: #665B00;
        padding: 12px 14px;
        border-radius: 10px;
        font-weight: 700;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #E1E1E1;
    }

    [data-testid="stSidebar"] h2 {
        color: #C00000;
    }

    .stButton > button {
        border-radius: 9px;
        font-weight: 700;
    }

    div[data-testid="stForm"] {
        border: 0 !important;
        padding: 0 !important;
    }

    .small-note {
        color: #666;
        font-size: 0.83rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================
def money(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.0f} đ".replace(",", ".")


def percent(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.2f}%"


def safe_float(value, default=0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
        return default


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        return str(value) if value else default
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
        return genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=60000) if types else None,
        )
    except Exception:
        return None


def call_gemini(instruction: str, data: dict) -> str:
    client = get_gemini_client()
    if client is None or types is None:
        return (
            "Chưa cấu hình Gemini API. Kết quả tính toán vẫn hoạt động bình thường. "
            "Hãy thêm GEMINI_API_KEY vào Streamlit Secrets để sử dụng phân tích AI."
        )

    system_instruction = """
Bạn là Trợ lý Tài chính Nhỏ, hỗ trợ tiểu thương, hộ kinh doanh, nông hộ,
cơ sở sản xuất nhỏ và hợp tác xã.

Quy tắc bắt buộc:
1. Chỉ dùng số liệu trong dữ liệu được cung cấp.
2. Không tự tạo hoặc thay đổi số liệu.
3. Phân biệt KẾT QUẢ, NHẬN XÉT và CẢNH BÁO.
4. Giải thích thuật ngữ tài chính bằng tiếng Việt đời thường.
5. Không khẳng định chắc chắn lợi nhuận tương lai.
6. Với đầu tư, nhấn mạnh kết quả phụ thuộc vào giả định dòng tiền và tỷ lệ chiết khấu.
7. Không thay thế kế toán, kiểm toán hoặc quyết định tín dụng chuyên môn.
8. Trình bày ngắn gọn, rõ ràng, có cấu trúc.
"""

    prompt = (
        f"{instruction}\n\n"
        "DỮ LIỆU ĐÃ ĐƯỢC HỆ THỐNG TÍNH TOÁN:\n"
        f"{json.dumps(data, ensure_ascii=False, indent=2, default=str)}"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=800,
                temperature=0.2,
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

    grid = np.concatenate([
        np.linspace(-0.99, -0.01, 120),
        np.linspace(0.0, 5.0, 400),
    ])
    prev_r, prev_v = grid[0], f(grid[0])

    for current_r in grid[1:]:
        current_v = f(current_r)
        if np.isfinite(prev_v) and np.isfinite(current_v):
            if prev_v == 0:
                return float(prev_r)
            if current_v == 0 or prev_v * current_v < 0:
                low, high = prev_r, current_r
                flow, fhigh = prev_v, current_v
                for _ in range(120):
                    mid = (low + high) / 2
                    fmid = f(mid)
                    if not np.isfinite(fmid):
                        break
                    if abs(fmid) < 1e-9:
                        return float(mid)
                    if flow * fmid <= 0:
                        high, fhigh = mid, fmid
                    else:
                        low, flow = mid, fmid
                return float((low + high) / 2)
        prev_r, prev_v = current_r, current_v
    return None


def payback_period(cashflows: list[float]) -> Optional[float]:
    cumulative = float(cashflows[0])
    if cumulative >= 0:
        return 0.0
    for index in range(1, len(cashflows)):
        previous = cumulative
        cumulative += cashflows[index]
        if cumulative >= 0:
            step = cashflows[index]
            if step == 0:
                return float(index)
            fraction = (-previous) / step
            return (index - 1) + float(np.clip(fraction, 0, 1))
    return None


def wacc(equity_weight, debt_weight, cost_equity, cost_debt, tax_rate):
    total = equity_weight + debt_weight
    if total <= 0:
        return None
    e = equity_weight / total
    d = debt_weight / total
    return e * cost_equity + d * cost_debt * (1 - tax_rate)


def money_input(label: str, value: float = 0.0, key: Optional[str] = None, step: int = 100000):
    """Native Streamlit number input with thousands grouping."""
    return st.number_input(
        label,
        min_value=0.0,
        value=float(value),
        step=float(step),
        format="%,.0f",
        key=key,
        help="Nhập số tiền. Hệ thống tự hiển thị dấu phân cách hàng nghìn.",
    )


def metric_card(title: str, value: str, meaning: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-name">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="meaning">{meaning}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def navigate(page_name: str):
    st.session_state["page"] = page_name
    st.rerun()


# ============================================================
# KHỞI TẠO TRẠNG THÁI
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Tổng quan"
if "cashflows" not in st.session_state:
    st.session_state.cashflows = pd.DataFrame(
        columns=["Ngày", "Loại", "Nhóm", "Nội dung", "Số tiền"]
    )


# ============================================================
# HEADER
# ============================================================
st.markdown(f'<div class="main-title">{APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Số hóa dòng tiền • Tính khấu hao • Đánh giá hiệu quả đầu tư</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## DANH MỤC")
st.sidebar.markdown(
    """
    <div class="guide-box">
    <b>Hướng dẫn nhanh</b><br>
    Chọn một chức năng bên dưới → nhập dữ liệu → xem kết quả →
    dùng nút <b>Phân tích bằng AI</b> khi cần giải thích sâu hơn.
    </div>
    """,
    unsafe_allow_html=True,
)

menu_items = [
    ("🏠 Tổng quan", "Tổng quan"),
    ("💰 Sổ tay Dòng tiền", "Dòng tiền"),
    ("⚙️ Tính Khấu hao", "Khấu hao"),
    ("📈 Đánh giá Đầu tư", "Đầu tư"),
]

for label, value in menu_items:
    if st.sidebar.button(label, key=f"side_{value}", use_container_width=True):
        navigate(label)

st.sidebar.divider()
if gemini_available():
    st.sidebar.success("Gemini AI: Đã sẵn sàng")
else:
    st.sidebar.info("Gemini AI: Chưa cấu hình")

page = st.session_state.page


# ============================================================
# TỔNG QUAN
# ============================================================
if page == "🏠 Tổng quan":
    st.markdown('<div class="section-head">TỔNG QUAN CHƯƠNG TRÌNH</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="guide-box">
        <b>TRỢ LÝ TÀI CHÍNH NHỎ</b> giúp người dùng ghi chép và nhìn lại hoạt động tài chính
        theo cách đơn giản: nhập số liệu thực tế, hệ thống tự tính các chỉ tiêu và biểu đồ,
        sau đó có thể dùng AI để giải thích bằng ngôn ngữ dễ hiểu.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💰 SỔ TAY DÒNG TIỀN", key="home_cash", use_container_width=True):
            navigate("💰 Sổ tay Dòng tiền")
        st.markdown(
            """
            <div class="info-box">
            <b>Công dụng:</b> ghi lại tiền thu và tiền chi theo ngày, nhóm giao dịch và nội dung.
            Hệ thống tự cộng tổng tiền vào, tổng tiền ra, dòng tiền ròng và vẽ biểu đồ theo tháng.<br><br>
            <b>Cách dùng:</b> nhập từng giao dịch hoặc tải Excel/CSV. Sau đó xem trạng thái dòng tiền,
            biểu đồ và dùng AI để nhận xét những khoản thu/chi đáng chú ý.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("⚙️ TÍNH KHẤU HAO", key="home_dep", use_container_width=True):
            navigate("⚙️ Tính Khấu hao")
        st.markdown(
            """
            <div class="info-box">
            <b>Công dụng:</b> xác định phần giá trị của máy móc, thiết bị hoặc tài sản được phân bổ
            vào chi phí theo thời gian; đồng thời theo dõi đã khấu hao bao nhiêu và còn lại bao nhiêu.<br><br>
            <b>Cách dùng:</b> nhập tên tài sản, nguyên giá, giá trị thu hồi, ngày mua và thời gian sử dụng.
            Hệ thống tự xác định số tháng đã sử dụng, khấu hao đã phát sinh và giá trị còn lại.
            </div>
            """,
            unsafe_allow_html=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        if st.button("📈 ĐÁNH GIÁ ĐẦU TƯ", key="home_inv", use_container_width=True):
            navigate("📈 Đánh giá Đầu tư")
        st.markdown(
            """
            <div class="info-box">
            <b>Công dụng:</b> đánh giá một dự án theo vốn đầu tư và dòng tiền dự kiến.
            Hệ thống tính NPV, IRR, thời gian hoàn vốn và WACC khi đủ dữ liệu.<br><br>
            <b>Cách dùng:</b> nhập vốn đầu tư, dòng tiền từng năm và tỷ lệ chiết khấu.
            Sau đó đọc cả con số chuyên môn lẫn phần giải thích bằng tiếng Việt để hiểu ý nghĩa.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="info-box">
            <b>Nguyên tắc sử dụng:</b> dữ liệu nhập vào là cơ sở của mọi kết quả.
            Cần kiểm tra số liệu trước khi sử dụng để báo cáo hoặc ra quyết định.<br><br>
            <b>AI:</b> chỉ hỗ trợ giải thích và nhận xét từ dữ liệu đã tính, không thay thế kế toán,
            kiểm toán hoặc thẩm định chuyên môn.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Luồng sử dụng đề xuất")
    flow = st.columns(4)
    for i, (title, desc) in enumerate([
        ("1. NHẬP", "Ghi nhận số liệu thực tế"),
        ("2. TÍNH", "Hệ thống tự động tính"),
        ("3. XEM", "Đọc số liệu và biểu đồ"),
        ("4. HIỂU", "Dùng AI để giải thích"),
    ]):
        with flow[i]:
            st.markdown(
                f'<div class="metric-card"><div class="metric-name">{title}</div>'
                f'<div class="meaning">{desc}</div></div>',
                unsafe_allow_html=True,
            )


# ============================================================
# MODULE DÒNG TIỀN
# ============================================================
elif page == "💰 Sổ tay Dòng tiền":
    st.markdown('<div class="section-head">SỔ TAY DÒNG TIỀN</div>', unsafe_allow_html=True)
    st.caption("Ghi nhận dòng tiền thực tế để theo dõi tiền vào, tiền ra và dòng tiền ròng.")

    left, right = st.columns([1, 1.75])

    with left:
        st.markdown('<div class="sub-head">NHẬP DỮ LIỆU</div>', unsafe_allow_html=True)
        with st.form("cash_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                d = st.date_input("Ngày", value=date.today())
            with c2:
                typ = st.selectbox("Loại giao dịch", ["Thu", "Chi"])

            group = st.selectbox(
                "Nhóm giao dịch",
                [
                    "Bán hàng", "Nguyên liệu", "Lương", "Điện/nước",
                    "Vận chuyển", "Thuê mặt bằng", "Mua tài sản", "Khác",
                ],
            )
            content = st.text_input("Nội dung")
            amount = money_input("Số tiền (đồng)", key="cash_amount", step=10000)
            submitted = st.form_submit_button("➕ THÊM GIAO DỊCH", use_container_width=True)

        uploaded = st.file_uploader(
            "Hoặc tải file Excel/CSV",
            type=["xlsx", "csv"],
            key="cash_upload",
        )
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
                required = {"Ngày", "Loại", "Nhóm", "Nội dung", "Số tiền"}
                missing = required.difference(frame.columns)
                if missing:
                    st.error("File thiếu cột: " + ", ".join(sorted(missing)))
                else:
                    st.session_state.cashflows = frame.copy()
                    st.success("Đã nạp dữ liệu từ file.")
            except Exception as exc:
                st.error(f"Không đọc được file: {exc}")

        if submitted:
            if amount <= 0:
                st.error("Số tiền phải lớn hơn 0.")
            else:
                row = pd.DataFrame(
                    [[pd.Timestamp(d), typ, group, content, amount]],
                    columns=st.session_state.cashflows.columns,
                )
                st.session_state.cashflows = pd.concat(
                    [st.session_state.cashflows, row], ignore_index=True
                )
                st.success("Đã thêm giao dịch.")

        if st.button("🗑️ XÓA TOÀN BỘ GIAO DỊCH", key="clear_cash", use_container_width=True):
            st.session_state.cashflows = pd.DataFrame(
                columns=["Ngày", "Loại", "Nhóm", "Nội dung", "Số tiền"]
            )
            st.rerun()

    frame = st.session_state.cashflows.copy()
    if not frame.empty:
        frame["Ngày"] = pd.to_datetime(frame["Ngày"], errors="coerce")
        frame["Số tiền"] = pd.to_numeric(frame["Số tiền"], errors="coerce").fillna(0)

    with right:
        st.markdown('<div class="sub-head">KẾT QUẢ VÀ PHÂN TÍCH</div>', unsafe_allow_html=True)
        if frame.empty:
            st.info("Chưa có dữ liệu. Hãy nhập giao dịch hoặc tải file.")
        else:
            total_in = float(frame.loc[frame["Loại"].eq("Thu"), "Số tiền"].sum())
            total_out = float(frame.loc[frame["Loại"].eq("Chi"), "Số tiền"].sum())
            net = total_in - total_out

            m1, m2, m3 = st.columns(3)
            with m1:
                metric_card("Tổng tiền vào", money(total_in), "Tổng các khoản thu đã ghi nhận")
            with m2:
                metric_card("Tổng tiền ra", money(total_out), "Tổng các khoản chi đã ghi nhận")
            with m3:
                metric_card("Dòng tiền ròng", money(net), "Tiền vào trừ tiền ra")

            st.markdown("### Trạng thái dòng tiền")
            if net > 0:
                st.markdown(
                    '<div class="positive-box">Dòng tiền đang dương. Tiền thu vào lớn hơn tiền chi ra, kết quả kinh doanh có dấu hiệu sinh lời</div>',
                    unsafe_allow_html=True,
                )
            elif net < 0:
                st.markdown(
                    '<div class="negative-box">Dòng tiền đang âm. Tiền thu vào đang ít hơn tiền chi ra, có thể xem xét lại các khoản chi phí và kết quả kinh doanh</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="neutral-box">Dòng tiền đang cân bằng. Tiền thu vào bằng tiền chi ra trong số liệu đã nhập.</div>',
                    unsafe_allow_html=True,
                )

            monthly = (
                frame.assign(Tháng=frame["Ngày"].dt.to_period("M").astype(str))
                .groupby(["Tháng", "Loại"], as_index=False)["Số tiền"].sum()
            )
            chart1, chart2 = st.columns(2)
            with chart1:
                fig = px.bar(
                    monthly, x="Tháng", y="Số tiền", color="Loại", barmode="group",
                    title="Tiền vào – tiền ra theo tháng",
                )
                fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with chart2:
                net_month = (
                    frame.assign(
                        Tháng=frame["Ngày"].dt.to_period("M").astype(str),
                        signed=np.where(frame["Loại"].eq("Thu"), frame["Số tiền"], -frame["Số tiền"]),
                    )
                    .groupby("Tháng", as_index=False)["signed"].sum()
                    .rename(columns={"signed": "Dòng tiền ròng"})
                )
                fig2 = px.line(net_month, x="Tháng", y="Dòng tiền ròng", markers=True, title="Dòng tiền ròng theo tháng")
                fig2.update_layout(margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig2, use_container_width=True)

            with st.expander("Xem bảng dữ liệu"):
                st.dataframe(
                    frame,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Số tiền": st.column_config.NumberColumn("Số tiền", format="%,d"),
                    },
                )

            a1, a2 = st.columns(2)
            with a1:
                if st.button("🤖 PHÂN TÍCH BẰNG AI", key="cash_ai", use_container_width=True):
                    data = {
                        "tong_tien_vao": total_in,
                        "tong_tien_ra": total_out,
                        "dong_tien_rong": net,
                        "so_giao_dich": int(len(frame)),
                    }
                    st.markdown(call_gemini(
                        "Phân tích dòng tiền. Nêu điểm nổi bật, nhóm chi đáng chú ý nếu có và cảnh báo thực tế.",
                        data,
                    ))
            with a2:
                excel = dataframe_to_excel_bytes({"So_tay_dong_tien": frame})
                st.download_button(
                    "⬇️ XUẤT EXCEL", excel, "so_tay_dong_tien.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )


# ============================================================
# MODULE KHẤU HAO
# ============================================================
elif page == "⚙️ Tính Khấu hao":
    st.markdown('<div class="section-head">TÍNH KHẤU HAO</div>', unsafe_allow_html=True)
    st.caption("Mặc định sử dụng phương pháp khấu hao đường thẳng.")

    left, right = st.columns([1, 1.75])
    with left:
        st.markdown('<div class="sub-head">NHẬP DỮ LIỆU TÀI SẢN</div>', unsafe_allow_html=True)
        asset = st.text_input("Tên tài sản", placeholder="Ví dụ: Máy cày")
        cost = money_input("Nguyên giá (đồng)", key="dep_cost", step=1000000)
        salvage = money_input("Giá trị thu hồi (đồng)", key="dep_salvage", step=1000000)
        purchase_date = st.date_input("Ngày mua tài sản", value=date.today())
        years = st.number_input(
            "Thời gian sử dụng (năm)", min_value=1, max_value=100, value=5, step=1,
            format="%d", key="dep_years",
        )

        valid = cost > 0 and 0 <= salvage <= cost and purchase_date <= date.today()
        if not valid:
            if purchase_date > date.today():
                st.error("Ngày mua tài sản không được lớn hơn ngày hiện tại.")
            elif salvage > cost:
                st.error("Giá trị thu hồi không được lớn hơn nguyên giá.")

    with right:
        st.markdown('<div class="sub-head">KẾT QUẢ KHẤU HAO VÀ GIÁ TRỊ CÒN LẠI</div>', unsafe_allow_html=True)
        if not valid:
            st.info("Hãy hoàn thiện dữ liệu hợp lệ để xem kết quả.")
        else:
            annual = (cost - salvage) / years
            monthly = annual / 12
            elapsed_days = max((date.today() - purchase_date).days, 0)
            elapsed_months = elapsed_days / 30.4375
            total_months = years * 12
            used_months = min(elapsed_months, total_months)
            accumulated = min(annual * used_months / 12, cost - salvage)
            book_value = max(cost - accumulated, salvage)
            remaining_months = max(total_months - used_months, 0)

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                metric_card("Khấu hao năm", money(annual), "Mức khấu hao trung bình mỗi năm")
            with r2:
                metric_card("Khấu hao tháng", money(monthly), "Mức khấu hao trung bình mỗi tháng")
            with r3:
                metric_card("Đã khấu hao", money(accumulated), "Giá trị khấu hao ước tính đã phát sinh")
            with r4:
                metric_card("Còn lại", money(book_value), "Giá trị còn lại trên cơ sở khấu hao đường thẳng")

            s1, s2 = st.columns(2)
            with s1:
                metric_card("Thời gian đã sử dụng", f"{used_months:.1f} tháng", "Tính từ ngày mua đến hôm nay")
            with s2:
                metric_card("Thời gian sử dụng còn lại", f"{remaining_months:.1f} tháng", "Thời gian còn lại theo vòng đời đã nhập")

            st.markdown("### Bảng phân bổ khấu hao")
            schedule = []
            for y in range(1, int(years) + 1):
                remaining = max(cost - annual * y, salvage)
                schedule.append({
                    "Năm": y,
                    "Khấu hao năm": annual,
                    "Giá trị còn lại cuối năm": remaining,
                })
            schedule_df = pd.DataFrame(schedule)
            st.dataframe(
                schedule_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Khấu hao năm": st.column_config.NumberColumn("Khấu hao năm", format="%,d đ"),
                    "Giá trị còn lại cuối năm": st.column_config.NumberColumn("Giá trị còn lại cuối năm", format="%,d đ"),
                },
            )

            fig = px.line(
                schedule_df, x="Năm", y="Giá trị còn lại cuối năm",
                markers=True, title="Giá trị còn lại theo thời gian",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)

            b1, b2 = st.columns(2)
            with b1:
                if st.button("🤖 PHÂN TÍCH BẰNG AI", key="dep_ai", use_container_width=True):
                    data = {
                        "tai_san": asset or "Chưa đặt tên",
                        "nguyen_gia": cost,
                        "gia_tri_thu_hoi": salvage,
                        "ngay_mua": str(purchase_date),
                        "thoi_gian_su_dung_nam": years,
                        "da_su_dung_thang": round(used_months, 1),
                        "thoi_gian_con_lai_thang": round(remaining_months, 1),
                        "khau_hao_nam": annual,
                        "khau_hao_thang": monthly,
                        "khau_hao_da_phat_sinh": accumulated,
                        "gia_tri_con_lai": book_value,
                    }
                    st.markdown(call_gemini(
                        "Giải thích khấu hao, số tháng đã sử dụng, khấu hao đã phát sinh và giá trị còn lại bằng ngôn ngữ dễ hiểu.",
                        data,
                    ))
            with b2:
                export = dataframe_to_excel_bytes({
                    "Tai_san": pd.DataFrame([{
                        "Tên tài sản": asset,
                        "Nguyên giá": cost,
                        "Giá trị thu hồi": salvage,
                        "Ngày mua": purchase_date,
                        "Thời gian sử dụng (năm)": years,
                        "Đã sử dụng (tháng)": round(used_months, 1),
                        "Khấu hao đã phát sinh": accumulated,
                        "Khấu hao còn lại": max((cost - salvage) - accumulated, 0),
                        "Giá trị còn lại": book_value,
                    }]),
                    "Phan_bo_khau_hao": schedule_df,
                })
                st.download_button(
                    "⬇️ XUẤT EXCEL", export, "tinh_khau_hao.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )


# ============================================================
# MODULE ĐÁNH GIÁ ĐẦU TƯ
# ============================================================
elif page == "📈 Đánh giá Đầu tư":
    st.markdown('<div class="section-head">ĐÁNH GIÁ HIỆU QUẢ ĐẦU TƯ</div>', unsafe_allow_html=True)
    st.caption("Nhập dòng tiền dự kiến để hệ thống tính toán các chỉ tiêu và giải thích ý nghĩa.")

    left, right = st.columns([1, 1.75])
    with left:
        st.markdown('<div class="sub-head">THÔNG TIN VÀ GIẢ ĐỊNH ĐẦU VÀO</div>', unsafe_allow_html=True)
        project = st.text_input("Tên dự án", placeholder="Ví dụ: Đầu tư máy sấy nông sản")
        initial = money_input("Vốn đầu tư ban đầu (đồng)", key="inv_initial", step=1000000)
        n_periods = st.number_input("Số kỳ dự kiến", min_value=1, max_value=50, value=5, step=1, format="%d")
        discount = st.number_input(
            "Tỷ lệ chiết khấu / mức sinh lời yêu cầu (%)",
            min_value=-99.0, max_value=500.0, value=10.0, step=0.5,
            format="%.2f%%",
        ) / 100

        st.markdown("#### Dòng tiền từng kỳ")
        cashflows = [-initial]
        for i in range(1, int(n_periods) + 1):
            cashflows.append(
                money_input(f"Dòng tiền kỳ {i} (đồng)", key=f"inv_cf_{i}", step=1000000)
            )

        st.markdown("#### WACC - CHI PHÍ SỬ DỤNG VỐN BÌNH QUÂN GIA QUYỀN")
        equity_weight = st.number_input("Vốn chủ sở hữu (%)", 0.0, 100.0, 100.0, 1.0, format="%.1f%%") / 100
        debt_weight = st.number_input("Vốn vay (%)", 0.0, 100.0, 0.0, 1.0, format="%.1f%%") / 100
        cost_equity = st.number_input("Chi phí vốn chủ sở hữu (%)", 0.0, 100.0, 10.0, 0.5, format="%.2f%%") / 100
        cost_debt = st.number_input("Chi phí vốn vay (%)", 0.0, 100.0, 8.0, 0.5, format="%.2f%%") / 100
        tax_rate = st.number_input("Thuế suất (%)", 0.0, 100.0, 20.0, 0.5, format="%.2f%%") / 100

        st.markdown('<div class="small-note">Tỷ trọng vốn chủ sở hữu + vốn vay có thể khác 100%; hệ thống sẽ chuẩn hóa khi tính WACC.</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="sub-head">CÁC CHỈ TIÊU VÀ Ý NGHĨA</div>', unsafe_allow_html=True)

        try:
            npv_value = npv(discount, cashflows)
        except Exception:
            npv_value = np.nan
        irr_value = irr_bisection(cashflows)
        payback = payback_period(cashflows)
        wacc_value = wacc(equity_weight, debt_weight, cost_equity, cost_debt, tax_rate)

        # Từ ngữ + định nghĩa chuyên môn
        terms = st.columns(4)
        with terms[0]:
            metric_card(
                "NPV – GIÁ TRỊ HIỆN TẠI RÒNG",
                money(npv_value),
                "Là tổng chênh lệch giữa dòng tiền thu vào và dòng tiền chi ra, quy đổi về hiện tại theo một tỷ suất chiết khấu nhất định.",
            )
        with terms[1]:
            metric_card(
                "IRR – TỶ SUẤT HOÀN VỐN NỘI BỘ",
                percent(irr_value),
                "Là mức tỷ suất làm cho NPV bằng 0; dùng để so sánh khả năng sinh lời của dự án với mức sinh lời yêu cầu.",
            )
        with terms[2]:
            metric_card(
                "PAYBACK – THỜI GIAN HOÀN VỐN",
                f"{payback:.1f} năm" if payback is not None else "Chưa hoàn vốn",
                "Là khoảng thời gian dự kiến để dòng tiền tích lũy bù đắp vốn đầu tư ban đầu.",
            )
        with terms[3]:
            metric_card(
                "WACC – CHI PHÍ SỬ DỤNG VỐN BÌNH QUÂN",
                percent(wacc_value),
                "Là chi phí vốn bình quân theo tỷ trọng vốn chủ và vốn vay, có điều chỉnh thuế đối với nợ vay.",
            )

        st.markdown("### Kết luận dễ hiểu")
        if npv_value > 0:
            st.markdown(
                '<div class="positive-box"><b>NPV dương:</b> Dự án đang có tín hiệu tạo thêm giá trị theo mức chiết khấu đã nhập.</div>',
                unsafe_allow_html=True,
            )
        elif npv_value < 0:
            st.markdown(
                '<div class="negative-box"><b>NPV âm:</b> Dự án chưa đạt mức sinh lời yêu cầu theo mức chiết khấu đã nhập.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="neutral-box"><b>NPV bằng 0:</b> Dự án vừa đạt mức sinh lời yêu cầu theo giả định hiện tại.</div>',
                unsafe_allow_html=True,
            )

        if irr_value is not None:
            if irr_value > discount:
                st.success(f"IRR ({percent(irr_value)}) cao hơn mức chiết khấu ({percent(discount)}): tín hiệu tích cực.")
            else:
                st.warning(f"IRR ({percent(irr_value)}) không cao hơn mức chiết khấu ({percent(discount)}).")

        cum = np.cumsum(cashflows)
        chart_df = pd.DataFrame({"Kỳ": range(len(cum)), "Dòng tiền tích lũy": cum})
        fig = px.line(chart_df, x="Kỳ", y="Dòng tiền tích lũy", markers=True, title="Dòng tiền tích lũy")
        fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("🤖 PHÂN TÍCH BẰNG AI", key="inv_ai", use_container_width=True):
                data = {
                    "du_an": project or "Chưa đặt tên",
                    "von_dau_tu_ban_dau": initial,
                    "dong_tien": cashflows,
                    "ty_le_chiet_khau": discount,
                    "NPV": npv_value,
                    "IRR": irr_value,
                    "thoi_gian_hoan_von_nam": payback,
                    "WACC": wacc_value,
                }
                st.markdown(call_gemini(
                    "Phân tích dự án theo 3 phần: Kết quả tính toán; Ý nghĩa dễ hiểu; Cảnh báo và giả định cần kiểm tra.",
                    data,
                ))
        with b2:
            export = dataframe_to_excel_bytes({
                "Dong_tien": pd.DataFrame({
                    "Kỳ": range(len(cashflows)),
                    "Dòng tiền": cashflows,
                    "Dòng tiền tích lũy": cum,
                }),
                "Ket_qua": pd.DataFrame([
                    ["NPV - Giá trị hiện tại ròng", npv_value],
                    ["IRR - Tỷ suất hoàn vốn nội bộ", irr_value],
                    ["Payback - Thời gian hoàn vốn (năm)", payback],
                    ["WACC - Chi phí sử dụng vốn bình quân", wacc_value],
                    ["Tỷ lệ chiết khấu", discount],
                ], columns=["Chỉ tiêu", "Giá trị"]),
            })
            st.download_button(
                "⬇️ XUẤT BÁO CÁO EXCEL", export, "danh_gia_dau_tu.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
