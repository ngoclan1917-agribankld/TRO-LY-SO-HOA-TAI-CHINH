import io
import json
from datetime import date

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


APP_TITLE = "Trợ lý Số hóa Tài chính Cơ sở"
MODEL_NAME = "gemini-3.7-flash"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# STYLE
# =========================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .sub-title {
        color: #5f6368;
        font-size: 1.02rem;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0.6rem;
    }
    .result-box {
        padding: 0.9rem 1rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(49,51,63,.18);
        background: rgba(250,250,250,.75);
    }
    .metric-label {
        color: #5f6368;
        font-size: 0.86rem;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
def money(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:,.0f} đ".replace(",", ".")


def percent(v: float) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v * 100:.2f}%"


def parse_float(value, default=0.0) -> float:
    try:
        x = float(value)
        if not np.isfinite(x):
            return default
        return x
    except Exception:
        return default


def get_gemini_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return ""


def ai_available() -> bool:
    return genai is not None and bool(get_gemini_key())


@st.cache_resource
def get_gemini_client():
    key = get_gemini_key()
    if not key or genai is None:
        return None
    return genai.Client(api_key=key)


def call_gemini(instruction: str, data: dict) -> str:
    client = get_gemini_client()
    if client is None:
        return (
            "Chưa cấu hình Gemini API. Kết quả tính toán vẫn được hiển thị "
            "bình thường. Hãy thêm GEMINI_API_KEY vào Streamlit Secrets."
        )

    system_instruction = """
Bạn là “Trợ lý tài chính bình dân” cho tiểu thương, hộ kinh doanh,
nông hộ, cơ sở sản xuất nhỏ và hợp tác xã.

Nguyên tắc:
1. Chỉ sử dụng số liệu được cung cấp.
2. Không tự tạo, sửa hoặc suy diễn số liệu thành số liệu thực tế.
3. Phân biệt rõ: KẾT QUẢ TÍNH TOÁN, NHẬN XÉT, CẢNH BÁO.
4. Giải thích thuật ngữ tài chính bằng tiếng Việt dễ hiểu.
5. Không khẳng định chắc chắn lợi nhuận tương lai.
6. Với đầu tư, nêu rõ kết luận phụ thuộc vào các giả định dòng tiền
   và tỷ lệ chiết khấu.
7. Không đưa ra quyết định tín dụng hoặc khuyến nghị vay vốn thay cho
   cán bộ chuyên môn.
8. Trình bày ngắn gọn, ưu tiên các ý chính.
"""

    prompt = (
        f"{instruction}\n\n"
        "DỮ LIỆU ĐÃ ĐƯỢC HỆ THỐNG TÍNH TOÁN (JSON):\n"
        f"{json.dumps(data, ensure_ascii=False, indent=2)}"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        return response.text or "AI không trả về nội dung."
    except Exception as exc:
        return f"Không thể gọi Gemini API lúc này: {exc}"


def dataframe_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = str(name)[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return output.getvalue()


def xirr_like_payback(cashflows: list[float]):
    cumulative = cashflows[0]
    if cumulative >= 0:
        return 0.0

    for i in range(1, len(cashflows)):
        prev = cumulative
        cumulative += cashflows[i]

        if cumulative >= 0:
            step_cash = cashflows[i]
            if step_cash == 0:
                return float(i)
            fraction = (-prev) / step_cash
            return (i - 1) + float(np.clip(fraction, 0, 1))

    return None


def npv(rate: float, cashflows: list[float]) -> float:
    return float(sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows)))


def irr_bisection(cashflows: list[float]) -> float | None:
    """Robust IRR approximation without depending on a separate financial package."""
    if len(cashflows) < 2:
        return None

    values = np.array(cashflows, dtype=float)
    if not (np.any(values > 0) and np.any(values < 0)):
        return None

    def f(r):
        if r <= -0.999999:
            return np.inf
        try:
            return npv(r, cashflows)
        except Exception:
            return np.inf

    # Scan for a sign change.
    grid = np.concatenate(
        [
            np.linspace(-0.99, -0.01, 100),
            np.linspace(0.0, 5.0, 300),
        ]
    )
    prev_r = grid[0]
    prev_v = f(prev_r)

    for r in grid[1:]:
        curr_v = f(r)
        if np.isfinite(prev_v) and np.isfinite(curr_v):
            if prev_v == 0:
                return prev_r
            if curr_v == 0 or prev_v * curr_v < 0:
                lo, hi = prev_r, r
                flo, fhi = prev_v, curr_v
                for _ in range(200):
                    mid = (lo + hi) / 2
                    fm = f(mid)
                    if not np.isfinite(fm):
                        break
                    if abs(fm) < 1e-8:
                        return float(mid)
                    if flo * fm <= 0:
                        hi, fhi = mid, fm
                    else:
                        lo, flo = mid, fm
                return float((lo + hi) / 2)
        prev_r, prev_v = r, curr_v

    return None


def compute_wacc(
    equity_weight: float,
    debt_weight: float,
    cost_equity: float,
    cost_debt: float,
    tax_rate: float,
):
    total = equity_weight + debt_weight
    if total <= 0:
        return None
    e = equity_weight / total
    d = debt_weight / total
    return e * cost_equity + d * cost_debt * (1 - tax_rate)


# =========================
# SIDEBAR
# =========================
st.sidebar.title("📱 Điều hướng")
page = st.sidebar.radio(
    "Chọn chức năng",
    [
        "🏠 Tổng quan",
        "💰 Sổ tay Dòng tiền",
        "⚙️ Tính Khấu hao",
        "📈 Đánh giá Đầu tư",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Trợ lý Số hóa Tài chính Cơ sở\n\n"
    "Nguyên tắc: Đầu vào → Xử lý → Kết quả"
)


# =========================
# HOME
# =========================
if page == "🏠 Tổng quan":
    st.markdown(f'<div class="main-title">📊 {APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        "Ứng dụng “Bình dân học vụ số” giúp số hóa dòng tiền, khấu hao "
        "và đánh giá hiệu quả đầu tư."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("### 💰 Dòng tiền\nTheo dõi tiền vào – tiền ra theo ngày, tháng, năm.")
    with c2:
        st.info("### ⚙️ Khấu hao\nPhân bổ giá trị tài sản thành chi phí dễ hiểu.")
    with c3:
        st.info("### 📈 Đầu tư\nTính NPV, IRR, thời gian hoàn vốn và WACC.")

    st.divider()
    st.subheader("Cách sử dụng")
    st.markdown(
        """
        **1. Nhập dữ liệu ở bên trái** → **2. Hệ thống tự tính ở giữa**
        → **3. Xem kết quả và giải thích ở bên phải.**

        Ứng dụng không thay thế kế toán, kiểm toán hoặc quyết định tín dụng.
        Các chỉ tiêu đầu tư phụ thuộc vào dữ liệu và giả định do người dùng nhập.
        """
    )

    if ai_available():
        st.success("Gemini API: Đã sẵn sàng")
    else:
        st.warning(
            "Gemini API chưa được cấu hình. Các module tính toán vẫn hoạt động; "
            "phần phân tích AI cần GEMINI_API_KEY."
        )


# =========================
# CASH FLOW
# =========================
elif page == "💰 Sổ tay Dòng tiền":
    st.markdown("## 💰 Sổ tay Dòng tiền")
    st.caption("Ghi chép “tiền tươi thóc thật” một cách đơn giản.")

    if "cashflows" not in st.session_state:
        st.session_state.cashflows = pd.DataFrame(
            columns=["Ngày", "Loại", "Nhóm", "Nội dung", "Số tiền"]
        )

    left, mid, right = st.columns([1.15, 1.0, 1.55])

    with left:
        st.markdown("### 1. Nhập dữ liệu")
        with st.form("cash_form", clear_on_submit=True):
            d = st.date_input("Ngày", value=date.today())
            typ = st.selectbox("Loại giao dịch", ["Thu", "Chi"])
            group = st.selectbox(
                "Nhóm",
                [
                    "Bán hàng",
                    "Nguyên liệu",
                    "Lương",
                    "Điện/nước",
                    "Vận chuyển",
                    "Thuê mặt bằng",
                    "Mua tài sản",
                    "Khác",
                ],
            )
            content = st.text_input("Nội dung")
            amount = st.number_input("Số tiền (đồng)", min_value=0.0, step=10000.0)
            submitted = st.form_submit_button("➕ Thêm giao dịch")

        if submitted and amount > 0:
            new_row = pd.DataFrame(
                [[pd.Timestamp(d), typ, group, content, amount]],
                columns=st.session_state.cashflows.columns,
            )
            st.session_state.cashflows = pd.concat(
                [st.session_state.cashflows, new_row], ignore_index=True
            )
            st.success("Đã thêm giao dịch.")

        uploaded = st.file_uploader(
            "Hoặc tải Excel/CSV",
            type=["xlsx", "csv"],
            key="cash_upload",
        )
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    df_up = pd.read_csv(uploaded)
                else:
                    df_up = pd.read_excel(uploaded)
                st.session_state.cashflows = df_up.copy()
                st.success("Đã nạp dữ liệu từ file.")
            except Exception as exc:
                st.error(f"Không đọc được file: {exc}")

    df = st.session_state.cashflows.copy()

    if not df.empty:
        df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
        df["Số tiền"] = pd.to_numeric(df["Số tiền"], errors="coerce").fillna(0)

    with mid:
        st.markdown("### 2. Hệ thống tính")
        if df.empty:
            total_in = total_out = net = 0.0
        else:
            total_in = float(df.loc[df["Loại"].eq("Thu"), "Số tiền"].sum())
            total_out = float(df.loc[df["Loại"].eq("Chi"), "Số tiền"].sum())
            net = total_in - total_out

        st.metric("Tổng tiền vào", money(total_in))
        st.metric("Tổng tiền ra", money(total_out))
        st.metric("Dòng tiền ròng", money(net))

        if net > 0:
            st.success("Dòng tiền đang dương.")
        elif net < 0:
            st.warning("Dòng tiền đang âm: tiền ra lớn hơn tiền vào.")
        else:
            st.info("Dòng tiền đang cân bằng.")

    with right:
        st.markdown("### 3. Kết quả")
        if df.empty:
            st.info("Chưa có dữ liệu. Hãy nhập ít nhất một giao dịch.")
        else:
            monthly = (
                df.assign(Tháng=df["Ngày"].dt.to_period("M").astype(str))
                .groupby(["Tháng", "Loại"], as_index=False)["Số tiền"]
                .sum()
            )
            fig = px.bar(
                monthly,
                x="Tháng",
                y="Số tiền",
                color="Loại",
                barmode="group",
                title="Tiền vào – tiền ra theo tháng",
            )
            st.plotly_chart(fig, use_container_width=True)

            net_month = (
                df.assign(
                    Tháng=df["Ngày"].dt.to_period("M").astype(str),
                    signed=np.where(df["Loại"].eq("Thu"), df["Số tiền"], -df["Số tiền"]),
                )
                .groupby("Tháng", as_index=False)["signed"]
                .sum()
                .rename(columns={"signed": "Dòng tiền ròng"})
            )
            fig2 = px.line(
                net_month,
                x="Tháng",
                y="Dòng tiền ròng",
                markers=True,
                title="Dòng tiền ròng theo tháng",
            )
            st.plotly_chart(fig2, use_container_width=True)

            if st.button("🤖 Phân tích bằng AI", key="cash_ai"):
                data = {
                    "tong_tien_vao": total_in,
                    "tong_tien_ra": total_out,
                    "dong_tien_rong": net,
                    "so_giao_dich": int(len(df)),
                }
                instruction = (
                    "Hãy phân tích tình hình dòng tiền. Nêu 2-3 điểm đáng chú ý "
                    "và 1-2 cảnh báo thực tế nếu có. Trình bày bằng ngôn ngữ bình dân."
                )
                st.markdown(call_gemini(instruction, data))

            export_bytes = dataframe_to_excel_bytes(
                {"So_tay_dong_tien": df}
            )
            st.download_button(
                "⬇️ Xuất Excel",
                data=export_bytes,
                file_name="so_tay_dong_tien.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.markdown("### Dữ liệu")
        st.dataframe(df, use_container_width=True, hide_index=True)


# =========================
# DEPRECIATION
# =========================
elif page == "⚙️ Tính Khấu hao":
    st.markdown("## ⚙️ Tính Khấu hao")
    st.caption("Mặc định sử dụng phương pháp khấu hao đường thẳng.")

    left, mid, right = st.columns([1.15, 1.0, 1.55])

    with left:
        st.markdown("### 1. Nhập dữ liệu")
        asset = st.text_input("Tên tài sản", placeholder="Ví dụ: Máy cày")
        cost = st.number_input("Nguyên giá (đồng)", min_value=0.0, step=1000000.0)
        salvage = st.number_input(
            "Giá trị thu hồi (đồng)", min_value=0.0, step=1000000.0
        )
        years = st.number_input(
            "Thời gian sử dụng (năm)", min_value=1, max_value=100, value=5, step=1
        )

        valid_dep = cost > 0 and 0 <= salvage <= cost and years > 0

    with mid:
        st.markdown("### 2. Hệ thống tính")
        if valid_dep:
            annual = (cost - salvage) / years
            monthly = annual / 12
            st.metric("Khấu hao năm", money(annual))
            st.metric("Khấu hao tháng", money(monthly))
            st.metric("Giá trị thu hồi", money(salvage))
        else:
            annual = monthly = 0.0
            st.warning("Kiểm tra nguyên giá, giá trị thu hồi và thời gian sử dụng.")

    with right:
        st.markdown("### 3. Kết quả")
        if valid_dep:
            st.success(
                f"Mỗi tháng có thể phân bổ khoảng **{money(monthly)}** "
                "giá trị tài sản vào chi phí."
            )

            schedule = []
            for y in range(1, int(years) + 1):
                end_value = max(cost - annual * y, salvage)
                schedule.append(
                    {
                        "Năm": y,
                        "Khấu hao năm": annual,
                        "Giá trị còn lại cuối năm": end_value,
                    }
                )

            schedule_df = pd.DataFrame(schedule)
            st.dataframe(
                schedule_df.style.format(
                    {
                        "Khấu hao năm": lambda x: money(x),
                        "Giá trị còn lại cuối năm": lambda x: money(x),
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            fig = px.line(
                schedule_df,
                x="Năm",
                y="Giá trị còn lại cuối năm",
                markers=True,
                title="Giá trị còn lại theo thời gian",
            )
            st.plotly_chart(fig, use_container_width=True)

            if st.button("🤖 Phân tích bằng AI", key="dep_ai"):
                data = {
                    "tai_san": asset or "Chưa đặt tên",
                    "nguyen_gia": cost,
                    "gia_tri_thu_hoi": salvage,
                    "thoi_gian_su_dung_nam": years,
                    "khau_hao_nam": annual,
                    "khau_hao_thang": monthly,
                }
                instruction = (
                    "Giải thích khấu hao của tài sản này bằng ngôn ngữ dễ hiểu. "
                    "Nêu ý nghĩa đối với việc tính chi phí sản xuất/kinh doanh."
                )
                st.markdown(call_gemini(instruction, data))


# =========================
# INVESTMENT
# =========================
elif page == "📈 Đánh giá Đầu tư":
    st.markdown("## 📈 Đánh giá Hiệu quả Đầu tư")
    st.caption("Tự động tính NPV, IRR, thời gian hoàn vốn và WACC khi đủ dữ liệu.")

    left, mid, right = st.columns([1.25, 1.0, 1.45])

    with left:
        st.markdown("### 1. Nhập dữ liệu")
        project = st.text_input(
            "Tên dự án", placeholder="Ví dụ: Đầu tư máy sấy nông sản"
        )
        initial = st.number_input(
            "Vốn đầu tư ban đầu (đồng)", min_value=0.0, step=1000000.0
        )
        n_periods = st.number_input(
            "Số kỳ dự kiến", min_value=1, max_value=50, value=5, step=1
        )
        discount = st.number_input(
            "Tỷ lệ chiết khấu / mức sinh lời yêu cầu (%)",
            min_value=-99.0,
            max_value=500.0,
            value=10.0,
            step=0.5,
        ) / 100.0

        cashflows = [-initial]
        for i in range(1, int(n_periods) + 1):
            cf = st.number_input(
                f"Dòng tiền năm {i} (đồng)",
                value=0.0,
                step=1000000.0,
                key=f"cf_{i}",
            )
            cashflows.append(cf)

        st.markdown("#### WACC (tùy chọn)")
        equity_weight = st.number_input(
            "Vốn chủ sở hữu (%)", 0.0, 100.0, 100.0, 1.0
        ) / 100.0
        debt_weight = st.number_input(
            "Vốn vay (%)", 0.0, 100.0, 0.0, 1.0
        ) / 100.0
        cost_equity = st.number_input(
            "Chi phí vốn chủ sở hữu (%)", 0.0, 100.0, 10.0, 0.5
        ) / 100.0
        cost_debt = st.number_input(
            "Chi phí vốn vay (%)", 0.0, 100.0, 8.0, 0.5
        ) / 100.0
        tax_rate = st.number_input(
            "Thuế suất (%)", 0.0, 100.0, 20.0, 0.5
        ) / 100.0

    with mid:
        st.markdown("### 2. Hệ thống tính")

        npv_value = npv(discount, cashflows)
        irr_value = irr_bisection(cashflows)
        payback = xirr_like_payback(cashflows)
        wacc_value = compute_wacc(
            equity_weight,
            debt_weight,
            cost_equity,
            cost_debt,
            tax_rate,
        )

        st.metric("NPV", money(npv_value))
        st.metric("IRR", percent(irr_value) if irr_value is not None else "Không xác định")
        st.metric(
            "Thời gian hoàn vốn",
            f"{payback:.1f} năm" if payback is not None else "Chưa hoàn vốn",
        )
        st.metric(
            "WACC",
            percent(wacc_value) if wacc_value is not None else "Chưa đủ dữ liệu",
        )

    with right:
        st.markdown("### 3. Kết quả dễ hiểu")

        if npv_value > 0:
            st.success(
                "NPV dương: dự án đang có tín hiệu **tạo thêm giá trị** "
                "theo mức chiết khấu đã nhập."
            )
        elif npv_value < 0:
            st.warning(
                "NPV âm: dự án **chưa đạt mức sinh lời yêu cầu** "
                "theo mức chiết khấu đã nhập."
            )
        else:
            st.info("NPV bằng 0: dự án vừa đạt mức sinh lời yêu cầu.")

        if irr_value is not None:
            if irr_value > discount:
                st.success(
                    f"IRR ({percent(irr_value)}) cao hơn tỷ lệ chiết khấu "
                    f"({percent(discount)}): tín hiệu tích cực."
                )
            else:
                st.warning(
                    f"IRR ({percent(irr_value)}) không cao hơn tỷ lệ chiết khấu "
                    f"({percent(discount)})."
                )

        cum = np.cumsum(cashflows)
        chart_df = pd.DataFrame(
            {"Kỳ": range(len(cum)), "Dòng tiền tích lũy": cum}
        )
        fig = px.line(
            chart_df,
            x="Kỳ",
            y="Dòng tiền tích lũy",
            markers=True,
            title="Dòng tiền tích lũy",
        )
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🤖 Phân tích bằng AI", key="invest_ai"):
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
            instruction = (
                "Phân tích hiệu quả tài chính dự án theo 3 phần: "
                "Kết quả tính toán, Nhận xét dễ hiểu, Cảnh báo/điểm cần kiểm tra. "
                "Không khẳng định chắc chắn dự án có lợi nhuận trong thực tế."
            )
            st.markdown(call_gemini(instruction, data))

        export_df = pd.DataFrame(
            {
                "Kỳ": range(len(cashflows)),
                "Dòng tiền": cashflows,
                "Dòng tiền tích lũy": cum,
            }
        )
        summary_df = pd.DataFrame(
            [
                ["NPV", npv_value],
                ["IRR", irr_value if irr_value is not None else np.nan],
                ["Thời gian hoàn vốn (năm)", payback if payback is not None else np.nan],
                ["WACC", wacc_value if wacc_value is not None else np.nan],
                ["Tỷ lệ chiết khấu", discount],
            ],
            columns=["Chỉ tiêu", "Giá trị"],
        )
        export_bytes = dataframe_to_excel_bytes(
            {
                "Dong_tien": export_df,
                "Ket_qua": summary_df,
            }
        )
        st.download_button(
            "⬇️ Xuất báo cáo Excel",
            data=export_bytes,
            file_name="danh_gia_hieu_qua_dau_tu.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
