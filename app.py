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
MODEL_NAME = "gemini-3.7-flash"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# GIAO DIỆN
# =========================
st.markdown(
    """
<style>
:root {
    --red: #b71c1c;
    --red2: #d32f2f;
    --green: #1b5e20;
    --blue: #1565c0;
    --line: #d7dce1;
    --soft: #f7f8fa;
    --text: #202124;
}
.block-container { padding-top: 0.08rem; max-width: 1450px; }
.app-title {
    color: var(--red); text-align: center; font-size: 2.05rem;
    line-height: 1.08; font-weight: 900; margin: 0 0 .65rem 0;
    letter-spacing: .15px;
}
.section-header {
    font-size: 1.42rem; font-weight: 900; color: var(--red);
    border-bottom: 2px solid var(--red); padding-bottom: .30rem;
    margin: .38rem 0 .72rem;
}
.stTextInput label, .stNumberInput label, .stDateInput label,
.stSelectbox label, .stFileUploader label, .stTextArea label,
.stRadio label, .stButton button, .stDownloadButton button {
    font-weight: 800 !important;
}
.sidebar-title { font-size: 1.52rem !important; font-weight: 900 !important; color: var(--red) !important; }
.sidebar-help { font-size: .9rem; line-height: 1.55; color: #555; }
.card-link {
    display: block; text-decoration: none !important; color: inherit !important;
    border: 1px solid var(--line); border-radius: 14px; padding: .95rem 1rem;
    background: #fff; box-shadow: 0 4px 12px rgba(0,0,0,.08);
    min-height: 145px; transition: transform .15s ease, box-shadow .15s ease;
}
.card-link:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(0,0,0,.12); border-color: #c9a1a1; }
.card-title { color: var(--red); font-size: 1.34rem; font-weight: 900; margin-bottom: .45rem; }
.card-text { line-height: 1.52; font-size: .95rem; }
.rule-box {
    border: 1px solid #e0bdbd; border-left: 5px solid var(--red);
    border-radius: 12px; padding: .95rem 1.05rem; background: #fff8f8;
    margin: .82rem 0 1rem;
}
.metric-box {
    border: 1px solid var(--line); border-radius: 12px; padding: .72rem .85rem;
    background: #fff; min-height: 92px; box-shadow: 0 2px 8px rgba(0,0,0,.04);
}
.metric-label { color: #656b73; font-size: .84rem; font-weight: 800; }
.metric-value { color: #111827; font-size: 1.28rem; font-weight: 900; margin-top: .18rem; }
.term-title { font-size: 1.03rem; font-weight: 900; color: var(--red); }
.term-desc { font-size: .9rem; line-height: 1.45; margin: .22rem 0 .58rem; }
.ai-box {
    border: 1px solid #c8d7eb; border-left: 5px solid var(--blue);
    padding: .9rem 1rem; border-radius: 10px; background: #f6f9ff;
}
.data-table-title { font-size: 1.08rem; font-weight: 900; color: #333; margin-top: .55rem; margin-bottom: .3rem; }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)

# =========================
# TIỆN ÍCH
# =========================
def money(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.0f}".replace(",", ".") + " đ"


def percent(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.2f}%"


def clean_money_text(value):
    if value is None:
        return 0.0
    raw = str(value).strip().replace("đ", "").replace("Đ", "")
    digits = re.sub(r"[^0-9]", "", raw)
    return float(digits) if digits else 0.0


def format_money_editor(value):
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = 0
    return f"{number:,}".replace(",", ".") if number else ""


def normalize_money_widget(key):
    raw = st.session_state.get(key, "")
    digits = re.sub(r"[^0-9]", "", str(raw)).lstrip("0")
    st.session_state[key] = f"{int(digits):,}".replace(",", ".") if digits else ""


def money_input(label, key, default=0, placeholder="0"):
    # 0 chỉ là placeholder chìm; người dùng click vào và nhập số mới ngay.
    if key not in st.session_state:
        st.session_state[key] = format_money_editor(default)
    st.text_input(
        label,
        key=key,
        placeholder=placeholder,
        on_change=normalize_money_widget,
        args=(key,),
    )
    return clean_money_text(st.session_state.get(key, ""))


def date_input_vn(label, key, default=None):
    if default is None:
        default = date.today()
    return st.date_input(label, value=default, format="DD/MM/YYYY", key=key)


def dataframe_to_excel_bytes(sheets):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=str(name)[:31], index=False)
    return output.getvalue()


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
        return "Chưa cấu hình Gemini API. Các chức năng tính toán vẫn hoạt động bình thường."
    system_instruction = """
Bạn là Trợ lý tài chính bình dân dành cho tiểu thương, hộ kinh doanh, nông hộ,
cơ sở sản xuất nhỏ và hợp tác xã.
Chỉ sử dụng số liệu được cung cấp; không tự tạo hoặc sửa số liệu.
Phân biệt rõ: Kết quả tính toán, Nhận xét, Cảnh báo.
Giải thích thuật ngữ bằng tiếng Việt dễ hiểu.
Không khẳng định chắc chắn lợi nhuận tương lai.
Không thay thế kế toán, kiểm toán hoặc thẩm định chuyên môn.
"""
    prompt = instruction + "\n\nDỮ LIỆU ĐÃ TÍNH:\n" + json.dumps(data, ensure_ascii=False, indent=2, default=str)
    try:
        cfg = types.GenerateContentConfig(system_instruction=system_instruction) if types else None
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=cfg)
        return response.text or "AI không trả về nội dung."
    except Exception as exc:
        return f"Không thể gọi Gemini API lúc này: {exc}"


def npv(rate, cashflows):
    if rate <= -1:
        return float("nan")
    return float(sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows)))


def irr_bisection(cashflows):
    values = np.asarray(cashflows, dtype=float)
    if len(values) < 2 or not (np.any(values > 0) and np.any(values < 0)):
        return None
    grid = np.concatenate([np.linspace(-0.99, -0.01, 100), np.linspace(0, 5, 300)])
    prev_r = float(grid[0])
    prev_v = npv(prev_r, cashflows)
    for current in grid[1:]:
        current = float(current)
        current_v = npv(current, cashflows)
        if np.isfinite(prev_v) and np.isfinite(current_v) and prev_v * current_v <= 0:
            low, high = prev_r, current
            flow = prev_v
            for _ in range(120):
                mid = (low + high) / 2
                fmid = npv(mid, cashflows)
                if abs(fmid) < 1e-9:
                    return float(mid)
                if flow * fmid <= 0:
                    high = mid
                else:
                    low, flow = mid, fmid
            return float((low + high) / 2)
        prev_r, prev_v = current, current_v
    return None


def payback_period(cashflows):
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
            fraction = min(max((-previous) / step, 0), 1)
            return float(i - 1 + fraction)
    return None


def compute_wacc(equity, debt, cost_equity, cost_debt, tax):
    total = equity + debt
    if total <= 0:
        return None
    e = equity / total
    d = debt / total
    return e * cost_equity + d * cost_debt * (1 - tax)


# =========================
# ĐIỀU HƯỚNG
# =========================
pages = {
    "🏠 Tổng quan": "tong-quan",
    "💰 Sổ tay Dòng tiền": "dong-tien",
    "⚙️ Tính Khấu hao": "khau-hao",
    "📈 Đánh giá Hiệu quả Đầu tư": "dau-tu",
}
slug_to_page = {v: k for k, v in pages.items()}

if "page" not in st.session_state:
    st.session_state.page = "🏠 Tổng quan"

requested = st.query_params.get("page")
if requested in slug_to_page:
    st.session_state.page = slug_to_page[requested]

with st.sidebar:
    st.markdown('<div class="sidebar-title">DANH MỤC</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-help"><b>Hướng dẫn nhanh:</b><br>'
        'Chọn một mục → nhập dữ liệu → xem kết quả.<br>'
        'Nút AI chỉ dùng khi cần giải thích hoặc nhận xét.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    selected = st.radio(
        "Chọn chức năng trong danh mục",
        list(pages.keys()),
        index=list(pages.keys()).index(st.session_state.page),
        label_visibility="collapsed",
    )
    if selected != st.session_state.page:
        st.session_state.page = selected
        st.query_params["page"] = pages[selected]
        st.rerun()

# =========================
# TỔNG QUAN
# =========================
if st.session_state.page == "🏠 Tổng quan":
    st.markdown('<div class="section-header">TỔNG QUAN CHƯƠNG TRÌNH</div>', unsafe_allow_html=True)
    cards = st.columns(3)
    card_data = [
        ("💰 SỔ TAY DÒNG TIỀN", "Ghi lại từng khoản thu và chi, theo dõi tổng tiền vào, tổng tiền ra và dòng tiền ròng", "dong-tien"),
        ("⚙️ TÍNH KHẤU HAO", "Tính khấu hao đường thẳng cho máy móc, thiết bị và tài sản.", "khau-hao"),
        ("📈 ĐÁNH GIÁ ĐẦU TƯ", "Tính NPV- Giá trị hiện tại ròng, IRR - Tỷ suất hoàn vốn nội bộ, thời gian hoàn vốn và WACC- Chi phí sử dụng vốn bình quân gia quyền khi đủ dữ liệu", "dau-tu"),
    ]
    for col, (title, desc, slug) in zip(cards, card_data):
        with col:
            # Anchor không có target => luôn chuyển trong cùng tab.
            st.markdown(
                f'<a class="card-link" href="?page={slug}" aria-label="Mở {title}">'
                f'<div class="card-title">{title}</div><div class="card-text">{desc}</div></a>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="rule-box"><b>Nguyên tắc sử dụng:</b> dữ liệu nhập vào là cơ sở của mọi kết quả. '
        'Cần kiểm tra số liệu trước khi sử dụng để báo cáo hoặc ra quyết định.<br><br>'
        '<b>AI:</b> chỉ hỗ trợ giải thích và nhận xét từ dữ liệu đã tính, không thay thế kế toán, kiểm toán hoặc thẩm định chuyên môn.</div>',
        unsafe_allow_html=True,
    )
    st.info("Gemini AI đã sẵn sàng." if ai_available() else "Gemini AI chưa được cấu hình; các chức năng tính toán vẫn hoạt động bình thường.")

# =========================
# SỔ TAY DÒNG TIỀN
# =========================
elif st.session_state.page == "💰 Sổ tay Dòng tiền":
    st.markdown('<div class="section-header">SỔ TAY DÒNG TIỀN</div>', unsafe_allow_html=True)
    if "cashflows" not in st.session_state:
        st.session_state.cashflows = pd.DataFrame(columns=["Ngày", "Loại", "Nhóm", "Nội dung", "Số tiền"])
    if "cash_calculated" not in st.session_state:
        st.session_state.cash_calculated = False

    st.markdown('<div class="section-header">NHẬP DỮ LIỆU</div>', unsafe_allow_html=True)
    if st.session_state.pop("cash_reset_form", False):
        # Reset ở đầu một lượt chạy, trước khi widget được tạo.
        st.session_state["cash_content"] = ""
        st.session_state["cash_amount"] = ""

    cols = st.columns([1.15, 1.0, 1.35, 1.75, 1.3, .8])
    with cols[0]:
        d = date_input_vn("Ngày", "cash_date")
    with cols[1]:
        typ = st.selectbox("Loại giao dịch", ["Thu", "Chi"], key="cash_type")
    with cols[2]:
        group = st.selectbox("Nhóm", ["Bán hàng", "Nguyên liệu", "Lương", "Điện/nước", "Vận chuyển", "Thuê mặt bằng", "Mua tài sản", "Khác"], key="cash_group")
    with cols[3]:
        content = st.text_input("Nội dung", key="cash_content", placeholder="Ví dụ: Bán cà phê, mua nguyên liệu, tiền điện...")
    with cols[4]:
        amount = money_input("Số tiền", "cash_amount", 0)
    with cols[5]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Thêm", key="cash_add", use_container_width=True):
            row = pd.DataFrame([[pd.Timestamp(d), typ, group, content.strip(), amount]], columns=st.session_state.cashflows.columns)
            st.session_state.cashflows = pd.concat([st.session_state.cashflows, row], ignore_index=True)
            st.session_state.cash_reset_form = True
            st.session_state.cash_calculated = False
            st.rerun()

    # Bảng tổng hợp luôn hiển thị ngay dưới vùng nhập liệu.
    st.markdown('<div class="data-table-title">BẢNG TỔNG HỢP DÒNG TIỀN ĐÃ NHẬP</div>', unsafe_allow_html=True)
    cash_display = st.session_state.cashflows.copy()
    if not cash_display.empty:
        cash_display["Ngày"] = pd.to_datetime(cash_display["Ngày"], errors="coerce", dayfirst=True)
        cash_display["Số tiền"] = pd.to_numeric(cash_display["Số tiền"].apply(clean_money_text), errors="coerce").fillna(0)
        cash_display["Ngày"] = cash_display["Ngày"].dt.strftime("%d/%m/%Y")
        cash_display["Số tiền"] = cash_display["Số tiền"].map(money)
    st.dataframe(cash_display, use_container_width=True, hide_index=True)

    upload_cols = st.columns([1.2, 2.3, 1.0])
    with upload_cols[0]:
        uploaded = st.file_uploader("Tải dữ liệu Excel/CSV", type=["xlsx", "csv"], key="cash_upload")
    with upload_cols[1]:
        st.caption("File nên có các cột: Ngày, Loại, Nhóm, Nội dung, Số tiền.")
    with upload_cols[2]:
        if st.button("Tính toán", key="cash_calculate", use_container_width=True):
            st.session_state.cash_calculated = True

    if uploaded is not None and st.session_state.get("cash_last_uploaded_name") != uploaded.name:
        try:
            imported = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            required = {"Ngày", "Loại", "Số tiền"}
            if not required.issubset(imported.columns):
                st.error("File thiếu cột bắt buộc: Ngày, Loại, Số tiền.")
            else:
                for col in ["Nhóm", "Nội dung"]:
                    if col not in imported.columns:
                        imported[col] = ""
                st.session_state.cashflows = imported[["Ngày", "Loại", "Nhóm", "Nội dung", "Số tiền"]].copy()
                st.session_state.cash_last_uploaded_name = uploaded.name
                st.session_state.cash_calculated = False
                st.rerun()
        except Exception as exc:
            st.error(f"Không thể đọc file: {exc}")

    df = st.session_state.cashflows.copy()
    if not df.empty:
        df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce", dayfirst=True)
        df["Số tiền"] = pd.to_numeric(df["Số tiền"].apply(clean_money_text), errors="coerce").fillna(0)
        df = df.dropna(subset=["Ngày"])

    st.markdown('<div class="section-header">KẾT QUẢ</div>', unsafe_allow_html=True)
    if not st.session_state.get("cash_calculated", False):
        st.info("Nhập hoặc thêm dữ liệu, sau đó bấm Tính toán để xem kết quả.")
    else:
        total_in = float(df.loc[df["Loại"].eq("Thu"), "Số tiền"].sum()) if not df.empty else 0.0
        total_out = float(df.loc[df["Loại"].eq("Chi"), "Số tiền"].sum()) if not df.empty else 0.0
        net = total_in - total_out

        metrics = st.columns(3)
        for col, label, value in zip(metrics, ["TỔNG TIỀN VÀO", "TỔNG TIỀN RA", "DÒNG TIỀN RÒNG"], [total_in, total_out, net]):
            with col:
                st.markdown(f'<div class="metric-box"><div class="metric-label">{label}</div><div class="metric-value">{money(value)}</div></div>', unsafe_allow_html=True)

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
            fig.update_yaxes(tickformat=",.0f", separatethousands=True)
            st.plotly_chart(fig, use_container_width=True)

            net_month = df.assign(Tháng=df["Ngày"].dt.to_period("M").astype(str), signed=np.where(df["Loại"].eq("Thu"), df["Số tiền"], -df["Số tiền"])).groupby("Tháng", as_index=False)["signed"].sum().rename(columns={"signed": "Dòng tiền ròng"})
            fig2 = px.line(net_month, x="Tháng", y="Dòng tiền ròng", markers=True, text="Dòng tiền ròng", title="Dòng tiền ròng theo tháng")
            fig2.update_traces(texttemplate="%{text:,.0f}", textposition="top center", hovertemplate="%{y:,.0f} đ<extra></extra>")
            fig2.update_yaxes(tickformat=",.0f", separatethousands=True)
            st.plotly_chart(fig2, use_container_width=True)

            action_cols = st.columns(2)
            with action_cols[0]:
                export_df = df.copy()
                export_df["Ngày"] = export_df["Ngày"].dt.strftime("%d/%m/%Y")
                export = dataframe_to_excel_bytes({"So_tay_dong_tien": export_df})
                st.download_button("Tải Excel", export, "so_tay_dong_tien.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with action_cols[1]:
                if st.button("Phân tích bằng AI", key="cash_ai", use_container_width=True):
                    payload = {"tong_tien_vao": total_in, "tong_tien_ra": total_out, "dong_tien_rong": net, "so_giao_dich": int(len(df))}
                    st.markdown(f'<div class="ai-box">{call_gemini("Phân tích ngắn gọn xu hướng dòng tiền, điểm đáng chú ý và cảnh báo nếu có.", payload)}</div>', unsafe_allow_html=True)

# =========================
# KHẤU HAO
# =========================
elif st.session_state.page == "⚙️ Tính Khấu hao":
    st.markdown('<div class="section-header">TÍNH KHẤU HAO</div>', unsafe_allow_html=True)
    if "assets" not in st.session_state:
        st.session_state.assets = []
    if "dep_calculated" not in st.session_state:
        st.session_state.dep_calculated = False

    st.markdown('<div class="section-header">NHẬP THÔNG TIN TÀI SẢN</div>', unsafe_allow_html=True)
    if st.session_state.pop("dep_reset_form", False):
        st.session_state["dep_name"] = ""
        st.session_state["dep_cost"] = ""
        st.session_state["dep_salvage"] = ""

    cols = st.columns([1.55, 1.25, 1.25, .95, 1.25, .8])
    with cols[0]:
        asset_name = st.text_input("Tên tài sản", key="dep_name", placeholder="Ví dụ: Máy cày, máy bơm, máy sấy...")
    with cols[1]:
        cost = money_input("Nguyên giá", "dep_cost", 0)
    with cols[2]:
        salvage = money_input("Giá trị thu hồi", "dep_salvage", 0)
    with cols[3]:
        years = st.number_input("Thời gian sử dụng (năm)", min_value=1, max_value=100, value=5, step=1, key="dep_years")
    with cols[4]:
        purchase = date_input_vn("Ngày mua", "dep_purchase")
    with cols[5]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Thêm", key="dep_add", use_container_width=True):
            if not asset_name.strip():
                st.error("Vui lòng nhập Tên tài sản.")
            elif salvage > cost:
                st.error("Giá trị thu hồi không được lớn hơn nguyên giá.")
            elif cost <= 0:
                st.error("Nguyên giá phải lớn hơn 0.")
            else:
                st.session_state.assets.append({
                    "Tên tài sản": asset_name.strip(),
                    "Nguyên giá": cost,
                    "Giá trị thu hồi": salvage,
                    "Thời gian sử dụng (năm)": int(years),
                    "Ngày mua": purchase.strftime("%d/%m/%Y"),
                })
                st.session_state.dep_reset_form = True
                st.session_state.dep_calculated = False
                st.rerun()

    # Bảng tài sản luôn hiển thị ngay dưới vùng nhập liệu.
    st.markdown('<div class="data-table-title">BẢNG TỔNG HỢP TÀI SẢN ĐÃ NHẬP</div>', unsafe_allow_html=True)
    assets_df = pd.DataFrame(st.session_state.assets)
    if not assets_df.empty:
        display_assets = assets_df.copy()
        display_assets["Nguyên giá"] = display_assets["Nguyên giá"].map(money)
        display_assets["Giá trị thu hồi"] = display_assets["Giá trị thu hồi"].map(money)
    else:
        display_assets = assets_df
    st.dataframe(display_assets, use_container_width=True, hide_index=True)

    calc_col = st.columns([1, 5])
    with calc_col[0]:
        if st.button("Tính toán", key="dep_calculate", use_container_width=True):
            st.session_state.dep_calculated = True

    if st.session_state.assets and not st.session_state.get("dep_calculated", False):
        st.info("Đã có dữ liệu tài sản. Bấm Tính toán để cập nhật kết quả khấu hao.")

    if st.session_state.assets and st.session_state.get("dep_calculated", False):
        today = date.today()
        rows = []
        for asset in st.session_state.assets:
            purchase_date = datetime.strptime(asset["Ngày mua"], "%d/%m/%Y").date()
            months_used = max(0, (today.year - purchase_date.year) * 12 + today.month - purchase_date.month - int(today.day < purchase_date.day))
            total_months = int(asset["Thời gian sử dụng (năm)"] * 12)
            used_months = min(months_used, total_months)
            annual_dep = (asset["Nguyên giá"] - asset["Giá trị thu hồi"]) / asset["Thời gian sử dụng (năm)"]
            monthly_dep = annual_dep / 12
            incurred = min(used_months * monthly_dep, asset["Nguyên giá"] - asset["Giá trị thu hồi"])
            remaining_dep = max(0, (asset["Nguyên giá"] - asset["Giá trị thu hồi"]) - incurred)
            remaining_months = max(0, total_months - used_months)
            book_value = max(asset["Giá trị thu hồi"], asset["Nguyên giá"] - incurred)
            rows.append({**asset, "Đã sử dụng": f"{used_months / 12:.1f} năm", "Thời gian còn lại": f"{remaining_months / 12:.1f} năm", "Khấu hao năm": annual_dep, "Khấu hao tháng": monthly_dep, "Khấu hao đã phát sinh": incurred, "Khấu hao còn lại": remaining_dep, "Giá trị còn lại": book_value})

        dep_df = pd.DataFrame(rows)
        st.markdown('<div class="section-header">KẾT QUẢ KHẤU HAO</div>', unsafe_allow_html=True)
        display_df = dep_df.copy()
        for column in ["Nguyên giá", "Giá trị thu hồi", "Khấu hao năm", "Khấu hao tháng", "Khấu hao đã phát sinh", "Khấu hao còn lại", "Giá trị còn lại"]:
            display_df[column] = display_df[column].map(money)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(f"Ngày tính: {today.strftime('%d/%m/%Y')}")

        metric_cols = st.columns(3)
        values = [float(dep_df["Khấu hao đã phát sinh"].sum()), float(dep_df["Khấu hao còn lại"].sum()), float(dep_df["Giá trị còn lại"].sum())]
        labels = ["KHẤU HAO ĐÃ PHÁT SINH", "KHẤU HAO CÒN LẠI", "GIÁ TRỊ CÒN LẠI"]
        for col, label, value in zip(metric_cols, labels, values):
            with col:
                st.markdown(f'<div class="metric-box"><div class="metric-label">{label}</div><div class="metric-value">{money(value)}</div></div>', unsafe_allow_html=True)

        actions = st.columns(2)
        with actions[0]:
            export = dataframe_to_excel_bytes({"Khau_hao": dep_df})
            st.download_button("Tải Excel", export, "tinh_khau_hao.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with actions[1]:
            if st.button("Phân tích bằng AI", key="dep_ai", use_container_width=True):
                st.markdown(f'<div class="ai-box">{call_gemini("Giải thích ngắn gọn khấu hao và ảnh hưởng của khấu hao đến chi phí kinh doanh.", {"tai_san": rows})}</div>', unsafe_allow_html=True)
    elif not st.session_state.assets:
        st.info("Chưa có tài sản. Nhập thông tin và bấm Thêm.")

# =========================
# ĐÁNH GIÁ ĐẦU TƯ
# =========================
else:
    st.markdown('<div class="section-header">ĐÁNH GIÁ HIỆU QUẢ ĐẦU TƯ</div>', unsafe_allow_html=True)
    if "investment_calculated" not in st.session_state:
        st.session_state.investment_calculated = False

    st.markdown('<div class="section-header">THÔNG TIN DỰ ÁN</div>', unsafe_allow_html=True)
    project_cols = st.columns([1.7, 1.3, 1.0])
    with project_cols[0]:
        project = st.text_input("Tên dự án", key="project", placeholder="Ví dụ: Đầu tư máy sấy nông sản...")
    with project_cols[1]:
        initial = money_input("Vốn đầu tư ban đầu", "initial", 0)
    with project_cols[2]:
        periods = st.number_input("Số kỳ dự kiến", min_value=1, max_value=50, value=5, step=1, key="periods")

    st.markdown('<div class="section-header">DÒNG TIỀN TỪNG KỲ</div>', unsafe_allow_html=True)
    cf_cols = st.columns(int(periods) + 1)
    for i in range(1, int(periods) + 1):
        with cf_cols[i - 1]:
            money_input(f"Kỳ {i}", f"cf_{i}", 0)
    with cf_cols[-1]:
        discount = st.number_input("Tỷ lệ chiết khấu (%)", min_value=-99.0, max_value=500.0, value=10.0, step=0.5, key="discount") / 100.0

    st.markdown('<div class="section-header">DỮ LIỆU WACC — NẾU CÓ</div>', unsafe_allow_html=True)
    wacc_cols = st.columns(5)
    with wacc_cols[0]:
        equity = st.number_input("Vốn chủ sở hữu (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key="equity") / 100.0
    with wacc_cols[1]:
        debt = st.number_input("Vốn vay (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="debt") / 100.0
    with wacc_cols[2]:
        cost_equity = st.number_input("Chi phí vốn chủ (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="cost_equity") / 100.0
    with wacc_cols[3]:
        cost_debt = st.number_input("Chi phí vốn vay (%)", min_value=0.0, max_value=100.0, value=8.0, step=0.5, key="cost_debt") / 100.0
    with wacc_cols[4]:
        tax = st.number_input("Thuế suất (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.5, key="tax") / 100.0

    calc_cols = st.columns([1, 5])
    with calc_cols[0]:
        if st.button("Tính toán", key="investment_calculate", use_container_width=True):
            st.session_state.investment_calculated = True

    if not st.session_state.get("investment_calculated", False):
        st.info("Nhập dữ liệu, sau đó bấm Tính toán để xem NPV, IRR, thời gian hoàn vốn và WACC.")
    else:
        cashflows = [-initial] + [clean_money_text(st.session_state.get(f"cf_{i}", "")) for i in range(1, int(periods) + 1)]
        npv_value = npv(discount, cashflows)
        irr_value = irr_bisection(cashflows)
        payback = payback_period(cashflows)
        wacc_value = compute_wacc(equity, debt, cost_equity, cost_debt, tax)

        st.markdown('<div class="section-header">KẾT QUẢ VÀ Ý NGHĨA</div>', unsafe_allow_html=True)
        concepts = [
            ("NPV — GIÁ TRỊ HIỆN TẠI RÒNG", "Là tổng chênh lệch giữa dòng tiền thu vào và dòng tiền chi ra, được quy đổi tất cả về giá trị tại thời điểm hiện tại theo một tỷ suất chiết khấu nhất định.", money(npv_value)),
            ("IRR — TỶ SUẤT HOÀN VỐN NỘI BỘ", "Là mức tỷ suất làm cho NPV của dự án bằng 0; có thể hiểu đơn giản là mức sinh lời nội tại của chuỗi dòng tiền.", percent(irr_value)),
            ("PAYBACK — THỜI GIAN HOÀN VỐN", "Là khoảng thời gian dự kiến để dòng tiền tích lũy thu hồi đủ số vốn đầu tư ban đầu.", f"{payback:.1f} năm" if payback is not None else "Chưa hoàn vốn"),
            ("WACC — CHI PHÍ SỬ DỤNG VỐN BÌNH QUÂN GIA QUYỀN", "Là chi phí bình quân của các nguồn vốn tài trợ cho dự án, có xét tỷ trọng vốn chủ sở hữu và vốn vay.", percent(wacc_value)),
        ]
        concept_cols = st.columns(4)
        for col, (title, description, value) in zip(concept_cols, concepts):
            with col:
                st.markdown(f'<div class="metric-box"><div class="term-title">{title}</div><div class="term-desc">{description}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

        if npv_value > 0:
            st.success("NPV dương: dự án đang có tín hiệu tạo thêm giá trị theo tỷ suất chiết khấu đã nhập.")
        elif npv_value < 0:
            st.warning("NPV âm: dự án chưa đạt mức sinh lời yêu cầu theo tỷ suất chiết khấu đã nhập.")
        else:
            st.info("NPV bằng 0: dự án vừa đạt mức sinh lời yêu cầu theo giả định hiện tại.")

        if irr_value is not None:
            if irr_value > discount:
                st.success(f"IRR ({percent(irr_value)}) cao hơn tỷ lệ chiết khấu ({percent(discount)}): tín hiệu tích cực.")
            else:
                st.warning(f"IRR ({percent(irr_value)}) không cao hơn tỷ lệ chiết khấu ({percent(discount)}).")

        cumulative = np.cumsum(cashflows)
        chart_df = pd.DataFrame({"Kỳ": range(len(cumulative)), "Dòng tiền tích lũy": cumulative})
        fig = px.line(chart_df, x="Kỳ", y="Dòng tiền tích lũy", markers=True, text="Dòng tiền tích lũy", title="Dòng tiền tích lũy")
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="top center", hovertemplate="%{y:,.0f} đ<extra></extra>")
        fig.update_yaxes(tickformat=",.0f", separatethousands=True)
        st.plotly_chart(fig, use_container_width=True)

        action_cols = st.columns(2)
        with action_cols[0]:
            summary_df = pd.DataFrame({
                "Chỉ tiêu": ["NPV", "IRR", "Thời gian hoàn vốn", "WACC", "Tỷ lệ chiết khấu"],
                "Giá trị": [npv_value, irr_value if irr_value is not None else np.nan, payback if payback is not None else np.nan, wacc_value if wacc_value is not None else np.nan, discount],
            })
            details_df = pd.DataFrame({"Kỳ": range(len(cashflows)), "Dòng tiền": cashflows, "Dòng tiền tích lũy": cumulative})
            export = dataframe_to_excel_bytes({"Ket_qua": summary_df, "Dong_tien": details_df})
            st.download_button("Tải báo cáo Excel", export, "danh_gia_hieu_qua_dau_tu.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with action_cols[1]:
            if st.button("Phân tích bằng AI", key="investment_ai", use_container_width=True):
                payload = {"du_an": project, "von_dau_tu_ban_dau": initial, "dong_tien": cashflows, "ty_le_chiet_khau": discount, "NPV": npv_value, "IRR": irr_value, "Payback": payback, "WACC": wacc_value}
                st.markdown(f'<div class="ai-box">{call_gemini("Phân tích hiệu quả dự án theo 3 phần: kết quả, nhận xét dễ hiểu và cảnh báo/điểm cần kiểm tra.", payload)}</div>', unsafe_allow_html=True)
