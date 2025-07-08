import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json
from streamlit_plotly_events import plotly_events

# --- 상수 정의 ---
NAN_VALUE_MIN_HEIGHT = 0.02
LANTHANIDES_PERIOD = 9
ACTINIDES_PERIOD = 10
LANTHANIDES_RANGE = (57, 71)
ACTINIDES_RANGE = (89, 103)
BAR_WIDTH_X = 0.8
BAR_WIDTH_Y = 0.8
DEFAULT_COLORSCALE = 'Jet'


# --- 1. 데이터 불러오기 ---
@st.cache_data
def load_data():
    df = pd.read_csv("elements.csv", encoding='utf-8')

    # 숫자형 컬럼 변환 시 오류 무시
    numeric_cols = [
        'group', 'period', 'atomic_number', 'atomic_weight', 'electronegativity',
        'covalent_radius_exp', 'van_der_waals_radius_exp',
        'electron_affinity', 'first_ionization_energy',
        'melting_point_K', 'boiling_point_K', 'density_g_cm3',
        'thermal_conductivity'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 란타넘족, 악티늄족 분리 및 가상 주기/족 할당
    lanthanides = df[(df['atomic_number'] >= LANTHANIDES_RANGE[0]) & (df['atomic_number'] <= LANTHANIDES_RANGE[1])].copy()
    actinides = df[(df['atomic_number'] >= ACTINIDES_RANGE[0]) & (df['atomic_number'] <= ACTINIDES_RANGE[1])].copy()

    df_main = df[~((df['atomic_number'] >= LANTHANIDES_RANGE[0]) & (df['atomic_number'] <= LANTHANIDES_RANGE[1])) &
                 ~((df['atomic_number'] >= ACTINIDES_RANGE[0]) & (df['atomic_number'] <= ACTINIDES_RANGE[1]))].copy()

    lanthanides.loc[:, 'period'] = LANTHANIDES_PERIOD
    lanthanides.loc[:, 'group'] = np.arange(3, 3 + len(lanthanides)) # 란타넘족은 3족부터 시작

    actinides.loc[:, 'period'] = ACTINIDES_PERIOD
    actinides.loc[:, 'group'] = np.arange(3, 3 + len(actinides)) # 악티늄족도 3족부터 시작

    df_final = pd.concat([df_main, lanthanides, actinides], ignore_index=True)

    # NaN을 허용하는 정수형으로 변환
    df_final['group'] = df_final['group'].astype(pd.Int64Dtype())
    df_final['period'] = df_final['period'].astype(pd.Int64Dtype())

    return df_final

@st.cache_data # JSON 파일도 캐싱하여 로드 시간을 줄임
def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return {}
    except json.JSONDecodeError:
        st.error(f"오류: '{file_path}' 파일이 유효한 JSON 형식이 아닙니다.")
        return {}
    except Exception as e:
        st.error(f"파일 로드 중 예상치 못한 오류 발생: {e}")
        return {}

# --- 2. 막대 생성 함수 (hovertext 매개변수 포함) ---
def create_bar_mesh(x_center, y_center, z_height, bar_width_x, bar_width_y, hovertext,
                    color=None, intensity=None, colorscale=None, cmin=None, cmax=None):
    x_coords = [x_center - bar_width_x / 2, x_center + bar_width_x / 2]
    y_coords = [y_center - bar_width_y / 2, y_center + bar_width_y / 2]
    z_coords = [0, z_height]

    vertices = np.array([
        [x_coords[0], y_coords[0], z_coords[0]],
        [x_coords[1], y_coords[0], z_coords[0]],
        [x_coords[1], y_coords[1], z_coords[0]],
        [x_coords[0], y_coords[1], z_coords[0]],
        [x_coords[0], y_coords[0], z_coords[1]],
        [x_coords[1], y_coords[0], z_coords[1]],
        [x_coords[1], y_coords[1], z_coords[1]],
        [x_coords[0], y_coords[1], z_coords[1]],
    ])

    faces = np.array([
        [0, 1, 2], [0, 2, 3], # Bottom
        [4, 5, 6], [4, 6, 7], # Top
        [0, 1, 5], [0, 5, 4], # Front
        [3, 2, 6], [3, 6, 7], # Back
        [0, 3, 7], [0, 7, 4], # Left
        [1, 2, 6], [1, 6, 5], # Right
    ])

    mesh_kwargs = dict(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=1.0,
        hoverinfo='text',
        hovertext=hovertext,
    )

    if color:
        mesh_kwargs['color'] = color
    else:
        mesh_kwargs['intensity'] = [intensity] * len(vertices) if intensity is not None else None
        mesh_kwargs['colorscale'] = colorscale
        mesh_kwargs['cmin'] = cmin
        mesh_kwargs['cmax'] = cmax
        mesh_kwargs['showscale'] = True
        mesh_kwargs['colorbar'] = dict(
            title="값",
            thickness=20,
            x=0.9,
            xpad=10
        )

    return go.Mesh3d(**mesh_kwargs)

# --- 각 정보별 Z축 압축 계수를 저장할 딕셔너리 ---
z_axis_compression_factors = {
    "원자량": 1.0,
    "공유 반지름(실험값)": 1.0,
    "반데르발 반지름(실험값)": 1.0,
    "전기음성도": 4.0,
    "전자친화도": 0.8,
    "제1 이온화 에너지(kJ/mol)": 1.8,
    "녹는점(K)": 0.6,
    "끓는점(K)": 0.6,
    "밀도(g/cm3)": 1.2,
    "열 전도율(W/(m·K))": 0.7
}

# --- 3. 3D 그래프 그리기 함수 ---
def draw_plotly_3d(data, column_name, display_label, compression_factor):
    fig = go.Figure()

    valid_data_for_color = data[column_name].dropna()
    valid_data_for_color = valid_data_for_color[valid_data_for_color > 0]

    min_val, max_val = 0.01, 1.0 # 기본값 설정
    if not valid_data_for_color.empty:
        min_val = valid_data_for_color.min()
        max_val = valid_data_for_color.max()
        if min_val == max_val: # 모든 값이 동일할 경우를 대비
            min_val = max_val * 0.9 if max_val != 0 else 0.01
            max_val = max_val * 1.1 if max_val != 0 else 1.0

    def scale_z_height(val, base_height_multiplier):
        if pd.isna(val) or val <= 0:
            return NAN_VALUE_MIN_HEIGHT
        return np.log10(max(val, 1e-9)) * base_height_multiplier

    max_z_scaled = 0
    for _, row in data.iterrows():
        z_original = row[column_name]
        if pd.notna(z_original) and z_original > 0:
            current_scaled_height = scale_z_height(z_original, compression_factor)
            if current_scaled_height > max_z_scaled:
                max_z_scaled = current_scaled_height

    z_range_max_display = max(max_z_scaled + 0.5, NAN_VALUE_MIN_HEIGHT + 0.5, 1.0)

    for _, row in data.iterrows():
        x = row['group']
        y = row['period']
        z_original = row[column_name]

        symbol = row['symbol']
        kor_name = row['kor_name']
        atomic_num = row['atomic_number']

        z_height = scale_z_height(z_original, compression_factor)

        is_nan_element = pd.isna(z_original) or z_original < 0

        display_value = "정보 없음" if is_nan_element else z_original
        bar_hovertext = f"원소: {kor_name}<br>원자 번호: {atomic_num}<br>{display_label}: {display_value}"

        if is_nan_element:
            bar = create_bar_mesh(x, y, z_height, BAR_WIDTH_X, BAR_WIDTH_Y,
                                  hovertext=bar_hovertext, color='lightgrey')
        else:
            bar = create_bar_mesh(x, y, z_height, BAR_WIDTH_X, BAR_WIDTH_Y,
                                  hovertext=bar_hovertext,
                                  intensity=z_original, colorscale=DEFAULT_COLORSCALE,
                                  cmin=min_val, cmax=max_val)
        fig.add_trace(bar)

        symbol_z_offset = z_height + 0.1
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[symbol_z_offset],
            mode='text', text=[symbol],
            textfont=dict(size=10, color='black'),
            showlegend=False, hoverinfo='text', hovertext=[bar_hovertext]
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Group', yaxis_title='Period', zaxis_title=f"{display_label} 상대값",
            xaxis=dict(
                tickmode='array', tickvals=list(range(1, 19)), ticktext=[str(i) for i in range(1, 19)],
                gridcolor='lightgrey', showgrid=True, zeroline=False, range=[18.5, 0.5], autorange=False
            ),
            yaxis=dict(
                tickmode='array', tickvals=list(range(1, 11)), ticktext=[str(i) for i in range(1, 8)] + ['', 'Lan.', 'Act.'],
                gridcolor='lightgrey', showgrid=True, zeroline=False
            ),
            zaxis=dict(
                range=[0, z_range_max_display],
                gridcolor='lightgrey', showgrid=True, zeroline=False
            ),
            aspectmode='data', bgcolor='white', camera=dict(eye=dict(x=1.8, y=1.8, z=0.8))
        ),
        height=700, margin=dict(l=0, r=0, b=0, t=30), showlegend=False,
        title_text=f"3D 주기율표 시각화 ({display_label})", title_x=0.5,
        paper_bgcolor='white', plot_bgcolor='white'
    )
    return fig

# --- 4. 2D 주기율표 그리기 함수 ---
def draw_2d_periodic_table(df_data):
    fig_2d = go.Figure()

    period_grid = np.full((10, 18), np.nan)

    for _, row in df_data.iterrows():
        group = row['group']
        period = row['period']

        if pd.isna(group) or pd.isna(period):
            continue

        group_int = int(group)
        period_int = int(period)

        if 1 <= group_int <= 18 and 1 <= period_int <= 10:
            row_idx = period_int - 1
            col_idx = group_int - 1
            period_grid[row_idx, col_idx] = group_int

    fig_2d.add_trace(go.Heatmap(
        z=period_grid,
        x=list(range(1, 19)),
        y=list(range(1, 11)),
        colorscale='Portland',
        showscale=False,
        hoverinfo='skip',
        zmin=1, zmax=18
    ))

    for _, row in df_data.iterrows():
        group = row['group']
        period = row['period']
        symbol = row['symbol']
        kor_name = row['kor_name']
        atomic_num = row['atomic_number']

        display_period = period
        display_group = group
        lanact_label = ""

        if period == LANTHANIDES_PERIOD:
            display_period = 6
            lanact_label = '란타넘족'
        elif period == ACTINIDES_PERIOD:
            display_period = 7
            lanact_label = '악티늄족'

        if pd.isna(group) or group == 0 or pd.isna(period):
            continue

        hovertext = f"원소: {kor_name}<br>원자 번호: {atomic_num}<br>족: {lanact_label if lanact_label else int(display_group)}<br>주기: {int(display_period)}"

        fig_2d.add_trace(go.Scattergl(
            x=[int(group)], y=[int(period)],
            mode='text', text=[symbol],
            textfont=dict(size=15, color='black', family='Arial Black'),
            hoverinfo='text', hovertext=[hovertext],
            showlegend=False,
            customdata=[row['atomic_number']],
            name=str(row['atomic_number'])
        ))

    fig_2d.update_layout(
        title='2D 주기율표 (원소를 클릭하여 상세 정보 확인)', title_x=0.5,
        xaxis=dict(
            title='Group', tickmode='array', tickvals=list(range(1, 19)), ticktext=[str(g) for g in range(1, 19)],
            showgrid=True, zeroline=False, side='top', range=[0.5, 18.5], ticklen=0
        ),
        yaxis=dict(
            title='Period', tickmode='array', tickvals=list(range(1, 8)) + [LANTHANIDES_PERIOD, ACTINIDES_PERIOD],
            ticktext=[str(p) for p in range(1, 8)] + ['Lan.', 'Act.'],
            autorange='reversed', showgrid=True, zeroline=False, range=[ACTINIDES_PERIOD + 0.5, 0.5], ticklen=0
        ),
        plot_bgcolor='white', paper_bgcolor='white', height=600,
        margin=dict(l=40, r=40, b=40, t=40), hovermode='closest',
    )
    return fig_2d

# --- 5. Streamlit 앱 ---
st.set_page_config(layout="wide")
st.title("3D 주기율표")
st.write("3D 주기율표에서 원소의 정보을 시각화하고, 아래 2D 주기율표를 클릭하여 상세 정보를 확인하세요.")

df = load_data()
models = load_json('models.json')

st.header("3D 주기율표")

value_options = {
    "원자량": "atomic_weight",
    "공유 반지름(실험값)": "covalent_radius_exp",
    "반데르발 반지름(실험값)": "van_der_waals_radius_exp",
    "전기음성도": "electronegativity",
    "전자친화도": "electron_affinity",
    "제1 이온화 에너지(kJ/mol)": "first_ionization_energy",
    "녹는점(K)": "melting_point_K",
    "끓는점(K)": "boiling_point_K",
    "밀도(g/cm3)": "density_g_cm3",
    "열 전도율(W/(m·K))": "thermal_conductivity"
}

selected_display_name = st.selectbox("3D 주기율표 시각화할 값 선택", list(value_options.keys()), key="3d_value_selector")
selected_column_name = value_options[selected_display_name]

current_compression_factor = z_axis_compression_factors.get(selected_display_name, 1.0)

st.markdown("---")

# --- st.session_state 초기화 및 동기화 (앱 시작 시에만) ---
if 'selected_elements_for_3d' not in st.session_state:
    st.session_state.selected_elements_for_3d = set(df['atomic_number'])

if 'temp_selected_elements_state' not in st.session_state:
    st.session_state.temp_selected_elements_state = {an: True for an in df['atomic_number']}
else:
    for an in df['atomic_number']:
        if an not in st.session_state.temp_selected_elements_state:
            st.session_state.temp_selected_elements_state[an] = (an in st.session_state.selected_elements_for_3d)

if 'show_element_selector' not in st.session_state:
    st.session_state.show_element_selector = False

if st.button("표시 원소 선택", key="toggle_element_selector"):
    st.session_state.show_element_selector = not st.session_state.show_element_selector

if st.session_state.show_element_selector:
    st.write("3D 주기율표에 표시할 원소를 선택하세요 (하단의 적용 버튼을 눌러 적용)")

    col_all_buttons_L, col_all_buttons_R, _ = st.columns([0.15, 0.15, 0.7])
    with col_all_buttons_L:
        if st.button("모두 활성화", key="select_all_elements_btn"):
            st.session_state.temp_selected_elements_state = {an: True for an in df['atomic_number']}
            st.rerun()
    with col_all_buttons_R:
        if st.button("모두 비활성화", key="deselect_all_elements_btn"):
            st.session_state.temp_selected_elements_state = {an: False for an in df['atomic_number']}
            st.rerun()

    st.markdown("#### 주기별 선택")
    periods_to_display = list(range(1, 8)) + [LANTHANIDES_PERIOD, ACTINIDES_PERIOD]
    period_cols = st.columns(len(periods_to_display))

    for i, p_val in enumerate(periods_to_display):
        with period_cols[i]:
            period_label = f"{p_val}주기" if p_val <= 7 else ('란타넘족' if p_val == LANTHANIDES_PERIOD else '악티늄족')
            st.markdown(f"<div style='text-align: center;'>{period_label}</div>", unsafe_allow_html=True)

            period_elements_atomic_numbers = df[df['period'] == p_val]['atomic_number'].tolist()

            if st.button("활성화", key=f"period_{p_val}_activate_btn"):
                for an in period_elements_atomic_numbers:
                    st.session_state.temp_selected_elements_state[an] = True
                st.rerun()

            if st.button("비활성화", key=f"period_{p_val}_deactivate_btn"):
                for an in period_elements_atomic_numbers:
                    st.session_state.temp_selected_elements_state[an] = False
                st.rerun()

    st.markdown("#### 족별 선택")
    group_cols = st.columns(18)

    for i, g in enumerate(range(1, 19)):
        with group_cols[i]:
            st.markdown(f"<div style='text-align: center;'>{g}족</div>", unsafe_allow_html=True)
            group_elements_atomic_numbers = df[(df['group'] == g) & ~(df['period'].isin([LANTHANIDES_PERIOD, ACTINIDES_PERIOD]))]['atomic_number'].tolist()

            if st.button("활성화", key=f"group_{g}_activate_btn"):
                for an in group_elements_atomic_numbers:
                    st.session_state.temp_selected_elements_state[an] = True
                st.rerun()

            if st.button("비활성화", key=f"group_{g}_deactivate_btn"):
                for an in group_elements_atomic_numbers:
                    st.session_state.temp_selected_elements_state[an] = False
                st.rerun()

    st.markdown("---")

    # --- 개별 원소 선택 (st.form 내부, "적용하기" 버튼으로 제출) ---
    with st.form(key='element_selection_form'):
        st.markdown("#### 개별 원소 선택")

        elements_for_checkboxes = df.sort_values(by=['period', 'group']).reset_index(drop=True)
        grouped_by_period = elements_for_checkboxes.groupby('period')

        header_weights = [0.15] + [1] * 18
        group_label_cols = st.columns(header_weights)

        with group_label_cols[0]:
            st.write("")
        for i in range(1, 19):
            with group_label_cols[i]:
                st.markdown(f"<div style='text-align: center; font-weight: bold;'>{i}족</div>", unsafe_allow_html=True)

        for period_num, period_df in grouped_by_period:
            row_cols_weights = [0.15] + [1] * (15 if period_num in [LANTHANIDES_PERIOD, ACTINIDES_PERIOD] else 18)
            row_cols = st.columns(row_cols_weights)

            with row_cols[0]:
                if period_num <= 7:
                    st.write(f"{period_num}주기")
                elif period_num == LANTHANIDES_PERIOD:
                    st.write(f"란타넘족")
                elif period_num == ACTINIDES_PERIOD:
                    st.write(f"악티늄족")

            if period_num in range(1, 8):
                for i in range(1, 19):
                    element_in_group = period_df[period_df['group'] == i]
                    with row_cols[i]:
                        if not element_in_group.empty:
                            element = element_in_group.iloc[0]
                            atomic_num = element['atomic_number']
                            symbol = element['symbol']
                            st.session_state.temp_selected_elements_state[atomic_num] = st.checkbox(
                                f"{symbol}",
                                value=st.session_state.temp_selected_elements_state.get(atomic_num, False),
                                key=f"individual_checkbox_{atomic_num}_form"
                            )
                        else:
                            st.write("")
            elif period_num == LANTHANIDES_PERIOD:
                lanthanide_elements = period_df.sort_values(by='atomic_number')
                for i, element in enumerate(lanthanide_elements.itertuples()):
                    with row_cols[i + 1]:
                        atomic_num = element.atomic_number
                        symbol = element.symbol
                        st.session_state.temp_selected_elements_state[atomic_num] = st.checkbox(
                            f"{symbol}",
                            value=st.session_state.temp_selected_elements_state.get(atomic_num, False),
                            key=f"individual_checkbox_{atomic_num}_lan"
                        )
            elif period_num == ACTINIDES_PERIOD:
                actinide_elements = period_df.sort_values(by='atomic_number')
                for i, element in enumerate(actinide_elements.itertuples()):
                    with row_cols[i + 1]:
                        atomic_num = element.atomic_number
                        symbol = element.symbol
                        st.session_state.temp_selected_elements_state[atomic_num] = st.checkbox(
                            f"{symbol}",
                            value=st.session_state.temp_selected_elements_state.get(atomic_num, False),
                            key=f"individual_checkbox_{atomic_num}_act"
                        )

        submit_button_final = st.form_submit_button(label="적용하기")

        if submit_button_final:
            st.session_state.selected_elements_for_3d = {
                an for an, is_checked in st.session_state.temp_selected_elements_state.items() if is_checked
            }
            st.success("선택사항이 3D 주기율표에 적용되었습니다.")
            st.rerun()

filtered_df_for_3d = df[df['atomic_number'].isin(st.session_state.selected_elements_for_3d)].copy()

fig_3d = draw_plotly_3d(filtered_df_for_3d, column_name=selected_column_name, display_label=selected_display_name, compression_factor=current_compression_factor)
st.plotly_chart(fig_3d, use_container_width=True)


st.markdown("---")

st.header("2D 주기율표 (원소를 클릭하여 상세 정보 확인)")

fig_2d = draw_2d_periodic_table(df)

if 'selected_element_info_2d' not in st.session_state:
    st.session_state['selected_element_info_2d'] = None

clicked_points_2d = plotly_events(
    fig_2d,
    click_event=True,
    select_event=False,
    hover_event=False,
    override_height=600,
    key="periodic_table_2d_chart_events"
)

if clicked_points_2d:
    clicked_point_data_2d = clicked_points_2d[0]
    selected_atomic_number = None

    if 'customdata' in clicked_point_data_2d and clicked_point_data_2d['customdata']:
        selected_atomic_number = clicked_point_data_2d['customdata'][0]
    elif 'x' in clicked_point_data_2d and 'y' in clicked_point_data_2d:
        clicked_x = clicked_point_data_2d['x']
        clicked_y = clicked_point_data_2d['y']

        df_displayable = df[(df['group'].notna()) & (df['period'].notna())].copy()

        target_group = round(clicked_x)
        target_period = round(clicked_y)

        potential_elements = df_displayable[
            (df_displayable['group'] == target_group) &
            (df_displayable['period'] == target_period)
        ]

        if not potential_elements.empty:
            selected_atomic_number = potential_elements.iloc[0]['atomic_number']
        else:
            distances = np.sqrt(
                (df_displayable['group'] - clicked_x)**2 +
                (df_displayable['period'] - clicked_y)**2
            )
            if not distances.empty:
                selected_atomic_number = df_displayable.loc[distances.idxmin()]['atomic_number']

    if selected_atomic_number is not None:
        selected_element_df_2d = df[df['atomic_number'] == selected_atomic_number]
        if not selected_element_df_2d.empty:
            st.session_state['selected_element_info_2d'] = selected_element_df_2d.iloc[0].to_dict()
        else:
            st.session_state['selected_element_info_2d'] = None
    else:
        st.session_state['selected_element_info_2d'] = None

st.markdown("---")

st.header("원소 상세 정보")

info_placeholder = st.empty()

element_info_to_display = st.session_state['selected_element_info_2d']

if element_info_to_display:
    with info_placeholder.container():
        st.subheader(f"✨ 원소 정보: {element_info_to_display['kor_name']} ({element_info_to_display['symbol']})")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.write("### 원자 모형")
            image_path = models.get(element_info_to_display['kor_name'])
            if image_path:
                st.image(image_path,
                         caption=f"{element_info_to_display['kor_name']} 원자 모형",
                         width=380)
            else:
                st.info("해당 원소의 원자 모형 이미지를 찾을 수 없습니다.")

        with col2:
            st.write("### 상세 정보")
            st.write(f"**원소 기호:** {element_info_to_display['symbol']}")
            st.write(f"**원소 이름:** {element_info_to_display['kor_name']}")

            def display_numeric_info(label, value, unit=""):
                if pd.notna(value):
                    if (label == "족" and isinstance(value, str)):
                        return f"**{label}:** {value}{unit}"
                    elif label in ["원자 번호", "주기"] and pd.api.types.is_numeric_dtype(type(value)):
                        return f"**{label}:** {int(value)}{unit}"
                    return f"**{label}:** {value}{unit}"
                return f"**{label}:** 정보 없음"

            period = element_info_to_display.get('period')
            group = element_info_to_display.get('group')

            display_period_info = period
            display_group_info = group

            if period == LANTHANIDES_PERIOD:
                display_period_info = 6
                display_group_info = '란타넘족'
            elif period == ACTINIDES_PERIOD:
                display_period_info = 7
                display_group_info = '악티늄족'

            st.write(display_numeric_info("원자 번호", element_info_to_display.get('atomic_number')))
            st.write(display_numeric_info("족", display_group_info))
            st.write(display_numeric_info("주기", display_period_info))
            st.write(display_numeric_info("원자량", element_info_to_display.get('atomic_weight')))
            st.write(display_numeric_info("공유 반지름", element_info_to_display.get('covalent_radius_exp'), unit=" pm"))
            st.write(display_numeric_info("반데르발 반지름", element_info_to_display.get('van_der_waals_radius_exp'), unit=" pm"))
            st.write(display_numeric_info("전기음성도", element_info_to_display.get('electronegativity')))
            st.write(display_numeric_info("전자친화도", element_info_to_display.get('electron_affinity')))
            st.write(display_numeric_info("제1 이온화 에너지 (kJ/mol)", element_info_to_display.get('first_ionization_energy')))
            st.write(display_numeric_info("녹는점 (K)", element_info_to_display.get('melting_point_K')))
            st.write(display_numeric_info("끓는점 (K)", element_info_to_display.get('boiling_point_K')))
            st.write(display_numeric_info("밀도 (g/cm³)", element_info_to_display.get('density_g_cm3')))
            st.write(display_numeric_info("열 전도율 (W/(m·K))", element_info_to_display.get('thermal_conductivity')))
else:
    with info_placeholder.container():
        st.info("2D 주기율표에서 원소를 클릭하여 자세한 정보를 확인하세요.")

st.markdown("---")
st.markdown("틀린 정보가 있을 수 있습니다. ssss2000.0229@gmail.com 으로 알려주세요.")