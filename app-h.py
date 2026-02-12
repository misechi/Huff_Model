import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math

# --- メッシュコード解析 & 緯度経度変換関数（境界情報付き） ---
def get_mesh_geometry(code):
    c = str(code).strip()
    length = len(c)
    
    # 緯度の起点
    lat_min = int(c[0:2]) / 1.5
    lon_min = int(c[2:4]) + 100
    
    if length >= 6:
        lat_min += (int(c[4]) * 5) / 60
        lon_min += (int(c[5]) * 7.5) / 60
    if length >= 8:
        lat_min += (int(c[6]) * 30) / 3600
        lon_min += (int(c[7]) * 45) / 3600
    
    u_lat, u_lon = 30/3600, 45/3600 # 3次基準
    if length >= 9:
        y, x = (1, 0) if c[8] in "34" else (0, 0)
        if c[8] in "24": x = 1
        u_lat, u_lon = 15/3600, 22.5/3600
        lat_min += y * u_lat
        lon_min += x * u_lon
    if length >= 10:
        y, x = (1, 0) if c[9] in "34" else (0, 0)
        if c[9] in "24": x = 1
        u_lat, u_lon = 7.5/3600, 11.25/3600
        lat_min += y * u_lat
        lon_min += x * u_lon
    if length >= 11:
        y, x = (1, 0) if c[10] in "34" else (0, 0)
        if c[10] in "24": x = 1
        u_lat, u_lon = 3.75/3600, 5.625/3600
        lat_min += y * u_lat
        lon_min += x * u_lon

    lat_max = lat_min + u_lat
    lon_max = lon_min + u_lon
    c_lat, c_lon = lat_min + (u_lat/2), lon_min + (u_lon/2)
    diag_m = hubeny_distance(lat_min, lon_min, lat_max, lon_max)
    
    return c_lat, c_lon, diag_m, lat_min, lat_max, lon_min, lon_max

# --- ヒュベニの公式 ---
def hubeny_distance(lat1, lon1, lat2, lon2):
    a, e2 = 6378137.0, 0.00669437999019758
    dy, dx = math.radians(lat1 - lat2), math.radians(lon1 - lon2)
    mu = math.radians((lat1 + lat2) / 2.0)
    w = math.sqrt(1.0 - e2 * math.sin(mu)**2)
    m = a * (1.0 - e2) / w**3
    n = a / w
    return math.sqrt((dy * m)**2 + (dx * n * math.cos(mu))**2)

st.set_page_config(page_title="ハフモデル分析・最新版", layout="wide")
st.title("🗺️ ハフモデルシミュレーター")

st.sidebar.header("⚙️ 計算パラメータ")
alpha = st.sidebar.slider("魅力度係数 (α)", 0.5, 2.0, 1.0, 0.1)
beta = st.sidebar.slider("距離抵抗係数 (β)", 1.0, 3.0, 2.0, 0.1)

st.header("1. 解析エリアデータの読み込み")
uploaded_file = st.file_uploader("テンプレート形式のCSVをアップロード（1列目:コード, 3列目:人口）", type="csv")

if uploaded_file is not None:
    try:
        # まず標準的な UTF-8 で読み込む
        uploaded_file.seek(0)
        raw_df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        # エラーが出たら 日本語Windows標準の Shift-JIS(cp932) で読み直す
        uploaded_file.seek(0)
        raw_df = pd.read_csv(uploaded_file, encoding="cp932")    # ここに続きの処理を書く
    
    # ユーザー指定の形式に従い、1列目と3列目を抽出してリネーム
    df = pd.DataFrame({
        'mesh_code': raw_df.iloc[:, 0],
        'population': raw_df.iloc[:, 2]
    })
    
    # データ制限（20x20を想定した400件）
    if len(df) > 400:
        st.warning("⚠️ データの総数が400件を超えています。上位400件のみ処理します。")
        df = df.head(400)

    # 緯度経度・ジオメトリ算出
    with st.spinner('地理データを計算中...'):
        results = df['mesh_code'].apply(get_mesh_geometry)
        df['c_lat'], df['c_lon'], df['diag_m'], df['l_min'], df['l_max'], df['ln_min'], df['ln_max'] = zip(*results)
    
    # 柔軟なグリッド座標（X, Y）の割り当て
    all_lons = sorted(df['c_lon'].unique())
    all_lats = sorted(df['c_lat'].unique())
    lon_map = {lon: i + 1 for i, lon in enumerate(all_lons)}
    lat_map = {lat: i + 1 for i, lat in enumerate(all_lats)}
    df['X'] = df['c_lon'].map(lon_map)
    df['Y'] = df['c_lat'].map(lat_map)

    st.header("2. 店舗位置の設定")
    st.info("Googleマップの座標（例: 35.62243, 139.71959）をそのまま貼り付けてください。")
    
    num_stores = st.number_input("比較店舗数", 2, 5, 3)
    stores = []
    cols = st.columns(num_stores)
    
    for i in range(num_stores):
        with cols[i]:
            st.subheader(f"店舗 {i+1}")
            s_name = st.text_input(f"店名 {i+1}", f"店舗{i+1}", key=f"sn_{i}")
            s_latlon = st.text_input(f"座標 {i+1}", "35.6224, 139.7195", key=f"sl_{i}")
            s_aj = st.number_input(f"魅力度(Aj) {i+1}", 100, 10000, 1000, key=f"sa_{i}")
            
            try:
                # 引用符やスペースを除去してパース
                s_latlon = s_latlon.strip().replace('"', '').replace('”', '').replace(' ', '')
                lat_str, lon_str = s_latlon.split(',')
                stores.append({"name": s_name, "lat": float(lat_str), "lon": float(lon_str), "aj": s_aj})
            except:
                st.error("形式エラー: '緯度, 経度' で入力してください。")

    if len(stores) == num_stores:
        # 距離・確率・来客数の計算
        for i, s in enumerate(stores):
            d_col = f'dist_{s["name"]}(m)'
            # ヒュベニ距離
            df[d_col] = df.apply(lambda r: hubeny_distance(r['c_lat'], r['c_lon'], s['lat'], s['lon']), axis=1)
            # 店舗がメッシュ内なら「対角線長/4」で補正
            is_inside = (df['l_min'] <= s['lat']) & (s['lat'] < df['l_max']) & \
                        (df['ln_min'] <= s['lon']) & (s['lon'] < df['ln_max'])
            df[d_col] = np.where(is_inside, df['diag_m'] / 4, df[d_col])
            # 引力G
            df[f'G_{i}'] = (s['aj']**alpha) * (df[d_col]**-beta)

        df['total_G'] = df[[f'G_{i}' for i in range(len(stores))]].sum(axis=1)
        for i, s in enumerate(stores):
            df[f'prob_{s["name"]}'] = df[f'G_{i}'] / df['total_G']
            df[f'expected_{s["name"]}'] = df[f'prob_{s["name"]}'] * df['population']

        # 可視化：RGB合成マップ
        st.header("3. 総合勢力図（RGBヒートマップ）")
        grid_x, grid_y = len(all_lons), len(all_lats)
        rgb_map = np.zeros((grid_y, grid_x, 3))
        max_pop = df['population'].max()

        for _, row in df.iterrows():
            iy, ix = int(row['Y'])-1, int(row['X'])-1
            br = row['population'] / max_pop if max_pop > 0 else 0
            # 店舗1=R, 店舗2=B, 店舗3=G
            r = row[f'prob_{stores[0]["name"]}'] * br if num_stores >= 1 else 0
            b = row[f'prob_{stores[1]["name"]}'] * br if num_stores >= 2 else 0
            g = row[f'prob_{stores[2]["name"]}'] * br if num_stores >= 3 else 0
            rgb_map[iy, ix] = [r, g, b]

        fig = px.imshow(rgb_map, x=list(range(1, grid_x+1)), y=list(range(1, grid_y+1)), 
                        origin='lower', title="勢力図（赤:店1, 青:店2, 緑:店3 / 明るさ:人口密度）")
        st.plotly_chart(fig, use_container_width=True)

        # ダウンロード
        st.header("4. 結果出力")
        drop_list = ['c_lat', 'c_lon', 'diag_m', 'l_min', 'l_max', 'ln_min', 'ln_max', 'total_G'] + [f'G_{i}' for i in range(len(stores))]
        out_df = df.drop(columns=drop_list)
        st.download_button("📥 計算結果(距離・確率込)をCSVで保存", out_df.to_csv(index=False).encode('utf-8-sig'), "huff_result.csv", "text/csv")
        st.dataframe(out_df.head())
else:
    st.info("CSVファイルをアップロードしてください。")
