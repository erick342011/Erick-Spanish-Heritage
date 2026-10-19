import html
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Hello World Motion Lab", page_icon="✨", layout="wide")

st.title("✨ Hello World Motion Lab")
st.caption("Type a name, choose a style, and add a looping animation.")

# Sidebar controls
st.sidebar.header("Customize your app")
name = st.sidebar.text_input("Enter your name", value="Hello World", max_chars=80)

font_family = st.sidebar.selectbox(
    "Font type",
    ["Arial", "Courier New", "Georgia", "Times New Roman", "Verdana", "Trebuchet MS", "Comic Sans MS"],
)
font_size = st.sidebar.slider("Font size", min_value=20, max_value=100, value=48, step=2)

backgrounds = {
    "Light": ("#ffffff", "#172033"),
    "Dark": ("#10131c", "#ffffff"),
    "Ocean": ("#dff6ff", "#064663"),
    "Sunset": ("#fff0e6", "#7b2d26"),
    "Forest": ("#e8f5e9", "#1b4332"),
    "Lavender": ("#f1eaff", "#452c63"),
}
background = st.sidebar.selectbox("Background", list(backgrounds.keys()))
name_mode = st.sidebar.selectbox("Name display mode", ["Centered", "Ticker Tape", "Repeated"])
effect = st.sidebar.selectbox(
    "Dynamic action",
    ["None", "Rain", "Snow", "Hail", "Wind gust", "Confetti", "Tornado"],
)
speed = st.sidebar.slider("Animation speed", min_value=1, max_value=10, value=5,
                          help="Higher values make the animation move faster.")

bg_color, text_color = backgrounds[background]
safe_name = html.escape(name.strip() or "Hello World")
safe_font = font_family.replace('"', "")
# A CSS duration is shorter for higher speed values.
duration = 11 - speed

# Each effect is made with CSS so it loops inside the embedded display
# without repeatedly rerunning the whole Streamlit script.
effect_css = ""
effect_markup = ""

if effect == "Rain":
    effect_css = f"""
    .particle {{ position:absolute; top:-15%; color:#2498e8; font-size:22px;
      animation: rain {duration * 0.22:.2f}s linear infinite; }}
    @keyframes rain {{ to {{ transform:translateY(125vh); }} }}
    """
    effect_markup = "".join(
        f'<span class="particle" style="left:{(i * 17 + 3) % 100}%; animation-delay:-{(i % 9) * 0.23}s;">💧</span>'
        for i in range(34)
    )
elif effect == "Snow":
    effect_css = f"""
    .particle {{ position:absolute; top:-10%; color:#77bfff; font-size:24px;
      animation: snow {duration * 0.45:.2f}s linear infinite; }}
    @keyframes snow {{ to {{ transform:translate3d(45px, 120vh, 0) rotate(360deg); }} }}
    """
    effect_markup = "".join(
        f'<span class="particle" style="left:{(i * 13 + 5) % 100}%; animation-delay:-{(i % 11) * 0.31}s;">❄</span>'
        for i in range(28)
    )
elif effect == "Hail":
    effect_css = f"""
    .particle {{ position:absolute; top:-12%; font-size:20px;
      animation: hail {duration * 0.28:.2f}s linear infinite; }}
    @keyframes hail {{ to {{ transform:translate3d(20px, 125vh, 0); }} }}
    """
    effect_markup = "".join(
        f'<span class="particle" style="left:{(i * 19 + 2) % 100}%; animation-delay:-{(i % 8) * 0.2}s;">🧊</span>'
        for i in range(30)
    )
elif effect == "Wind gust":
    effect_css = f"""
    .particle {{ position:absolute; left:-25%; font-size:28px; white-space:nowrap;
      animation: gust {duration * 0.55:.2f}s ease-in-out infinite; }}
    @keyframes gust {{ 0% {{ transform:translateX(0); opacity:0; }}
      15% {{ opacity:.9; }} 85% {{ opacity:.9; }} 100% {{ transform:translateX(150vw); opacity:0; }} }}
    """
    effect_markup = "".join(
        f'<span class="particle" style="top:{8 + (i * 17) % 82}%; animation-delay:-{(i % 7) * 0.6}s;">〰️ 💨</span>'
        for i in range(9)
    )
elif effect == "Confetti":
    effect_css = f"""
    .particle {{ position:absolute; top:-12%; font-size:24px;
      animation: confetti {duration * 0.42:.2f}s linear infinite; }}
    @keyframes confetti {{ to {{ transform:translate3d(35px, 125vh, 0) rotate(720deg); }} }}
    """
    pieces = ["🎉", "🎊", "🟣", "🟡", "🟢", "🔵", "🟠", "⭐"]
    effect_markup = "".join(
        f'<span class="particle" style="left:{(i * 11 + 1) % 100}%; animation-delay:-{(i % 12) * 0.24}s;">{pieces[i % len(pieces)]}</span>'
        for i in range(40)
    )
elif effect == "Tornado":
    effect_css = f"""
    .vortex {{ position:absolute; left:50%; top:50%; width:240px; height:240px;
      transform:translate(-50%,-50%); animation: spin {duration * 0.7:.2f}s linear infinite; }}
    .vortex span {{ position:absolute; left:50%; top:50%; font-size:34px;
      transform:rotate(calc(var(--i) * 30deg)) translateY(-90px); }}
    @keyframes spin {{ to {{ transform:translate(-50%,-50%) rotate(360deg); }} }}
    """
    effect_markup = '<div class="vortex">' + "".join(
        f'<span style="--i:{i};">{ "🌪️" if i % 3 == 0 else "💨" }</span>'
        for i in range(12)
    ) + "</div>"

if name_mode == "Ticker Tape":
    name_markup = f'<div class="name ticker">{safe_name}</div>'
elif name_mode == "Repeated":
    name_markup = '<div class="repeat-grid">' + "".join(
        f'<div class="name repeated">{safe_name}</div>' for _ in range(6)
    ) + "</div>"
else:
    name_markup = f'<div class="name centered">{safe_name}</div>'

html_page = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:"{safe_font}", sans-serif; }}
  .stage {{
    position:relative; isolation:isolate; overflow:hidden;
    width:100%; height: min(72vh, 620px); min-height:360px;
    border-radius:18px; background:{bg_color}; color:{text_color};
    border:1px solid rgba(120,120,120,.25);
  }}
  .effects {{ position:absolute; inset:0; overflow:hidden; pointer-events:none; z-index:1; }}
  .name-layer {{ position:absolute; inset:0; display:flex; align-items:center;
    justify-content:center; z-index:2; padding:24px; overflow:hidden; }}
  .name {{ font-family:"{safe_font}", sans-serif; font-size:{font_size}px;
    font-weight:700; line-height:1.2; overflow-wrap:anywhere; text-shadow:0 2px 10px rgba(100,100,100,.12); }}
  .centered {{ text-align:center; max-width:100%; }}
  .ticker {{ white-space:nowrap; animation:ticker 12s linear infinite; }}
  @keyframes ticker {{ from {{ transform:translateX(100%); }} to {{ transform:translateX(-100%); }} }}
  .repeat-grid {{ width:100%; display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
    gap:18px; text-align:center; }}
  .repeated {{ font-size:{max(18, int(font_size * 0.62))}px; }}
  {effect_css}
  @media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ animation-duration:30s !important; }}
  }}
</style>
</head>
<body>
  <div class="stage">
    <div class="effects">{effect_markup}</div>
    <div class="name-layer">{name_markup}</div>
  </div>
</body>
</html>
"""

components.html(html_page, height=640, scrolling=False)

st.caption("Tip: Change the controls in the sidebar to try different looks and animations.")
