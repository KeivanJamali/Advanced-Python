import matplotlib.pyplot as plt
import matplotlib as mpl
import arabic_reshaper
from bidi.algorithm import get_display

def configure_matplotlib_for_persian(font_family="Arial", font_size=12):
    """
    Configure Matplotlib to properly render Persian (RTL) and English (LTR) text.

    Args:
        font_family: str, name of the font (must support Persian/Arabic glyphs)
        font_size: int, default font size
    """
    mpl.rcParams['font.family'] = font_family
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['axes.unicode_minus'] = False  # fix for minus sign in Persian

def reshape_text(text):
    """
    Reshape Persian/Arabic text for Matplotlib display.
    Keeps English words unchanged, fixes RTL/LTR rendering.

    Args:
        text: str, input text (Persian/English mix)

    Returns:
        str, reshaped and bidi-corrected text
    """
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        return bidi_text
    except Exception:
        # Fallback in case reshaping fails (e.g., text is English only)
        return text
