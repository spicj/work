
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, gamma

st.set_page_config(page_title="Distribution Explorer", layout="wide")
st.title("Distribution Explorer")

# -------- Sidebar (stable widgets) --------
with st.sidebar:
    st.header("Controls")
    dist_name = st.selectbox("Distribution", ["Normal", "Poisson", "Gamma"], index=0, key="dist")
    show_cdf  = st.checkbox("Show CDF", value=False, key="show_cdf")

    # Keep bounds stable while dragging (don't compute them from other live widgets)
    x_min = st.number_input("x-axis min", value=-5.0 if dist_name == "Normal" else 0.0, step=0.5, key="xmin")
    x_max = st.number_input("x-axis max", value=5.0 if dist_name == "Normal" else 20.0, step=0.5, key="xmax")
    num_points = st.slider("Resolution (points)", 200, 2000, 800, 100, key="npts")

    if dist_name == "Normal":
        mu    = st.slider("μ (Mean/Average)", -10, 10, 0, 1, key="mu")
        sigma = st.slider("σ (Standard Deviation)", 0.1, 10.0, 1.0, 0.1, key="sigma")

    elif dist_name == "Poisson":
        lam   = st.slider("λ (Mean/Rate)", 0.1, 1.0, 0.1, 0.1, key="lam")
        kmax  = st.slider("Max k (x-limit)", 0, 20, 7, 1, key="kmax")

    else:  # Gamma
        k     = st.slider("k (Shape)", 0.1, 20.0, 2.0, 0.1, key="kshape")
        theta = st.slider("θ (Scale)", 0.1, 10.0, 2.0, 0.1, key="theta")

# -------- Fragment: only this part reruns on slider drag --------
@st.fragment
def draw_chart():
    fig, ax = plt.subplots(figsize=(6, 4))

    if st.session_state.dist == "Normal":
        x = np.linspace(st.session_state.xmin, st.session_state.xmax, st.session_state.npts)
        pdf = norm.pdf(x, loc=st.session_state.mu, scale=st.session_state.sigma)
        ax.plot(x, pdf, label="PDF", color="#1f77b4")
        if st.session_state.show_cdf:
            ax.plot(x, norm.cdf(x, loc=st.session_state.mu, scale=st.session_state.sigma),
                    label="CDF", color="#ff7f0e", linestyle="--")
        ax.set_title(f"Normal(μ={st.session_state.mu:.2f}, σ={st.session_state.sigma:.2f})")

    elif st.session_state.dist == "Poisson":
        k_vals = np.arange(0, st.session_state.kmax + 1)
        pmf = poisson.pmf(k_vals, mu=st.session_state.lam)
        ax.vlines(k_vals, [0], pmf, colors="#1f77b4", lw=2, label="PMF")
        ax.scatter(k_vals, pmf, color="#1f77b4", s=16)
        if st.session_state.show_cdf:
            ax.step(k_vals, poisson.cdf(k_vals, mu=st.session_state.lam), where="post",
                    label="CDF", color="#ff7f0e", linestyle="--")
        ax.set_xlim(0, st.session_state.kmax)
        ax.set_title(f"Poisson(λ={st.session_state.lam:.2f})")

    else:  # Gamma
        xmin = max(0.0, st.session_state.xmin)
        xmax = max(xmin + 1e-6, st.session_state.xmax)
        x = np.linspace(xmin, xmax, st.session_state.npts)
        pdf = gamma.pdf(x, a=st.session_state.kshape, scale=st.session_state.theta)
        ax.plot(x, pdf, label="PDF", color="#1f77b4")
        if st.session_state.show_cdf:
            ax.plot(x, gamma.cdf(x, a=st.session_state.kshape, scale=st.session_state.theta),
                    label="CDF", color="#ff7f0e", linestyle="--")
        m = st.session_state.kshape * st.session_state.theta
        v = st.session_state.kshape * (st.session_state.theta**2)
        ax.set_title(f"Gamma(k={st.session_state.kshape:.2f}, θ={st.session_state.theta:.2f}) | mean={m:.2f}, var={v:.2f}")

    ax.set_xlabel("x" if st.session_state.dist != "Poisson" else "Event number k")
    ax.set_ylabel("Probability Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)

# Layout: three columns, graph in the middle
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    draw_chart()


