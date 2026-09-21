import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Wall-Following Robot — MDP Dashboard", layout="wide")

st.title("Wall-Following Robot Navigation — MDP & Value Iteration")
st.caption(
    "Dataset: SCITOS-G5 wall-following robot (Kaggle) · "
    "Features: SD_front, SD_left, SD_right, SD_back · "
    "Classes: Move-Forward, Slight-Right-Turn, Sharp-Right-Turn, Slight-Left-Turn"
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DEFAULT_PATH = "sensor_readings_4.csv"
COLS = ["SD_front", "SD_left", "SD_right", "SD_back", "Class"]

st.sidebar.header("1. Data")
uploaded = st.sidebar.file_uploader("Upload sensor_readings_4.csv (optional)", type="csv")


@st.cache_data
def load_data(file_or_path):
    return pd.read_csv(file_or_path, header=None, names=COLS)


try:
    if uploaded is not None:
        df = load_data(uploaded)
    else:
        df = load_data(DEFAULT_PATH)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DEFAULT_PATH}` next to the app, and no file was uploaded. "
        "Upload `sensor_readings_4.csv` in the sidebar to continue."
    )
    st.stop()

classes = sorted(df["Class"].unique())
feature_cols = ["SD_front", "SD_left", "SD_right", "SD_back"]

st.sidebar.success(f"Loaded {len(df):,} rows")

# ---------------------------------------------------------------------------
# Sidebar: MDP controls
# ---------------------------------------------------------------------------
st.sidebar.header("2. MDP settings")
n_states = st.sidebar.slider("Number of states (K-Means clusters)", 4, 30, 12)
gamma = st.sidebar.slider("Discount factor (gamma)", 0.5, 0.99, 0.9, 0.01)
tolerance = st.sidebar.select_slider(
    "Convergence threshold (tolerance)",
    options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    value=1e-4,
)
max_iterations = st.sidebar.slider("Max iterations", 100, 5000, 2000, 100)
seed = st.sidebar.number_input("Random seed (clustering)", value=42, step=1)

tab1, tab2, tab3 = st.tabs(
    ["📊 Dataset Overview", "🧭 MDP & Value Iteration", "🗺️ Learned Policy"]
)

# ---------------------------------------------------------------------------
# TAB 1: EDA
# ---------------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Classes", len(classes))
    c3.metric("Features", len(feature_cols))
    c4.metric("Missing values", int(df.isna().sum().sum()))

    st.subheader("Sample rows")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class distribution")
        st.bar_chart(df["Class"].value_counts())
    with col2:
        st.subheader("Summary statistics")
        st.dataframe(df[feature_cols].describe(), use_container_width=True)

    st.subheader("Sensor reading distributions by movement class")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, col in zip(axes, feature_cols):
        df.boxplot(column=col, by="Class", ax=ax, rot=30)
        ax.set_title(col)
        ax.set_xlabel("")
    plt.suptitle("")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Correlation between sensors")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    corr = df[feature_cols].corr()
    im = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(feature_cols)))
    ax2.set_xticklabels(feature_cols, rotation=45)
    ax2.set_yticks(range(len(feature_cols)))
    ax2.set_yticklabels(feature_cols)
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            ax2.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig2.colorbar(im)
    st.pyplot(fig2)

# ---------------------------------------------------------------------------
# Build MDP (shared by tabs 2 and 3)
# ---------------------------------------------------------------------------
@st.cache_data
def build_mdp(df, n_states, seed):
    X = df[feature_cols].values
    classes = sorted(df["Class"].unique())
    action_idx = {a: i for i, a in enumerate(classes)}
    n_actions = len(classes)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_states, random_state=int(seed), n_init=10)
    state_labels = km.fit_predict(Xs)

    work = df.copy()
    work["state"] = state_labels
    work["clearance"] = X.mean(axis=1)  # avg distance to walls -> safety proxy
    work["action_idx"] = work["Class"].map(action_idx)

    # Rewards: mean clearance observed for each (state, action) pair.
    # Higher clearance = further from walls = safer navigation.
    R = np.full((n_states, n_actions), work["clearance"].mean())
    for (s, a), g in work.groupby(["state", "action_idx"]):
        R[int(s), int(a)] = g["clearance"].mean()

    # Transitions: empirical, built from consecutive time-ordered rows.
    s_arr = work["state"].values
    a_arr = work["action_idx"].values
    trans_counts = np.zeros((n_states, n_actions, n_states))
    for t in range(len(work) - 1):
        trans_counts[s_arr[t], a_arr[t], s_arr[t + 1]] += 1

    T = np.zeros_like(trans_counts)
    for s in range(n_states):
        for a in range(n_actions):
            tot = trans_counts[s, a].sum()
            T[s, a] = trans_counts[s, a] / tot if tot > 0 else np.ones(n_states) / n_states

    cluster_centers = scaler.inverse_transform(km.cluster_centers_)
    centers_df = pd.DataFrame(cluster_centers, columns=feature_cols)
    centers_df.index.name = "state"

    state_counts = work["state"].value_counts().sort_index()

    return R, T, classes, action_idx, centers_df, state_counts


def value_iteration(R, T, gamma, tolerance, max_iterations):
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    history = []
    for it in range(1, max_iterations + 1):
        Q = R + gamma * np.einsum("saj,j->sa", T, V)
        V_new = Q.max(axis=1)
        delta = np.max(np.abs(V_new - V))
        history.append(delta)
        V = V_new
        if delta < tolerance:
            break
    return V, Q, it, delta, history


R, T, classes, action_idx, centers_df, state_counts = build_mdp(df, n_states, seed)

# ---------------------------------------------------------------------------
# TAB 2: MDP & Value Iteration
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("How this MDP is built from the data")
    st.markdown(
        f"""
- **States ({n_states}):** rows are clustered with K-Means on the 4 standardized sensor
  readings (`SD_front`, `SD_left`, `SD_right`, `SD_back`). Each cluster = one discretized state.
- **Actions ({len(classes)}):** the movement classes — {", ".join(classes)}.
- **Rewards:** mean *clearance* (average of the 4 sensor readings — how far the robot is from
  any wall) observed for each (state, action) pair. Higher clearance = safer = higher reward.
- **Transition probabilities:** estimated empirically from consecutive rows in the (time-ordered)
  dataset — how often action *a* in state *s* led to state *s′*.
"""
    )

    with st.expander("Cluster centers (mean sensor readings per state)"):
        st.dataframe(centers_df.style.format("{:.3f}"), use_container_width=True)
        st.bar_chart(state_counts.rename("rows per state"))

    if st.button("▶ Run Value Iteration", type="primary"):
        V, Q, iters, delta, history = value_iteration(R, T, gamma, tolerance, max_iterations)
        st.session_state["V"] = V
        st.session_state["Q"] = Q
        st.session_state["iters"] = iters
        st.session_state["delta"] = delta
        st.session_state["history"] = history

    if "V" in st.session_state:
        V = st.session_state["V"]
        Q = st.session_state["Q"]
        iters = st.session_state["iters"]
        delta = st.session_state["delta"]
        history = st.session_state["history"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Iterations to converge", iters)
        c2.metric("Final delta", f"{delta:.2e}")
        c3.metric("Converged?", "Yes ✅" if delta < tolerance else "Hit max iterations ⚠️")

        st.subheader("Optimal value function V*(s)")
        value_df = pd.DataFrame({"state": range(n_states), "value": V}).set_index("state")
        st.bar_chart(value_df)
        st.dataframe(value_df.T, use_container_width=True)

        st.subheader("Convergence curve")
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        ax3.plot(history)
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("max |V_new - V_old|")
        ax3.set_yscale("log")
        ax3.axhline(tolerance, color="red", linestyle="--", label="tolerance")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

        # Downloadable CSV, matching the slide's stated output artifact
        out_df = pd.DataFrame({"state": range(n_states), "optimal_value": V})
        csv_buf = io.StringIO()
        out_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇ Download optimal_value_function.csv",
            data=csv_buf.getvalue(),
            file_name="optimal_value_function.csv",
            mime="text/csv",
        )
    else:
        st.info("Adjust settings in the sidebar, then click **Run Value Iteration** above.")

# ---------------------------------------------------------------------------
# TAB 3: Policy
# ---------------------------------------------------------------------------
with tab3:
    if "Q" not in st.session_state:
        st.info("Run Value Iteration in the previous tab first.")
    else:
        Q = st.session_state["Q"]
        best_actions = Q.argmax(axis=1)
        policy_df = pd.DataFrame(
            {
                "state": range(n_states),
                "best_action": [classes[a] for a in best_actions],
                "Q_value": Q.max(axis=1),
            }
        ).set_index("state")

        st.subheader("Greedy optimal policy π*(s) = argmax_a Q(s, a)")
        st.dataframe(policy_df, use_container_width=True)

        st.subheader("Action chosen per state")
        st.bar_chart(policy_df["best_action"].value_counts())

        st.subheader("State profile vs. recommended action")
        merged = centers_df.copy()
        merged["recommended_action"] = policy_df["best_action"].values
        st.dataframe(merged.style.format({c: "{:.3f}" for c in feature_cols}), use_container_width=True)

st.divider()
st.caption(
    "Built for the MDP / Value Iteration wall-following robot exercise. "
    "Dataset source: Kaggle (UCI Wall-Following Robot Navigation)."
)
