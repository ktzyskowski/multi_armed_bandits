# import plotly.express as px
# import plotly.graph_objects as go

# import streamlit as st
# from mab.bandit.nonstationary_bandit import NonstationaryMultiArmedBandit
# from mab.bandit.stationary_bandit import MultiArmedBandit
# from mab.policy.epsilon_greedy_policy import EpsilonGreedyPolicy
# from mab.policy.random_policy import RandomPolicy
# from mab.testbed import Testbed

# st.title("k-Armed Bandit Testbed")


# @st.dialog("Configure Testbed")
# def configure_testbed():
#     # st.write(f"Why is {} your favorite?")
#     # reason = st.text_input("Because...")
#     if st.button("Submit"):
#         # st.session_state.vote
#         st.rerun()


# @st.cache_data
# def run_testbed(n_runs: int, n_steps: int):
#     testbed = Testbed(
#         policy_factory=lambda random_seed: EpsilonGreedyPolicy(
#             eps=0.1,
#             k=k,
#             random_seed=random_seed,
#         ),
#         bandit_factory=lambda random_seed: NonstationaryMultiArmedBandit(
#             k=k,
#             random_seed=random_seed,
#         ),
#     )
#     rewards = testbed.run(n_runs, n_steps)
#     return rewards


# with st.sidebar:
#     # testbed config
#     n_runs = st.number_input("Number of runs:", 1, 2_000, 2_000)
#     n_steps = st.number_input("Number of steps:", 1, 10_000, 1_000)
#     # number of bandit arms
#     k = st.number_input("Number of bandit arms ($k$):", 1, 10, 10)

#     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
#     # st.divider()
#     # random_policy_enabled = st.checkbox("Random Policy")
#     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
#     st.divider()
#     epsilon_greedy_policy_enabled = st.checkbox("$\\epsilon$-Greedy Policy")
#     epsilon_greedy_policy_config = {
#         "epsilon": st.number_input(
#             "$\\epsilon$",
#             0.0,
#             1.0,
#             0.1,
#             disabled=not epsilon_greedy_policy_enabled,
#             key="epsilon_greedy_policy",
#         )
#     }

# run_started = st.button("Run", type="primary", width="stretch")

# if run_started:
#     with st.spinner(f"Running bandits...", show_time=True):
#         rewards = run_simulator(n_runs, n_steps)
#         average_rewards = rewards.mean(axis=0)

#         fig = go.Figure()
#         fig.add_trace(go.Scatter(y=average_rewards, mode="lines", name="random"))
#         fig.update_layout(xaxis_title="Step", yaxis_title="Average Reward")

#         # fig = px.line(average_rewards, title="Average Reward", name="random")
#         st.plotly_chart(fig)
