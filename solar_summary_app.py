import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Replace these with dynamic inputs or calculated values from your project
usable_area_m2 = 45.0
solar_capacity_kw = 6.9
installation_cost = 483000
annual_savings = 66240
payback_period_years = 7.3

def plot_payback(installation_cost, annual_savings, years=10):
    years_range = np.arange(1, years + 1)
    cumulative_savings = annual_savings * years_range
    net_cost = installation_cost - cumulative_savings

    fig, ax = plt.subplots()
    ax.plot(years_range, net_cost, marker='o', label='Net Investment Over Time')
    ax.axhline(0, color='green', linestyle='--', label='Break-even')
    ax.set_xlabel('Years')
    ax.set_ylabel('Net Cost (₹)')
    ax.set_title('Payback Period Visualization')
    ax.legend()
    ax.grid(True)
    return fig

def plot_annual_savings(annual_savings, years=10):
    years_range = np.arange(1, years + 1)
    savings = np.full(years, annual_savings)

    fig, ax = plt.subplots()
    ax.bar(years_range, savings, color='orange')
    ax.set_xlabel('Years')
    ax.set_ylabel('Annual Savings (₹)')
    ax.set_title('Annual Savings Over Time')
    return fig

def main():
    st.title("🌞 Solar Feasibility Summary & Visualization")

    st.markdown("### Summary Report")
    summary_md = f"""
    **Rooftop Specifications:**
    - Usable Area: {usable_area_m2} m²
    - Estimated Solar Capacity: {solar_capacity_kw} kW

    **Financial Overview:**
    - Installation Cost: ₹{installation_cost:,}
    - Annual Savings: ₹{annual_savings:,}
    - Payback Period: {payback_period_years} years
    """
    st.markdown(summary_md)

    st.markdown("---")
    st.markdown("### Payback Period Visualization")
    fig1 = plot_payback(installation_cost, annual_savings)
    st.pyplot(fig1)

    st.markdown("### Annual Savings Over Time")
    fig2 = plot_annual_savings(annual_savings)
    st.pyplot(fig2)

    st.markdown("---")
    st.markdown("### Personalized Recommendations")

    budget = st.number_input("Your budget for installation (₹):", min_value=0, value=installation_cost)
    incentive_eligible = st.checkbox("Eligible for government incentives or subsidies?", value=False)

    if st.button("Get Recommendations"):
        reco = []
        if budget < installation_cost:
            reco.append("Consider applying for financing options or loans to cover the upfront cost.")
        else:
            reco.append("Your budget covers the installation cost — you are ready to proceed!")

        if incentive_eligible:
            reco.append("You should apply for available government incentives to reduce costs further.")
        else:
            reco.append("Check local regulations for any upcoming incentives or rebates.")

        reco.append("Regular maintenance ensures maximum efficiency and long-term savings.")

        st.markdown("**Recommendations:**")
        for r in reco:
            st.write(f"- {r}")

if __name__ == "__main__":
    main()
