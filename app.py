from sklearn.linear_model import LinearRegression
from io import BytesIO
import pandas as pd
import plotly.graph_objects as go
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import ScalarFormatter

# Black-Scholes Formula Class


class BlackScholes:
    @staticmethod
    def calculate(S, K, T, r, sigma, option_type="call"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# Binomial Tree Class
class BinomialTree:
    def __init__(self, S, K, T, r, steps, option_type="call", american=False):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.steps = steps
        self.dt = T / steps
        self.discount = np.exp(-r * self.dt)
        self.option_type = option_type
        self.american = american

    def calculate_price(self, u, d, q):
        # Initialize stock prices at maturity
        stock_prices = self.S * \
            d**np.arange(self.steps, -1, -1) * u**np.arange(0, self.steps + 1)

        if self.option_type == "call":
            option_values = np.maximum(0, stock_prices - self.K)
        else:  # Put
            option_values = np.maximum(0, self.K - stock_prices)

        # Backward induction
        for i in range(self.steps - 1, -1, -1):
            option_values = self.discount * \
                (q * option_values[1:] + (1 - q) * option_values[:-1])

            if self.american:
                stock_prices = self.S * \
                    d**np.arange(i, -1, -1) * u**np.arange(0, i + 1)
                if self.option_type == "call":
                    intrinsic_values = np.maximum(0, stock_prices - self.K)
                else:  # Put
                    intrinsic_values = np.maximum(0, self.K - stock_prices)
                option_values = np.maximum(option_values, intrinsic_values)

        return option_values[0]


# CRR Model Class
class CRRModel:
    def __init__(self, option):
        self.option = option

    def calculate(self, steps):
        dt = self.option.T / steps
        u = np.exp(self.option.sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp(self.option.r * dt) - d) / (u - d)
        tree = BinomialTree(self.option.S, self.option.K, self.option.T,
                            self.option.r, steps, self.option.option_type, self.option.american)
        return tree.calculate_price(u, d, q)


x = BinomialTree()

x.calculate_price()
# Tian Model Class


class TianModel:
    def __init__(self, option):
        self.option = option

    def calculate(self, steps):
        dt = self.option.T / steps
        M = np.exp(self.option.r * dt)
        V = np.exp(self.option.sigma**2 * dt)
        u = M * np.exp(self.option.sigma * np.sqrt(dt))
        d = M * np.exp(-self.option.sigma * np.sqrt(dt))
        q = (M - d) / (u - d)
        tree = BinomialTree(self.option.S, self.option.K, self.option.T,
                            self.option.r, steps, self.option.option_type, self.option.american)
        return tree.calculate_price(u, d, q)


class LeisenReimerModel:
    def __init__(self, option):
        self.option = option

    def calculate(self, steps):
        # Ensure steps are odd for Leisen-Reimer model
        if steps % 2 == 0:
            steps += 1

        dt = self.option.T / steps
        a = np.exp(self.option.r * dt)

        # Compute d1 and d2 using standard Black-Scholes logic
        d1 = (np.log(self.option.S / self.option.K) + (self.option.r + 0.5 * self.option.sigma**2) * self.option.T) / \
             (self.option.sigma * np.sqrt(self.option.T))
        d2 = d1 - self.option.sigma * np.sqrt(self.option.T)

        def peizer_pratt(z, n):
            exponent = -((z / (n + 1 / 3)) ** 2) * (n + 1 / 6)
            sqrt_term = np.sqrt(1 / 4 - 1 / 4 * np.exp(exponent))
            return 0.5 + np.sign(z) * sqrt_term

        # Use Peizer-Pratt to find p and p_bar
        p = peizer_pratt(d2, steps)
        p_bar = peizer_pratt(d1, steps)

        # Calculate up and down move factors
        u = a * (p_bar / p)
        d = a * ((1 - p_bar) / (1 - p))
        q = p  # Use calculated probability

        # Binomial tree backward induction
        tree = BinomialTree(self.option.S, self.option.K, self.option.T, self.option.r, steps, self.option.option_type,
                            self.option.american)
        price = tree.calculate_price(u, d, q)
        return price


# Pegged CRR Model Class
class PeggedCRRModel:
    def __init__(self, option):
        self.option = option

    def calculate(self, steps):
        dt = self.option.T / steps
        u = np.exp(self.option.sigma * np.sqrt(dt) + dt *
                   np.log(self.option.K / self.option.S))
        d = 1 / u
        q = (np.exp(self.option.r * dt) - d) / (u - d)
        tree = BinomialTree(self.option.S, self.option.K, self.option.T,
                            self.option.r, steps, self.option.option_type, self.option.american)
        return tree.calculate_price(u, d, q)


# Option Class
class Option:
    def __init__(self, S, K, T, r, sigma, option_type="call", american=False):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.american = american

# --- ConvergenceAnalysis Class ---

# --- ConvergenceAnalysis Class ---


class ConvergenceAnalysis:
    def __init__(self, time_steps, model, bs_price):
        self.time_steps = time_steps
        self.model = model
        self.bs_price = bs_price

    def calculate_errors(self):
        """Calculate absolute errors for the selected model."""
        errors = []
        for n in self.time_steps:
            model_price = self.model.calculate(n)
            error = abs(model_price - self.bs_price)
            errors.append(error)
        return errors

    def analyze_convergence(self, errors):
        """Perform regression analysis on log-log error data to determine convergence order."""
        # Prepare log-log data
        log_n = np.log(self.time_steps).reshape(-1, 1)
        log_errors = np.log(errors).reshape(-1, 1)

        # Fit a linear regression model
        regression = LinearRegression()
        regression.fit(log_n, log_errors)
        slope = regression.coef_[0][0]

        return slope

    def plot_convergence(self, errors, model_name, option_type, american, slope):
        """Plot the error convergence for the selected model."""
        fig = go.Figure()

        # Add error trace
        fig.add_trace(go.Scatter(
            x=self.time_steps,
            y=errors,
            mode='lines+markers',
            name="Absolute Error",
            line=dict(color='orange', width=2)
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Error Convergence Analysis ({model_name} Model)<br>Convergence Order: ~{
                    slope:.2f}",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Number of Time Steps",
            yaxis_title="Absolute Error",
            yaxis_type="log",  # Logarithmic scale for better visualization
            xaxis_type="log",
            template="plotly_white",
            legend=dict(orientation="h", x=0.5, y=-0.3, xanchor="center")
        )

        # Display the plot
        st.plotly_chart(fig)
# --- PlotModel Class ---


class PlotModel:
    @staticmethod
    def plot_results(time_steps, prices, bs_price, model_name, option_type, american):
        fig = go.Figure()

        # Add model prices trace
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=prices,
            mode='lines',
            name=f"{model_name} Model",
            line=dict(color='blue', width=2)
        ))

        # Add Black-Scholes Price as a horizontal line if applicable
        if bs_price is not None:
            fig.add_trace(go.Scatter(
                x=[time_steps[0], time_steps[-1]],
                y=[bs_price, bs_price],
                mode='lines',
                name="Black-Scholes Price",
                line=dict(color='red', dash='dash')
            ))

        # Update layout for centered title and legend at the bottom
        fig.update_layout(
            title=dict(
                text=f"Convergence of {option_type.capitalize()} Option Prices ({model_name} Model, {
                    'American' if american else 'European'})",
                x=0.5,  # Center the title horizontally
                xanchor="center",
                font=dict(size=16)
            ),
            xaxis_title="Number of Time Steps",
            yaxis_title="Option Price",
            template="plotly_white",
            legend=dict(
                orientation="h",  # Horizontal orientation
                x=0.5,  # Center the legend horizontally
                y=-0.2,  # Place the legend outside the plot at the bottom
                xanchor="center",
                yanchor="top"
            )
        )

        # Render the plot in Streamlit
        st.plotly_chart(fig)


class PlotModel:
    @staticmethod
    def plot_results(time_steps, prices, bs_price, model_name, option_type, american):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_steps, y=prices, mode='lines', name=f"{
                      model_name} Model", line=dict(color='blue', width=2)))
        if bs_price is not None:
            fig.add_trace(go.Scatter(x=[time_steps[0], time_steps[-1]], y=[bs_price, bs_price],
                          mode='lines', name="Black-Scholes Price", line=dict(color='red', dash='dash')))
        fig.update_layout(
            title={
                "text": f"Convergence of {option_type.capitalize()} Option Prices ({model_name} Model, {'American' if american else 'European'})",
                "x": 0.5, "xanchor": "center", "yanchor": "top"},
            xaxis_title="Number of Time Steps",
            yaxis_title="Option Price",
            legend=dict(orientation="h", y=-0.3, x=0.5,
                        xanchor='center', title_text=""),
            template="plotly_white",
        )
        st.plotly_chart(fig)
        return fig  # Return figure for later use


class App:
    # --- Streamlit App Main Function ---
    def main(self):
        st.title("Option Pricing Models")
        st.markdown("## Compare Convergence of Binomial Models")

        # Sidebar: Option Parameters
        st.sidebar.header("Option Parameters")
        S = st.sidebar.number_input(
            "Spot Price (S)", min_value=1.0, value=100.0)
        K = st.sidebar.number_input(
            "Strike Price (K)", min_value=1.0, value=110.0)
        T = st.sidebar.number_input(
            "Time to Maturity (T, in years)", min_value=0.01, value=0.5)
        r = st.sidebar.number_input(
            "Risk-Free Rate (r)", min_value=0.0, value=0.02)
        sigma = st.sidebar.number_input(
            "Volatility (\u03c3)", min_value=0.01, value=0.4)
        option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
        american = st.sidebar.selectbox(
            "Option Style", ["European", "American"]) == "American"

        st.sidebar.header("Model and Steps")
        model_name = st.sidebar.selectbox(
            "Pricing Model", ["CRR", "Tian", "Leisen-Reimer", "Pegged CRR"], index=0)
        min_steps = st.sidebar.number_input("Min Steps", value=25, step=1)
        max_steps = st.sidebar.number_input("Max Steps", value=225, step=1)
        step_size = st.sidebar.number_input(
            "Step Size", min_value=1, value=1, step=1)

        # error printing msg
        if min_steps >= max_steps:
            st.error("Min Steps must be less than Max Steps.")
            st.stop()

        if (max_steps - min_steps) < step_size:
            st.error(
                "Step Size is too large compared to the range of steps. Adjust it.")
            st.stop()

        # Instantiate Option
        option = Option(S, K, T, r, sigma, option_type, american)

        # Select Model
        if model_name == "CRR":
            model = CRRModel(option)
        elif model_name == "Tian":
            model = TianModel(option)
        elif model_name == "Leisen-Reimer":
            model = LeisenReimerModel(option)
        elif model_name == "Pegged CRR":
            model = PeggedCRRModel(option)

        time_steps = np.arange(min_steps, max_steps + 1, step_size)

        # Calculate Prices
        prices = []
        progress_bar = st.progress(0)
        for i, n in enumerate(time_steps):
            prices.append(model.calculate(n))
            progress_bar.progress((i + 1) / len(time_steps))

        # Black-Scholes Price
        bs_price = BlackScholes.calculate(
            S, K, T, r, sigma, option_type) if not american else None

        # Plot Results
        fig = PlotModel.plot_results(time_steps, prices, bs_price,
                                     model_name, option_type, american)

        # Final Option Price
        final_price = prices[-1]
        st.success(f"The final price of the option using {
                   model_name} model is: **{final_price:.4f}**")

        # --- Convergence Analysis Button ---
        if st.button("Convergence Analysis"):
            st.subheader("Error Convergence Analysis")
            if not american and bs_price is not None:
                # Perform convergence analysis
                convergence_analysis = ConvergenceAnalysis(
                    time_steps, model, bs_price)
                errors = convergence_analysis.calculate_errors()
                slope = convergence_analysis.analyze_convergence(errors)

                # Plot convergence with the convergence rate
                convergence_analysis.plot_convergence(
                    errors, model_name, option_type, american, slope)

                # Display the convergence rate
                st.info(f"Estimated Convergence Order (p): **{slope:.2f}**")

                # Add interpretation
                st.markdown("""
                **Interpretation**:
                - A slope of **-1** corresponds to **linear convergence** (p = 1).
                - A slope of **-2** corresponds to **quadratic convergence** (p = 2).
                - A slope of **-0.5** means the error converges slower than linear (p = 0.5).
                - The sign of the slope is negative because errors decrease as the number of time steps \( n \) increases.
                """)

            else:
                st.warning(
                    "Convergence analysis requires European options with a valid Black-Scholes price.")

        # --- Generate Report ---
        if st.button("Generate Report"):
            st.session_state['report_generated'] = True
            convergence_analysis = ConvergenceAnalysis(
                time_steps, model, bs_price) if not american and bs_price else None
            errors = convergence_analysis.calculate_errors() if convergence_analysis else None
            st.session_state['report'] = self.create_report(
                S, K, T, r, sigma, option_type, american, time_steps, prices, bs_price, fig, errors, convergence_analysis
            )

        if st.session_state.get('report_generated', False):
            st.download_button("Download Report", data=st.session_state['report'],
                               file_name="option_pricing_report.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # --- Report Generation ---

    def create_report(self, S, K, T, r, sigma, option_type, american, time_steps, prices, bs_price, fig, errors, convergence_analysis):
        output = BytesIO()

        # Generate the Convergence Plot and Save as Image
        convergence_plot_buffer = BytesIO()
        if errors:
            convergence_fig = go.Figure()
            convergence_fig.add_trace(go.Scatter(
                x=time_steps,
                y=errors,
                mode='lines+markers',
                name="Absolute Error",
                line=dict(color='orange', width=2)
            ))
            convergence_fig.update_layout(
                title="Convergence Error Plot",
                xaxis_title="Number of Time Steps",
                yaxis_title="Absolute Error",
                yaxis_type="log",  # Logarithmic scale
                xaxis_type="log",
                template="plotly_white"
            )
            convergence_fig.write_image(convergence_plot_buffer, format="png")

        # Write to Excel
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # --- Sheet 1: Parameters and Final Results ---
            params = {"Spot Price": S, "Strike Price": K, "Time to Maturity": T, "Risk-Free Rate": r, "Volatility": sigma,
                      "Option Type": option_type, "Option Style": "American" if american else "European", "Final Price": prices[-1]}
            if not american and bs_price is not None:
                params["Black-Scholes Price"] = bs_price
            pd.DataFrame([params]).to_excel(
                writer, sheet_name="Parameters", index=False)

            # Insert the primary plot into the worksheet
            workbook = writer.book
            worksheet = writer.sheets['Parameters']
            plot_buffer = BytesIO()
            fig.write_image(plot_buffer, format="png")
            plot_buffer.seek(0)
            worksheet.insert_image(
                'D5', 'plot.png', {'image_data': plot_buffer})

            # --- Sheet 2: Time Steps and Option Prices ---
            pd.DataFrame({"Time Steps": time_steps, "Model Prices": prices}).to_excel(
                writer, sheet_name="Model Values", index=False)

            # --- Sheet 3: Convergence Errors and Plot ---
            if errors:
                # Insert error values into a DataFrame
                pd.DataFrame({"Time Steps": time_steps, "Convergence Errors": errors}).to_excel(
                    writer, sheet_name="Convergence Errors", index=False)

                # Insert the convergence plot into the worksheet
                worksheet_errors = writer.sheets['Convergence Errors']
                convergence_plot_buffer.seek(0)
                worksheet_errors.insert_image('D5', 'convergence_plot.png', {
                                              'image_data': convergence_plot_buffer})

        return output.getvalue()


if __name__ == "__main__":
    app = App()
    app.main()
