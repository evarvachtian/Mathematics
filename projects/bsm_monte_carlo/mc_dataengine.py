import numpy as np
import time
import matplotlib.pyplot as plt
from statistics import NormalDist
from dataclasses import dataclass

#Set up dataclasses for our option and simulation constants
@dataclass(frozen=True)
class OptionParameters:
    S0: float #Initial stock price
    K: float #Strike price
    T: float #Time to maturity (years)
    sigma: float #volatility
    r: float #risk free rate of interest

@dataclass(frozen=True)
class SimulationParameters:
    N_values: tuple[int, ...] #number of simulated paths
    M_values: tuple[int, ...] #number of time steps
    seed: int #seed used for random number generation
    convergence_repetitions: int #number of times the monte carlo simulation is ran
    complexity_repetitions: int #number of times that runtime is averaged

def main():
    #Create objects for our dataclasses
    euoption = OptionParameters(
        S0 = 120,
        K = 125,
        T = 1,
        sigma = 0.22,
        r = 0.03,
    )

    simulation = SimulationParameters(
        N_values=(1,2,5,
                  10,20,50,
                  100,200,500,
                  1000,2000,5000,
                  10000,20000,50000,
                  100000,200000,500000,
                  1000000),
        M_values=(1,10,100),
        seed = 572141,
        convergence_repetitions = 50,
        complexity_repetitions = 2
    )

    call_bsm, put_bsm = bsm(euoption)

    convergence_results, call_slope, put_slope = compute_convergence(simulation, euoption, call_bsm, put_bsm)

    complexity_results = compute_complexity(simulation, euoption)

    plot_error(convergence_results, call_slope, put_slope)

    plot_convergence(convergence_results, call_bsm, put_bsm)

    plot_runtime(complexity_results, simulation)

def bsm(euoption: OptionParameters) -> tuple[float,float]:
    #Compute the Black-Scholes-Merton analytical solution for the given option parameters

    std_normal = NormalDist()

    d1 = (
    np.log(euoption.S0 / euoption.K)
    + (euoption.r + 0.5 * euoption.sigma**2) * euoption.T
) / (euoption.sigma * np.sqrt(euoption.T))

    d2 = d1 - euoption.sigma * np.sqrt(euoption.T)

    call_price = (
    euoption.S0 * std_normal.cdf(d1)
    - euoption.K * np.exp(-euoption.r * euoption.T) * std_normal.cdf(d2)
)

    put_price = (
    euoption.K * np.exp(-euoption.r * euoption.T) * std_normal.cdf(-d2)
    - euoption.S0 * std_normal.cdf(-d1)
)

    return call_price, put_price

def monte_carlo(euoption: OptionParameters, N: int, M: int, seed: int) -> tuple[float,float]:
    rng = np.random.default_rng(seed=seed)

    delta_t = euoption.T / M
    sqrt_delta_t = np.sqrt(delta_t)

    #Create an array of length N of stock prices, all initially equal to S0
    S = np.full(N, float(euoption.S0))

    #Simulate N different possibilities of what the final stock price is (ST)
    for _ in range(M):
        Z = rng.standard_normal(N)
        S *= np.exp(
            (euoption.r - 0.5 * euoption.sigma**2) * delta_t + euoption.sigma * sqrt_delta_t * Z
        )

    #Calculate option payoffs based off of the final stock prices
    call_payoff = np.maximum(S - euoption.K, 0)
    put_payoff = np.maximum(euoption.K - S, 0)

    #Discount option payoffs and return their mean
    discount = np.exp(-euoption.r * euoption.T)

    mc_call_price = discount * np.mean(call_payoff)    
    mc_put_price = discount * np.mean(put_payoff)

    return mc_call_price, mc_put_price

def compute_convergence(simulation: SimulationParameters, euoption: OptionParameters, call_bsm: float, put_bsm: float) -> np.ndarray:
    convergence_results = []

    
    for N in simulation.N_values:
        call_errors = []
        put_errors = []

        call_prices = []
        put_prices = []

        for repetition in range(simulation.convergence_repetitions):
            #ensure a new seed is used for each monte carlo simulation
            seed = simulation.seed + repetition
            #compute call and put prices
            mc_call_price, mc_put_price = monte_carlo(euoption, N, M=1, seed=seed)

            #append prices to lists
            call_prices.append(mc_call_price)
            put_prices.append(mc_put_price)

            #compute absolute error
            call_error = abs(mc_call_price - call_bsm)
            put_error = abs(mc_put_price - put_bsm)

            #append errors to lists
            call_errors.append(call_error)
            put_errors.append(put_error)

        #compute mean call and put errors
        mean_call_error = np.mean(call_errors)
        mean_put_error = np.mean(put_errors)

        #compute mean call and put prices
        mean_call_price = np.mean(call_prices)
        mean_put_price = np.mean(put_prices)

        #append one row to convergence_results
        convergence_results.append([N,
                                    mean_call_error,
                                    mean_put_error,
                                    mean_call_price,
                                    mean_put_price])
        
    #convert results to numpy array
    convergence_results = np.array(convergence_results)

    #calculate linear regression in log-log space, used for convergence
    N = convergence_results[:, 0]
    call_error_fit = convergence_results[:, 1]
    put_error_fit = convergence_results[:, 2]

    call_slope, _ = np.polyfit(np.log10(N), np.log10(call_error_fit), 1)
    put_slope, _ = np.polyfit(np.log10(N), np.log10(put_error_fit), 1)


    return convergence_results, call_slope, put_slope

def compute_complexity(simulation: SimulationParameters, euoption: OptionParameters) -> np.ndarray:
    complexity_results = []

    #calculate runtime
    for M in simulation.M_values:
        for N in simulation.N_values:
            runtimes = []

            #average runtimes to get a smoother result
            for repetition in range(simulation.complexity_repetitions):
                seed = simulation.seed + repetition
                start_time = time.perf_counter()
                monte_carlo(euoption, N, M, seed)
                stop_time = time.perf_counter()

                runtimes.append(stop_time - start_time)

            median_runtime = np.median(runtimes)
            complexity_results.append([N, M, median_runtime])

    #convert results to numpy array
    complexity_results = np.array(complexity_results)

    return complexity_results

def plot_error(convergence_results: np.ndarray, call_slope: float, put_slope: float):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4.5), constrained_layout=True)
    #Configure call option figure
    ax1.plot(convergence_results[:, 0], convergence_results[:, 1], color='#008080', lw=2)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_title('European Call Option')
    ax1.set_xlabel(r'Number of Simulations ($N$)')
    ax1.set_ylabel('Absolute Error')
    ax1.minorticks_off()
    ##Emperical results for slope
    ax1.text(0.95,
             0.95,
             f"Slope: {call_slope:.3f}",
             transform=ax1.transAxes,
             ha='right',
             va='top',
             bbox=dict(facecolor='white', alpha=0.8)
    )
    



    #Configure put option figure
    ax2.plot(convergence_results[:, 0], convergence_results[:, 2], color='#E06666', lw=2)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_title('European Put Option')
    ax2.set_xlabel(r'Number of Simulations ($N$)')
    ax2.set_ylabel('Absolute Error')
    ax2.minorticks_off()
    ##Emperical results for slope
    ax2.text(0.95,
             0.95,
             f"Slope: {put_slope:.3f}",
             transform=ax2.transAxes,
             ha='right',
             va='top',
             bbox=dict(facecolor='white', alpha=0.8)
    )

    #Plot generation
    fig.suptitle(r'Monte Carlo Error Convergence ($M=1$)', fontsize=22)
    plt.savefig('plot_error.png', dpi=300)
    plt.close(fig)

def plot_convergence(convergence_results: np.ndarray, call_bsm: float, put_bsm: float):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4.5), constrained_layout=True)

    #Call option subplot
    ax1.plot(convergence_results[:, 0], convergence_results[:, 3], color='#008080', lw=2)
    ax1.set_xscale('log')
    ax1.set_title('European Call Option')
    ax1.set_xlabel(r'Number of Simulations ($N$)')
    ax1.set_ylabel('Mean Price ($)')
    ax1.minorticks_off()
    #add bsm price
    ax1.axhline(y=call_bsm, color='black', linestyle='--', linewidth=1.5)
    

    #Put option subplot
    ax2.plot(convergence_results[:, 0], convergence_results[:, 4], color='#E06666', lw=2)
    ax2.set_xscale('log')
    ax2.set_title('European Put Option')
    ax2.set_xlabel(r'Number of Simulations ($N$)')
    ax2.set_ylabel('Mean Price ($)')
    ax2.minorticks_off()
    #add bsm price
    ax2.axhline(y=put_bsm, color='black', linestyle='--', linewidth=1.5)

    #Plot generation
    fig.suptitle(r'Monte Carlo Price Convergence ($M=1$)', fontsize=22)
    fig.text(0.95,
             0.95,
             'BSM: --',
             color='black',
             fontweight='bold', 
             ha='right',
             va='top',
             bbox=dict(facecolor='white', alpha=0.8) 
    )
    plt.savefig('plot_convergence.png', dpi=300)
    plt.close(fig)

def plot_runtime(complexity_results: np.ndarray, simulation: SimulationParameters):
    fig, ax = plt.subplots(figsize=(12,4.5), constrained_layout=True)

    #Hardcoded, but add extra colors for every potential new M value
    colors = {
        1: '#D3D3D3',
        10: '#708090',
        100: '#222222'
    }

    #plot lines for each M value
    for M in simulation.M_values:
        mask = complexity_results[:, 1] == M

        ax.plot(
            complexity_results[mask, 0],
            complexity_results[mask, 2],
            color=colors[M],
            lw=2,
            label=fr"$M={M}$",
        )

    #plot settings
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Number of Simulations ($N$)')
    ax.set_ylabel(r'Execution time ($s$)')
    ax.minorticks_off()

    #legend
    ax.legend(title='Time Steps', frameon=False, loc='upper left')

    #plot
    fig.suptitle(r'Computation Time', fontsize=22)
    plt.savefig('plot_time.png', dpi=300)
    plt.close(fig)



if __name__ == "__main__":
    main()