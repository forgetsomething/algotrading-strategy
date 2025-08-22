# Multi-Objective Optimization with NSGA-III

NSGA-III (Non-dominated Sorting Genetic Algorithm III) represents a significant advancement in multi-objective optimization, specifically designed to handle many-objective optimization problems (typically 4+ objectives). In algorithmic trading, this is particularly valuable for portfolio optimization where we need to balance multiple competing objectives simultaneously.

## Introduction to NSGA-III

### Why NSGA-III?

Traditional optimization approaches often fail in trading because they focus on single objectives (like maximizing returns). Real-world trading requires balancing:

- **Return Maximization**
- **Risk Minimization** 
- **Drawdown Control**
- **Sharpe Ratio Optimization**
- **Volatility Management**
- **Transaction Cost Minimization**

NSGA-III excels at finding trade-offs between these competing objectives.

## Theoretical Foundation

### Key Concepts

1. **Pareto Optimality**: Solutions where improving one objective requires worsening another
2. **Reference Directions**: Systematic way to maintain diversity in high-dimensional objective spaces
3. **Non-dominated Sorting**: Ranking solutions based on dominance relationships

```python
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.core.problem import Problem

class PortfolioOptimizationProblem(Problem):
    """
    Multi-objective portfolio optimization problem using NSGA-III
    """
    
    def __init__(self, 
                 returns_data,
                 transaction_costs=0.001,
                 n_objectives=4):
        
        self.returns_data = returns_data
        self.n_assets = returns_data.shape[1]
        self.transaction_costs = transaction_costs
        self.cov_matrix = returns_data.cov().values
        self.mean_returns = returns_data.mean().values
        
        # Define the optimization problem
        super().__init__(
            n_var=self.n_assets,  # Number of assets (decision variables)
            n_obj=n_objectives,   # Number of objectives
            n_constr=1,           # Portfolio weights sum to 1
            xl=0.0,               # Lower bound (no short selling)
            xu=1.0                # Upper bound (max 100% in any asset)
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate portfolio for multiple objectives
        """
        n_portfolios = X.shape[0]
        
        # Normalize weights to sum to 1
        weights = X / X.sum(axis=1, keepdims=True)
        
        # Calculate objectives for each portfolio
        objectives = np.zeros((n_portfolios, self.n_obj))
        constraints = np.zeros((n_portfolios, 1))
        
        for i, w in enumerate(weights):
            # Objective 1: Negative expected return (minimize negative = maximize positive)
            expected_return = np.dot(w, self.mean_returns)
            objectives[i, 0] = -expected_return
            
            # Objective 2: Portfolio volatility (minimize)
            portfolio_variance = np.dot(w, np.dot(self.cov_matrix, w))
            portfolio_volatility = np.sqrt(portfolio_variance * 252)  # Annualized
            objectives[i, 1] = portfolio_volatility
            
            # Objective 3: Maximum drawdown (minimize)
            portfolio_returns = np.dot(self.returns_data.values, w)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.abs(drawdown.min())
            objectives[i, 2] = max_drawdown
            
            # Objective 4: Transaction costs (minimize)
            # Assuming equal initial weights and calculating turnover
            initial_weights = np.ones(self.n_assets) / self.n_assets
            turnover = np.sum(np.abs(w - initial_weights))
            transaction_cost = turnover * self.transaction_costs
            objectives[i, 3] = transaction_cost
            
            # Constraint: weights sum to 1 (should be 0 for feasible solutions)
            constraints[i, 0] = abs(w.sum() - 1.0)
        
        out["F"] = objectives
        out["G"] = constraints
```

## Advanced Implementation

### Complete NSGA-III Portfolio Optimizer

```python
class NSGA3PortfolioOptimizer:
    def __init__(self, 
                 returns_data,
                 transaction_costs=0.001,
                 population_size=100,
                 n_generations=200):
        
        self.returns_data = returns_data
        self.transaction_costs = transaction_costs
        self.population_size = population_size
        self.n_generations = n_generations
        
        # Initialize the optimization problem
        self.problem = PortfolioOptimizationProblem(
            returns_data=returns_data,
            transaction_costs=transaction_costs,
            n_objectives=4
        )
        
        # Configure NSGA-III algorithm
        self.algorithm = NSGA3(
            pop_size=population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=1.0/self.problem.n_var, eta=20),
            eliminate_duplicates=True
        )
        
        self.result = None
        self.pareto_front = None
    
    def optimize(self, verbose=True):
        """
        Run the NSGA-III optimization
        """
        if verbose:
            print("Starting NSGA-III optimization...")
            print(f"Population size: {self.population_size}")
            print(f"Generations: {self.n_generations}")
            print(f"Number of assets: {self.problem.n_assets}")
        
        # Run optimization
        self.result = minimize(
            self.problem,
            self.algorithm,
            ('n_gen', self.n_generations),
            verbose=verbose
        )
        
        # Extract Pareto front
        self.pareto_front = self.result.F
        self.pareto_solutions = self.result.X
        
        if verbose:
            print(f"Optimization completed!")
            print(f"Number of Pareto optimal solutions: {len(self.pareto_front)}")
        
        return self.result
    
    def analyze_pareto_front(self):
        """
        Analyze the Pareto front solutions
        """
        if self.pareto_front is None:
            raise ValueError("Run optimization first!")
        
        # Convert to DataFrame for easier analysis
        objectives_df = pd.DataFrame(
            self.pareto_front,
            columns=['Negative_Return', 'Volatility', 'Max_Drawdown', 'Transaction_Costs']
        )
        
        # Convert negative return back to positive
        objectives_df['Expected_Return'] = -objectives_df['Negative_Return']
        objectives_df = objectives_df.drop('Negative_Return', axis=1)
        
        # Calculate Sharpe ratios for each solution
        objectives_df['Sharpe_Ratio'] = (
            objectives_df['Expected_Return'] / objectives_df['Volatility']
        )
        
        # Normalize weights for solutions
        normalized_weights = self.pareto_solutions / self.pareto_solutions.sum(axis=1, keepdims=True)
        
        weights_df = pd.DataFrame(
            normalized_weights,
            columns=[f'Asset_{i+1}' for i in range(self.problem.n_assets)]
        )
        
        return objectives_df, weights_df
    
    def select_portfolio(self, preference_weights=None):
        """
        Select a single portfolio from the Pareto front based on preferences
        """
        if preference_weights is None:
            # Default: equal weights for all objectives (except negative return)
            preference_weights = np.array([0.4, 0.2, 0.2, 0.2])  # Return, Vol, DD, TC
        
        objectives_df, weights_df = self.analyze_pareto_front()
        
        # Normalize objectives for comparison
        normalized_objectives = objectives_df.copy()
        for col in ['Volatility', 'Max_Drawdown', 'Transaction_Costs']:
            # For objectives to minimize, lower is better
            normalized_objectives[col] = (
                1 - (objectives_df[col] - objectives_df[col].min()) / 
                (objectives_df[col].max() - objectives_df[col].min())
            )
        
        # For return, higher is better
        normalized_objectives['Expected_Return'] = (
            (objectives_df['Expected_Return'] - objectives_df['Expected_Return'].min()) /
            (objectives_df['Expected_Return'].max() - objectives_df['Expected_Return'].min())
        )
        
        # Calculate utility scores
        utility_scores = (
            normalized_objectives['Expected_Return'] * preference_weights[0] +
            normalized_objectives['Volatility'] * preference_weights[1] +
            normalized_objectives['Max_Drawdown'] * preference_weights[2] +
            normalized_objectives['Transaction_Costs'] * preference_weights[3]
        )
        
        # Select portfolio with highest utility
        best_idx = utility_scores.idxmax()
        
        selected_portfolio = {
            'weights': weights_df.iloc[best_idx].to_dict(),
            'objectives': objectives_df.iloc[best_idx].to_dict(),
            'utility_score': utility_scores.iloc[best_idx]
        }
        
        return selected_portfolio, best_idx
```

## Advanced Multi-Objective Strategies

### Dynamic Objective Weighting

```python
class DynamicNSGA3Optimizer:
    """
    NSGA-III optimizer with dynamic objective weighting based on market conditions
    """
    
    def __init__(self, returns_data, market_regime_data):
        self.returns_data = returns_data
        self.market_regime_data = market_regime_data
        self.regime_optimizers = {}
        
    def train_regime_specific_optimizers(self):
        """
        Train separate optimizers for different market regimes
        """
        unique_regimes = self.market_regime_data.unique()
        
        for regime in unique_regimes:
            # Filter data for this regime
            regime_mask = self.market_regime_data == regime
            regime_returns = self.returns_data[regime_mask]
            
            if len(regime_returns) > 50:  # Minimum data requirement
                # Create regime-specific optimizer
                optimizer = NSGA3PortfolioOptimizer(
                    returns_data=regime_returns,
                    population_size=80,
                    n_generations=150
                )
                
                # Optimize for this regime
                optimizer.optimize(verbose=False)
                
                self.regime_optimizers[regime] = optimizer
                
                print(f"Trained optimizer for regime {regime} with {len(regime_returns)} samples")
    
    def get_regime_adaptive_portfolio(self, current_regime, preference_weights=None):
        """
        Get portfolio optimized for current market regime
        """
        if current_regime not in self.regime_optimizers:
            raise ValueError(f"No optimizer trained for regime {current_regime}")
        
        optimizer = self.regime_optimizers[current_regime]
        portfolio, idx = optimizer.select_portfolio(preference_weights)
        
        return portfolio
```

### Real-time Portfolio Rebalancing

```python
class RealTimeNSGA3Rebalancer:
    """
    Real-time portfolio rebalancing using NSGA-III
    """
    
    def __init__(self, initial_weights, returns_data, rebalance_threshold=0.05):
        self.current_weights = initial_weights
        self.returns_data = returns_data
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_history = []
        
    def should_rebalance(self, current_prices, target_weights):
        """
        Determine if rebalancing is needed
        """
        # Calculate current actual weights based on price movements
        current_values = current_prices * self.current_weights
        total_value = current_values.sum()
        actual_weights = current_values / total_value
        
        # Calculate weight drift
        weight_drift = np.abs(actual_weights - target_weights).max()
        
        return weight_drift > self.rebalance_threshold
    
    def rebalance_portfolio(self, market_data_window, transaction_cost=0.001):
        """
        Rebalance portfolio using NSGA-III optimization
        """
        # Create optimizer with recent data
        optimizer = NSGA3PortfolioOptimizer(
            returns_data=market_data_window,
            transaction_costs=transaction_cost,
            population_size=60,
            n_generations=100
        )
        
        # Optimize
        result = optimizer.optimize(verbose=False)
        
        # Select new portfolio considering current weights
        new_portfolio, _ = optimizer.select_portfolio()
        
        # Calculate rebalancing trades
        new_weights = np.array(list(new_portfolio['weights'].values()))
        trades = new_weights - self.current_weights
        
        # Store rebalancing information
        rebalance_info = {
            'timestamp': pd.Timestamp.now(),
            'old_weights': self.current_weights.copy(),
            'new_weights': new_weights,
            'trades': trades,
            'expected_return': new_portfolio['objectives']['Expected_Return'],
            'volatility': new_portfolio['objectives']['Volatility'],
            'max_drawdown': new_portfolio['objectives']['Max_Drawdown']
        }
        
        self.rebalance_history.append(rebalance_info)
        self.current_weights = new_weights
        
        return rebalance_info
```

## Performance Analysis and Visualization

### Comprehensive Performance Analytics

```python
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class NSGA3PerformanceAnalyzer:
    """
    Comprehensive analysis tools for NSGA-III optimization results
    """
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.objectives_df, self.weights_df = optimizer.analyze_pareto_front()
    
    def plot_pareto_front_3d(self, objectives_indices=[0, 1, 2]):
        """
        Plot 3D Pareto front
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        obj_data = self.optimizer.pareto_front
        
        # Plot Pareto front
        scatter = ax.scatter(
            obj_data[:, objectives_indices[0]],
            obj_data[:, objectives_indices[1]], 
            obj_data[:, objectives_indices[2]],
            c=self.objectives_df['Sharpe_Ratio'],
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        
        ax.set_xlabel('Expected Return')
        ax.set_ylabel('Volatility')
        ax.set_zlabel('Max Drawdown')
        ax.set_title('3D Pareto Front (Colored by Sharpe Ratio)')
        
        plt.colorbar(scatter, label='Sharpe Ratio')
        plt.tight_layout()
        plt.show()
    
    def plot_parallel_coordinates(self):
        """
        Plot parallel coordinates for all objectives
        """
        plt.figure(figsize=(14, 8))
        
        # Normalize data for parallel coordinates
        normalized_data = self.objectives_df.copy()
        for col in normalized_data.columns:
            if col != 'Sharpe_Ratio':
                normalized_data[col] = (
                    (normalized_data[col] - normalized_data[col].min()) /
                    (normalized_data[col].max() - normalized_data[col].min())
                )
        
        # Create parallel coordinates plot
        from pandas.plotting import parallel_coordinates
        parallel_coordinates(
            normalized_data.reset_index(),
            'index',
            colormap='viridis',
            alpha=0.6
        )
        
        plt.title('Parallel Coordinates Plot of Pareto Solutions')
        plt.xlabel('Objectives')
        plt.ylabel('Normalized Values')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def analyze_portfolio_characteristics(self):
        """
        Analyze characteristics of Pareto optimal portfolios
        """
        # Portfolio concentration analysis
        portfolio_concentrations = []
        for i in range(len(self.weights_df)):
            weights = self.weights_df.iloc[i].values
            # Calculate Herfindahl-Hirschman Index
            hhi = np.sum(weights**2)
            portfolio_concentrations.append(hhi)
        
        self.objectives_df['Portfolio_Concentration'] = portfolio_concentrations
        
        # Risk-return efficient frontier
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Risk-Return scatter
        plt.subplot(2, 3, 1)
        scatter = plt.scatter(
            self.objectives_df['Volatility'],
            self.objectives_df['Expected_Return'],
            c=self.objectives_df['Sharpe_Ratio'],
            cmap='RdYlGn',
            s=50,
            alpha=0.7
        )
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        plt.title('Risk-Return Efficient Frontier')
        plt.colorbar(scatter, label='Sharpe Ratio')
        
        # Subplot 2: Drawdown vs Return
        plt.subplot(2, 3, 2)
        plt.scatter(
            self.objectives_df['Max_Drawdown'],
            self.objectives_df['Expected_Return'],
            c=self.objectives_df['Volatility'],
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        plt.xlabel('Max Drawdown')
        plt.ylabel('Expected Return')
        plt.title('Drawdown vs Return')
        
        # Subplot 3: Portfolio concentration
        plt.subplot(2, 3, 3)
        plt.hist(portfolio_concentrations, bins=20, alpha=0.7, color='skyblue')
        plt.xlabel('Portfolio Concentration (HHI)')
        plt.ylabel('Frequency')
        plt.title('Portfolio Concentration Distribution')
        
        # Subplot 4: Transaction costs vs Return
        plt.subplot(2, 3, 4)
        plt.scatter(
            self.objectives_df['Transaction_Costs'],
            self.objectives_df['Expected_Return'],
            c=self.objectives_df['Sharpe_Ratio'],
            cmap='plasma',
            s=50,
            alpha=0.7
        )
        plt.xlabel('Transaction Costs')
        plt.ylabel('Expected Return')
        plt.title('Transaction Costs vs Return')
        
        # Subplot 5: Correlation matrix of objectives
        plt.subplot(2, 3, 5)
        corr_matrix = self.objectives_df[['Expected_Return', 'Volatility', 'Max_Drawdown', 'Transaction_Costs']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Objectives Correlation Matrix')
        
        # Subplot 6: Sharpe ratio distribution
        plt.subplot(2, 3, 6)
        plt.hist(self.objectives_df['Sharpe_Ratio'], bins=20, alpha=0.7, color='lightgreen')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Frequency')
        plt.title('Sharpe Ratio Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return self.objectives_df
```

## Practical Implementation Example

### Complete Trading System Integration

```python
def implement_nsga3_trading_system():
    """
    Complete implementation example
    """
    
    # 1. Load market data
    import yfinance as yf
    
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    
    print(f"Loaded data for {len(tickers)} assets from {start_date} to {end_date}")
    print(f"Returns data shape: {returns.shape}")
    
    # 2. Initialize and run NSGA-III optimization
    optimizer = NSGA3PortfolioOptimizer(
        returns_data=returns,
        transaction_costs=0.001,
        population_size=100,
        n_generations=200
    )
    
    # Run optimization
    result = optimizer.optimize()
    
    # 3. Analyze results
    analyzer = NSGA3PerformanceAnalyzer(optimizer)
    
    # Plot results
    analyzer.plot_pareto_front_3d()
    analyzer.plot_parallel_coordinates()
    portfolio_analysis = analyzer.analyze_portfolio_characteristics()
    
    # 4. Select portfolio based on preferences
    # Conservative investor: emphasize low risk
    conservative_weights = np.array([0.2, 0.4, 0.3, 0.1])  # Return, Vol, DD, TC
    conservative_portfolio, _ = optimizer.select_portfolio(conservative_weights)
    
    # Aggressive investor: emphasize returns
    aggressive_weights = np.array([0.6, 0.2, 0.1, 0.1])
    aggressive_portfolio, _ = optimizer.select_portfolio(aggressive_weights)
    
    print("\n=== CONSERVATIVE PORTFOLIO ===")
    print(f"Expected Return: {conservative_portfolio['objectives']['Expected_Return']:.4f}")
    print(f"Volatility: {conservative_portfolio['objectives']['Volatility']:.4f}")
    print(f"Max Drawdown: {conservative_portfolio['objectives']['Max_Drawdown']:.4f}")
    print(f"Sharpe Ratio: {conservative_portfolio['objectives']['Sharpe_Ratio']:.4f}")
    print("Weights:")
    for asset, weight in conservative_portfolio['weights'].items():
        print(f"  {asset}: {weight:.3f}")
    
    print("\n=== AGGRESSIVE PORTFOLIO ===")
    print(f"Expected Return: {aggressive_portfolio['objectives']['Expected_Return']:.4f}")
    print(f"Volatility: {aggressive_portfolio['objectives']['Volatility']:.4f}")
    print(f"Max Drawdown: {aggressive_portfolio['objectives']['Max_Drawdown']:.4f}")
    print(f"Sharpe Ratio: {aggressive_portfolio['objectives']['Sharpe_Ratio']:.4f}")
    print("Weights:")
    for asset, weight in aggressive_portfolio['weights'].items():
        print(f"  {asset}: {weight:.3f}")
    
    return optimizer, analyzer, conservative_portfolio, aggressive_portfolio

if __name__ == "__main__":
    optimizer, analyzer, conservative, aggressive = implement_nsga3_trading_system()
```

## Key Advantages of NSGA-III

1. **Many-Objective Optimization**: Handles 4+ objectives effectively
2. **Reference Direction Diversity**: Maintains solution diversity in high-dimensional spaces
3. **Computational Efficiency**: Scales better than NSGA-II for many objectives
4. **Parameter Robustness**: Less sensitive to parameter tuning
5. **Real-world Applicability**: Practical for complex portfolio optimization

## Next Steps

In the following chapters, we'll explore:

- [AGE-MOEA2](age-moea2.md): Alternative multi-objective approach
- [Portfolio Implementation](../portfolio/multi-objective.md): Practical portfolio construction
- [Risk Management Integration](../../management/risk/model-based-var.md): Risk-aware optimization

---

**Previous**: [Single Objective Optimization ←](../single-objective/ga-pso-es.md)

**Next**: [AGE-MOEA2 →](age-moea2.md)

**Related Topics**:
- [Deep Learning Portfolio Optimization](../portfolio/deep-learning.md)
- [Risk Management](../../management/risk/model-based-var.md)
