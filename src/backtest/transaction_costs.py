"""
Transaction Cost Engine.
Computes rebalance costs based on portfolio turnover and configurable cost rate in basis points.
"""

class TransactionCostModel:
    """Calculates rebalancing transaction friction."""

    def __init__(self, cost_bps: float = 10.0):
        self.cost_bps = cost_bps
        self.cost_rate = cost_bps / 10000.0 # Convert bps to decimal (10 bps = 0.0010)

    def calculate_cost(self, turnover: float) -> float:
        """Calculate transaction cost as decimal return reduction."""
        return float(turnover * self.cost_rate)

    def calculate_net_return(self, gross_return: float, turnover: float) -> tuple[float, float]:
        """Return (net_return, transaction_cost)."""
        cost = self.calculate_cost(turnover)
        net = gross_return - cost
        return float(net), float(cost)
