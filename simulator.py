class MarketSimulator():
    """Simulate the financial market for a given strategy"""

    def __init__(self, asset_returns, costs, market_volumes):
        self.asset_returns = asset_returns
        self.transaction_costs = transaction_costs
        self.holding_costs = holding_costs
        self.market_volumes = market_volumes

