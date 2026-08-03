import numpy as np
import pandas as pd

from research.labels.events import drawdown_class


def test_drawdown_severity_boundaries() -> None:
    thresholds = {1: 0.03, 2: 0.05, 3: 0.08, 4: 0.12}
    values = pd.Series([-0.01, -0.03, -0.051, -0.08, -0.20, np.nan])
    assert drawdown_class(values, thresholds).tolist() == [0, 1, 2, 3, 4, 0]
