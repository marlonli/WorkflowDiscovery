# WorkflowDiscovery
An algorithm for discovering interpretable medical workflow models from an alignment matrix.
## Requirements
- Python 3.6
- [Pandas](https://pandas.pydata.org/)
- [graphviz](https://pypi.org/project/graphviz/)
## Example usage
A sample input CSV file is given in the reporsitory.
```python
import workflowModel as workflow
threshold = 0.5
span = "max"
#span = 5
workflow.model("Synthetic_activityTraces_1000.csv", threshold, span)
```
We use [Process-oriented Iterative Multiple Alignment(PIMA)](https://arxiv.org/pdf/1709.05440.pdf) algorithm to perform trace alignment.