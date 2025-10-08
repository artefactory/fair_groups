# 📄 Fairness-aware Grouping for Continuous Sensitive Variables

This repository contains the code and experiments accompanying our paper:

**"Fairness-Aware Grouping for Continuous Sensitive Variables: Application for Debiasing Face Analysis with respect to Skin Tone"**  
*Accepted to ECAI 2025*

---

## 🔍 Overview

The main goal is to analyze fairness in computer vision models with respect to skin tone variations using a novel data-driven approach and then extend it to bias mitigation. 

See notebooks folder for full examples of the usage.

---

## 🎯 Usage

### From source

1. Clone the repository:
```bash
git clone git@github.com:artefactory/fair_groups.git
cd fair_groups
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

### Example

Here's a quick example showing how to use the package to create fair groups and visualize the results with confidence intervals:

```python
import numpy as np
from fair_groups.partition_estimation import FairGroups
from fair_groups.visualization import plot_partition_with_ci

# Create fair groups
fair_groups = FairGroups(n_groups=5)
fair_groups.fit(s, y)

# Visualize the partition with confidence intervals
plot_partition_with_ci(fair_groups.partition, fair_groups.phi_by_group_ci, 'S')
```

This code will create a plot showing:
- The partition boundaries on the x-axis
- The positive outcome rates for each group
- Confidence intervals for the positive outcome rates

---

## 📌 Note to Reviewers

Thank you for reviewing our work. If you encounter any issues or need clarification while evaluating the code, please don't hesitate to contact us (our contact information is available in the paper).

---

## Citation

If you consider this package or any of its feature useful for your research, consider citing our [paper](https://arxiv.org/abs/2507.11247).

```bibtex
@article{shilova2025fairness,
  title={Fairness-Aware Grouping for Continuous Sensitive Variables: Application for Debiasing Face Analysis with respect to Skin Tone},
  author={Shilova, Veronika and Malherbe, Emmanuel and Palma, Giovanni and Risser, Laurent and Loubes, Jean-Michel},
  journal={arXiv preprint arXiv:2507.11247},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
