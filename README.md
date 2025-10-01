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

### Using pip

```bash
pip install -i https://test.pypi.org/simple/ fair-partition
```

### From source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fair-partition.git
cd fair-partition
```

2. Build and install the package:
```bash
make build
```

3. Import newly-built local version of the package:
```bash
pip install dist/fair_partition-$(VERSION)-py3-none-any.whl
```

### Example

Here's a quick example showing how to use the package to create fair groups and visualize the results with confidence intervals:

```python
import numpy as np
from fair_partition.partition_estimation import FairGroups
from fair_partition.visualization import plot_partition_with_ci

# Create fair groups
fair_groups = FairGroups(nb_groups=5)
fair_groups.fit(s, y)

# Visualize the partition with confidence intervals
plot_partition_with_ci(fair_groups.partition, fair_groups.phi_by_group_ci, 'S')
```

This will create a plot showing:
- The partition boundaries on the x-axis
- The positive outcome rates for each group
- Confidence intervals for the positive outcome rates

---

## 🚧 Status: Work in Progress

This repository is currently being cleaned and finalized. The full code, documentation, and instructions will be made available shortly. In the meantime, feel free to reach out if you have questions.

---

## 📌 Note to Reviewers

Thank you for reviewing our work. If you encounter any issues or need clarification while evaluating the code, please don't hesitate to contact us (our contact information is available in the paper).

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
