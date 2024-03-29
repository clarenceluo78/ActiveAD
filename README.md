# ActiveAD: Active Learning for Tabular Anomaly Detection

Detecting anomalies in tabular data is critical in many fields, including cybersecurity, finance, and healthcare. However, labeling data for anomaly detection is often labor-intensive and costly. *Active learning* (AL) emerges as a promising approach to mitigate these challenges, aiming to reduce the labeling cost while maintaining high detection performance. In our project, we propose a pipeline, ActiveAD, for active anomaly detection that combines various anomaly detection models and active learning querying strategies to improve the efficiency and effectiveness of identifying anomalies with limited labeled data. Benefiting from the recent study on anomaly detection benchmark, we also offer a comprehensive comparison of different active learning method performance on diverse datasets. Extensive experiments reveal the strengths and weaknesses of each method and the impact of outliers, providing valuable insights into their suitability under different conditions. 

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project aims to combine the strengths of machine learning algorithms and human domain knowledge to create a more efficient and accurate anomaly detection system. ActiveAD focuses on the following aspects:

- Active learning with human feedback
- Guided instance selection
- Interactive model refinement
- Explainable AI integration
- Evaluation and validation
- Ethical considerations

## Installation

To set up the ActiveAD project, follow these steps:

1. Clone the repository
2. Change to the project directory
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage

Provide a brief description of how to use your ActiveAD project, including any relevant code examples or command-line instructions.

## Experiments

Describe the experiments you've conducted using the ActiveAD framework, including:

- Dataset(s) used
- Anomaly detection models/algorithms
- Active learning strategies
- Human-in-the-loop setup and collaboration

## Results

Summarize the results of your experiments, highlighting key findings such as:

- Improvement in detection performance
- Reduction in the number of required queries
- Efficiency of human-AI collaboration
- Comparison with baseline or state-of-the-art methods

## Contributing

If you'd like to contribute to the ActiveAD project, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b your-feature-branch`)
3. Commit your changes (`git commit -m 'Add your feature description'`)
4. Push your branch (`git push origin your-feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the [MIT License](LICENSE.md).

## Acknowledgments