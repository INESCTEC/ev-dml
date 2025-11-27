# Dynamic incentives for electric vehicles charging at supermarket stations: causal insights on demand flexibility

This repository is part of the research work for the paper "Dynamic incentives for electric vehicles charging at supermarket stations: causal insights on demand flexibility". The code provided here is intended to support the findings and conclusions presented in the paper. The code is written in Python and leverages various libraries for data processing, statistical analysis, and causal inference.

---

## Project Structure

- `estimate_effects_single.py`: Main script for estimating the causal effects of dynamic incentives on electric vehicle charging demand.
- `estimate_effects_neighboring.py`: Main script for estimating the causal effects of dynamic incentives on electric vehicle charging demand considering neighboring hours.
- `dataset_store1.csv`: CSV file with necessary data from an EV charging station (e.g., consumption, discounts) to execute the scripts.
- `requirements.txt`: File listing the required Python libraries and their versions.
- `LICENSE`: License information for the project.
- `README.md`: Overview and instructions for the project.

## Dependencies

The project requires a set of Python libraries to run the scripts successfully, which are available in the requirements.txt file.
You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

The scripts were tested with the following Python version:
```bash
Python 3.8.18
```

## Usage

1. **Data Preparation**: The script starts by loading and preprocessing the data, including removing missing values and scaling features.

2. **Simple Regression Approach**: It performs a simple regression analysis considering model diagnosis and interactions.

3. **Causal Model Estimation**: The script uses the DoWhy library to instantiate a causal model, identify the estimand, and estimate the causal effect using the `econml` library's `CausalForestDML` method.

4. **Model Evaluation**: The script evaluates the model's performance and computes the effect time series and marginal effects.

5. **Refutation Tests**: Finally, the script performs refutation tests to validate the causal estimates.

## Running the Script

To run the script, execute one of the following commands:

```bash
python estimate_effects_single.py
```

```bash
python estimate_effects_neighboring.py
```

## Output

The script generates various debug logs and outputs, including:

- OLS regression summary.
- Causal effect estimates and model scores.
- Effect time series and marginal effects.
- Results of refutation tests.

## License

See the LICENSE file for more details.

## Contact

For any questions or inquiries, please contact the authors of the paper.

- Carlos A. M. Silva (carlos.silva@inesctec.pt)
- José R. Andrade (jose.r.andrade@inesctec.pt)
- Ricardo J. Bessa (ricardo.j.bessa@inesctec.pt)

---
