# Automating Deep Learning Workflows
Automating Deep Learning Workflows: Parameter Management and LLM-based Result Interpretation

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/radubauzh/automating_deep_learning_workflows.git
   cd ../automating_deep_learning_workflows
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
3. Install all required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Environment Setup

1. Create a file named `.env` in the project root.  
2. Add your OpenAI API key to it:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
3. Adjust any file paths or environment variables as needed for your system.

## Usage

1. Choose a Conda environment name you wish to use (default is DL).

2. Start the GUI, run:
   ```
   python GUI.py
   ```

3. In the GUI, you can specify:
- Conda environment (default "DL"): Type the name of the environment you want to use.
- Other hyperparameters: batch size, learning rates, L2 lambdas, etc.
4. Click "Generate JSON and Run Script" to generate a config file and run main.py within the specified environment.


## Additional Notes

- GPU usage: Ensure you install a PyTorch version compatible with your CUDA setup.
- If you encounter any issues, please check your Python and package versions, or review file paths.

## License and Support

- This project is licensed under the MIT License.
- For support, open an issue or pull request.
