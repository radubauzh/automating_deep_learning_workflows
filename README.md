# Automating Deep Learning Workflows
Automating Deep Learning Workflows: Parameter Management and LLM-based Result Interpretation

## Installation

1. Clone this repository.
2. (Optional but recommended) Create and activate a virtual environment:
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

- To start the GUI, run:>:
   ```
   python GUI.py
   ```
## Additional Notes

- GPU usage: Ensure you install a PyTorch version compatible with your CUDA setup.
- If you encounter any issues, please check your Python and package versions, or review file paths.

## License and Support

- This project is licensed under the MIT License.
- For support, open an issue or pull request.
