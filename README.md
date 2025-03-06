# LLM Podcast Explorer

## Project Setup Guide

This guide will help you set up the LLM Podcast Explorer project using `uv`. Follow the steps below to get started.

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.8 or higher
- `uv` package
- `pip` (Python package installer)
- `git` (version control system)

### Installation Steps

1. **Clone the Repository**

    Open your terminal and run the following command to clone the repository:

    ```bash
    git clone https://github.com/yourusername/llm-podcast-explorer.git
    cd llm-podcast-explorer
    ```

2. **Create a Virtual Environment**

    It is recommended to create a virtual environment to manage dependencies. Run the following commands:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    Install the required packages using `uv`:

    ```bash
    uv pip install -r pyproject.toml
    ```

5. **Run the streamlit app**

    With static data:

    ```bash
    ANALYSIS_MODE=static streamlit run llm_podcast_explorer/streamlit_app.py
    ```

    Active analyis mode:
    
    Requires valid openai api key: https://platform.openai.com/account/api-keys

    ```bash
    ANALYSIS_MODE=active, OPENAI_API_KEY=<apikey> streamlit run llm_podcast_explorer/streamlit_app.py
    ```

    This will start the development server and you can access the application at `http://127.0.0.1:8501`.


### Contributing

If you would like to contribute to this project, please fork the repository and create a pull request. We welcome all contributions!

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

