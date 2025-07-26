# Artha-X Financial Agent Application (arthax)

Artha-X is a web-based financial agent application designed to provide a user interface for financial interactions and data presentation. It serves as the front-end component of a larger financial management system, handling user requests and displaying processed information.

## Features

* **Web-based Interface**: Accessible via a browser.
* **User Interaction**: Designed to handle user inputs and requests for financial operations.
* **Data Presentation**: Displays financial data and analysis results in a user-friendly format.
* **Modular Design**: Structured to integrate with backend services for complex financial logic.
* **voice based** : speaks multilingual and answers your questions from finance.

## Project Structure

The `arthax` directory is organized as follows:

arthax/
├── app.py              # The main Flask application entry point.
├── requirements.txt    # Lists all Python dependencies for the application.
└── templates/          # Contains Jinja2 HTML templates for the web interface.
    └── index.html      # The main HTML template for the application's home page.


## Technologies Used

* **Python**: The core programming language.
* **Flask**: A micro web framework for building the web application.
* **Jinja2**: Templating engine for rendering dynamic HTML content.
* **HTML/CSS/JavaScript**: For front-end structure, styling, and interactivity.

## Setup and Installation

To set up and run the `arthax` application locally, follow these steps:

1.  **Clone the Repository:**
    If you haven't already, clone the `agentXcelerate` repository which contains the `arthax` application:
    ```bash
    git clone https://github.com/tharunsajjanam/agentXcelerate
    cd agentXcelerate
    ```

2.  **Navigate to the `arthax` directory:**
    ```bash
    cd arthax
    ```

3.  **Create a Python Virtual Environment (Recommended):**
    It's best practice to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv venv
    ```

4.  **Activate the Virtual Environment:**
    * On Linux/macOS:
        ```bash
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

5.  **Install Dependencies:**
    Install all required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```


## Prerequisites: Setting Up fi-mcp-dev

The `arthax` application relies on the `fi-mcp-dev` service for its backend financial operations. You need to set up and run the developer version of `fi-mcp-dev` *before* starting the `arthax` application.

1.  **Clone the `fi-mcp-dev` repository:**
    Open a **new terminal window or tab** and navigate to your preferred development directory (e.g., `~/my_agentXcelerate_workspace` if that's where you cloned `agentXcelerate`, but outside the `agentXcelerate` folder itself if `fi-mcp-dev` is a separate service).
    ```bash
    git clone [https://github.com/epiFi/fi-mcp-dev.git](https://github.com/epiFi/fi-mcp-dev.git)
    cd fi-mcp-dev
    ```

2.  **Follow `fi-mcp-dev` setup instructions:**
    Refer to the `README.md` or any other documentation within the `fi-mcp-dev` repository for specific instructions on how to install its dependencies and run its service. This typically involves:
    * Installing required programming languages (e.g., Go, if it's a Go project).
    * Installing project-specific dependencies.
    * Configuring environment variables or configuration files.
    * Running the `fi-mcp-dev` service.

    *(Example commands, these might vary based on `fi-mcp-dev`'s actual setup):*
    ```bash
    # Example: If it's a Go project
    go mod download
    go run main.go
    # Or start a specific server
    ```
    Ensure `fi-mcp-dev` is running successfully and accessible by `arthax` (e.g., listening on a specific port).

## Usage

To run the `arthax` Flask application:

1.  **Ensure your virtual environment is activated.**
2.  **Set the Flask application environment variable:**
    ```bash
    export FLASK_APP=app.py
    ```
    (On Windows, use `set FLASK_APP=app.py`)

3.  **Run the Flask application:**
    ```bash
    flask run
    ```

    The application will typically be accessible at `http://127.0.0.1:5000/` in your web browser.

## HACKATHON
    this is for only hackathon not for any public usage. 
