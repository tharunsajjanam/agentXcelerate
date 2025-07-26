# Artha-X Financial Agent Application (arthax)

Artha-X is a web-based financial agent application designed to provide a user interface for financial interactions and data presentation. It serves as the front-end component of a larger financial management system, handling user requests and displaying processed information.


## Features

* **Web-based Interface**: Provides an accessible user interface through a standard web browser.
* **User Interaction & Query Handling**: Designed to capture and process user inputs and specific requests related to financial operations.
* **Dynamic Data Presentation**: Displays real-time or historical financial data, visualized through charts and tables, and presents analysis results in an intuitive format.
* **Voice-Based Interaction**: Enables users to interact with the application using voice commands.
    * **Multilingual Support**: Processes and responds to financial queries in multiple languages.
    * **Natural Language Understanding (NLU)**: Interprets complex financial questions posed in natural language.
* **Backend Service Integration**: Seamlessly connects with robust backend services (e.g., `fi-mcp-dev`) for executing high-volume financial computations, accessing core financial logic, and retrieving data.
* **Modular Architecture**: Structured for easy extension and integration of new features and financial modules.

## Next Scope of Features (Future Enhancements)

* **Personalized Financial Insights**: Offer tailored advice and recommendations based on individual user spending patterns, investment goals, and risk profiles.
* **Predictive Analytics**: Implement models for forecasting future financial trends, budget adherence, and potential investment outcomes.
* **Advanced Data Visualization**: Incorporate more interactive and customizable dashboards for deeper financial data exploration.
* **Integration with External Financial APIs**: Connect with third-party banking, investment, or market data APIs to provide a holistic financial view.
* **Transaction Categorization & Analysis**: Automatically categorize transactions and provide detailed spending analysis, identifying areas for optimization.
* **Goal Tracking & Planning**: Allow users to set financial goals (e.g., savings, debt repayment) and track their progress, offering actionable plans.
* **Security Enhancements**: Implement advanced authentication (e.g., MFA) and data encryption measures to ensure robust security for sensitive financial information.
* **Notifications & Alerts**: Provide timely alerts for unusual activity, budget overruns, bill reminders, or market changes.
* **Mobile Responsiveness**: Optimize the web interface for seamless use across various devices including smartphones and tablets.


## Project Structure

The `arthax` directory is organized as follows:

```
arthax/
├── app.py              # The main Flask application entry point.
├── requirements.txt    # Lists all Python dependencies for the application.
└── templates/          # Contains Jinja2 HTML templates for the web interface.
    └── index.html      # The main HTML template for the application's home page.
```


## Technologies Used

* **Python**: The core programming language for the application logic.
* **Flask**: A lightweight micro web framework used to build the web application's API and serve web pages.
* **Jinja2**: The templating engine employed for rendering dynamic HTML content within the Flask application.
* **HTML/CSS/JavaScript**: Standard front-end technologies for structuring, styling, and adding interactivity to the user interface.
* **Google Cloud Platform (GCP)**: The cloud infrastructure hosting and supporting various services.
* **Vertex AI**: Google Cloud's machine learning platform, likely used for:
    * **Natural Language Processing (NLP)**: For understanding user queries.
    * **Custom Models**: For financial forecasting, risk assessment, or other data analysis.
* **Other Potential GCP Services (Implied/Common for such apps)**:
    * **Cloud Run / App Engine**: For deploying and scaling the `arthax` web service.
    * **Cloud SQL / Firestore**: For database management and storing financial data.
    * **Cloud Functions**: For serverless backend logic or API interactions.
    * **Pub/Sub**: For asynchronous communication between services.

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

6.  **Configure Environment Variables (`.env` file):**
    Create a `.env` file in the `arthax/` directory to store your Google Cloud project configuration. This file is typically ignored by Git (as specified in the `.gitignore` file) for security reasons and should **not** be committed to version control.

    ```bash
  
    nano .env
    ```
    Change the following lines to the `.env` file, replacing `YOUR_PROJECT_ID` with your actual Google Cloud Project ID and `YOUR_GCP_REGION` with the desired GCP region (e.g., `us-central1`, `asia-south1`):

    ```
    PROJECT_ID=YOUR_PROJECT_ID
    LOCATION=YOUR_GCP_REGION
    ```
    Save and exit `nano` (`Ctrl+O`, Enter, `Ctrl+X`).

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
    This is for hackathon use only, not for public usage.
